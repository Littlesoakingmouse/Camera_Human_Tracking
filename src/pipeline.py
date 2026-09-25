"""Three-pass orchestration for offline multi-camera person tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .detector_tracker import DetectorTracker, TrackObservation
from .global_tracker import GlobalAssignment, GlobalTracker
from .intra_camera_matcher import (
    IntraCameraAssignment,
    IntraCameraMatcher,
    bbox_iou,
)
from .matcher import CrossCameraMatcher, similarity_dataframe
from .reid import ReIDModel
from .track_database import (
    LocalTrack,
    ObservationKey,
    SegmentKey,
    TrackDatabase,
    TrackSegment,
)
from .track_segmenter import SegmentationResult, TrackSegmenter
from .utils.common import ProjectError, project_path
from .utils.video import VideoReader, create_video_writer
from .visualizer import Visualizer


@dataclass(frozen=True)
class PipelineSummary:
    local_tracks_by_camera: dict[str, int]
    camera_persons_by_camera: dict[str, int]
    intra_camera_matches: int
    global_persons: int
    cross_camera_matches: int
    output_directory: Path


class MultiCameraTrackingPipeline:
    """Run local tracking, track repair, cross-camera matching, and replay."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.output_dir = project_path(config, config["output"]["output_directory"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detector_tracker = DetectorTracker(config)
        self.reid = ReIDModel(config)
        print(f"Using device: detector={self.detector_tracker.device}, reid={self.reid.device}")
        self.database = TrackDatabase()
        self.observations: dict[str, dict[int, list[TrackObservation]]] = {}
        self.camera_configs = self._camera_configs()
        self.frame_diagnostics: list[dict[str, Any]] = []

    def _camera_configs(self) -> list[dict[str, Any]]:
        cameras: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in self.config["video"].values():
            if not isinstance(entry, dict) or "id" not in entry or "path" not in entry:
                raise ProjectError("Each video entry must contain id and path.")
            camera = dict(entry)
            camera["id"] = str(camera["id"])
            if camera["id"] in seen_ids:
                raise ProjectError(f"Duplicate camera id in configuration: {camera['id']}")
            seen_ids.add(camera["id"])
            camera["resolved_path"] = project_path(self.config, camera["path"])
            camera["time_offset_seconds"] = float(camera.get("time_offset_seconds", 0.0))
            cameras.append(camera)
        if len(cameras) < 2:
            raise ProjectError("At least two cameras must be configured under video.")
        return cameras

    def _save_crop(self, camera_id: str, track_id: int, frame_index: int,
                   crop: np.ndarray) -> None:
        path = (self.output_dir / "crops" / camera_id / f"track_{track_id:04d}" /
                f"frame_{frame_index:06d}.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), crop):
            logging.warning("Could not save crop: %s", path)

    def _record_remaining_overlaps(self, observations: list[TrackObservation]) -> None:
        threshold = float(
            self.config.get("intra_camera", {}).get("duplicate_iou_threshold", 0.60)
        )
        for a, b in combinations(observations, 2):
            overlap = bbox_iou(a.bbox, b.bbox)
            if overlap < threshold:
                continue
            self.frame_diagnostics.append({
                "event_type": "post_nms_overlap",
                "camera_id": a.camera_id,
                "local_track_id": a.local_track_id,
                "segment_index": np.nan,
                "other_local_track_id": b.local_track_id,
                "other_segment_index": np.nan,
                "frame_index": a.frame_index,
                "accepted": False,
                "reason": "two_tracked_boxes_still_overlap_after_nms",
                "appearance_similarity": np.nan,
                "overlap_frames": 1,
                "overlap_ratio": np.nan,
                "median_iou": overlap,
                "duplicate_frame_ratio": 1.0,
                "median_center_distance_ratio": np.nan,
                "window_similarity": np.nan,
            })

    def _process_camera(self, camera: dict[str, Any]) -> None:
        camera_id = camera["id"]
        offset = camera["time_offset_seconds"]
        sample_interval = int(self.config["reid"]["sample_interval"])
        min_width = int(self.config["reid"]["min_crop_width"])
        min_height = int(self.config["reid"]["min_crop_height"])
        if sample_interval <= 0:
            raise ProjectError("reid.sample_interval must be greater than zero.")
        if min_width <= 0 or min_height <= 0:
            raise ProjectError("Minimum crop dimensions must be greater than zero.")

        self.detector_tracker.reset()
        frame_records: dict[int, list[TrackObservation]] = {}
        with VideoReader(camera["resolved_path"]) as reader:
            for frame_index, frame in reader:
                timestamp = frame_index / reader.metadata.fps + offset
                observations = self.detector_tracker.track_frame(
                    frame, camera_id, frame_index, timestamp
                )
                if observations:
                    frame_records[frame_index] = observations
                    self._record_remaining_overlaps(observations)
                for observation in observations:
                    self.database.update(observation)
                    if frame_index % sample_interval != 0:
                        continue
                    x1, y1, x2, y2 = observation.bbox
                    if x2 - x1 < min_width or y2 - y1 < min_height:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    embedding = self.reid.extract_embedding(crop)
                    self.database.add_embedding(
                        (camera_id, observation.local_track_id), observation, embedding
                    )
                    if bool(self.config["output"].get("save_crops", False)):
                        self._save_crop(
                            camera_id, observation.local_track_id, frame_index, crop
                        )
                if frame_index > 0 and frame_index % 250 == 0:
                    total = reader.metadata.frame_count
                    suffix = f"/{total}" if total > 0 else ""
                    logging.info("%s: processed %d%s frames", camera_id, frame_index, suffix)
        self.observations[camera_id] = frame_records

    def _save_local_tracks(self, tracks: list[LocalTrack]) -> None:
        columns = ["camera_id", "local_track_id", "start_frame", "end_frame",
                   "start_time", "end_time", "num_frames", "num_embeddings"]
        rows = [{
            "camera_id": track.camera_id,
            "local_track_id": track.local_track_id,
            "start_frame": track.start_frame,
            "end_frame": track.end_frame,
            "start_time": track.start_time,
            "end_time": track.end_time,
            "num_frames": track.num_frames,
            "num_embeddings": track.num_embeddings,
        } for track in tracks]
        pd.DataFrame(rows, columns=columns).to_csv(
            self.output_dir / "local_tracks.csv", index=False
        )

    def _save_track_segments(self, segmentation: SegmentationResult,
                             stitching: IntraCameraAssignment) -> None:
        columns = ["camera_id", "camera_person_id", "local_track_id", "segment_index",
                   "start_frame", "end_frame", "start_time", "end_time", "num_frames",
                   "num_embeddings", "retained_embeddings", "descriptor_quality",
                   "split_reason", "intra_camera_similarity"]
        rows = [{
            "camera_id": segment.camera_id,
            "camera_person_id": stitching.key_to_camera_id[segment.key],
            "local_track_id": segment.local_track_id,
            "segment_index": segment.segment_index,
            "start_frame": segment.start_frame,
            "end_frame": segment.end_frame,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
            "num_frames": segment.num_frames,
            "num_embeddings": segment.num_embeddings,
            "retained_embeddings": segment.retained_embeddings,
            "descriptor_quality": segment.descriptor_quality,
            "split_reason": segment.split_reason,
            "intra_camera_similarity": stitching.similarity_by_track.get(
                segment.key, np.nan
            ),
        } for segment in segmentation.segments]
        pd.DataFrame(rows, columns=columns).to_csv(
            self.output_dir / "track_segments.csv", index=False
        )

    def _save_camera_tracks(self, stitching: IntraCameraAssignment) -> None:
        columns = ["camera_id", "camera_person_id", "source_segments", "num_segments",
                   "start_frame", "end_frame", "start_time", "end_time", "num_frames",
                   "num_embeddings"]
        rows = []
        for track in stitching.camera_tracks:
            member_keys = stitching.groups[track.key]
            rows.append({
                "camera_id": track.camera_id,
                "camera_person_id": track.local_track_id,
                "source_segments": "|".join(
                    f"L{key[1]}:S{key[2]}" for key in member_keys
                ),
                "num_segments": len(member_keys),
                "start_frame": track.start_frame,
                "end_frame": track.end_frame,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "num_frames": track.num_frames,
                "num_embeddings": track.num_embeddings,
            })
        pd.DataFrame(rows, columns=columns).to_csv(
            self.output_dir / "camera_tracks.csv", index=False
        )

    def _save_intra_camera_matches(self, stitching: IntraCameraAssignment) -> None:
        columns = ["camera_id", "camera_person_id", "previous_local_track_id",
                   "previous_segment_index", "next_local_track_id", "next_segment_index",
                   "match_type", "appearance_similarity", "time_gap_seconds",
                   "overlap_frames", "overlap_ratio", "median_iou",
                   "duplicate_frame_ratio", "median_center_distance_ratio", "score"]
        rows = [{
            "camera_id": match.previous_track[0],
            "camera_person_id": stitching.key_to_camera_id[match.previous_track],
            "previous_local_track_id": match.previous_track[1],
            "previous_segment_index": match.previous_track[2],
            "next_local_track_id": match.next_track[1],
            "next_segment_index": match.next_track[2],
            "match_type": match.match_type,
            "appearance_similarity": match.appearance_similarity,
            "time_gap_seconds": match.time_gap_seconds,
            "overlap_frames": match.overlap_frames,
            "overlap_ratio": match.overlap_ratio,
            "median_iou": match.median_iou,
            "duplicate_frame_ratio": match.duplicate_frame_ratio,
            "median_center_distance_ratio": match.median_center_distance_ratio,
            "score": match.score,
        } for match in stitching.accepted_matches]
        pd.DataFrame(rows, columns=columns).sort_values(
            ["camera_id", "camera_person_id", "previous_local_track_id",
             "previous_segment_index"]
        ).to_csv(self.output_dir / "intra_camera_matches.csv", index=False)

    def _save_global_tracks(self, segments: list[TrackSegment],
                            stitching: IntraCameraAssignment,
                            assignment: GlobalAssignment) -> None:
        columns = ["global_id", "camera_id", "camera_person_id", "local_track_id",
                   "segment_index", "start_frame", "end_frame", "start_time", "end_time",
                   "intra_camera_similarity", "similarity_to_match"]
        rows = []
        for segment in segments:
            camera_person_id = stitching.key_to_camera_id[segment.key]
            camera_key = segment.camera_id, camera_person_id
            rows.append({
                "global_id": assignment.key_to_global_id[camera_key],
                "camera_id": segment.camera_id,
                "camera_person_id": camera_person_id,
                "local_track_id": segment.local_track_id,
                "segment_index": segment.segment_index,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "intra_camera_similarity": stitching.similarity_by_track.get(
                    segment.key, np.nan
                ),
                "similarity_to_match": assignment.similarity_by_track.get(
                    camera_key, np.nan
                ),
            })
        pd.DataFrame(rows, columns=columns).sort_values(
            ["global_id", "camera_id", "camera_person_id", "start_frame",
             "local_track_id", "segment_index"]
        ).to_csv(self.output_dir / "global_tracks.csv", index=False)

    def _save_diagnostics(self, segmentation: SegmentationResult,
                          stitching: IntraCameraAssignment) -> None:
        rows = list(self.frame_diagnostics)
        for event in segmentation.split_events:
            rows.append({
                "event_type": "track_split",
                "camera_id": event.camera_id,
                "local_track_id": event.local_track_id,
                "segment_index": event.left_segment_index,
                "other_local_track_id": event.local_track_id,
                "other_segment_index": event.right_segment_index,
                "frame_index": event.boundary_frame,
                "accepted": True,
                "reason": event.reason,
                "appearance_similarity": np.nan,
                "overlap_frames": 0,
                "overlap_ratio": 0.0,
                "median_iou": np.nan,
                "duplicate_frame_ratio": np.nan,
                "median_center_distance_ratio": np.nan,
                "window_similarity": event.window_similarity,
            })
        for decision in stitching.decisions:
            rows.append({
                "event_type": "intra_camera_decision",
                "camera_id": decision.previous_track[0],
                "local_track_id": decision.previous_track[1],
                "segment_index": decision.previous_track[2],
                "other_local_track_id": decision.next_track[1],
                "other_segment_index": decision.next_track[2],
                "frame_index": np.nan,
                "accepted": decision.accepted,
                "reason": decision.reason,
                "appearance_similarity": decision.appearance_similarity,
                "overlap_frames": decision.overlap_frames,
                "overlap_ratio": decision.overlap_ratio,
                "median_iou": decision.median_iou,
                "duplicate_frame_ratio": decision.duplicate_frame_ratio,
                "median_center_distance_ratio": decision.median_center_distance_ratio,
                "window_similarity": np.nan,
            })
        columns = ["event_type", "camera_id", "local_track_id", "segment_index",
                   "other_local_track_id", "other_segment_index", "frame_index",
                   "accepted", "reason", "appearance_similarity", "overlap_frames",
                   "overlap_ratio", "median_iou", "duplicate_frame_ratio",
                   "median_center_distance_ratio", "window_similarity"]
        pd.DataFrame(rows, columns=columns).to_csv(
            self.output_dir / "tracking_diagnostics.csv", index=False
        )

    @staticmethod
    def _identity_maps(segmentation: SegmentationResult,
                       stitching: IntraCameraAssignment,
                       assignment: GlobalAssignment) -> tuple[
                           dict[ObservationKey, int], dict[ObservationKey, int]
                       ]:
        camera_ids: dict[ObservationKey, int] = {}
        global_ids: dict[ObservationKey, int] = {}
        for observation_key, segment_key in segmentation.frame_to_segment.items():
            camera_person_id = stitching.key_to_camera_id[segment_key]
            camera_ids[observation_key] = camera_person_id
            global_ids[observation_key] = assignment.key_to_global_id[
                (segment_key[0], camera_person_id)
            ]
        return camera_ids, global_ids

    @staticmethod
    def _deduplicate_frame(observations: list[TrackObservation],
                           camera_ids: dict[ObservationKey, int]) -> list[TrackObservation]:
        best_by_camera_person: dict[int | tuple[str, int], TrackObservation] = {}
        for observation in observations:
            key = observation.camera_id, observation.local_track_id, observation.frame_index
            group_key: int | tuple[str, int] = camera_ids.get(
                key, (observation.camera_id, observation.local_track_id)
            )
            current = best_by_camera_person.get(group_key)
            if current is None or observation.confidence > current.confidence:
                best_by_camera_person[group_key] = observation
        return sorted(
            best_by_camera_person.values(), key=lambda item: item.local_track_id
        )

    def _render_camera(self, camera: dict[str, Any],
                       global_ids: dict[ObservationKey, int],
                       camera_ids: dict[ObservationKey, int]) -> None:
        output_cfg = self.config["output"]
        save_global = bool(output_cfg.get("save_global_tracking_video", True))
        save_local = bool(output_cfg.get("save_local_tracking_video", False))
        if not save_global and not save_local:
            return
        camera_id = camera["id"]
        visualizer = Visualizer(
            bool(output_cfg.get("show_local_id", True)),
            bool(output_cfg.get("show_confidence", False)),
            bool(output_cfg.get("show_camera_id", True)),
        )
        codec = str(output_cfg.get("codec", "mp4v"))
        with VideoReader(camera["resolved_path"]) as reader:
            global_writer = create_video_writer(
                self.output_dir / "videos" / f"{camera_id}_global_tracking.mp4",
                reader.metadata, codec
            ) if save_global else None
            local_writer = create_video_writer(
                self.output_dir / "videos" / f"{camera_id}_local_tracking.mp4",
                reader.metadata, codec
            ) if save_local else None
            try:
                for frame_index, frame in reader:
                    raw = self.observations[camera_id].get(frame_index, [])
                    observations = self._deduplicate_frame(raw, camera_ids)
                    if global_writer is not None:
                        global_writer.write(visualizer.draw(
                            frame.copy(), observations, global_ids, camera_ids
                        ))
                    if local_writer is not None:
                        local_writer.write(visualizer.draw(
                            frame.copy(), observations, camera_ids=camera_ids
                        ))
            finally:
                if global_writer is not None:
                    global_writer.release()
                if local_writer is not None:
                    local_writer.release()

    def run(self) -> PipelineSummary:
        total_steps = len(self.camera_configs) + 5
        for index, camera in enumerate(self.camera_configs, start=2):
            print(f"[{index}/{total_steps}] Processing camera {camera['id']}")
            self._process_camera(camera)
        self.database.finalize()
        tracks = self.database.values()
        self._save_local_tracks(tracks)

        split_step = len(self.camera_configs) + 2
        print(f"[{split_step}/{total_steps}] Splitting appearance-inconsistent local tracks")
        split_cfg = self.config.get("tracklet_split", {})
        segmentation = TrackSegmenter(
            int(split_cfg.get("window_size", 4)),
            float(split_cfg.get("change_similarity_threshold", 0.65)),
            int(split_cfg.get("min_separation_frames", 20)),
            int(split_cfg.get("min_segment_embeddings", 4)),
            float(split_cfg.get("outlier_similarity_threshold", 0.55)),
            bool(split_cfg.get("enabled", True)),
        ).split(tracks)

        stitching_step = len(self.camera_configs) + 3
        print(f"[{stitching_step}/{total_steps}] Stitching same-camera track segments")
        intra_cfg = self.config.get("intra_camera", {})
        stitching = IntraCameraMatcher(
            float(intra_cfg.get("similarity_threshold", 0.78)),
            float(intra_cfg.get("max_time_gap_seconds", 120.0)),
            float(intra_cfg.get("time_penalty_weight", 0.05)),
            int(intra_cfg.get("min_embeddings", 2)),
            bool(intra_cfg.get("enabled", True)),
            float(intra_cfg.get("duplicate_similarity_threshold", 0.82)),
            float(intra_cfg.get("duplicate_iou_threshold", 0.60)),
            float(intra_cfg.get("duplicate_frame_ratio_threshold", 0.70)),
            float(intra_cfg.get("duplicate_center_distance_ratio", 0.25)),
            int(intra_cfg.get("duplicate_max_overlap_frames", 45)),
            float(intra_cfg.get("duplicate_max_overlap_ratio", 0.30)),
            int(intra_cfg.get("duplicate_min_shared_frames", 2)),
        ).assign(segmentation.segments)
        self._save_track_segments(segmentation, stitching)
        self._save_camera_tracks(stitching)
        self._save_intra_camera_matches(stitching)
        self._save_diagnostics(segmentation, stitching)

        matching_step = len(self.camera_configs) + 4
        print(f"[{matching_step}/{total_steps}] Performing cross-camera matching")
        matching_cfg = self.config["matching"]
        matcher = CrossCameraMatcher(
            float(matching_cfg["similarity_threshold"]),
            float(matching_cfg["max_time_gap_seconds"]),
            bool(matching_cfg["use_time_constraint"]),
        )
        matches = matcher.match_all(stitching.camera_tracks)
        similarity_dataframe(
            segmentation.segments, include_same_camera=True
        ).to_csv(self.output_dir / "local_similarity_matrix.csv")
        similarity_dataframe(stitching.camera_tracks, id_prefix="C").to_csv(
            self.output_dir / "similarity_matrix.csv"
        )
        assignment = GlobalTracker().assign(stitching.camera_tracks, matches)
        self._save_global_tracks(segmentation.segments, stitching, assignment)
        camera_ids, global_ids = self._identity_maps(
            segmentation, stitching, assignment
        )

        print(f"[{total_steps}/{total_steps}] Rendering output videos")
        for camera in self.camera_configs:
            self._render_camera(camera, global_ids, camera_ids)

        counts = {
            camera["id"]: len(self.database.for_camera(camera["id"]))
            for camera in self.camera_configs
        }
        camera_counts = {
            camera["id"]: sum(
                track.camera_id == camera["id"] for track in stitching.camera_tracks
            )
            for camera in self.camera_configs
        }
        return PipelineSummary(
            counts, camera_counts, len(stitching.accepted_matches),
            len(assignment.global_tracks), len(assignment.accepted_matches),
            self.output_dir,
        )
