"""Three-pass orchestration for offline multi-camera person tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .detector_tracker import DetectorTracker, TrackObservation
from .global_tracker import GlobalAssignment, GlobalTracker
from .matcher import CrossCameraMatcher, similarity_dataframe
from .reid import ReIDModel
from .track_database import LocalTrack, TrackDatabase
from .utils.common import ProjectError, project_path
from .utils.video import VideoReader, create_video_writer
from .visualizer import Visualizer


@dataclass(frozen=True)
class PipelineSummary:
    local_tracks_by_camera: dict[str, int]
    global_persons: int
    cross_camera_matches: int
    output_directory: Path


class MultiCameraTrackingPipeline:
    """Run local tracking, cross-camera matching, and annotated replay."""

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
                    self.database.add_embedding((camera_id, observation.local_track_id), embedding)
                    if bool(self.config["output"].get("save_crops", False)):
                        self._save_crop(camera_id, observation.local_track_id, frame_index, crop)
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
        pd.DataFrame(rows, columns=columns).to_csv(self.output_dir / "local_tracks.csv", index=False)

    def _save_global_tracks(self, tracks: list[LocalTrack], assignment: GlobalAssignment) -> None:
        columns = ["global_id", "camera_id", "local_track_id", "start_time", "end_time",
                   "similarity_to_match"]
        rows = []
        for track in tracks:
            rows.append({
                "global_id": assignment.key_to_global_id[track.key],
                "camera_id": track.camera_id,
                "local_track_id": track.local_track_id,
                "start_time": track.start_time,
                "end_time": track.end_time,
                "similarity_to_match": assignment.similarity_by_track.get(track.key, np.nan),
            })
        pd.DataFrame(rows, columns=columns).sort_values(
            ["global_id", "camera_id", "local_track_id"]
        ).to_csv(self.output_dir / "global_tracks.csv", index=False)

    def _render_camera(self, camera: dict[str, Any], assignment: GlobalAssignment) -> None:
        output_cfg = self.config["output"]
        save_global = bool(output_cfg.get("save_global_tracking_video", True))
        save_local = bool(output_cfg.get("save_local_tracking_video", False))
        if not save_global and not save_local:
            return
        camera_id = camera["id"]
        visualizer = Visualizer(
            bool(output_cfg.get("show_local_id", True)),
            bool(output_cfg.get("show_confidence", False)),
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
                    observations = self.observations[camera_id].get(frame_index, [])
                    if global_writer is not None:
                        annotated = visualizer.draw(frame.copy(), observations,
                                                    assignment.key_to_global_id)
                        global_writer.write(annotated)
                    if local_writer is not None:
                        local_writer.write(visualizer.draw(frame.copy(), observations))
            finally:
                if global_writer is not None:
                    global_writer.release()
                if local_writer is not None:
                    local_writer.release()

    def run(self) -> PipelineSummary:
        total_steps = len(self.camera_configs) + 3
        for index, camera in enumerate(self.camera_configs, start=2):
            print(f"[{index}/{total_steps}] Processing camera {camera['id']}")
            self._process_camera(camera)
        self.database.finalize()
        tracks = self.database.values()
        self._save_local_tracks(tracks)

        matching_step = len(self.camera_configs) + 2
        print(f"[{matching_step}/{total_steps}] Performing cross-camera matching")
        matching_cfg = self.config["matching"]
        matcher = CrossCameraMatcher(
            float(matching_cfg["similarity_threshold"]),
            float(matching_cfg["max_time_gap_seconds"]),
            bool(matching_cfg["use_time_constraint"]),
        )
        matches = matcher.match_all(tracks)
        similarity_dataframe(tracks).to_csv(self.output_dir / "similarity_matrix.csv")
        assignment = GlobalTracker().assign(tracks, matches)
        self._save_global_tracks(tracks, assignment)

        print(f"[{total_steps}/{total_steps}] Rendering output videos")
        for camera in self.camera_configs:
            self._render_camera(camera, assignment)

        counts = {camera["id"]: len(self.database.for_camera(camera["id"]))
                  for camera in self.camera_configs}
        return PipelineSummary(
            counts, len(assignment.global_tracks), len(assignment.accepted_matches), self.output_dir
        )
