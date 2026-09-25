"""Model-free tests for re-entry, duplicate overlap, and CSV expansion."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector_tracker import TrackObservation  # noqa: E402
from src.global_tracker import GlobalTracker  # noqa: E402
from src.intra_camera_matcher import IntraCameraMatcher  # noqa: E402
from src.matcher import CrossCameraMatcher  # noqa: E402
from src.pipeline import MultiCameraTrackingPipeline  # noqa: E402
from src.track_database import EmbeddingSample, LocalTrack, TrackSegment  # noqa: E402
from src.track_segmenter import SegmentationResult  # noqa: E402
from src.utils.common import l2_normalize  # noqa: E402


def make_segment(camera: str, track_id: int, segment_index: int,
                 start_frame: int, end_frame: int, vector: list[float],
                 bbox: tuple[int, int, int, int]) -> TrackSegment:
    observations = {
        frame: TrackObservation(camera, frame, frame / 10.0, track_id, bbox, 0.90)
        for frame in range(start_frame, end_frame + 1)
    }
    embedding = l2_normalize(np.asarray(vector, dtype=np.float32))
    sample_frames = sorted({start_frame, (start_frame + end_frame) // 2, end_frame})
    samples = [
        EmbeddingSample(frame, frame / 10.0, bbox, 0.90, embedding.copy())
        for frame in sample_frames
    ]
    segment = TrackSegment(
        camera, track_id, segment_index, start_frame, end_frame,
        start_frame / 10.0, end_frame / 10.0,
        observations=observations, embedding_samples=samples,
    )
    segment.finalize()
    return segment


def raw_track(segment: TrackSegment) -> LocalTrack:
    track = LocalTrack(
        segment.camera_id, segment.local_track_id, segment.start_frame,
        segment.end_frame, segment.start_time, segment.end_time,
        num_frames=segment.num_frames, observations=segment.observations,
        embedding_samples=segment.embedding_samples,
    )
    track.finalize()
    return track


def main() -> int:
    segments = [
        # Regular exit/re-entry pair.
        make_segment("cam1", 3, 1, 0, 100, [1.0, 0.0, 0.0], (20, 20, 80, 180)),
        make_segment("cam1", 12, 1, 300, 450, [0.99, 0.04, 0.0], (25, 20, 85, 180)),
        # Duplicate handoff: overlaps L12 for 21/121 frames with nearly identical boxes.
        make_segment("cam1", 32, 1, 430, 550, [0.98, 0.05, 0.0], (27, 22, 87, 182)),
        # A real simultaneous person with distant geometry must not be merged.
        make_segment("cam1", 40, 1, 435, 540, [0.0, 1.0, 0.0], (300, 20, 360, 180)),
        make_segment("cam2", 7, 1, 120, 250, [0.99, 0.02, 0.0], (30, 20, 90, 180)),
    ]

    stitching = IntraCameraMatcher(
        similarity_threshold=0.90,
        max_time_gap_seconds=60.0,
        time_penalty_weight=0.05,
        min_embeddings=2,
    ).assign(segments)

    cid_l3 = stitching.key_to_camera_id[("cam1", 3, 1)]
    assert cid_l3 == stitching.key_to_camera_id[("cam1", 12, 1)]
    assert cid_l3 == stitching.key_to_camera_id[("cam1", 32, 1)]
    assert cid_l3 != stitching.key_to_camera_id[("cam1", 40, 1)]
    assert {match.match_type for match in stitching.accepted_matches} == {
        "reentry", "duplicate_overlap"
    }
    duplicate = next(
        match for match in stitching.accepted_matches
        if match.match_type == "duplicate_overlap"
    )
    assert duplicate.overlap_frames == 21
    aggregate = next(
        track for track in stitching.camera_tracks
        if track.camera_id == "cam1" and track.local_track_id == cid_l3
    )
    assert aggregate.num_frames == 352  # union of 0..100 and 300..550

    matches = CrossCameraMatcher(0.80, 30.0, True).match_all(stitching.camera_tracks)
    global_assignment = GlobalTracker().assign(stitching.camera_tracks, matches)
    cam2_cid = stitching.key_to_camera_id[("cam2", 7, 1)]
    assert global_assignment.key_to_global_id[("cam1", cid_l3)] == (
        global_assignment.key_to_global_id[("cam2", cam2_cid)]
    )

    frame_to_segment = {
        (segment.camera_id, segment.local_track_id, frame): segment.key
        for segment in segments for frame in segment.observations
    }
    segmentation = SegmentationResult(segments, frame_to_segment, [])
    raw_tracks = [raw_track(segment) for segment in segments]
    with tempfile.TemporaryDirectory() as temporary_directory:
        pipeline = object.__new__(MultiCameraTrackingPipeline)
        pipeline.output_dir = Path(temporary_directory)
        pipeline.frame_diagnostics = []
        pipeline._save_local_tracks(raw_tracks)
        pipeline._save_track_segments(segmentation, stitching)
        pipeline._save_camera_tracks(stitching)
        pipeline._save_intra_camera_matches(stitching)
        pipeline._save_global_tracks(segments, stitching, global_assignment)
        pipeline._save_diagnostics(segmentation, stitching)
        assert "camera_person_id" in pd.read_csv(
            pipeline.output_dir / "track_segments.csv"
        ).columns
        assert "segment_index" in pd.read_csv(
            pipeline.output_dir / "global_tracks.csv"
        ).columns
        matches_csv = pd.read_csv(pipeline.output_dir / "intra_camera_matches.csv")
        assert "duplicate_overlap" in set(matches_csv.match_type)

    print("PASS: re-entry and duplicate overlap share one CID; real overlap stays separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
