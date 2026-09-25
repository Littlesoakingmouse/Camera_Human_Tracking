"""Model-free tests for appearance change-point splitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector_tracker import TrackObservation  # noqa: E402
from src.track_database import EmbeddingSample, LocalTrack  # noqa: E402
from src.track_segmenter import TrackSegmenter  # noqa: E402
from src.utils.common import l2_normalize  # noqa: E402


def make_track(track_id: int, vectors: list[list[float]]) -> LocalTrack:
    frames = [index * 5 for index in range(len(vectors))]
    observations = {
        frame: TrackObservation("cam1", frame, frame / 10.0, track_id,
                                (20, 20, 80, 180), 0.9)
        for frame in range(frames[-1] + 1)
    }
    samples = [
        EmbeddingSample(
            frame, frame / 10.0, (20, 20, 80, 180), 0.9,
            l2_normalize(np.asarray(vector, dtype=np.float32)),
        )
        for frame, vector in zip(frames, vectors)
    ]
    track = LocalTrack(
        "cam1", track_id, 0, frames[-1], 0.0, frames[-1] / 10.0,
        num_frames=len(observations), observations=observations,
        embedding_samples=samples,
    )
    track.finalize()
    return track


def main() -> int:
    switched = make_track(
        1,
        [[1.0, 0.02, 0.0]] * 6 + [[0.02, 1.0, 0.0]] * 6,
    )
    stable = make_track(
        2,
        [[1.0, 0.02 + 0.005 * (index % 3), 0.0] for index in range(12)],
    )
    result = TrackSegmenter(
        window_size=4,
        change_similarity_threshold=0.65,
        min_separation_frames=20,
        min_segment_embeddings=4,
        outlier_similarity_threshold=0.55,
    ).split([switched, stable])

    switched_segments = [segment for segment in result.segments if segment.local_track_id == 1]
    stable_segments = [segment for segment in result.segments if segment.local_track_id == 2]
    assert len(switched_segments) == 2
    assert len(stable_segments) == 1
    assert len(result.split_events) == 1
    assert switched_segments[0].embedding is not None
    assert switched_segments[1].embedding is not None
    assert float(np.dot(switched_segments[0].embedding, switched_segments[1].embedding)) < 0.10
    print("PASS: abrupt identity change splits; mild view variation remains one segment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
