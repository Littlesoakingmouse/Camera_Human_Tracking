"""Model-free tests for appearance change-point splitting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector_tracker import TrackObservation  # noqa: E402
from src.pipeline import crop_quality  # noqa: E402
from src.track_database import EmbeddingSample, LocalTrack  # noqa: E402
from src.track_segmenter import TrackSegmenter  # noqa: E402
from src.utils.common import l2_normalize  # noqa: E402


def make_track(track_id: int, vectors: list[list[float]],
               reliable: list[bool] | None = None) -> LocalTrack:
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
            reliable=is_reliable,
        )
        for frame, vector, is_reliable in zip(
            frames, vectors, reliable or [True] * len(vectors)
        )
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
    # Person A is briefly hidden behind person B: the occluded crops look like B.
    occluded = make_track(
        3,
        [[1.0, 0.02, 0.0]] * 6 + [[0.02, 1.0, 0.0]] * 5 + [[1.0, 0.02, 0.0]] * 6,
        [True] * 6 + [False] * 5 + [True] * 6,
    )
    result = TrackSegmenter(
        window_size=4,
        change_similarity_threshold=0.65,
        min_separation_frames=20,
        min_segment_embeddings=4,
        outlier_similarity_threshold=0.55,
    ).split([switched, stable, occluded])

    switched_segments = [segment for segment in result.segments if segment.local_track_id == 1]
    stable_segments = [segment for segment in result.segments if segment.local_track_id == 2]
    assert len(switched_segments) == 2
    assert len(stable_segments) == 1
    assert len(result.split_events) == 1
    assert switched_segments[0].embedding is not None
    assert switched_segments[1].embedding is not None
    assert float(np.dot(switched_segments[0].embedding, switched_segments[1].embedding)) < 0.10
    occluded_segments = [segment for segment in result.segments if segment.local_track_id == 3]
    assert len(occluded_segments) == 1
    assert occluded_segments[0].num_embeddings == 17
    assert float(occluded_segments[0].embedding[0]) > 0.99

    def box(x1: int, y1: int, x2: int, y2: int) -> TrackObservation:
        return TrackObservation("cam1", 0, 0.0, x1, (x1, y1, x2, y2), 0.9)

    back, front, edge = box(100, 100, 200, 400), box(150, 150, 250, 450), box(0, 100, 80, 400)
    (back_occ, back_ok), (front_occ, front_ok), (_, edge_ok) = crop_quality(
        [back, front, edge], 1280, 720, 0.30, 2
    )
    assert back_occ > 0.30 and not back_ok
    assert front_occ == 0.0 and front_ok
    assert not edge_ok
    print("PASS: abrupt identity change splits; mild view variation remains one segment")
    print("PASS: occluded crops neither split a track nor pollute its descriptor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
