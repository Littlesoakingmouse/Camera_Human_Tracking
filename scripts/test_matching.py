"""Model-free unit smoke test for matching and global ID assignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.global_tracker import GlobalTracker  # noqa: E402
from src.matcher import CrossCameraMatcher  # noqa: E402
from src.track_database import LocalTrack  # noqa: E402
from src.utils.common import l2_normalize  # noqa: E402


def make_track(camera: str, track_id: int, vector: list[float]) -> LocalTrack:
    track = LocalTrack(camera, track_id, 0, 20, 0.0, 2.0, num_frames=21)
    track.embedding = l2_normalize(np.asarray(vector, dtype=np.float32))
    return track


def main() -> int:
    tracks = [
        make_track("cam1", 1, [1.0, 0.0, 0.0]),
        make_track("cam1", 2, [0.0, 1.0, 0.0]),
        make_track("cam2", 8, [0.99, 0.05, 0.0]),
        make_track("cam2", 4, [0.04, 0.99, 0.0]),
        make_track("cam2", 9, [0.0, 0.0, 1.0]),
    ]
    matcher = CrossCameraMatcher(0.80, 30.0, True)
    matches = matcher.match_all(tracks)
    assignment = GlobalTracker().assign(tracks, matches)
    matched_pairs = {frozenset((item.track_a, item.track_b)) for item in matches}
    assert frozenset((("cam1", 1), ("cam2", 8))) in matched_pairs
    assert frozenset((("cam1", 2), ("cam2", 4))) in matched_pairs
    assert assignment.key_to_global_id[("cam1", 1)] == assignment.key_to_global_id[("cam2", 8)]
    assert assignment.key_to_global_id[("cam1", 2)] == assignment.key_to_global_id[("cam2", 4)]
    assert assignment.key_to_global_id[("cam2", 9)] not in {
        assignment.key_to_global_id[("cam1", 1)], assignment.key_to_global_id[("cam1", 2)]
    }
    for match in matches:
        print(f"{match.track_a} <-> {match.track_b}: similarity={match.similarity:.4f}")
    print(f"PASS: {len(matches)} matches, {len(assignment.global_tracks)} global identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
