"""Model-free test for constrained agglomerative global association."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.global_association import AgglomerativeGlobalTracker, pairwise_similarity  # noqa: E402
from src.track_database import EmbeddingSample, LocalTrack  # noqa: E402
from src.utils.common import l2_normalize  # noqa: E402


def make_track(camera: str, track_id: int, vectors: list[np.ndarray],
               start: float = 0.0) -> LocalTrack:
    track = LocalTrack(camera, track_id, 0, 10, start, start + 1.0)
    track.embedding_samples = [
        EmbeddingSample(i, start, (0, 0, 10, 20), 0.9, l2_normalize(v))
        for i, v in enumerate(vectors)
    ]
    track.num_frames = 10
    track.finalize()
    return track


def basis(index: int, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vector = np.zeros(16, dtype=np.float32)
    vector[index] = 1.0
    return vector + noise * rng.standard_normal(16).astype(np.float32)


def test_clusters_same_person_across_cameras() -> None:
    tracks = [
        make_track("cam1", 1, [basis(0, 0.1, s) for s in range(3)]),
        make_track("cam2", 1, [basis(0, 0.1, s + 10) for s in range(3)]),
        make_track("cam3", 1, [basis(0, 0.1, s + 20) for s in range(3)]),
        make_track("cam1", 2, [basis(5, 0.1, s + 30) for s in range(3)]),
        make_track("cam2", 2, [basis(5, 0.1, s + 40) for s in range(3)]),
    ]
    result = AgglomerativeGlobalTracker(0.7, 30, False).assign(tracks)
    gid = result.key_to_global_id
    assert gid[("cam1", 1)] == gid[("cam2", 1)] == gid[("cam3", 1)]
    assert gid[("cam1", 2)] == gid[("cam2", 2)]
    assert gid[("cam1", 1)] != gid[("cam1", 2)]
    assert len(result.global_tracks) == 2


def test_same_camera_never_merged() -> None:
    tracks = [make_track("cam1", 1, [basis(0)]), make_track("cam1", 2, [basis(0)])]
    result = AgglomerativeGlobalTracker(0.5, 30, False).assign(tracks)
    assert len(result.global_tracks) == 2


def test_bridge_does_not_chain() -> None:
    # B sits between A and C; A and C are dissimilar. Average linkage must not
    # put all three together even though A~B and B~C pass the threshold.
    a = basis(0)
    c = basis(1)
    b = l2_normalize(a + c)
    tracks = [make_track("cam1", 1, [a]), make_track("cam2", 1, [b]),
              make_track("cam3", 1, [c])]
    result = AgglomerativeGlobalTracker(0.7, 30, False, "descriptor").assign(tracks)
    assert len(result.global_tracks) == 2, result.global_tracks


def test_time_constraint() -> None:
    tracks = [make_track("cam1", 1, [basis(0)], start=0.0),
              make_track("cam2", 1, [basis(0)], start=500.0)]
    assert len(AgglomerativeGlobalTracker(0.7, 30, True).assign(tracks).global_tracks) == 2
    assert len(AgglomerativeGlobalTracker(0.7, 30, False).assign(tracks).global_tracks) == 1


def test_set_similarity_prefers_closest_views() -> None:
    front, back = basis(0), basis(1)
    person_cam1 = make_track("cam1", 1, [front, back])
    person_cam2 = make_track("cam2", 1, [front])
    sims = pairwise_similarity([person_cam1, person_cam2], "set_topk", top_fraction=0.5)
    descriptor = pairwise_similarity([person_cam1, person_cam2], "descriptor")
    assert sims[0, 1] > descriptor[0, 1]


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print(f"PASS {name}")
