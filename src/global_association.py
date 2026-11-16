"""Constrained agglomerative clustering of camera identities into global IDs.

Pairwise Hungarian matching followed by greedy union (``GlobalTracker``) decides
every camera pair in isolation, so A~B and B~C can merge A with a very
dissimilar C. Here every CID starts as its own cluster and the two clusters with
the highest *average-linkage* similarity are merged repeatedly while:

- the linkage is at least ``similarity_threshold``;
- no camera appears twice in the merged cluster (cannot-link);
- with ``use_time_constraint``, every cross-cluster pair satisfies the time gap.

Average linkage scores a merge by all member pairs, so one strong bridge edge
cannot pull two otherwise unrelated groups together.
"""

from __future__ import annotations

import numpy as np

from .global_tracker import GlobalAssignment
from .matcher import CandidateMatch, track_time_gap
from .track_database import LocalTrack, preferred_samples


def _sample_matrix(track: LocalTrack, max_samples: int) -> np.ndarray | None:
    samples = preferred_samples(track.embedding_samples)
    if samples:
        if len(samples) > max_samples:
            picks = np.linspace(0, len(samples) - 1, max_samples).round().astype(int)
            samples = [samples[index] for index in picks]
        return np.stack([sample.vector for sample in samples]).astype(np.float32)
    if track.embedding is not None:
        return np.asarray(track.embedding, dtype=np.float32).reshape(1, -1)
    return None


def pairwise_similarity(tracks: list[LocalTrack], mode: str = "set_topk",
                        top_fraction: float = 0.25, max_samples: int = 64) -> np.ndarray:
    """Similarity between every pair of identities.

    ``descriptor``: cosine of the two mean descriptors (previous behaviour).
    ``set_topk``: mean of the best ``top_fraction`` of all sample-pair cosines.
    Two cameras rarely see the same pose; comparing the closest views of each
    identity is more robust than comparing two averages of different poses.
    """
    count = len(tracks)
    result = np.full((count, count), np.nan)
    if mode == "descriptor":
        vectors = [None if t.embedding is None else np.asarray(t.embedding, np.float32)
                   for t in tracks]
        for i in range(count):
            for j in range(i + 1, count):
                if vectors[i] is not None and vectors[j] is not None:
                    result[i, j] = result[j, i] = float(vectors[i] @ vectors[j])
        return result
    if mode != "set_topk":
        raise ValueError(f"Unknown similarity mode: {mode}")
    matrices = [_sample_matrix(track, max_samples) for track in tracks]
    for i in range(count):
        for j in range(i + 1, count):
            if matrices[i] is None or matrices[j] is None:
                continue
            values = np.sort((matrices[i] @ matrices[j].T).ravel())[::-1]
            keep = max(1, int(np.ceil(top_fraction * values.size)))
            result[i, j] = result[j, i] = float(values[:keep].mean())
    return result


class AgglomerativeGlobalTracker:
    """Average-linkage clustering with camera and time cannot-link constraints."""

    def __init__(self, similarity_threshold: float, max_time_gap_seconds: float,
                 use_time_constraint: bool, similarity_mode: str = "set_topk",
                 top_fraction: float = 0.25) -> None:
        self.threshold = float(similarity_threshold)
        self.max_time_gap = float(max_time_gap_seconds)
        self.use_time_constraint = bool(use_time_constraint)
        self.mode = similarity_mode
        self.top_fraction = float(top_fraction)
        self.similarity: np.ndarray | None = None

    def assign(self, tracks: list[LocalTrack]) -> GlobalAssignment:
        tracks = sorted(tracks, key=lambda track: track.key)
        count = len(tracks)
        sim = pairwise_similarity(tracks, self.mode, self.top_fraction)
        self.similarity = sim
        allowed = np.isfinite(sim)
        cameras = np.array([track.camera_id for track in tracks])
        allowed &= cameras[:, None] != cameras[None, :]
        if self.use_time_constraint:
            gaps = np.array([[track_time_gap(a, b) for b in tracks] for a in tracks])
            allowed &= gaps <= self.max_time_gap
        weight = np.where(allowed, sim, 0.0)

        clusters: dict[int, list[int]] = {index: [index] for index in range(count)}
        accepted: list[CandidateMatch] = []
        best_link: dict[int, float] = {}
        # Cluster-level state, updated in place (Lance-Williams for average linkage):
        # total similarity between two clusters, their sizes, and whether any
        # member pair is forbidden (one forbidden pair blocks the merge).
        total = weight.copy()
        blocked = ~allowed
        np.fill_diagonal(blocked, True)
        size = np.ones(count)
        active = np.ones(count, dtype=bool)
        while count > 1:
            linkage = total / np.outer(size, size)
            invalid = blocked | ~active[:, None] | ~active[None, :]
            linkage[invalid] = -np.inf
            flat = int(np.argmax(linkage))
            a, b = divmod(flat, count)
            best = float(linkage[a, b])
            if not np.isfinite(best) or best < self.threshold:
                break
            a, b = min(a, b), max(a, b)
            total[a, :] += total[b, :]
            total[:, a] = total[a, :]
            blocked[a, :] |= blocked[b, :]
            blocked[:, a] = blocked[a, :]
            blocked[a, a] = True
            size[a] += size[b]
            active[b] = False
            # Report the strongest member edge so CSVs stay comparable to the old tracker.
            block = weight[np.ix_(clusters[a], clusters[b])]
            i, j = np.unravel_index(int(np.argmax(block)), block.shape)
            left, right = tracks[clusters[a][i]], tracks[clusters[b][j]]
            accepted.append(CandidateMatch(left.key, right.key, best,
                                           track_time_gap(left, right)))
            for member in clusters[a] + clusters[b]:
                best_link[member] = max(best_link.get(member, -1.0), best)
            clusters[a] = clusters[a] + clusters.pop(b)

        ordered = sorted((sorted(tracks[m].key for m in members)
                          for members in clusters.values()), key=lambda group: group[0])
        global_tracks = {gid: group for gid, group in enumerate(ordered, start=1)}
        key_to_global = {key: gid for gid, group in global_tracks.items() for key in group}
        similarity_by_track = {tracks[m].key: s for m, s in best_link.items()}
        return GlobalAssignment(key_to_global, global_tracks, similarity_by_track, accepted)
