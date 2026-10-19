"""Cosine similarity and pairwise cross-camera Hungarian matching."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

from .track_database import LocalTrack, TrackKey, TrackSegment

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class CandidateMatch:
    track_a: TrackKey
    track_b: TrackKey
    similarity: float
    time_gap_seconds: float


def cosine_similarity(feature_a: np.ndarray, feature_b: np.ndarray) -> float:
    """Compute cosine similarity defensively, whether or not inputs are normalized."""
    a = np.asarray(feature_a, dtype=np.float32).reshape(-1)
    b = np.asarray(feature_b, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"Embedding shape mismatch: {a.shape} versus {b.shape}")
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def similarity_matrix(tracks_a: list, tracks_b: list) -> np.ndarray:
    """Cosine similarity of every descriptor pair in one matrix product."""
    if not tracks_a or not tracks_b:
        return np.empty((len(tracks_a), len(tracks_b)), dtype=np.float64)
    a = np.stack([np.asarray(track.embedding, dtype=np.float32).reshape(-1) for track in tracks_a])
    b = np.stack([np.asarray(track.embedding, dtype=np.float32).reshape(-1) for track in tracks_b])
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Embedding shape mismatch: {a.shape[1]} versus {b.shape[1]}")
    norms_a = np.linalg.norm(a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(norms_a <= 1e-12) or np.any(norms_b <= 1e-12):
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return np.clip((a / norms_a) @ (b / norms_b).T, -1.0, 1.0).astype(np.float64)


def track_time_gap(a: LocalTrack, b: LocalTrack) -> float:
    """Return zero for overlapping intervals, otherwise the distance between them."""
    if a.end_time < b.start_time:
        return b.start_time - a.end_time
    if b.end_time < a.start_time:
        return a.start_time - b.end_time
    return 0.0


class CrossCameraMatcher:
    """Generate one-to-one matches independently for every camera pair."""

    def __init__(self, similarity_threshold: float, max_time_gap_seconds: float,
                 use_time_constraint: bool) -> None:
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1].")
        if max_time_gap_seconds < 0:
            raise ValueError("max_time_gap_seconds must be non-negative.")
        self.threshold = float(similarity_threshold)
        self.max_time_gap = float(max_time_gap_seconds)
        self.use_time_constraint = bool(use_time_constraint)

    def _eligible(self, a: LocalTrack, b: LocalTrack) -> tuple[bool, float, float]:
        if a.embedding is None or b.embedding is None:
            return False, float("nan"), float("inf")
        similarity = cosine_similarity(a.embedding, b.embedding)
        gap = track_time_gap(a, b)
        time_ok = not self.use_time_constraint or gap <= self.max_time_gap
        return time_ok and similarity >= self.threshold, similarity, gap

    def match_pair(self, tracks_a: list[LocalTrack], tracks_b: list[LocalTrack]) -> list[CandidateMatch]:
        """Run threshold-aware Hungarian assignment for two distinct cameras."""
        cameras_a = {track.camera_id for track in tracks_a}
        cameras_b = {track.camera_id for track in tracks_b}
        if cameras_a.intersection(cameras_b):
            raise ValueError("Cross-camera matching cannot compare tracks from the same camera.")
        usable_a = [track for track in tracks_a if track.embedding is not None]
        usable_b = [track for track in tracks_b if track.embedding is not None]
        if not usable_a or not usable_b:
            return []
        scores = similarity_matrix(usable_a, usable_b)
        gaps = np.array([
            [track_time_gap(track_a, track_b) for track_b in usable_b]
            for track_a in usable_a
        ], dtype=np.float64)
        eligible = scores >= self.threshold
        if self.use_time_constraint:
            eligible &= gaps <= self.max_time_gap
        cost = np.where(eligible, 1.0 - scores, 1e6)
        rows, columns = linear_sum_assignment(cost)
        matches: list[CandidateMatch] = []
        for row, column in zip(rows, columns):
            if cost[row, column] < 1e6:
                matches.append(CandidateMatch(
                    usable_a[row].key, usable_b[column].key,
                    float(scores[row, column]), float(gaps[row, column])
                ))
        return matches

    def match_all(self, tracks: list[LocalTrack]) -> list[CandidateMatch]:
        """Run Hungarian matching for every pair of cameras."""
        cameras = sorted({track.camera_id for track in tracks})
        by_camera = {camera: [track for track in tracks if track.camera_id == camera]
                     for camera in cameras}
        candidates: list[CandidateMatch] = []
        for camera_a, camera_b in combinations(cameras, 2):
            candidates.extend(self.match_pair(by_camera[camera_a], by_camera[camera_b]))
        return candidates


def similarity_dataframe(tracks: list[LocalTrack | TrackSegment], id_prefix: str = "L",
                         include_same_camera: bool = False) -> "pd.DataFrame":
    """Build a labeled similarity matrix; diagonal and unavailable cells are blank."""
    import pandas as pd

    labels = [
        (
            f"{track.camera_id}:{id_prefix}{track.local_track_id}:S{track.segment_index}"
            if isinstance(track, TrackSegment)
            else f"{track.camera_id}:{id_prefix}{track.local_track_id}"
        )
        for track in tracks
    ]
    matrix = np.full((len(tracks), len(tracks)), np.nan, dtype=np.float64)
    usable = [index for index, track in enumerate(tracks) if track.embedding is not None]
    if usable:
        selected = [tracks[index] for index in usable]
        matrix[np.ix_(usable, usable)] = similarity_matrix(selected, selected)
    np.fill_diagonal(matrix, np.nan)
    if not include_same_camera:
        cameras = np.array([track.camera_id for track in tracks])
        matrix[cameras[:, None] == cameras[None, :]] = np.nan
    return pd.DataFrame(matrix, index=labels, columns=labels)
