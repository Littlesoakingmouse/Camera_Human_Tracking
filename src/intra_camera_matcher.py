"""Offline same-camera stitching for re-entry and short duplicate tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .matcher import cosine_similarity
from .track_database import (
    LocalTrack,
    SegmentKey,
    TrackKey,
    TrackSegment,
    robust_descriptor,
)


@dataclass(frozen=True)
class IntraCameraMatch:
    previous_track: SegmentKey
    next_track: SegmentKey
    match_type: str
    appearance_similarity: float
    time_gap_seconds: float
    overlap_frames: int
    overlap_ratio: float
    median_iou: float
    duplicate_frame_ratio: float
    median_center_distance_ratio: float
    score: float

    @property
    def similarity(self) -> float:
        """Backward-compatible alias used by existing reports."""
        return self.appearance_similarity


@dataclass(frozen=True)
class IntraCameraDecision:
    previous_track: SegmentKey
    next_track: SegmentKey
    match_type: str
    accepted: bool
    reason: str
    appearance_similarity: float
    overlap_frames: int
    overlap_ratio: float
    median_iou: float
    duplicate_frame_ratio: float
    median_center_distance_ratio: float


@dataclass(frozen=True)
class IntraCameraAssignment:
    key_to_camera_id: dict[SegmentKey, int]
    camera_tracks: list[LocalTrack]
    groups: dict[TrackKey, list[SegmentKey]]
    similarity_by_track: dict[SegmentKey, float]
    accepted_matches: list[IntraCameraMatch]
    decisions: list[IntraCameraDecision]


@dataclass(frozen=True)
class _OverlapMetrics:
    frames: int
    ratio: float
    median_iou: float
    duplicate_frame_ratio: float
    median_center_distance_ratio: float


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def center_distance_ratio(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    center_a = ((a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0)
    center_b = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
    distance = float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))
    diagonal_a = float(np.hypot(a[2] - a[0], a[3] - a[1]))
    diagonal_b = float(np.hypot(b[2] - b[0], b[3] - b[1]))
    denominator = max(1.0, min(diagonal_a, diagonal_b))
    return distance / denominator


class IntraCameraMatcher:
    """Join appearance-compatible segments into stable per-camera identities."""

    def __init__(
        self,
        similarity_threshold: float,
        max_time_gap_seconds: float,
        time_penalty_weight: float = 0.05,
        min_embeddings: int = 2,
        enabled: bool = True,
        duplicate_similarity_threshold: float = 0.82,
        duplicate_iou_threshold: float = 0.60,
        duplicate_frame_ratio_threshold: float = 0.70,
        duplicate_center_distance_ratio: float = 0.25,
        duplicate_max_overlap_frames: int = 45,
        duplicate_max_overlap_ratio: float = 0.30,
        duplicate_min_shared_frames: int = 2,
    ) -> None:
        for name, value in (
            ("similarity_threshold", similarity_threshold),
            ("duplicate_similarity_threshold", duplicate_similarity_threshold),
            ("duplicate_iou_threshold", duplicate_iou_threshold),
            ("duplicate_frame_ratio_threshold", duplicate_frame_ratio_threshold),
            ("duplicate_max_overlap_ratio", duplicate_max_overlap_ratio),
        ):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"intra_camera.{name} must be in [-1, 1].")
        if max_time_gap_seconds < 0 or time_penalty_weight < 0:
            raise ValueError("Intra-camera time settings must be non-negative.")
        if min_embeddings < 1 or duplicate_min_shared_frames < 1:
            raise ValueError("Intra-camera sample minimums must be at least one.")
        if duplicate_max_overlap_frames < 1 or duplicate_center_distance_ratio < 0:
            raise ValueError("Invalid duplicate-overlap geometry setting.")
        self.threshold = float(similarity_threshold)
        self.max_time_gap = float(max_time_gap_seconds)
        self.time_penalty_weight = float(time_penalty_weight)
        self.min_embeddings = int(min_embeddings)
        self.enabled = bool(enabled)
        self.duplicate_similarity_threshold = float(duplicate_similarity_threshold)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.duplicate_frame_ratio_threshold = float(duplicate_frame_ratio_threshold)
        self.duplicate_center_distance_ratio = float(duplicate_center_distance_ratio)
        self.duplicate_max_overlap_frames = int(duplicate_max_overlap_frames)
        self.duplicate_max_overlap_ratio = float(duplicate_max_overlap_ratio)
        self.duplicate_min_shared_frames = int(duplicate_min_shared_frames)

    def _usable(self, track: TrackSegment) -> bool:
        return track.embedding is not None and track.num_embeddings >= self.min_embeddings

    def _overlap_metrics(self, a: TrackSegment, b: TrackSegment) -> _OverlapMetrics:
        shared = sorted(set(a.observations).intersection(b.observations))
        if not shared:
            return _OverlapMetrics(0, 0.0, float("nan"), 0.0, float("nan"))
        ious = np.asarray([
            bbox_iou(a.observations[frame].bbox, b.observations[frame].bbox)
            for frame in shared
        ])
        centers = np.asarray([
            center_distance_ratio(a.observations[frame].bbox, b.observations[frame].bbox)
            for frame in shared
        ])
        shorter = max(1, min(a.num_frames, b.num_frames))
        return _OverlapMetrics(
            frames=len(shared),
            ratio=len(shared) / shorter,
            median_iou=float(np.median(ious)),
            duplicate_frame_ratio=float(np.mean(ious >= self.duplicate_iou_threshold)),
            median_center_distance_ratio=float(np.median(centers)),
        )

    def _duplicate_reason(self, similarity: float, metrics: _OverlapMetrics) -> str | None:
        checks = (
            (similarity >= self.duplicate_similarity_threshold, "appearance_below_threshold"),
            (metrics.frames >= self.duplicate_min_shared_frames, "too_few_shared_frames"),
            (metrics.frames <= self.duplicate_max_overlap_frames, "overlap_too_long"),
            (metrics.ratio <= self.duplicate_max_overlap_ratio, "overlap_ratio_too_high"),
            (metrics.median_iou >= self.duplicate_iou_threshold, "median_iou_too_low"),
            (
                metrics.duplicate_frame_ratio >= self.duplicate_frame_ratio_threshold,
                "duplicate_frame_ratio_too_low",
            ),
            (
                metrics.median_center_distance_ratio <= self.duplicate_center_distance_ratio,
                "box_centers_too_far",
            ),
        )
        for accepted, reason in checks:
            if not accepted:
                return reason
        return None

    def _match_camera(
        self, tracks: list[TrackSegment]
    ) -> tuple[list[IntraCameraMatch], list[IntraCameraDecision]]:
        if not self.enabled:
            return [], []
        usable = [track for track in tracks if self._usable(track)]
        if len(usable) < 2:
            return [], []

        invalid_cost = 1e6
        cost = np.full((len(usable), len(usable)), invalid_cost, dtype=np.float64)
        candidates: dict[tuple[int, int], IntraCameraMatch] = {}
        decisions: list[IntraCameraDecision] = []

        for row, previous in enumerate(usable):
            for column, current in enumerate(usable):
                if row == column:
                    continue
                previous_order = (
                    previous.start_time, previous.start_frame,
                    previous.local_track_id, previous.segment_index,
                )
                current_order = (
                    current.start_time, current.start_frame,
                    current.local_track_id, current.segment_index,
                )
                if previous_order >= current_order:
                    continue
                similarity = cosine_similarity(previous.embedding, current.embedding)
                metrics = self._overlap_metrics(previous, current)

                if metrics.frames:
                    reason = self._duplicate_reason(similarity, metrics)
                    if reason is not None:
                        decisions.append(IntraCameraDecision(
                            previous.key, current.key, "duplicate_overlap", False, reason,
                            similarity, metrics.frames, metrics.ratio, metrics.median_iou,
                            metrics.duplicate_frame_ratio,
                            metrics.median_center_distance_ratio,
                        ))
                        continue
                    score = 0.70 * similarity + 0.30 * metrics.median_iou
                    match_type = "duplicate_overlap"
                    gap = 0.0
                else:
                    if previous.end_time >= current.start_time:
                        continue
                    gap = current.start_time - previous.end_time
                    if gap > self.max_time_gap or similarity < self.threshold:
                        continue
                    normalized_gap = gap / self.max_time_gap if self.max_time_gap > 0 else 0.0
                    score = similarity - self.time_penalty_weight * normalized_gap
                    match_type = "reentry"

                match = IntraCameraMatch(
                    previous.key, current.key, match_type, similarity, gap,
                    metrics.frames, metrics.ratio, metrics.median_iou,
                    metrics.duplicate_frame_ratio, metrics.median_center_distance_ratio,
                    score,
                )
                candidates[(row, column)] = match
                cost[row, column] = 1.0 - score

        rows, columns = linear_sum_assignment(cost)
        selected = {
            (int(row), int(column))
            for row, column in zip(rows, columns) if cost[row, column] < invalid_cost
        }
        matches = [candidates[key] for key in selected]
        for key, match in candidates.items():
            if match.match_type != "duplicate_overlap":
                continue
            accepted = key in selected
            decisions.append(IntraCameraDecision(
                match.previous_track, match.next_track, match.match_type, accepted,
                "accepted" if accepted else "not_selected_by_hungarian",
                match.appearance_similarity, match.overlap_frames, match.overlap_ratio,
                match.median_iou, match.duplicate_frame_ratio,
                match.median_center_distance_ratio,
            ))
        return matches, decisions

    @staticmethod
    def _aggregate_group(camera_id: str, camera_person_id: int,
                         members: list[TrackSegment]) -> LocalTrack:
        observations = {}
        samples_by_frame = {}
        fallback_embeddings: list[np.ndarray] = []
        for track in members:
            for frame, observation in track.observations.items():
                previous = observations.get(frame)
                if previous is None or observation.confidence > previous.confidence:
                    observations[frame] = observation
            for sample in track.embedding_samples:
                previous = samples_by_frame.get(sample.frame_index)
                if previous is None or sample.confidence > previous.confidence:
                    samples_by_frame[sample.frame_index] = sample
            if not track.embedding_samples and track.embedding is not None:
                fallback_embeddings.append(track.embedding)

        aggregate = LocalTrack(
            camera_id=camera_id,
            local_track_id=camera_person_id,
            start_frame=min(track.start_frame for track in members),
            end_frame=max(track.end_frame for track in members),
            start_time=min(track.start_time for track in members),
            end_time=max(track.end_time for track in members),
            num_frames=len(observations) if observations else sum(track.num_frames for track in members),
            embeddings=fallback_embeddings,
            observations=observations,
            embedding_samples=[samples_by_frame[key] for key in sorted(samples_by_frame)],
        )
        if aggregate.embedding_samples:
            aggregate.embedding, _, _ = robust_descriptor(
                aggregate.embedding_samples, outlier_similarity_threshold=0.55
            )
        else:
            aggregate.finalize()
        return aggregate

    def assign(self, tracks: list[TrackSegment]) -> IntraCameraAssignment:
        by_camera: dict[str, list[TrackSegment]] = {}
        track_by_key = {track.key: track for track in tracks}
        for track in tracks:
            by_camera.setdefault(track.camera_id, []).append(track)

        key_to_camera_id: dict[SegmentKey, int] = {}
        camera_tracks: list[LocalTrack] = []
        groups: dict[TrackKey, list[SegmentKey]] = {}
        similarity_by_track: dict[SegmentKey, float] = {}
        accepted_matches: list[IntraCameraMatch] = []
        decisions: list[IntraCameraDecision] = []

        for camera_id in sorted(by_camera):
            ordered_tracks = sorted(
                by_camera[camera_id],
                key=lambda track: (
                    track.start_time, track.local_track_id, track.segment_index
                ),
            )
            matches, camera_decisions = self._match_camera(ordered_tracks)
            accepted_matches.extend(matches)
            decisions.extend(camera_decisions)
            parent = {track.key: track.key for track in ordered_tracks}

            def find(key: SegmentKey) -> SegmentKey:
                while parent[key] != key:
                    parent[key] = parent[parent[key]]
                    key = parent[key]
                return key

            for match in matches:
                root_previous, root_current = find(match.previous_track), find(match.next_track)
                if root_previous != root_current:
                    parent[root_current] = root_previous
                for key in (match.previous_track, match.next_track):
                    similarity_by_track[key] = max(
                        similarity_by_track.get(key, -1.0), match.appearance_similarity
                    )

            raw_groups: dict[SegmentKey, list[SegmentKey]] = {}
            for track in ordered_tracks:
                raw_groups.setdefault(find(track.key), []).append(track.key)
            ordered_groups = sorted(
                raw_groups.values(),
                key=lambda keys: min(
                    (
                        track_by_key[key].start_time,
                        track_by_key[key].local_track_id,
                        track_by_key[key].segment_index,
                    )
                    for key in keys
                ),
            )
            for camera_person_id, member_keys in enumerate(ordered_groups, start=1):
                member_keys = sorted(
                    member_keys,
                    key=lambda key: (
                        track_by_key[key].start_time,
                        track_by_key[key].local_track_id,
                        track_by_key[key].segment_index,
                    ),
                )
                camera_key = camera_id, camera_person_id
                groups[camera_key] = member_keys
                for key in member_keys:
                    key_to_camera_id[key] = camera_person_id
                camera_tracks.append(self._aggregate_group(
                    camera_id, camera_person_id,
                    [track_by_key[key] for key in member_keys],
                ))

        return IntraCameraAssignment(
            key_to_camera_id=key_to_camera_id,
            camera_tracks=sorted(camera_tracks, key=lambda track: track.key),
            groups=groups,
            similarity_by_track=similarity_by_track,
            accepted_matches=accepted_matches,
            decisions=decisions,
        )
