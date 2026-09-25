"""Split raw tracker IDs at sustained appearance change points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track_database import (
    LocalTrack,
    ObservationKey,
    SegmentKey,
    TrackSegment,
)
from .utils.common import l2_normalize


@dataclass(frozen=True)
class SplitEvent:
    camera_id: str
    local_track_id: int
    boundary_frame: int
    left_segment_index: int
    right_segment_index: int
    window_similarity: float
    reason: str = "appearance_change"


@dataclass(frozen=True)
class SegmentationResult:
    segments: list[TrackSegment]
    frame_to_segment: dict[ObservationKey, SegmentKey]
    split_events: list[SplitEvent]


class TrackSegmenter:
    """Offline change-point detection over ordered Re-ID embedding samples."""

    def __init__(
        self,
        window_size: int = 4,
        change_similarity_threshold: float = 0.65,
        min_separation_frames: int = 20,
        min_segment_embeddings: int = 4,
        outlier_similarity_threshold: float = 0.55,
        enabled: bool = True,
    ) -> None:
        if window_size < 1:
            raise ValueError("tracklet_split.window_size must be at least one.")
        if min_segment_embeddings < 1:
            raise ValueError("tracklet_split.min_segment_embeddings must be at least one.")
        if min_separation_frames < 0:
            raise ValueError("tracklet_split.min_separation_frames must be non-negative.")
        for name, value in (
            ("change_similarity_threshold", change_similarity_threshold),
            ("outlier_similarity_threshold", outlier_similarity_threshold),
        ):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"tracklet_split.{name} must be in [-1, 1].")
        self.window_size = int(window_size)
        self.change_threshold = float(change_similarity_threshold)
        self.min_separation_frames = int(min_separation_frames)
        self.min_segment_embeddings = int(min_segment_embeddings)
        self.outlier_threshold = float(outlier_similarity_threshold)
        self.enabled = bool(enabled)

    @staticmethod
    def _window_descriptor(vectors: list[np.ndarray]) -> np.ndarray:
        return l2_normalize(np.mean(np.stack(vectors), axis=0))

    def _candidate_boundaries(self, track: LocalTrack) -> list[tuple[int, float]]:
        samples = sorted(track.embedding_samples, key=lambda sample: sample.frame_index)
        minimum = max(self.window_size, self.min_segment_embeddings)
        if not self.enabled or len(samples) < 2 * minimum:
            return []

        candidates: list[tuple[int, float]] = []
        for index in range(self.window_size, len(samples) - self.window_size + 1):
            left = self._window_descriptor([
                sample.vector for sample in samples[index - self.window_size:index]
            ])
            right = self._window_descriptor([
                sample.vector for sample in samples[index:index + self.window_size]
            ])
            similarity = float(np.clip(np.dot(left, right), -1.0, 1.0))
            if similarity < self.change_threshold:
                candidates.append((index, similarity))

        if not candidates:
            return []

        # Consecutive low-similarity positions describe one transition. Keep its minimum.
        runs: list[list[tuple[int, float]]] = [[candidates[0]]]
        for candidate in candidates[1:]:
            if candidate[0] == runs[-1][-1][0] + 1:
                runs[-1].append(candidate)
            else:
                runs.append([candidate])
        minima = [min(run, key=lambda item: item[1]) for run in runs]

        selected: list[tuple[int, float]] = []
        previous_index = 0
        previous_frame = track.start_frame - self.min_separation_frames
        for index, similarity in minima:
            boundary_frame = (samples[index - 1].frame_index + samples[index].frame_index) // 2
            if index - previous_index < self.min_segment_embeddings:
                continue
            if len(samples) - index < self.min_segment_embeddings:
                continue
            if boundary_frame - previous_frame < self.min_separation_frames:
                continue
            selected.append((index, similarity))
            previous_index = index
            previous_frame = boundary_frame
        return selected

    def _segments_for_track(self, track: LocalTrack) -> tuple[list[TrackSegment], list[SplitEvent]]:
        samples = sorted(track.embedding_samples, key=lambda sample: sample.frame_index)
        boundaries = self._candidate_boundaries(track)
        sample_edges = [0, *[index for index, _ in boundaries], len(samples)]
        boundary_frames = [
            (samples[index - 1].frame_index + samples[index].frame_index) // 2
            for index, _ in boundaries
        ]
        observation_frames = sorted(track.observations)
        segments: list[TrackSegment] = []

        for offset, (sample_start, sample_end) in enumerate(
            zip(sample_edges[:-1], sample_edges[1:]), start=1
        ):
            lower = boundary_frames[offset - 2] if offset > 1 else track.start_frame - 1
            upper = boundary_frames[offset - 1] if offset <= len(boundary_frames) else track.end_frame
            observations = {
                frame: track.observations[frame]
                for frame in observation_frames if lower < frame <= upper
            }
            segment_samples = samples[sample_start:sample_end]
            if observations:
                start_observation = observations[min(observations)]
                end_observation = observations[max(observations)]
                start_frame, end_frame = min(observations), max(observations)
                start_time, end_time = start_observation.timestamp, end_observation.timestamp
            elif segment_samples:
                start_frame, end_frame = segment_samples[0].frame_index, segment_samples[-1].frame_index
                start_time, end_time = segment_samples[0].timestamp, segment_samples[-1].timestamp
            else:
                start_frame, end_frame = track.start_frame, track.end_frame
                start_time, end_time = track.start_time, track.end_time
            segment = TrackSegment(
                camera_id=track.camera_id,
                local_track_id=track.local_track_id,
                segment_index=offset,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                observations=observations,
                embedding_samples=list(segment_samples),
                split_reason="appearance_change" if boundaries else "none",
            )
            segment.finalize(self.outlier_threshold)
            if segment.embedding is None and track.embedding is not None and len(sample_edges) == 2:
                segment.embedding = track.embedding
                segment.descriptor_quality = float("nan")
            segments.append(segment)

        events = []
        for segment_index, ((sample_index, similarity), boundary_frame) in enumerate(
            zip(boundaries, boundary_frames), start=1
        ):
            del sample_index
            events.append(
                SplitEvent(
                    camera_id=track.camera_id,
                    local_track_id=track.local_track_id,
                    boundary_frame=boundary_frame,
                    left_segment_index=segment_index,
                    right_segment_index=segment_index + 1,
                    window_similarity=similarity,
                )
            )
        return segments, events

    def split(self, tracks: list[LocalTrack]) -> SegmentationResult:
        segments: list[TrackSegment] = []
        events: list[SplitEvent] = []
        frame_to_segment: dict[ObservationKey, SegmentKey] = {}
        for track in tracks:
            track_segments, track_events = self._segments_for_track(track)
            segments.extend(track_segments)
            events.extend(track_events)
            for segment in track_segments:
                for frame_index in segment.observations:
                    frame_to_segment[(segment.camera_id, segment.local_track_id, frame_index)] = (
                        segment.key
                    )
        return SegmentationResult(
            segments=sorted(
                segments,
                key=lambda item: (
                    item.camera_id, item.local_track_id, item.segment_index
                ),
            ),
            frame_to_segment=frame_to_segment,
            split_events=events,
        )
