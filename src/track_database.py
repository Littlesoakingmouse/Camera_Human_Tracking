"""Aggregation of frame observations and embeddings into tracks and segments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detector_tracker import TrackObservation
from .utils.common import l2_normalize


TrackKey = tuple[str, int]
SegmentKey = tuple[str, int, int]
ObservationKey = tuple[str, int, int]


@dataclass(frozen=True)
class EmbeddingSample:
    """A Re-ID descriptor tied to the observation that produced it."""

    frame_index: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    confidence: float
    vector: np.ndarray = field(repr=False)


def robust_descriptor(
    samples: list[EmbeddingSample], outlier_similarity_threshold: float = -1.0
) -> tuple[np.ndarray | None, float, int]:
    """Return an L2 descriptor, median quality, and number of retained samples."""
    if not samples:
        return None, float("nan"), 0
    vectors = np.stack([l2_normalize(sample.vector) for sample in samples])
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)
    medoid_index = int(np.argmax(np.mean(similarities, axis=1)))
    similarity_to_medoid = similarities[medoid_index]
    keep = similarity_to_medoid >= outlier_similarity_threshold
    if not np.any(keep):
        keep[medoid_index] = True
    retained = vectors[keep]
    descriptor = l2_normalize(np.mean(retained, axis=0))
    quality = float(np.median(np.clip(retained @ descriptor, -1.0, 1.0)))
    return descriptor, quality, int(retained.shape[0])


@dataclass
class LocalTrack:
    """One raw ByteTrack identity before appearance-based splitting."""

    camera_id: str
    local_track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    num_frames: int = 0
    # Kept for model-free tests and imported/precomputed tracks.
    embeddings: list[np.ndarray] = field(default_factory=list, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)
    observations: dict[int, TrackObservation] = field(default_factory=dict, repr=False)
    embedding_samples: list[EmbeddingSample] = field(default_factory=list, repr=False)

    @property
    def key(self) -> TrackKey:
        return self.camera_id, self.local_track_id

    @property
    def num_embeddings(self) -> int:
        return len(self.embedding_samples) if self.embedding_samples else len(self.embeddings)

    def finalize(self) -> None:
        """Mean-pool raw descriptors for backward-compatible local-track diagnostics."""
        vectors = (
            [sample.vector for sample in self.embedding_samples]
            if self.embedding_samples else self.embeddings
        )
        if vectors:
            self.embedding = l2_normalize(np.mean(np.stack(vectors), axis=0))


@dataclass
class TrackSegment:
    """A temporally contiguous, appearance-consistent portion of one local track."""

    camera_id: str
    local_track_id: int
    segment_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    observations: dict[int, TrackObservation] = field(default_factory=dict, repr=False)
    embedding_samples: list[EmbeddingSample] = field(default_factory=list, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)
    descriptor_quality: float = float("nan")
    retained_embeddings: int = 0
    split_reason: str = "none"

    @property
    def key(self) -> SegmentKey:
        return self.camera_id, self.local_track_id, self.segment_index

    @property
    def num_frames(self) -> int:
        return len(self.observations)

    @property
    def num_embeddings(self) -> int:
        return len(self.embedding_samples)

    def finalize(self, outlier_similarity_threshold: float = 0.55) -> None:
        self.embedding, self.descriptor_quality, self.retained_embeddings = robust_descriptor(
            self.embedding_samples, outlier_similarity_threshold
        )


class TrackDatabase:
    """Mutable collection keyed by raw ``(camera_id, local_track_id)``."""

    def __init__(self) -> None:
        self.tracks: dict[TrackKey, LocalTrack] = {}

    def update(self, observation: TrackObservation) -> None:
        key = observation.camera_id, observation.local_track_id
        track = self.tracks.get(key)
        if track is None:
            track = LocalTrack(
                observation.camera_id, observation.local_track_id,
                observation.frame_index, observation.frame_index,
                observation.timestamp, observation.timestamp,
            )
            self.tracks[key] = track
        track.start_frame = min(track.start_frame, observation.frame_index)
        track.end_frame = max(track.end_frame, observation.frame_index)
        track.start_time = min(track.start_time, observation.timestamp)
        track.end_time = max(track.end_time, observation.timestamp)
        track.observations[observation.frame_index] = observation
        track.num_frames = len(track.observations)

    def add_embedding(self, key: TrackKey, observation: TrackObservation,
                      embedding: np.ndarray) -> None:
        if key not in self.tracks:
            raise KeyError(f"Unknown local track: {key}")
        self.tracks[key].embedding_samples.append(
            EmbeddingSample(
                frame_index=observation.frame_index,
                timestamp=observation.timestamp,
                bbox=observation.bbox,
                confidence=observation.confidence,
                vector=l2_normalize(embedding),
            )
        )

    def finalize(self) -> None:
        for track in self.tracks.values():
            track.finalize()

    def values(self) -> list[LocalTrack]:
        return sorted(self.tracks.values(), key=lambda item: (item.camera_id, item.local_track_id))

    def for_camera(self, camera_id: str) -> list[LocalTrack]:
        return [track for track in self.values() if track.camera_id == camera_id]
