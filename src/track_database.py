"""Aggregation of frame observations and embeddings into local tracks."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detector_tracker import TrackObservation
from .utils.common import l2_normalize


TrackKey = tuple[str, int]


@dataclass
class LocalTrack:
    camera_id: str
    local_track_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    num_frames: int = 0
    embeddings: list[np.ndarray] = field(default_factory=list, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)

    @property
    def key(self) -> TrackKey:
        return self.camera_id, self.local_track_id

    @property
    def num_embeddings(self) -> int:
        return len(self.embeddings)

    def finalize(self) -> None:
        """Mean-pool all descriptors and normalize the track descriptor."""
        if self.embeddings:
            self.embedding = l2_normalize(np.mean(np.stack(self.embeddings), axis=0))


class TrackDatabase:
    """Mutable collection keyed by (camera_id, local_track_id)."""

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
        track.num_frames += 1

    def add_embedding(self, key: TrackKey, embedding: np.ndarray) -> None:
        if key not in self.tracks:
            raise KeyError(f"Unknown local track: {key}")
        self.tracks[key].embeddings.append(l2_normalize(embedding))

    def finalize(self) -> None:
        for track in self.tracks.values():
            track.finalize()

    def values(self) -> list[LocalTrack]:
        return sorted(self.tracks.values(), key=lambda item: (item.camera_id, item.local_track_id))

    def for_camera(self, camera_id: str) -> list[LocalTrack]:
        return [track for track in self.values() if track.camera_id == camera_id]
