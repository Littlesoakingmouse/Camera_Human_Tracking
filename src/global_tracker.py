"""Conflict-aware global person ID assignment."""

from __future__ import annotations

from dataclasses import dataclass

from .matcher import CandidateMatch
from .track_database import LocalTrack, TrackKey


@dataclass(frozen=True)
class GlobalAssignment:
    key_to_global_id: dict[TrackKey, int]
    global_tracks: dict[int, list[TrackKey]]
    similarity_by_track: dict[TrackKey, float]
    accepted_matches: list[CandidateMatch]


class GlobalTracker:
    """Merge match edges while ensuring one local track per camera per identity."""

    def assign(self, tracks: list[LocalTrack], matches: list[CandidateMatch]) -> GlobalAssignment:
        keys = [track.key for track in tracks]
        parent = {key: key for key in keys}
        cameras = {key: {key[0]} for key in keys}

        def find(key: TrackKey) -> TrackKey:
            while parent[key] != key:
                parent[key] = parent[parent[key]]
                key = parent[key]
            return key

        accepted: list[CandidateMatch] = []
        similarities: dict[TrackKey, float] = {}
        # Strong edges win if pairwise assignments create a multi-camera conflict.
        for match in sorted(matches, key=lambda item: item.similarity, reverse=True):
            root_a, root_b = find(match.track_a), find(match.track_b)
            if root_a == root_b:
                continue
            if cameras[root_a].intersection(cameras[root_b]):
                continue
            parent[root_b] = root_a
            cameras[root_a].update(cameras[root_b])
            accepted.append(match)
            similarities[match.track_a] = max(similarities.get(match.track_a, -1.0), match.similarity)
            similarities[match.track_b] = max(similarities.get(match.track_b, -1.0), match.similarity)

        groups: dict[TrackKey, list[TrackKey]] = {}
        for key in keys:
            groups.setdefault(find(key), []).append(key)
        ordered_groups = sorted(
            (sorted(group) for group in groups.values()), key=lambda group: group[0]
        )
        global_tracks = {global_id: group for global_id, group in enumerate(ordered_groups, start=1)}
        key_to_global = {
            key: global_id for global_id, group in global_tracks.items() for key in group
        }
        return GlobalAssignment(key_to_global, global_tracks, similarities, accepted)
