"""Stable, readable OpenCV annotations."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

from .detector_tracker import TrackObservation


def stable_color(identity: int) -> tuple[int, int, int]:
    """Map an integer identity to a deterministic vivid BGR color."""
    hue = (identity * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(blue * 255), int(green * 255), int(red * 255)


class Visualizer:
    def __init__(self, show_local_id: bool = True, show_confidence: bool = False) -> None:
        self.show_local_id = show_local_id
        self.show_confidence = show_confidence

    def draw(self, frame: np.ndarray, observations: list[TrackObservation],
             global_ids: dict[tuple[str, int], int] | None = None) -> np.ndarray:
        """Draw observations in-place and return the frame."""
        for observation in observations:
            key = observation.camera_id, observation.local_track_id
            global_id = global_ids.get(key) if global_ids is not None else None
            identity = global_id if global_id is not None else observation.local_track_id
            color = stable_color(identity)
            x1, y1, x2, y2 = observation.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            parts = [f"GID: {global_id}"] if global_id is not None else []
            if self.show_local_id or global_id is None:
                parts.append(f"LID: {observation.local_track_id}")
            if self.show_confidence:
                parts.append(f"{observation.confidence:.2f}")
            label = " | ".join(parts)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            top = max(0, y1 - text_height - baseline - 6)
            cv2.rectangle(frame, (x1, top), (x1 + text_width + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, max(text_height + 1, y1 - baseline - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        return frame
