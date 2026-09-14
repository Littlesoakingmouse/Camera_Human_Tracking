"""Small, defensive OpenCV video helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .common import ProjectError


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    """Context-managed frame iterator that validates video metadata."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if not path.is_file():
            raise ProjectError(
                f"Input video not found:\n{path}\n\n"
                "Please place the video at this location or update its path in config.yaml."
            )
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise ProjectError(f"OpenCV could not open input video:\n{path}")
        fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or width <= 0 or height <= 0:
            self.capture.release()
            raise ProjectError(f"Input video has invalid FPS or resolution:\n{path}")
        self.metadata = VideoMetadata(fps, width, height, frame_count)

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame_index, frame
            frame_index += 1

    def close(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def create_video_writer(path: Path, metadata: VideoMetadata, codec: str = "mp4v") -> cv2.VideoWriter:
    """Create and validate an output writer using source FPS and resolution."""
    if len(codec) != 4:
        raise ProjectError("output.codec must contain exactly four characters (for example, mp4v).")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), metadata.fps, (metadata.width, metadata.height)
    )
    if not writer.isOpened():
        raise ProjectError(f"OpenCV could not create output video:\n{path}")
    return writer
