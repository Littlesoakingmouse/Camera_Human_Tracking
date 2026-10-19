"""YOLO person detection and Ultralytics ByteTrack integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from .utils.common import ProjectError, project_path, select_device


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int


@dataclass(frozen=True)
class TrackObservation:
    camera_id: str
    frame_index: int
    timestamp: float
    local_track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float


class DetectorTracker:
    """Batched YOLO detection with a standalone ByteTrack updated frame by frame.

    Detection is stateless, so frames can be inferred in batches. Tracking is
    stateful, so each frame's detections are fed to ByteTrack in order. This is
    the same association Ultralytics ``model.track`` performs, without its
    per-call overhead.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        detector_cfg = config["detector"]
        self.model_path = project_path(config, detector_cfg["model_path"])
        self.tracker_path = project_path(config, config["tracker"]["config_path"])
        self.confidence = float(detector_cfg["confidence_threshold"])
        self.iou_threshold = float(detector_cfg.get("iou_threshold", 0.60))
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ProjectError("detector.iou_threshold must be between 0 and 1.")
        self.person_class_id = int(detector_cfg["person_class_id"])
        self.device = select_device(detector_cfg.get("device", "auto"))
        self.batch_size = int(detector_cfg.get("batch_size", 8))
        if self.batch_size < 1:
            raise ProjectError("detector.batch_size must be at least one.")
        # FP16 roughly halves GPU memory traffic; it is ignored on CPU.
        self.half = bool(detector_cfg.get("half", True)) and self.device.type == "cuda"
        self.image_size = int(detector_cfg.get("image_size", 640))
        self._validate_files()
        try:
            from ultralytics import YOLO
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
        except ImportError as exc:
            raise ProjectError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        with self.tracker_path.open("r", encoding="utf-8") as stream:
            tracker_cfg = yaml.safe_load(stream)
        if tracker_cfg.get("tracker_type") != "bytetrack":
            raise ProjectError("tracker.config_path must describe a bytetrack tracker.")
        self._tracker_args = IterableSimpleNamespace(**tracker_cfg)
        self._tracker_class = BYTETracker
        # A verified local path prevents Ultralytics from treating the value as a model name to download.
        self.model = YOLO(str(self.model_path))
        self.tracker = BYTETracker(self._tracker_args)

    def _validate_files(self) -> None:
        if not self.model_path.is_file():
            raise ProjectError(
                f"YOLO model not found:\n{self.model_path}\n\n"
                "Please place your YOLO weights at this location or update "
                "detector.model_path in config.yaml."
            )
        if not self.tracker_path.is_file():
            raise ProjectError(f"ByteTrack configuration not found:\n{self.tracker_path}")

    def reset(self) -> None:
        """Discard tracker state before starting a different camera."""
        # A fresh tracker also restarts track IDs at 1, so no LID crosses cameras.
        self.tracker = self._tracker_class(self._tracker_args)

    @staticmethod
    def _clip_box(values: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in values)
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _predict(self, frames: list[np.ndarray]) -> list[Any]:
        return self.model.predict(
            source=frames, conf=self.confidence, iou=self.iou_threshold,
            classes=[self.person_class_id], imgsz=self.image_size,
            # Ultralytics 8.4 replaced the deprecated ``half`` flag with ``quantize``.
            quantize=16 if self.half else None,
            batch=len(frames), device=str(self.device), verbose=False,
        )

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[Detection]]:
        """Detect people in several BGR frames without updating tracker state."""
        detections: list[list[Detection]] = []
        for frame, result in zip(frames, self._predict(frames)):
            boxes = result.boxes
            frame_detections: list[Detection] = []
            if boxes is not None and len(boxes) > 0:
                height, width = frame.shape[:2]
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidence = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for raw_box, score, class_id in zip(xyxy, confidence, classes):
                    clipped = self._clip_box(raw_box, width, height)
                    if clipped is not None:
                        frame_detections.append(Detection(clipped, float(score), int(class_id)))
            detections.append(frame_detections)
        return detections

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in one BGR frame without updating tracker state."""
        return self.detect_batch([frame])[0]

    def track_batch(
        self, frames: list[np.ndarray], camera_id: str, frame_indices: list[int],
        timestamps: list[float],
    ) -> list[list[TrackObservation]]:
        """Detect a batch of consecutive frames, then update ByteTrack in frame order."""
        results = self._predict(frames)
        batch_observations: list[list[TrackObservation]] = []
        for frame, result, frame_index, timestamp in zip(
            frames, results, frame_indices, timestamps
        ):
            # The tracker must see every frame, including empty ones, to age lost tracks.
            tracks = self.tracker.update(result.boxes.cpu().numpy(), frame)
            observations: list[TrackObservation] = []
            if len(tracks):
                height, width = frame.shape[:2]
                # Columns: x1, y1, x2, y2, track_id, score, class, detection_index.
                for row in tracks:
                    track_id = int(row[4])
                    if track_id < 0:
                        continue
                    clipped = self._clip_box(row[:4], width, height)
                    if clipped is not None:
                        observations.append(TrackObservation(
                            camera_id, frame_index, timestamp, track_id, clipped, float(row[5])
                        ))
            batch_observations.append(observations)
        return batch_observations

    def track_frame(
        self, frame: np.ndarray, camera_id: str, frame_index: int, timestamp: float
    ) -> list[TrackObservation]:
        """Detect and track people in one BGR frame."""
        return self.track_batch([frame], camera_id, [frame_index], [timestamp])[0]
