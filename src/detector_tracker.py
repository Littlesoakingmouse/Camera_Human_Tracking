"""YOLO person detection and Ultralytics ByteTrack integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
    """Own a YOLO model and maintain ByteTrack state across successive frames."""

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
        self._validate_files()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ProjectError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        # A verified local path prevents Ultralytics from treating the value as a model name to download.
        self._yolo_class = YOLO
        self.model = YOLO(str(self.model_path))
        self._has_started = False

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
        """Discard predictor/tracker state before starting a different camera."""
        # A fresh model wrapper guarantees that no ByteTrack state crosses cameras.
        if self._has_started:
            self.model = self._yolo_class(str(self.model_path))
        else:
            self._has_started = True
            self.model.predictor = None

    @staticmethod
    def _clip_box(values: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in values)
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def detect_frame(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in one BGR frame without updating tracker state."""
        result = self.model.predict(
            source=frame, conf=self.confidence, iou=self.iou_threshold,
            classes=[self.person_class_id],
            device=str(self.device), verbose=False
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        height, width = frame.shape[:2]
        detections: list[Detection] = []
        for raw_box, score, class_id in zip(xyxy, confidence, classes):
            clipped = self._clip_box(raw_box, width, height)
            if clipped is not None:
                detections.append(Detection(clipped, float(score), int(class_id)))
        return detections

    def track_frame(
        self, frame: np.ndarray, camera_id: str, frame_index: int, timestamp: float
    ) -> list[TrackObservation]:
        """Detect and track people in one BGR frame."""
        result = self.model.track(
            source=frame,
            persist=True,
            tracker=str(self.tracker_path),
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=[self.person_class_id],
            device=str(self.device),
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        track_ids = boxes.id.detach().cpu().numpy().astype(int)
        height, width = frame.shape[:2]
        observations: list[TrackObservation] = []
        for raw_box, score, track_id in zip(xyxy, confidence, track_ids):
            if track_id < 0:
                continue
            clipped = self._clip_box(raw_box, width, height)
            if clipped is not None:
                observations.append(
                    TrackObservation(
                        camera_id, frame_index, timestamp, int(track_id), clipped, float(score)
                    )
                )
        return observations
