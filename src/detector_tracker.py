"""YOLO person detection and Ultralytics ByteTrack integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import yaml

from .utils.common import ProjectError, project_path, select_device


def _has_quantize_flag() -> bool:
    try:
        from importlib.metadata import version

        major, minor = (int(part) for part in version("ultralytics").split(".")[:2])
        return (major, minor) >= (8, 4)
    except Exception:  # unknown version: assume the current API
        return True


_ULTRALYTICS_HAS_QUANTIZE = _has_quantize_flag()



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
    # OSNet feature of the matched detection when the tracker computed one (BoT-SORT).
    embedding: np.ndarray | None = field(default=None, repr=False, compare=False)


class DetectorTracker:
    """Batched YOLO detection with a standalone ByteTrack updated frame by frame.

    Detection is stateless, so frames can be inferred in batches. Tracking is
    stateful, so each frame's detections are fed to ByteTrack in order. This is
    the same association Ultralytics ``model.track`` performs, without its
    per-call overhead.
    """

    def __init__(self, config: dict[str, Any], reid_model: Any = None) -> None:
        detector_cfg = config["detector"]
        # Emit the first frame of each track once it is confirmed (see _backfill).
        self.backfill_first_frame = bool(config["tracker"].get("backfill_first_frame", False))
        self.backfilled: list[TrackObservation] = []
        self._pending: dict[int, TrackObservation] = {}
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
            from ultralytics.trackers.bot_sort import BOTSORT
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace
        except ImportError as exc:
            raise ProjectError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc
        with self.tracker_path.open("r", encoding="utf-8") as stream:
            tracker_cfg = yaml.safe_load(stream)
        tracker_type = tracker_cfg.get("tracker_type")
        if tracker_type not in {"bytetrack", "botsort"}:
            raise ProjectError("tracker.config_path must describe a bytetrack or botsort tracker.")
        self.tracker_type = tracker_type
        # BoT-SORT with Re-ID uses this project's OSNet instead of an Ultralytics ReID
        # model: features of every detection are computed here, one OSNet pass per
        # frame batch, and handed to the tracker ("auto" = use the given features).
        # The same features become the Re-ID samples of the pipeline, so appearance
        # costs one forward per detection instead of two.
        self._reid = None
        if tracker_type == "botsort" and tracker_cfg.get("with_reid", False):
            if reid_model is None:
                raise ProjectError("BoT-SORT with_reid needs the OSNet model (pass reid_model).")
            self._reid = reid_model
            tracker_cfg = {**tracker_cfg, "model": "auto"}
        self._tracker_args = IterableSimpleNamespace(**tracker_cfg)
        self._tracker_class = BOTSORT if tracker_type == "botsort" else BYTETracker
        # A verified local path prevents Ultralytics from treating the value as a model name to download.
        self.model = YOLO(str(self.model_path))
        self.tracker = self._new_tracker()

    @property
    def provides_embeddings(self) -> bool:
        """True when observations carry the OSNet feature used for tracking."""
        return self._reid is not None

    def _new_tracker(self) -> Any:
        tracker = self._tracker_class(self._tracker_args)
        if self._reid is not None and not hasattr(tracker, "_input_for"):
            raise ProjectError("BoT-SORT with OSNet Re-ID needs ultralytics>=8.4.")
        return tracker

    def _detection_features(self, frames: list[np.ndarray],
                            results: list[Any]) -> list[np.ndarray | None]:
        """OSNet features for every detection of the batch, in one forward pass."""
        if self._reid is None:
            return [None] * len(frames)
        crops, owners, counts = [], [], []
        for frame_number, (frame, result) in enumerate(zip(frames, results)):
            height, width = frame.shape[:2]
            boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else []
            counts.append(len(boxes))
            for row, raw in enumerate(boxes):
                clipped = self._clip_box(raw, width, height)
                if clipped is None:
                    continue
                x1, y1, x2, y2 = clipped
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    crops.append(frame[y1:y2, x1:x2])
                    owners.append((frame_number, row))
        vectors = self._reid.extract_embeddings(crops) if crops else np.zeros((0, 512), np.float32)
        dimension = vectors.shape[1] if len(vectors) else 512
        # Tiny boxes keep a zero vector: Ultralytics then ignores appearance for them.
        features = [np.zeros((count, dimension), np.float32) for count in counts]
        for (frame_number, row), vector in zip(owners, vectors):
            features[frame_number][row] = vector
        return features

    @staticmethod
    def _feature(features: np.ndarray | None, index: int) -> np.ndarray | None:
        if features is None or not 0 <= index < len(features):
            return None
        vector = features[index]
        return vector if float(np.dot(vector, vector)) > 1e-12 else None

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
        self.tracker = self._new_tracker()
        self._pending.clear()
        self.backfilled.clear()

    @staticmethod
    def _clip_box(values: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in values)
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def _predict(self, frames: list[np.ndarray]) -> list[Any]:
        # Ultralytics 8.4 replaced the deprecated ``half`` flag with ``quantize``;
        # requirements allow 8.3, so pick the argument the installed version knows.
        precision = ({"quantize": 16 if self.half else None} if _ULTRALYTICS_HAS_QUANTIZE
                     else {"half": self.half})
        return self.model.predict(
            source=frames, conf=self.confidence, iou=self.iou_threshold,
            classes=[self.person_class_id], imgsz=self.image_size,
            batch=len(frames), device=str(self.device), verbose=False, **precision,
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
        batch_features = self._detection_features(frames, results)
        batch_observations: list[list[TrackObservation]] = []
        for frame, result, frame_index, timestamp, features in zip(
            frames, results, frame_indices, timestamps, batch_features
        ):
            # The tracker must see every frame, including empty ones, to age lost tracks.
            tracks = self.tracker.update(result.boxes.cpu().numpy(), frame, feats=features)
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
                            camera_id, frame_index, timestamp, track_id, clipped, float(row[5]),
                            self._feature(features, int(row[7])),
                        ))
            if self.backfill_first_frame:
                self._backfill(camera_id, frame_index, timestamp, frame, observations, features)
            batch_observations.append(observations)
        return batch_observations

    def _backfill(self, camera_id: str, frame_index: int, timestamp: float,
                  frame: np.ndarray, observations: list[TrackObservation],
                  features: np.ndarray | None = None) -> None:
        """Recover the first box of every track.

        Ultralytics creates a track as *unconfirmed* and only reports it from the
        frame where it is matched again, so the first detection of every person is
        dropped. At low frame rates a person is visible in few frames and this loss
        is large. Offline we remember the unconfirmed box and emit it once the track
        is confirmed; the pipeline inserts it at its own frame index.
        """
        reported = {observation.local_track_id for observation in observations}
        for track_id in reported & set(self._pending):
            self.backfilled.append(self._pending.pop(track_id))
        height, width = frame.shape[:2]
        current = self.tracker.frame_id
        for track in self.tracker.tracked_stracks:
            if track.is_activated or track.frame_id != current or track.track_id in reported:
                continue
            clipped = self._clip_box(np.asarray(track.xyxy), width, height)
            if clipped is not None:
                self._pending[int(track.track_id)] = TrackObservation(
                    camera_id, frame_index, timestamp, int(track.track_id), clipped,
                    float(track.score), self._feature(features, int(getattr(track, "idx", -1))),
                )
        # Unconfirmed tracks that found no match are dropped by the tracker.
        alive = {int(track.track_id) for track in self.tracker.tracked_stracks}
        for track_id in list(self._pending):
            if track_id not in alive:
                del self._pending[track_id]

    def drain_backfilled(self) -> list[TrackObservation]:
        """Return and clear first-frame boxes recovered since the last call."""
        items, self.backfilled = self.backfilled, []
        return items

    def track_frame(
        self, frame: np.ndarray, camera_id: str, frame_index: int, timestamp: float
    ) -> list[TrackObservation]:
        """Detect and track people in one BGR frame."""
        return self.track_batch([frame], camera_id, [frame_index], [timestamp])[0]
