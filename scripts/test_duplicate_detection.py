"""Regression test for the duplicate person detections seen in cam5 frame 3255."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector_tracker import DetectorTracker  # noqa: E402
from src.intra_camera_matcher import bbox_iou  # noqa: E402
from src.utils.common import load_config, project_path  # noqa: E402


def main() -> int:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    video_cfg = next(
        item for item in config["video"].values() if item["id"] == "cam5"
    )
    video_path = project_path(config, video_cfg["path"])
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 3255)
        ok, frame = capture.read()
    finally:
        capture.release()
    assert ok and frame is not None, f"Could not read cam5 frame 3255 from {video_path}"

    detector = DetectorTracker(config)
    detections = detector.detect_frame(frame)
    assert len(detections) == 2, (
        f"Expected exactly two people after NMS, got {len(detections)}: {detections}"
    )
    maximum_iou = max(
        bbox_iou(left.bbox, right.bbox)
        for index, left in enumerate(detections)
        for right in detections[index + 1:]
    )
    threshold = float(config["detector"]["iou_threshold"])
    assert maximum_iou < threshold, (
        f"A duplicate pair survived NMS: IoU={maximum_iou:.4f}, threshold={threshold:.2f}"
    )
    print(
        "PASS: cam5 frame 3255 contains 2 detections after NMS; "
        f"maximum pair IoU={maximum_iou:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
