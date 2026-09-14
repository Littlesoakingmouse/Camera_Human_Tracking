"""Smoke-test YOLO person detection on a few frames from one configured video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector_tracker import DetectorTracker  # noqa: E402
from src.utils.common import ProjectError, load_config, project_path  # noqa: E402
from src.utils.video import VideoReader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    parser.add_argument("--camera", default="cam1")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        camera = next((item for item in config["video"].values()
                       if str(item["id"]) == args.camera), None)
        if camera is None:
            raise ProjectError(f"Camera not found in configuration: {args.camera}")
        model = DetectorTracker(config)
        detection_count = 0
        with VideoReader(project_path(config, camera["path"])) as reader:
            for frame_index, frame in reader:
                detections = model.detect_frame(frame)
                detection_count += len(detections)
                print(f"frame={frame_index}: people={len(detections)}")
                if frame_index + 1 >= args.frames:
                    break
        print(f"Detector test complete: {detection_count} person detections")
        return 0
    except ProjectError as exc:
        print(f"ERROR:\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
