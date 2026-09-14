"""Smoke-test ByteTrack ID persistence on one configured video."""

from __future__ import annotations

import argparse
import collections
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
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        camera = next((item for item in config["video"].values()
                       if str(item["id"]) == args.camera), None)
        if camera is None:
            raise ProjectError(f"Camera not found in configuration: {args.camera}")
        tracker = DetectorTracker(config)
        tracker.reset()
        counts: collections.Counter[int] = collections.Counter()
        offset = float(camera.get("time_offset_seconds", 0.0))
        with VideoReader(project_path(config, camera["path"])) as reader:
            for frame_index, frame in reader:
                timestamp = frame_index / reader.metadata.fps + offset
                observations = tracker.track_frame(frame, args.camera, frame_index, timestamp)
                counts.update(item.local_track_id for item in observations)
                print(f"frame={frame_index}: ids={[item.local_track_id for item in observations]}")
                if frame_index + 1 >= args.frames:
                    break
        print("\nFrames observed per local ID:")
        for track_id, count in sorted(counts.items()):
            print(f"  LID {track_id}: {count}")
        return 0
    except ProjectError as exc:
        print(f"ERROR:\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
