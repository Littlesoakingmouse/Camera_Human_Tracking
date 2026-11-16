"""Model-free test: the first (unconfirmed) box of a track is recovered.

YOLO is replaced by synthetic detections, so this runs on CPU in a second.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics.engine.results import Boxes  # noqa: E402

from src.detector_tracker import DetectorTracker  # noqa: E402
from src.utils.common import load_config  # noqa: E402


class FakeResult:
    def __init__(self, boxes: Boxes) -> None:
        self.boxes = boxes


def fake_predict(boxes_per_frame: dict[int, list[list[float]]]):
    counter = {"frame": 0}

    def predict(frames: list[np.ndarray]) -> list[FakeResult]:
        results = []
        for _ in frames:
            rows = boxes_per_frame.get(counter["frame"], [])
            data = torch.tensor([row + [0.9, 0.0] for row in rows], dtype=torch.float32)
            results.append(FakeResult(Boxes(data.reshape(-1, 6), (480, 640))))
            counter["frame"] += 1
        return results

    return predict


class FakeReID:
    """Stands in for OSNet: every crop gets the same unit vector."""

    def extract_embeddings(self, crops: list[np.ndarray]) -> np.ndarray:
        vectors = np.zeros((len(crops), 512), np.float32)
        vectors[:, 0] = 1.0
        return vectors


def run(backfill: bool, tracker_file: str = "configs/bytetrack.yaml") -> dict[int, list[int]]:
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    config["tracker"]["config_path"] = tracker_file
    config["tracker"]["backfill_first_frame"] = backfill
    config["detector"]["device"] = "cpu"
    tracker = DetectorTracker(config, FakeReID())
    tracker.reset()
    # A person appears at frame 3 (not the tracker's first frame) and walks right.
    boxes = {f: [[100 + 10 * (f - 3), 100, 160 + 10 * (f - 3), 260]] for f in range(3, 10)}
    tracker._predict = fake_predict(boxes)
    frame = np.zeros((480, 640, 3), np.uint8)
    seen: dict[int, list[int]] = {}
    for f in range(10):
        for obs in tracker.track_batch([frame], "cam1", [f], [f / 25.0])[0]:
            seen.setdefault(obs.local_track_id, []).append(obs.frame_index)
        for obs in tracker.drain_backfilled():
            seen.setdefault(obs.local_track_id, []).append(obs.frame_index)
    return {k: sorted(v) for k, v in seen.items()}


def main() -> int:
    plain = run(backfill=False)
    filled = run(backfill=True)
    assert list(plain.values()) == [list(range(4, 10))], plain
    assert list(filled.values()) == [list(range(3, 10))], filled
    print("PASS: first frame of a track is dropped by default and recovered by backfill")
    botsort = run(backfill=True, tracker_file="configs/botsort.yaml")
    assert list(botsort.values()) == [list(range(3, 10))], botsort
    print("PASS: BoT-SORT with OSNet features keeps one ID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
