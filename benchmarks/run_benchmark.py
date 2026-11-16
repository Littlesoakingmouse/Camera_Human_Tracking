"""Run one pipeline version on the HDA benchmark and score it.

The runner imports the pipeline from ``--code-dir`` (the current checkout or a
git worktree of an older tag), so every version is measured with the same data,
the same metric code and the same hardware.

Usage:
    python -m benchmarks.run_benchmark --name v3.0
    python -m benchmarks.run_benchmark --name v2.1 --code-dir .versions/v2.1

Outputs in ``benchmarks/runs/<name>/`` (git-ignored, raw data) and a compact
``benchmarks/results/<name>.json`` (committed, used by the version report).
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from benchmarks.metrics import evaluate, remove_ignored  # noqa: E402

DATASET = PROJECT_ROOT / "data" / "benchmark" / "hda"
CAMERA_KEY_STRIDE = 10 ** 7


class PeakMemory:
    """Sample process RSS on a background thread."""

    def __init__(self, interval: float = 0.5) -> None:
        import psutil

        self._process = psutil.Process()
        self._interval = interval
        self._stop = threading.Event()
        self.peak = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._process.memory_info().rss)
            self._stop.wait(self._interval)

    def __enter__(self) -> "PeakMemory":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join()


def build_config(code_dir: Path, run_dir: Path, cameras: list[str],
                 overrides: dict[str, Any]) -> dict[str, Any]:
    with (code_dir / "configs" / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config = copy.deepcopy(config)
    # Weights are git-ignored, so worktrees of older tags read them from the main checkout.
    config["detector"]["model_path"] = str(PROJECT_ROOT / config["detector"]["model_path"])
    config["reid"]["model_path"] = str(PROJECT_ROOT / config["reid"]["model_path"])
    config["tracker"]["config_path"] = str(code_dir / config["tracker"]["config_path"])
    config["video"] = {
        camera: {"id": camera, "path": str(DATASET / camera / "video.avi"),
                 "time_offset_seconds": 0}
        for camera in cameras
    }
    output = config.setdefault("output", {})
    output.update({"output_directory": str(run_dir / "pipeline"), "save_crops": False,
                   "save_local_tracking_video": False, "save_global_tracking_video": False})
    # HDA cameras were recorded on different days: no common clock exists.
    config["matching"]["use_time_constraint"] = False
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    tracker_path = Path(config["tracker"]["config_path"])
    if not tracker_path.is_absolute():
        config["tracker"]["config_path"] = str(code_dir / tracker_path)
    return config


def per_frame_predictions(pipeline: Any, output_dir: Path) -> pd.DataFrame:
    """Join every tracked box with its final global id (works for all versions)."""
    frame_tracks = output_dir / "frame_tracks.csv"
    if frame_tracks.exists():  # v3.0+: the pipeline exports exactly what it draws
        pred = pd.read_csv(frame_tracks).rename(columns={
            "camera_id": "camera", "frame_index": "frame", "global_id": "id",
            "confidence": "conf"})
        return pred[["camera", "frame", "id", "x", "y", "w", "h", "conf"]]
    global_tracks = pd.read_csv(output_dir / "global_tracks.csv")
    by_segment = "start_frame" in global_tracks.columns
    lookup: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
    for row in global_tracks.itertuples(index=False):
        start = int(row.start_frame) if by_segment else -1
        end = int(row.end_frame) if by_segment else 1 << 40
        lookup.setdefault((row.camera_id, int(row.local_track_id)), []).append(
            (start, end, int(row.global_id))
        )
    rows = []
    for camera_id, frames in pipeline.observations.items():
        for frame_index, observations in frames.items():
            for obs in observations:
                spans = lookup.get((camera_id, obs.local_track_id))
                if not spans:
                    continue
                gid = next((g for s, e, g in spans if s <= frame_index <= e), None)
                if gid is None:  # between two segments: take the nearest one
                    gid = min(spans, key=lambda span: min(abs(span[0] - frame_index),
                                                           abs(span[1] - frame_index)))[2]
                if gid == 0:
                    continue
                x1, y1, x2, y2 = obs.bbox
                rows.append((camera_id, frame_index, gid, x1, y1, x2 - x1, y2 - y1,
                             obs.confidence))
    pred = pd.DataFrame(rows, columns=["camera", "frame", "id", "x", "y", "w", "h", "conf"])
    # The replay draws one box per identity per frame; score exactly that.
    return (pred.sort_values("conf", ascending=False)
            .drop_duplicates(["camera", "frame", "id"])
            .sort_values(["camera", "frame", "id"]).reset_index(drop=True))


def score(pred: pd.DataFrame, cameras: list[str]) -> dict[str, Any]:
    per_camera: dict[str, dict[str, float]] = {}
    gt_parts, pred_parts = [], []
    for index, camera in enumerate(cameras):
        gt_all = pd.read_csv(DATASET / camera / "gt.csv").rename(columns={"person_id": "id"})
        cam_pred = pred[pred.camera == camera]
        cam_pred = remove_ignored(cam_pred, gt_all)
        gt = gt_all[~gt_all["ignore"]]
        per_camera[camera] = evaluate(gt, cam_pred)
        offset = index * CAMERA_KEY_STRIDE
        gt_parts.append(gt.assign(frame=gt.frame + offset))
        pred_parts.append(cam_pred.assign(frame=cam_pred.frame + offset))
    # Multi-camera: the same person must keep one global id in every camera.
    mtmc = evaluate(pd.concat(gt_parts), pd.concat(pred_parts))
    keys = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "Precision", "Recall"]
    mean = {key: float(np.mean([m[key] for m in per_camera.values()])) for key in keys}
    for key in ["TP", "FP", "FN", "IDSW", "Frag", "GT_IDs", "Pred_IDs"]:
        mean[key] = int(sum(m[key] for m in per_camera.values()))
    return {"per_camera": per_camera, "per_camera_mean": mean, "mtmc": mtmc}


def git_describe(code_dir: Path) -> str:
    try:
        return subprocess.run(["git", "describe", "--tags", "--always", "--dirty"],
                              cwd=code_dir, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Version label, e.g. v3.0")
    parser.add_argument("--code-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--cameras", nargs="+",
                        default=["cam56", "cam57", "cam58", "cam59", "cam60"])
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Config overrides, e.g. matching.similarity_threshold=0.75")
    parser.add_argument("--use-cache", action="store_true",
                        help="v3.0+: reuse cached detection/Re-ID (for threshold tuning; "
                             "timings of cached runs are not comparable)")
    parser.add_argument("--score-only", action="store_true",
                        help="Re-score an existing run without running the pipeline")
    args = parser.parse_args()

    code_dir = Path(args.code_dir).resolve()
    run_dir = PROJECT_ROOT / "benchmarks" / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    overrides = {k: yaml.safe_load(v) for k, v in (item.split("=", 1) for item in args.set)}

    if args.score_only:
        pred = pd.read_csv(run_dir / "predictions.csv")
        performance = json.loads((run_dir / "result.json").read_text())["performance"]
    else:
        config = build_config(code_dir, run_dir, args.cameras, overrides)
        config["cache"] = {"enabled": bool(args.use_cache),
                           "directory": str(PROJECT_ROOT / "benchmarks" / "runs" / "_cache")}
        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        # Import the pipeline of the requested version, not of this checkout.
        sys.path.insert(0, str(code_dir))
        for module in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[module]
        common = importlib.import_module("src.utils.common")
        pipeline_module = importlib.import_module("src.pipeline")
        import torch

        loaded = common.load_config(config_path)
        stage = {"detect_track_reid": 0.0}
        original = pipeline_module.MultiCameraTrackingPipeline._process_camera

        def timed(self: Any, camera: dict[str, Any]) -> None:
            clock = time.perf_counter()
            original(self, camera)
            stage["detect_track_reid"] += time.perf_counter() - clock

        pipeline_module.MultiCameraTrackingPipeline._process_camera = timed
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with PeakMemory() as memory:
            clock = time.perf_counter()
            pipeline = pipeline_module.MultiCameraTrackingPipeline(loaded)
            setup = time.perf_counter() - clock
            clock = time.perf_counter()
            pipeline.run()
            total = time.perf_counter() - clock
        frames = sum(len(list((DATASET / c / "frames").glob("*.jpg"))) for c in args.cameras)
        pred = per_frame_predictions(pipeline, Path(loaded["output"]["output_directory"]))
        pred.to_csv(run_dir / "predictions.csv", index=False)
        performance = {
            "frames": frames,
            "model_load_seconds": setup,
            "total_seconds": total,
            "detect_track_reid_seconds": stage["detect_track_reid"],
            "association_seconds": total - stage["detect_track_reid"],
            "fps_end_to_end": frames / total,
            "fps_detect_track_reid": frames / max(stage["detect_track_reid"], 1e-9),
            "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2 ** 20
                                   if torch.cuda.is_available() else 0.0),
            "peak_ram_mb": memory.peak / 2 ** 20,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "python": platform.python_version(),
            "torch": torch.__version__,
        }

    metrics = score(pred, args.cameras)
    result = {
        "version": args.name,
        "code": git_describe(code_dir),
        "dataset": "HDA " + "+".join(args.cameras),
        "overrides": overrides,
        "cached_inference": bool(getattr(args, "use_cache", False)),
        "performance": performance,
        "accuracy": metrics,
        "counts": {"global_ids": int(pred["id"].nunique()), "boxes": int(len(pred))},
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(exist_ok=True)
    (results_dir / f"{args.name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    m, s, p = metrics["mtmc"], metrics["per_camera_mean"], performance
    print(f"\n{args.name}: MTMC IDF1 {m['IDF1']:.3f} HOTA {m['HOTA']:.3f} | per-cam HOTA "
          f"{s['HOTA']:.3f} MOTA {s['MOTA']:.3f} IDF1 {s['IDF1']:.3f} IDSW {s['IDSW']} | "
          f"{p['fps_end_to_end']:.1f} FPS, GPU {p['peak_gpu_memory_mb']:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
