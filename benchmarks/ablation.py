"""Ablation study of the v3.0 changes on the HDA benchmark.

Each variant removes one v3.0 change from the released config. Variants that
only change later stages (matching, filtering, motion link) reuse the cached
detection/Re-ID results, so the whole study takes about 15 minutes.

Usage:
    python -m benchmarks.ablation            # run all variants, write docs/ABLATION.md
    python -m benchmarks.ablation --report   # only rebuild the table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "benchmarks" / "runs"
OUTPUT = PROJECT_ROOT / "docs" / "ABLATION.md"

# (run name, description, overrides, needs fresh model inference)
VARIANTS: list[tuple[str, str, list[str], bool]] = [
    ("ablation-full", "v3.0 đầy đủ", [], True),
    ("ablation-bytetrack", "ByteTrack thay BoT-SORT + OSNet",
     ["tracker.config_path=configs/bytetrack.yaml"], True),
    ("ablation-no-backfill", "Không ghi bù frame đầu của track",
     ["tracker.backfill_first_frame=false"], True),
    ("ablation-frame-sampling", "Lấy mẫu ReID mỗi 5 frame (như v2.x)",
     ["reid.samples_per_second=null"], True),
    ("ablation-no-motion", "Không motion link trong camera",
     ["intra_camera.motion_link_enabled=false"], False),
    ("ablation-no-filter", "Không lọc tracklet ngắn",
     ["postprocess.min_camera_track_frames=0"], False),
    ("ablation-filter10", "Lọc tracklet < 10 frame (thay vì 5)",
     ["postprocess.min_camera_track_frames=10"], False),
    ("ablation-hungarian", "Hungarian từng cặp camera thay phân cụm",
     ["matching.method=hungarian"], False),
    ("ablation-descriptor", "Descriptor trung bình thay so khớp tập-với-tập",
     ["matching.similarity_mode=descriptor"], False),
]


def run_variants() -> None:
    for name, _, overrides, _ in VARIANTS:
        command = [sys.executable, "-m", "benchmarks.run_benchmark", "--name", name,
                   "--use-cache"]
        if overrides:
            command += ["--set", *overrides]
        print("$", " ".join(command[1:]), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def render() -> str:
    lines = [
        "# Ablation v3.0",
        "",
        "> Sinh bởi `python -m benchmarks.ablation`. Mỗi dòng bỏ **một** thay đổi của v3.0 "
        "khỏi config phát hành; mọi thứ khác giữ nguyên. Dữ liệu: HDA cam56–cam60.",
        "",
        "| Biến thể | MTMC IDF1 | MTMC HOTA | HOTA/camera | MOTA/camera | IDSW | GID dự đoán "
        "| FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    base = None
    for name, label, _, fresh in VARIANTS:
        path = RUNS / name / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        m, c = r["accuracy"]["mtmc"], r["accuracy"]["per_camera_mean"]
        if base is None:
            base = m["IDF1"]
        delta = f" ({100 * (m['IDF1'] - base):+.1f})" if name != "ablation-full" else ""
        fps = f"{r['performance']['fps_end_to_end']:.1f}" if fresh else "cache"
        lines.append(
            f"| {label} | {100 * m['IDF1']:.1f}{delta} | {100 * m['HOTA']:.1f} | "
            f"{100 * c['HOTA']:.1f} | {100 * c['MOTA']:.1f} | {c['IDSW']} | {m['Pred_IDs']} "
            f"| {fps} |"
        )
    lines += [
        "",
        "Cột FPS chỉ có ý nghĩa với biến thể phải chạy lại model (biến thể đầu tiên của "
        "mỗi cấu hình detection/ReID). Biến thể ghi `cache` dùng lại kết quả detection, "
        "ReID của lần chạy trước nên thời gian không so sánh được.",
        "",
        "Số trong ngoặc là chênh lệch MTMC IDF1 so với v3.0 đầy đủ: âm nghĩa là thay đổi bị bỏ "
        "có ích. Phân tích từng dòng nằm ở [CHANGELOG.md](CHANGELOG.md#v30-nhánh-versionv30).",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", action="store_true", help="Only rebuild the table")
    args = parser.parse_args()
    if not args.report:
        run_variants()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
