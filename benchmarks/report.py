"""Render benchmarks/results/*.json into docs/BENCHMARK_RESULTS.md.

Usage:
    python -m benchmarks.report
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "benchmarks" / "results"
OUTPUT = PROJECT_ROOT / "docs" / "BENCHMARK_RESULTS.md"


def _version_key(name: str) -> tuple:
    parts = name.lstrip("v").replace("-", ".").split(".")
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def load_results() -> list[dict]:
    results = [json.loads(path.read_text(encoding="utf-8")) for path in RESULTS.glob("*.json")]
    official = [r for r in results if not r.get("cached_inference") and not r.get("overrides")]
    return sorted(official, key=lambda r: _version_key(r["version"]))


def best_marks(rows: list[list[str]], column: int, higher: bool) -> None:
    values = [float(row[column]) for row in rows]
    best = max(values) if higher else min(values)
    for row, value in zip(rows, values):
        if value == best:
            row[column] = f"**{row[column]}**"


def render(results: list[dict]) -> str:
    lines = [
        "# Kết quả benchmark theo phiên bản",
        "",
        "> File này được sinh tự động bởi `python -m benchmarks.report` từ "
        "`benchmarks/results/*.json`. Đừng sửa tay; xem cách đọc ở "
        "[EVALUATION.md](EVALUATION.md).",
        "",
    ]
    if not results:
        return "\n".join(lines + ["Chưa có kết quả."]) + "\n"
    first = results[0]
    perf = first["performance"]
    lines += [
        f"- Dữ liệu: **{first['dataset']}**",
        f"- Phần cứng: {perf['device']}, Python {perf['python']}, PyTorch {perf['torch']}",
        f"- Số frame: {perf['frames']}",
        "",
        "## 1. Độ chính xác đa camera (MTMC)",
        "",
        "Một người phải giữ cùng một Global ID ở mọi camera. Đây là chỉ số chính.",
        "",
        "| Phiên bản | Code | IDF1 ↑ | IDP ↑ | IDR ↑ | HOTA ↑ | AssA ↑ | GID dự đoán | GID thật |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for r in results:
        m = r["accuracy"]["mtmc"]
        rows.append([r["version"], f"`{r['code']}`", _pct(m["IDF1"]), _pct(m["IDP"]),
                     _pct(m["IDR"]), _pct(m["HOTA"]), _pct(m["AssA"]),
                     str(m["Pred_IDs"]), str(m["GT_IDs"])])
    for column in (2, 3, 4, 5, 6):
        best_marks(rows, column, higher=True)
    lines += ["| " + " | ".join(row) + " |" for row in rows]

    lines += [
        "",
        "## 2. Độ chính xác trong từng camera (trung bình 5 camera)",
        "",
        "| Phiên bản | HOTA ↑ | DetA ↑ | AssA ↑ | MOTA ↑ | IDF1 ↑ | Precision ↑ | Recall ↑ "
        "| FP ↓ | FN ↓ | IDSW ↓ | Frag ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for r in results:
        m = r["accuracy"]["per_camera_mean"]
        rows.append([r["version"], _pct(m["HOTA"]), _pct(m["DetA"]), _pct(m["AssA"]),
                     _pct(m["MOTA"]), _pct(m["IDF1"]), _pct(m["Precision"]),
                     _pct(m["Recall"]), str(m["FP"]), str(m["FN"]), str(m["IDSW"]),
                     str(m["Frag"])])
    for column in range(1, 8):
        best_marks(rows, column, higher=True)
    for column in range(8, 12):
        best_marks(rows, column, higher=False)
    lines += ["| " + " | ".join(row) + " |" for row in rows]

    lines += [
        "",
        "## 3. Hiệu năng",
        "",
        "| Phiên bản | Tổng thời gian (s) ↓ | Detect+Track+ReID (s) ↓ | Ghép ID (s) ↓ "
        "| FPS end-to-end ↑ | GPU peak (MB) ↓ | RAM peak (MB) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for r in results:
        p = r["performance"]
        rows.append([r["version"], f"{p['total_seconds']:.0f}",
                     f"{p['detect_track_reid_seconds']:.0f}", f"{p['association_seconds']:.1f}",
                     f"{p['fps_end_to_end']:.1f}", f"{p['peak_gpu_memory_mb']:.0f}",
                     f"{p['peak_ram_mb']:.0f}"])
    for column in (1, 2, 3, 5, 6):
        best_marks(rows, column, higher=False)
    best_marks(rows, 4, higher=True)
    lines += ["| " + " | ".join(row) + " |" for row in rows]

    cameras = list(results[-1]["accuracy"]["per_camera"])
    lines += [
        "",
        "## 4. Chi tiết từng camera: IDF1 / HOTA (%)",
        "",
        "| Phiên bản | " + " | ".join(cameras) + " |",
        "|---|" + "---:|" * len(cameras),
    ]
    for r in results:
        cells = []
        for camera in cameras:
            m = r["accuracy"]["per_camera"].get(camera)
            cells.append(f"{_pct(m['IDF1'])} / {_pct(m['HOTA'])}" if m else "-")
        lines.append(f"| {r['version']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(render(load_results()), encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
