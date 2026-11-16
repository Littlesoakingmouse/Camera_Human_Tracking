"""Build the HDA benchmark: frames as JPEG sequences + MOT-style ground truth.

Usage:
    python benchmarks/prepare_hda.py --hda-root ../HDA_Dataset_V1.3

Output (git-ignored), one folder per camera:
    data/benchmark/hda/camXX/frames/000000.jpg ...
    data/benchmark/hda/camXX/video.avi   the same JPEGs in an MJPEG container
                                        (stream copy, no re-encoding; 2 FPS)
    data/benchmark/hda/camXX/gt.csv   frame,person_id,x,y,w,h,occluded,ignore

Frames wider than ``--max-width`` (default 1280) are downscaled to halve JPEG decode
cost; ground-truth boxes are scaled identically. The scale is computed from the real
JPEG size: the .seq header of cam56-59 claims 2560x1600 but stores 1280x800 frames.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.hda import iter_seq_jpegs, read_vbb  # noqa: E402

DEFAULT_CAMERAS = ["cam56", "cam57", "cam58", "cam59", "cam60"]


def person_number(label: str) -> int:
    return int(label.replace("person", ""))


def build_ground_truth(vbb_path: Path, scale: float) -> tuple[int, pd.DataFrame]:
    frame_count, objects = read_vbb(vbb_path)
    rows = []
    for obj in objects:
        ignore = not obj.is_person
        person_id = person_number(obj.label) if obj.is_person else -1
        for offset, (box, occluded) in enumerate(zip(obj.boxes, obj.occluded)):
            x, y, w, h = (float(v) * scale for v in box)
            if w <= 1 or h <= 1:
                continue
            rows.append((obj.start + offset, person_id, x, y, w, h, bool(occluded), ignore))
    frame = pd.DataFrame(rows, columns=["frame", "person_id", "x", "y", "w", "h",
                                        "occluded", "ignore"])
    return frame_count, frame.sort_values(["frame", "person_id"]).reset_index(drop=True)


def native_size(seq_path: Path) -> tuple[int, int]:
    payload = next(iter_seq_jpegs(seq_path))
    height, width = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR).shape[:2]
    return width, height


def extract_frames(seq_path: Path, out_dir: Path, scale: float, quality: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, payload in enumerate(iter_seq_jpegs(seq_path)):
        target = out_dir / f"{index:06d}.jpg"
        count = index + 1
        if target.exists():
            continue
        if scale == 1.0:
            target.write_bytes(payload)
            continue
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError(f"Could not encode {target}")
        target.write_bytes(encoded.tobytes())
    return count


def build_video(frames_dir: Path, target: Path, fps: float) -> None:
    """Pack the JPEG frames into an AVI so every pipeline version can read it."""
    if target.exists():
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-framerate", str(fps),
                        "-i", str(frames_dir / "%06d.jpg"), "-c:v", "copy", str(target)],
                       check=True)
        return
    # Fallback without ffmpeg: OpenCV re-encodes the frames as MJPEG.
    files = sorted(frames_dir.glob("*.jpg"))
    first = cv2.imread(str(files[0]))
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"MJPG"), fps,
                             (first.shape[1], first.shape[0]))
    for file in files:
        writer.write(cv2.imread(str(file)))
    writer.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hda-root", default="../HDA_Dataset_V1.3")
    parser.add_argument("--out", default="data/benchmark/hda")
    parser.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS)
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--fps", type=float, default=2.0,
                        help="HDA records about 2 FPS (read from the on-frame clock)")
    args = parser.parse_args()

    hda_root, out_root = Path(args.hda_root).resolve(), Path(args.out).resolve()
    summary = {"max_width": args.max_width, "cameras": {}}
    for camera in args.cameras:
        number = camera.replace("cam", "")
        seq_path = hda_root / "hda_image_sequences_matlab" / f"camera{number}.seq"
        vbb_path = hda_root / "hda_annotations" / f"{camera}.txt"
        width, height = native_size(seq_path)
        scale = min(1.0, args.max_width / width)
        annotated_frames, gt = build_ground_truth(vbb_path, scale)
        camera_dir = out_root / camera
        frames = extract_frames(seq_path, camera_dir / "frames", scale, args.quality)
        gt.to_csv(camera_dir / "gt.csv", index=False)
        build_video(camera_dir / "frames", camera_dir / "video.avi", args.fps)
        persons = gt[~gt.ignore]
        summary["cameras"][camera] = {
            "frames": frames, "annotated_frames": annotated_frames,
            "scale": scale, "width": round(width * scale), "height": round(height * scale),
            "person_boxes": int(len(persons)), "identities": int(persons.person_id.nunique()),
            "ignore_boxes": int(gt.ignore.sum()),
        }
        print(camera, summary["cameras"][camera], flush=True)
    (out_root / "dataset.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
