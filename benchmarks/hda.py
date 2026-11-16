"""Readers for the HDA Person Dataset (Norpix .seq video + Piotr VBB annotations).

The HDA cameras record JPEG frames inside Norpix ``.seq`` containers and are
annotated in the VBB text format of Piotr Dollar's toolbox. Labels such as
``person045`` are global: the same label in two cameras is the same person, so
the annotations support both per-camera MOT metrics and cross-camera identity
metrics. ``crowd`` and ``personUnk`` boxes are treated as ignore regions.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SEQ_HEADER_BYTES = 1024
JPEG_FORMAT = 201
# After each JPEG payload the HDA files store 16 bytes: uint32 frame index + padding.
SEQ_FRAME_TRAILER_BYTES = 16


@dataclass(frozen=True)
class SeqHeader:
    width: int
    height: int
    image_format: int
    frame_count: int


def read_seq_header(path: Path) -> SeqHeader:
    with path.open("rb") as stream:
        header = stream.read(SEQ_HEADER_BYTES)
    width, height, _bit_depth, _real_depth, _size, image_format, frames = struct.unpack(
        "<7I", header[548:576]
    )
    if image_format != JPEG_FORMAT:
        raise ValueError(f"{path}: only JPEG .seq files are supported (format {image_format}).")
    return SeqHeader(width, height, image_format, frames)


def iter_seq_jpegs(path: Path) -> Iterator[bytes]:
    """Yield the raw JPEG bytes of every frame without decoding them."""
    header = read_seq_header(path)
    with path.open("rb") as stream:
        stream.seek(SEQ_HEADER_BYTES)
        for _ in range(header.frame_count):
            raw_size = stream.read(4)
            if len(raw_size) < 4:
                return
            size = struct.unpack("<I", raw_size)[0]
            payload = stream.read(size - 4)
            if not payload.startswith(b"\xff\xd8"):
                raise ValueError(f"{path}: frame does not start with a JPEG marker.")
            yield payload
            stream.seek(SEQ_FRAME_TRAILER_BYTES, 1)


@dataclass
class VbbObject:
    label: str
    start: int  # zero-based first frame
    end: int  # zero-based last frame (inclusive)
    boxes: np.ndarray  # (N, 4) x, y, w, h
    occluded: np.ndarray  # (N,) bool

    @property
    def kind(self) -> str:
        return re.sub(r"\d+$", "", self.label)

    @property
    def is_person(self) -> bool:
        return self.kind == "person"


_OBJECT_RE = re.compile(r"lbl='([^']*)' str=(\d+) end=(\d+) hide=(\d+)")


def _matrix(line: str) -> np.ndarray:
    body = line.split("[", 1)[1].rsplit("]", 1)[0].strip()
    if not body:
        return np.empty((0, 4))
    return np.array([[float(v) for v in row.split()] for row in body.split(";") if row.strip()])


def read_vbb(path: Path) -> tuple[int, list[VbbObject]]:
    """Parse a VBB text file. VBB frames are one-based; results are zero-based."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frame_count = int(re.search(r"nFrame=(\d+)", lines[1]).group(1))
    objects: list[VbbObject] = []
    index = 0
    while index < len(lines):
        match = _OBJECT_RE.match(lines[index].strip())
        if not match:
            index += 1
            continue
        fields: dict[str, str] = {}
        cursor = index + 1
        while cursor < len(lines) and not lines[cursor].startswith("lbl="):
            key = lines[cursor].split("=", 1)[0].strip()
            if key in {"pos", "occl"}:
                fields[key] = lines[cursor]
            if lines[cursor].startswith("----"):
                break
            cursor += 1
        start, end = int(match.group(2)) - 1, int(match.group(3)) - 1
        boxes = _matrix(fields["pos"]).reshape(-1, 4)
        occl_body = fields["occl"].split("[", 1)[1].rsplit("]", 1)[0].split()
        occluded = np.array([v == "1" for v in occl_body], dtype=bool)
        if len(boxes) != end - start + 1:
            raise ValueError(f"{path}: {match.group(1)} has {len(boxes)} boxes for "
                             f"frames {start}..{end}")
        objects.append(VbbObject(match.group(1), start, end, boxes, occluded))
        index = cursor
    return frame_count, objects
