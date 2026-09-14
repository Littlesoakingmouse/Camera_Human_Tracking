"""Compare two images of person A with an image of person B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.matcher import cosine_similarity  # noqa: E402
from src.reid import ReIDModel  # noqa: E402
from src.utils.common import ProjectError, load_config  # noqa: E402


def read_image(path_value: str) -> np.ndarray:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ProjectError(f"Test image not found:\n{path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ProjectError(f"OpenCV could not read test image:\n{path}")
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("person_a_cam1")
    parser.add_argument("person_a_cam2")
    parser.add_argument("person_b")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/config.yaml"))
    args = parser.parse_args()
    try:
        model = ReIDModel(load_config(args.config))
        feature_a1 = model.extract_embedding(read_image(args.person_a_cam1))
        feature_a2 = model.extract_embedding(read_image(args.person_a_cam2))
        feature_b = model.extract_embedding(read_image(args.person_b))
        print(f"similarity(A_cam1, A_cam2) = {cosine_similarity(feature_a1, feature_a2):.4f}")
        print(f"similarity(A_cam1, B)      = {cosine_similarity(feature_a1, feature_b):.4f}")
        return 0
    except (ProjectError, ValueError) as exc:
        print(f"ERROR:\n{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
