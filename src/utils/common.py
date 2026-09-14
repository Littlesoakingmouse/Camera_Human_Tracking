"""Configuration, path, logging, and numerical helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


class ProjectError(RuntimeError):
    """An expected error that should be shown without a traceback."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML and attach the project root used to resolve relative paths."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ProjectError(f"Configuration file not found:\n{path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ProjectError(f"Invalid YAML in configuration file:\n{path}\n\n{exc}") from exc
    if not isinstance(config, dict):
        raise ProjectError(f"Configuration must be a YAML mapping:\n{path}")
    required = {"detector", "tracker", "reid", "matching", "video", "output"}
    missing = sorted(required.difference(config))
    if missing:
        raise ProjectError(f"Missing configuration sections: {', '.join(missing)}")
    # By convention configs/config.yaml lives directly below the project root.
    config["_project_root"] = path.parent.parent
    config["_config_path"] = path
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a user-configured path relative to the project root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA when requested/available, otherwise return a validated device."""
    requested = str(requested).strip().lower()
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ProjectError("CUDA was requested, but torch.cuda.is_available() is False.")
    return device


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return an L2-normalized float32 vector, rejecting empty/invalid input."""
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    if value.size == 0 or not np.all(np.isfinite(value)):
        raise ValueError("Embedding must be non-empty and finite.")
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise ValueError("Cannot normalize a zero-length embedding.")
    return value / norm


def configure_logging() -> None:
    """Install a concise console logging format."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
