"""Reusable local OSNet inference wrapper."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from .osnet import osnet_x1_0
from .osnet_ain import osnet_ain_x1_0
from .utils.common import ProjectError, l2_normalize, project_path, select_device


class ReIDModel:
    """Load local OSNet weights and extract normalized person descriptors."""

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config["reid"]
        self.model_path = project_path(config, cfg["model_path"])
        self.device = select_device(cfg.get("device", "auto"))
        self.input_height = int(cfg.get("input_height", 256))
        self.input_width = int(cfg.get("input_width", 128))
        self.architecture = str(cfg.get("architecture", "osnet_x1_0"))
        self.batch_size = int(cfg.get("batch_size", 64))
        if self.batch_size < 1:
            raise ProjectError("reid.batch_size must be at least one.")
        self.half = bool(cfg.get("half", True)) and self.device.type == "cuda"
        # Averaging with the mirrored crop removes left/right pose bias from the descriptor.
        self.flip_augmentation = bool(cfg.get("flip_augmentation", True))
        factories = {
            "osnet_x1_0": osnet_x1_0,
            "osnet_ain_x1_0": osnet_ain_x1_0,
        }
        if self.architecture not in factories:
            supported = ", ".join(sorted(factories))
            raise ProjectError(
                f"Unsupported Re-ID architecture: {self.architecture}. "
                f"Supported values: {supported}"
            )
        if not self.model_path.is_file():
            raise ProjectError(
                f"OSNet model not found:\n{self.model_path}\n\n"
                "Please place your OSNet weights at this location or update "
                "reid.model_path in config.yaml."
            )
        self.model = factories[self.architecture](num_classes=1)
        self._load_weights()
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        dtype = torch.float16 if self.half else torch.float32
        self._mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device, dtype=dtype
        ).view(1, 3, 1, 1)
        self._std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device, dtype=dtype
        ).view(1, 3, 1, 1)

    def _load_weights(self) -> None:
        try:
            try:
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            except TypeError:  # Compatibility with older supported PyTorch releases.
                checkpoint = torch.load(self.model_path, map_location="cpu")
        except Exception as exc:
            raise ProjectError(f"Could not load OSNet weights:\n{self.model_path}\n\n{exc}") from exc

        if isinstance(checkpoint, dict):
            for container_key in ("state_dict", "model_state_dict", "model"):
                candidate = checkpoint.get(container_key)
                if isinstance(candidate, dict):
                    checkpoint = candidate
                    break
        if not isinstance(checkpoint, dict):
            raise ProjectError("OSNet checkpoint does not contain a state dictionary.")

        current = self.model.state_dict()
        compatible: dict[str, torch.Tensor] = {}
        for raw_key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(raw_key)
            for prefix in ("module.", "model."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key in current and current[key].shape == value.shape:
                compatible[key] = value
        backbone_keys = [key for key in current if not key.startswith("classifier.")]
        loaded_backbone = sum(key in compatible for key in backbone_keys)
        if loaded_backbone < int(0.8 * len(backbone_keys)):
            raise ProjectError(
                f"The checkpoint is not compatible with {self.architecture} "
                f"model (loaded {loaded_backbone}/{len(backbone_keys)} backbone tensors)."
            )
        self.model.load_state_dict(compatible, strict=False)
        logging.info("Loaded %d/%d compatible OSNet tensors", len(compatible), len(current))

    def _forward(self, resized: list[np.ndarray]) -> np.ndarray:
        # Upload uint8 pixels and normalize on the device: 4x less PCIe traffic than float32.
        batch = torch.from_numpy(np.stack(resized)).to(self.device, non_blocking=True)
        batch = batch.permute(0, 3, 1, 2).to(self._mean.dtype).div_(255.0)
        batch = batch.sub_(self._mean).div_(self._std)
        with torch.inference_mode():
            output = self.model(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if self.flip_augmentation:
                flipped = self.model(torch.flip(batch, dims=[3]))
                if isinstance(flipped, (tuple, list)):
                    flipped = flipped[0]
                output = output + flipped
        features = output.detach().float().cpu().numpy().reshape(len(resized), -1)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.all(np.isfinite(features)) or np.any(norms <= 1e-12):
            raise ValueError("OSNet produced an invalid embedding.")
        return (features / norms).astype(np.float32)

    def extract_embeddings(self, person_crops: list[np.ndarray]) -> np.ndarray:
        """Convert BGR crops to an ``(N, D)`` array of L2-normalized OSNet embeddings."""
        if not person_crops:
            return np.empty((0, 0), dtype=np.float32)
        resized: list[np.ndarray] = []
        for crop in person_crops:
            if crop is None or crop.size == 0:
                raise ValueError("Person crop is empty.")
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized.append(cv2.resize(
                rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
            ))
        chunks = [
            self._forward(resized[start:start + self.batch_size])
            for start in range(0, len(resized), self.batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def extract_embedding(self, person_crop: np.ndarray) -> np.ndarray:
        """Convert a BGR crop to a normalized, one-dimensional OSNet embedding."""
        return l2_normalize(self.extract_embeddings([person_crop])[0])
