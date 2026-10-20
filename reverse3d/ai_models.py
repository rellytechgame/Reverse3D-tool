"""Reverse3D model packs (.r3dm): Voxel Enterprise's own AI models.

The packs are signed and encrypted; reverse3d-core verifies, licenses and runs
them. This module only lists and installs them, so the open tool never holds
the format, keys or decryption.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from reverse3d.errors import DependencyMissingError, InputValidationError

logger = logging.getLogger(__name__)


def _core_models() -> Any:
    from reverse3d.reconstruction.silhouette import core_engine

    core = core_engine()
    if core is None:
        raise DependencyMissingError(
            "Reverse3D model packs need Reverse3D Core.", hint="Included in reverse3d Studio."
        )
    try:
        import importlib

        return importlib.import_module("reverse3d_core.models")
    except ModuleNotFoundError as exc:
        raise DependencyMissingError(
            "This Reverse3D Core is too old for model packs.", hint="Update reverse3d Studio."
        ) from exc


def list_models() -> dict[str, object]:
    """Installed packs and whether this licence can use them."""
    return {"models": _core_models().list_packs()}


def install_model(path: Path) -> dict[str, object]:
    """Verify a .r3dm file and copy it into the models folder."""
    models = _core_models()
    if path.suffix.lower() != ".r3dm" or not path.is_file():
        raise InputValidationError("Choose a Reverse3D model pack (.r3dm file).")
    try:
        info = models.install(path)
    except models.PackError as exc:
        raise InputValidationError(str(exc)) from exc
    return {"installed": info.to_dict()}


# ─── Using the models while reconstructing ────────────────────────────────────

#: preferred first: the Professional model when the licence allows it
PREFERRED = ("revenus-1", "optiverse-1")
_PROFILES = ("object", "character", "architecture", "landscape")


class VisionModel:
    """A Voxel Enterprise vision model (Optiverse / Revenus) ready to run on photos."""

    def __init__(self, session: Any, info: Any) -> None:
        self.session = session
        self.name = str(info.name)
        self.version = str(info.version)
        spec = info.manifest.get("input", {})
        self.size = int(spec.get("size", 518))
        self.mean = np.asarray(spec.get("mean", (0.485, 0.456, 0.406)), np.float32)
        self.std = np.asarray(spec.get("std", (0.229, 0.224, 0.225)), np.float32)
        self.profiles = tuple(info.manifest.get("profiles", _PROFILES))
        #: set (signed) at export once the cut-out beats the reference on real photos
        self.mask_ready = bool(info.manifest.get("mask_ready", False))

    @property
    def label(self) -> str:
        return f"{self.name} {self.version}"

    def run(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Disparity and subject probability (both H x W) and scene probabilities."""
        import cv2

        h, w = rgb.shape[:2]
        side = max(h, w)
        # Letterbox to a square (the model sees undistorted proportions).
        canvas = np.empty((side, side, 3), np.uint8)
        canvas[:] = (self.mean * 255).astype(np.uint8)
        y0, x0 = (side - h) // 2, (side - w) // 2
        canvas[y0 : y0 + h, x0 : x0 + w] = rgb
        x = cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_AREA)
        x = ((x.astype(np.float32) / 255.0 - self.mean) / self.std).transpose(2, 0, 1)[None]
        depth, mask, scene = self.session.run(None, {"pixels": np.ascontiguousarray(x)})

        def back(a: np.ndarray) -> np.ndarray:
            full = cv2.resize(
                np.asarray(a[0], np.float32), (side, side), interpolation=cv2.INTER_LINEAR
            )
            return np.ascontiguousarray(full[y0 : y0 + h, x0 : x0 + w])

        logits = np.asarray(scene[0], np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        prob_mask = 1.0 / (1.0 + np.exp(-back(mask)))
        return (
            back(depth),
            prob_mask.astype(np.float32),
            {p: float(v) for p, v in zip(self.profiles, probs, strict=False)},
        )


class OwnSegmenter:
    """The subject cut-out of Optiverse / Revenus with the interface of the SAM segmenter
    (embed once per photo, then segment with a box and a point)."""

    def __init__(self, model: VisionModel) -> None:
        self.model = model

    def embed(self, rgb: np.ndarray) -> dict[str, Any]:
        return {"prob": self.model.run(rgb)[1]}

    def segment(
        self, state: dict[str, Any], box: tuple[int, int, int, int], point: tuple[float, float]
    ) -> np.ndarray:
        import cv2

        prob = state["prob"]
        x0, y0, x1, y1 = box
        inside = np.zeros(prob.shape, bool)
        inside[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] = True
        mask = ((prob > 0.5) & inside).astype(np.uint8)
        # Only the object the box and the point are about: no loose blobs.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n > 1:
            px = min(max(int(point[0]), 0), mask.shape[1] - 1)
            py = min(max(int(point[1]), 0), mask.shape[0] - 1)
            label = int(labels[py, px]) or 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (labels == label).astype(np.uint8)
        # A solid object has no holes in its outline (a printed logo can look like
        # background): fill everything the outside cannot reach.
        flood = np.pad(mask, 1).astype(np.uint8)
        outside = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, outside, (0, 0), 1)
        holes = flood[1:-1, 1:-1] == 0
        mask[holes] = 1
        return np.where(mask > 0, 255, 0).astype(np.uint8)


_loaded: VisionModel | bool | None = False


def own_model() -> VisionModel | None:
    """The best installed Voxel Enterprise model this licence can use, or None.

    R3D_SKIP=models keeps the third-party base models (benchmarks).
    """
    global _loaded
    if _loaded is not False:
        return _loaded if isinstance(_loaded, VisionModel) else None
    from reverse3d.reconstruction.silhouette import agent_skipped

    _loaded = None
    if agent_skipped("models"):
        return None
    try:
        models = _core_models()
    except DependencyMissingError:
        return None
    for name in PREFERRED:
        try:
            session, info = models.session(name)
        except Exception as exc:  # not installed, licence, damaged: try the next one
            logger.debug("Model %s unavailable: %s", name, exc)
            continue
        _loaded = VisionModel(session, info)
        logger.info("Voxel Enterprise model: %s", _loaded.label)
        return _loaded
    return None
