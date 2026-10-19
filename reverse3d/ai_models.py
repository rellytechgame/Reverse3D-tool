"""Reverse3D model packs (.r3dm): Voxel Enterprise's own AI models.

The packs are signed and encrypted; reverse3d-core verifies, licenses and runs
them. This module only lists and installs them, so the open tool never holds
the format, keys or decryption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reverse3d.errors import DependencyMissingError, InputValidationError


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
