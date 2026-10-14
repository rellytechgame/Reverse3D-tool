"""Safe mesh / point-cloud I/O on top of trimesh.

Security notes:

* Only allowlisted formats are parsed (no pickle-based or executable formats).
* ``.gltf`` and ``.obj`` may reference sibling files (buffers, textures, .mtl).
  :class:`SandboxedResolver` only serves files located inside the model's own
  directory, so a crafted model cannot read ``../../secrets`` (CWE-22).
* Face counts are bounded by ``Settings.max_mesh_faces`` after loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import trimesh
from trimesh.resolvers import FilePathResolver

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError, SecurityViolationError
from reverse3d.security import (
    MESH_EXTENSIONS,
    POINT_CLOUD_EXTENSIONS,
    prepare_output_file,
    validate_input_file,
)

logger = logging.getLogger(__name__)

Geometry = trimesh.Trimesh | trimesh.PointCloud


class SandboxedResolver(FilePathResolver):
    """A trimesh resolver that refuses to read or write outside its directory."""

    def __init__(self, source: str | Path) -> None:
        super().__init__(str(source))
        self._root = Path(self.parent).resolve()

    def _check(self, name: str) -> None:
        for variant in {name, unquote(name)}:
            target = (self._root / variant.strip()).resolve()
            if target != self._root and not target.is_relative_to(self._root):
                raise SecurityViolationError("Model references a file outside its own directory.")

    def get(self, name: str) -> bytes:
        self._check(name)
        return super().get(name)

    def write(self, name: str, data: bytes | str) -> None:
        self._check(name)
        super().write(name, data)


def _check_faces(mesh: trimesh.Trimesh, settings: Settings) -> None:
    if len(mesh.faces) > settings.max_mesh_faces:
        raise InputValidationError(
            f"Mesh has too many faces ({len(mesh.faces):,} > {settings.max_mesh_faces:,}).",
            hint="Decimate it first or raise REVERSE3D_MAX_MESH_FACES.",
        )


def load_geometry(path: str | Path, settings: Settings) -> Geometry:
    """Load a mesh or point cloud from an allowlisted format.

    Scenes (glTF/GLB with many nodes) are flattened into a single mesh with
    node transforms applied.

    Raises:
        InputValidationError: unsupported, unreadable, empty or oversize input.
    """
    allowed = MESH_EXTENSIONS | POINT_CLOUD_EXTENSIONS
    resolved = validate_input_file(
        path, allowed_extensions=allowed, max_bytes=settings.max_input_bytes
    )
    loaded: Any
    try:
        loaded = trimesh.load(
            str(resolved),
            file_type=resolved.suffix.lower().lstrip("."),
            resolver=SandboxedResolver(resolved),
            # Merge duplicated vertices: STL stores unshared triangle soups and
            # topology metrics (watertight, volume, holes) would be meaningless.
            process=True,
        )
    except SecurityViolationError:
        raise
    except Exception as exc:  # trimesh raises many exception types for bad files
        raise InputValidationError(f"Could not parse geometry file: {resolved.name}") from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise InputValidationError(f"Scene contains no geometry: {resolved.name}")
        loaded = loaded.to_mesh() if hasattr(loaded, "to_mesh") else loaded.dump(concatenate=True)

    if isinstance(loaded, trimesh.Trimesh):
        if len(loaded.faces) == 0:
            # A PLY with vertices only is a point cloud.
            if len(loaded.vertices) == 0:
                raise InputValidationError(f"Empty geometry: {resolved.name}")
            return trimesh.PointCloud(loaded.vertices)
        _check_faces(loaded, settings)
        return loaded
    if isinstance(loaded, trimesh.PointCloud):
        if len(loaded.vertices) == 0:
            raise InputValidationError(f"Empty point cloud: {resolved.name}")
        return loaded
    raise InputValidationError(f"Unsupported geometry type in {resolved.name}")


def load_mesh(path: str | Path, settings: Settings) -> trimesh.Trimesh:
    """Like :func:`load_geometry` but requires a triangle mesh."""
    geometry = load_geometry(path, settings)
    if not isinstance(geometry, trimesh.Trimesh):
        raise InputValidationError(
            "File contains a point cloud, not a mesh.",
            hint="Use the photogrammetry/meshing tools to build a surface first.",
        )
    return geometry


def save_mesh(mesh: trimesh.Trimesh, path: str | Path, *, overwrite: bool = False) -> Path:
    """Export ``mesh`` to an allowlisted format inferred from the extension."""
    out = prepare_output_file(path, allowed_extensions=MESH_EXTENSIONS, overwrite=overwrite)
    file_type = out.suffix.lower().lstrip(".")
    if file_type == "gltf":
        # Embed buffers so the export is a single self-contained file.
        data = trimesh.exchange.gltf.export_gltf(mesh.scene(), embed_buffers=True)
        out.write_bytes(data["model.gltf"])
    else:
        mesh.export(str(out), file_type=file_type)
    logger.info("Saved mesh %s (%d faces)", out.name, len(mesh.faces))
    return out


def sample_points(
    geometry: Geometry, count: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray | None]:
    """Sample up to ``count`` points (and normals for meshes) from ``geometry``."""
    if isinstance(geometry, trimesh.Trimesh):
        points, face_idx = trimesh.sample.sample_surface(geometry, count, seed=seed)
        normals = geometry.face_normals[face_idx]
        return np.asarray(points, dtype=np.float64), np.asarray(normals, dtype=np.float64)
    vertices = np.asarray(geometry.vertices, dtype=np.float64)
    if len(vertices) > count:
        rng = np.random.default_rng(seed)
        vertices = vertices[rng.choice(len(vertices), count, replace=False)]
    return vertices, None
