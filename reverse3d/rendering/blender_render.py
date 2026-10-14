"""Automated headless rendering with Blender (preferred) or PyVista (fallback).

Security model of the Blender launcher:

* Render parameters are validated here and passed to a **static** script
  (:mod:`reverse3d.rendering._blender_job`) through a JSON file. No Python
  source is generated from user/agent input, so there is nothing to inject.
* Blender runs with ``--factory-startup`` (ignores user add-ons and startup
  scripts) and ``--disable-autoexec`` (never runs scripts embedded in .blend
  files), under the hard timeout and scrubbed environment of
  :func:`reverse3d.security.run_command`.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError, InputValidationError, ProcessingError
from reverse3d.mesh_io import load_mesh, save_mesh
from reverse3d.security import (
    MESH_EXTENSIONS,
    prepare_output_file,
    resolve_executable,
    run_command,
    validate_input_file,
)

logger = logging.getLogger(__name__)

RenderEngine = Literal["cycles", "eevee", "workbench"]
Backend = Literal["auto", "blender", "pyvista"]

_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")
_JOB_SCRIPT = Path(__file__).with_name("_blender_job.py")
_RESULT_MARKER = "REVERSE3D_RESULT "
_BLENDER_NATIVE = {".obj", ".stl", ".ply", ".glb", ".gltf"}


@dataclass
class RenderOptions:
    """Validated render parameters (shared by the Blender and PyVista backends)."""

    width: int = 1600
    height: int = 1200
    engine: RenderEngine = "cycles"
    device: Literal["cpu", "gpu"] = "cpu"
    samples: int = 64
    color: str = "#9aa4b2"
    metallic: float = 0.0
    roughness: float = 0.45
    background: str = "#f4f5f7"
    transparent: bool = False
    elevation_deg: float = 25.0
    azimuth_deg: float = 35.0
    turntable_frames: int = 0
    ground: bool = True
    shading: Literal["auto", "smooth", "flat"] = "auto"
    vertex_colors: bool = True

    def validate(self) -> None:
        checks = [
            (64 <= self.width <= 8192, "width must be in [64, 8192]"),
            (64 <= self.height <= 8192, "height must be in [64, 8192]"),
            (
                self.engine in ("cycles", "eevee", "workbench"),
                "engine must be cycles, eevee or workbench",
            ),
            (self.device in ("cpu", "gpu"), "device must be cpu or gpu"),
            (1 <= self.samples <= 4096, "samples must be in [1, 4096]"),
            (bool(_HEX_COLOR.match(self.color)), "color must be #RRGGBB"),
            (bool(_HEX_COLOR.match(self.background)), "background must be #RRGGBB"),
            (0 <= self.metallic <= 1, "metallic must be in [0, 1]"),
            (0 <= self.roughness <= 1, "roughness must be in [0, 1]"),
            (-89 <= self.elevation_deg <= 89, "elevation_deg must be in [-89, 89]"),
            (0 <= self.azimuth_deg <= 360, "azimuth_deg must be in [0, 360]"),
            (0 <= self.turntable_frames <= 360, "turntable_frames must be in [0, 360]"),
            (self.shading in ("auto", "smooth", "flat"), "shading must be auto, smooth or flat"),
        ]
        for ok, message in checks:
            if not ok:
                raise InputValidationError(message)


@dataclass
class RenderResult:
    """Paths of the rendered images and the backend that produced them."""

    backend: str
    files: list[str] = field(default_factory=list)
    renderer_version: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _expected_outputs(out: Path, frames: int) -> list[Path]:
    if frames <= 0:
        return [out]
    return [out.with_name(f"{out.stem}_{i:04d}{out.suffix}") for i in range(frames)]


def _check_outputs_free(out: Path, frames: int, overwrite: bool) -> None:
    if overwrite:
        return
    for path in _expected_outputs(out, frames):
        if path.exists():
            raise InputValidationError(
                f"Output file already exists: {path.name}", hint="Pass overwrite=true."
            )


def render_with_blender(
    mesh_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    options: RenderOptions | None = None,
    *,
    blender_executable: Path | None = None,
    overwrite: bool = False,
) -> RenderResult:
    """Render ``mesh_path`` to PNG with Blender in background mode.

    With ``turntable_frames > 0`` the files are ``<stem>_0000.png ...``.

    Raises:
        DependencyMissingError: Blender not found.
        ExternalToolError: Blender failed or timed out.
    """
    options = options or RenderOptions()
    options.validate()
    src = validate_input_file(
        mesh_path, allowed_extensions=MESH_EXTENSIONS, max_bytes=settings.max_input_bytes
    )
    out = prepare_output_file(output_path, allowed_extensions={".png"}, overwrite=True)
    _check_outputs_free(out, options.turntable_frames, overwrite)
    blender = blender_executable or resolve_executable(
        ["blender", "blender.exe"], settings.blender_path, tool="blender"
    )

    with tempfile.TemporaryDirectory(prefix="reverse3d_render_") as tmp:
        tmp_dir = Path(tmp)
        mesh_for_blender = src
        if src.suffix.lower() not in _BLENDER_NATIVE:
            # Blender has no importer for e.g. .off: convert through trimesh first.
            mesh_for_blender = save_mesh(load_mesh(src, settings), tmp_dir / "mesh.ply")
        job = {**asdict(options), "mesh": str(mesh_for_blender), "output": str(out)}
        job_file = tmp_dir / "job.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")

        proc = run_command(
            blender,
            [
                "--background",
                "--factory-startup",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(_JOB_SCRIPT),
                "--",
                str(job_file),
            ],
            timeout_s=settings.subprocess_timeout_s,
            tool="blender",
        )

    payload: dict[str, object] = {}
    for line in (proc.stdout or "").splitlines():
        if line.startswith(_RESULT_MARKER):
            try:
                payload = json.loads(line[len(_RESULT_MARKER) :])
            except json.JSONDecodeError:
                payload = {}
    files = [str(p) for p in _expected_outputs(out, options.turntable_frames) if p.exists()]
    if not files:
        raise ProcessingError("Blender finished but produced no image.")
    version = payload.get("blender")
    logger.info("Blender rendered %d image(s)", len(files))
    return RenderResult(
        backend="blender", files=files, renderer_version=str(version) if version else None
    )


def render_mesh(
    mesh_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    options: RenderOptions | None = None,
    *,
    backend: Backend = "auto",
    overwrite: bool = False,
) -> RenderResult:
    """Render with the requested backend. ``auto`` uses Blender when available
    and falls back to PyVista (``[render]`` extra) otherwise."""
    if backend not in ("auto", "blender", "pyvista"):
        raise InputValidationError("backend must be auto, blender or pyvista")
    if backend in ("auto", "blender"):
        try:
            return render_with_blender(
                mesh_path, output_path, settings, options, overwrite=overwrite
            )
        except DependencyMissingError:
            if backend == "blender":
                raise
            logger.info("Blender not available, falling back to PyVista")
    from reverse3d.rendering.preview import render_with_pyvista

    return render_with_pyvista(mesh_path, output_path, settings, options, overwrite=overwrite)
