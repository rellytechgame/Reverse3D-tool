"""Lightweight offscreen renderer based on PyVista/VTK (``[render]`` extra).

Used when Blender is not installed. Quality is lower (no path tracing) but it
needs no external binary and is fast enough for previews in agent loops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError
from reverse3d.mesh_io import load_mesh
from reverse3d.rendering.blender_render import (
    RenderOptions,
    RenderResult,
    _check_outputs_free,
    _expected_outputs,
)
from reverse3d.security import prepare_output_file

logger = logging.getLogger(__name__)


def _require_pyvista() -> Any:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise DependencyMissingError(
            "Neither Blender nor PyVista is available for rendering.",
            hint='Install Blender (and add it to PATH) or: pip install "reverse3d[render]"',
        ) from exc
    return pv


def render_with_pyvista(
    mesh_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    options: RenderOptions | None = None,
    *,
    overwrite: bool = False,
) -> RenderResult:
    """Render ``mesh_path`` offscreen with PyVista. Supports turntables."""
    options = options or RenderOptions()
    options.validate()
    pv = _require_pyvista()
    mesh = load_mesh(mesh_path, settings)
    out = prepare_output_file(output_path, allowed_extensions={".png"}, overwrite=True)
    _check_outputs_free(out, options.turntable_frames, overwrite)

    plotter = pv.Plotter(off_screen=True, window_size=(options.width, options.height))
    plotter.set_background(options.background)
    plotter.add_mesh(
        pv.wrap(mesh),
        color=options.color,
        smooth_shading=options.shading != "flat",
        pbr=True,
        metallic=options.metallic,
        roughness=options.roughness,
    )
    if options.ground:
        plotter.enable_shadows()
    targets = _expected_outputs(out, options.turntable_frames)
    step = 360.0 / max(len(targets), 1)
    plotter.camera_position = "iso"
    plotter.camera.azimuth = options.azimuth_deg
    plotter.camera.elevation = options.elevation_deg
    plotter.reset_camera()
    files: list[str] = []
    for index, target in enumerate(targets):
        if index:
            plotter.camera.azimuth = options.azimuth_deg + step * index
        plotter.screenshot(str(target), transparent_background=options.transparent)
        files.append(str(target))
    plotter.close()
    logger.info("PyVista rendered %d image(s)", len(files))
    return RenderResult(backend="pyvista", files=files, renderer_version=pv.__version__)
