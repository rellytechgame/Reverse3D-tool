"""Automated rendering: headless Blender (Cycles/Eevee/Workbench) or PyVista."""

from reverse3d.rendering.blender_render import (
    RenderOptions,
    RenderResult,
    render_mesh,
    render_with_blender,
)

__all__ = ["RenderOptions", "RenderResult", "render_mesh", "render_with_blender"]
