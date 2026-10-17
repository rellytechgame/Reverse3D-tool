"""Environment diagnostics: which optional stacks and external tools are usable."""

from __future__ import annotations

import importlib.util
import platform
import sys
from importlib import metadata

from reverse3d._version import __version__
from reverse3d.config import Settings
from reverse3d.errors import Reverse3DError
from reverse3d.security import resolve_executable

_PACKAGES = {
    "numpy": "core",
    "opencv-python-headless": "core",
    "trimesh": "core",
    "shapely": "core",
    "fast-simplification": "core",
    "open3d": "recon",
    "torch": "depth",
    "transformers": "depth",
    "pyvista": "render",
    "mcp": "mcp",
    "anthropic": "agents",
}
_MODULES = {
    "opencv-python-headless": "cv2",
    "fast-simplification": "fast_simplification",
}


def _package_version(dist: str) -> str | None:
    module = _MODULES.get(dist, dist.replace("-", "_"))
    if importlib.util.find_spec(module) is None:
        return None
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "installed"


def core_status() -> dict[str, object]:
    """Whether reverse3d-core, the proprietary reconstruction engine, is active."""
    from reverse3d.reconstruction.silhouette import core_engine

    core = core_engine()
    if core is None:
        return {"available": False, "hint": "Included in reverse3d Studio."}
    return {"available": True, "version": str(getattr(core, "__version__", "?"))}


def environment_report(settings: Settings, *, include_offline: bool = True) -> dict[str, object]:
    """Return a JSON-serialisable description of the runtime environment.

    Never raises for missing optional components; they are reported as absent.
    """
    packages = {
        name: {"extra": extra, "version": _package_version(name)}
        for name, extra in _PACKAGES.items()
    }
    tools: dict[str, dict[str, object]] = {}
    for tool, names, override in (
        ("colmap", ["colmap", "COLMAP", "colmap.exe", "COLMAP.bat"], settings.colmap_path),
        ("blender", ["blender", "blender.exe"], settings.blender_path),
    ):
        try:
            tools[tool] = {
                "available": True,
                "path": str(resolve_executable(names, override, tool=tool)),
            }
        except Reverse3DError as exc:
            tools[tool] = {"available": False, "hint": exc.hint}

    from reverse3d._about import AUTHOR, AUTHOR_URL, OFFICIAL_REPOSITORY
    from reverse3d.hardware import hardware_report

    report: dict[str, object] = {
        "reverse3d": __version__,
        "author": f"{AUTHOR} ({AUTHOR_URL})",
        "official_source": OFFICIAL_REPOSITORY,
        # Voxel Enterprise's proprietary engine (bundled with reverse3d Studio).
        "core": core_status(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "workspace": str(settings.workspace),
        "settings": {
            "device": settings.device,
            "profile": settings.profile,
            "verbosity": settings.verbosity,
            "toolset": settings.toolset,
        },
        "hardware": hardware_report(settings),
        "limits": {
            "max_input_mb": settings.max_input_bytes // (1024 * 1024),
            "max_image_pixels": settings.max_image_pixels,
            "max_images": settings.max_images,
            "max_mesh_faces": settings.max_mesh_faces,
            "timeout_s": settings.subprocess_timeout_s,
        },
        "packages": packages,
        "external_tools": tools,
        "capabilities": {
            "vision": True,
            "mesh_repair": True,
            "primitive_fitting": True,
            "photogrammetry": bool(tools["colmap"]["available"]),
            "dense_meshing": packages["open3d"]["version"] is not None,
            "depth_estimation": packages["torch"]["version"] is not None
            and packages["transformers"]["version"] is not None,
            "render_blender": bool(tools["blender"]["available"]),
            "render_pyvista": packages["pyvista"]["version"] is not None,
            "mcp_server": packages["mcp"]["version"] is not None,
        },
    }
    if include_offline:
        from reverse3d.offline import offline_status

        status = offline_status(settings)
        report["offline"] = {
            k: status[k]
            for k in ("mode", "internet", "working_offline", "ready_for_offline", "missing")
        }
    return report
