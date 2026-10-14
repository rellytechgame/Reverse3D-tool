"""Token-efficient shaping of tool results for LLMs (``verbosity=compact``).

* Absolute paths inside the workspace become short relative paths.
* Floats are rounded to 6 significant digits.
* ``None`` values and empty containers are dropped.
* Long lists of objects are truncated with an ``{"_omitted": n}`` marker.
* Per-tool reducers keep only what an agent needs to decide the next step
  (e.g. only the images *with problems* from an image-set analysis).

Nothing is lost: recipes write the full report to disk, and
``REVERSE3D_VERBOSITY=full`` disables compaction.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path, PurePath
from typing import Any

_MAX_LIST = 12


def _relativize(value: str, root: str) -> str:
    if len(value) < 3 or not PurePath(value).is_absolute():
        return value
    try:
        rel = Path(value).relative_to(root)
    except ValueError:
        return value
    return rel.as_posix() or "."


def shrink(value: Any, root: str, *, max_list: int = _MAX_LIST) -> Any:
    """Recursively compact a JSON-like structure."""
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, str):
        return _relativize(value, root)
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            item = shrink(item, root, max_list=max_list)
            if item is None or item == [] or item == {}:
                continue
            out[key] = item
        return out
    if isinstance(value, (list, tuple)):
        items = [shrink(v, root, max_list=max_list) for v in value]
        if len(items) > max_list and any(isinstance(v, dict) for v in items):
            return [*items[:max_list], {"_omitted": len(items) - max_list}]
        return items
    return value


def _reduce_analysis(result: dict[str, Any]) -> dict[str, Any]:
    images = result.get("images", [])
    pairs = result.get("pairs", [])
    return {
        "image_count": result.get("image_count"),
        "suitable_for_photogrammetry": result.get("suitable_for_photogrammetry"),
        "recommendations": result.get("recommendations"),
        "images_with_issues": [
            {"name": i["name"], "issues": i["issues"]} for i in images if i.get("issues")
        ],
        "weak_links": result.get("weak_links"),
        "median_pair_inliers": sorted(p["inliers"] for p in pairs)[len(pairs) // 2]
        if pairs
        else None,
    }


def _reduce_clean(result: dict[str, Any]) -> dict[str, Any]:
    keys = ("faces", "watertight", "components", "boundary_edges", "obb_extents", "volume")
    return {
        "output_path": result.get("output_path"),
        "before": {k: result["before"].get(k) for k in keys},
        "after": {k: result["after"].get(k) for k in keys},
        "steps": result.get("steps"),
        "warnings": result.get("warnings"),
    }


def _reduce_photogrammetry(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k not in ("stages", "sparse_model_dir")}


def _reduce_fit(result: dict[str, Any]) -> dict[str, Any]:
    prims = [
        {"kind": p["kind"], "inlier_ratio": p["inlier_ratio"], **p["params"]}
        for p in result.get("primitives", [])
    ]
    return {**{k: v for k, v in result.items() if k != "primitives"}, "primitives": prims}


REDUCERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "reverse3d_analyze_images": _reduce_analysis,
    "reverse3d_clean_mesh": _reduce_clean,
    "reverse3d_photogrammetry": _reduce_photogrammetry,
    "reverse3d_fit_primitives": _reduce_fit,
}


def compact_result(tool: str, result: Any, workspace: Path) -> Any:
    """Apply the tool's reducer (if any) and :func:`shrink`."""
    reducer = REDUCERS.get(tool)
    if reducer is not None and isinstance(result, dict):
        result = reducer(result)
    return shrink(result, str(workspace))
