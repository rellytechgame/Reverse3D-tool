"""Mask, silhouette and contour extraction from photos, plans and stencils.

Typical inputs are a technical drawing, a scanned stencil/template ("careta"),
a product photo on a plain background, or an image with an alpha channel. The
output is a binary mask, vector contours (outer boundaries + holes) in JSON and
SVG, and optionally an extruded 3D solid built from those contours.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import InputValidationError, ProcessingError
from reverse3d.imaging import load_image, save_image
from reverse3d.security import (
    JSON_EXTENSIONS,
    prepare_output_dir,
    prepare_output_file,
    validate_input_file,
)

logger = logging.getLogger(__name__)

MaskMethod = Literal["otsu", "adaptive", "canny", "alpha"]
CONTOURS_SCHEMA_VERSION = 1


@dataclass
class Polygon2D:
    """A closed outline in pixel coordinates (x right, y down) with optional holes."""

    exterior: list[list[float]]
    holes: list[list[list[float]]] = field(default_factory=list)
    area_px: float = 0.0


@dataclass
class MaskResult:
    """Outputs of :func:`extract_mask`."""

    mask_path: str
    contours_path: str
    svg_path: str
    image_size: tuple[int, int]
    polygon_count: int
    total_area_px: float
    bbox_px: tuple[int, int, int, int] | None
    units_per_pixel: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _binary_mask(image: np.ndarray, method: MaskMethod, blur: int) -> np.ndarray:
    if method == "alpha":
        if image.ndim != 3 or image.shape[2] != 4:
            raise InputValidationError(
                "method='alpha' requires an image with an alpha channel (RGBA PNG)."
            )
        return np.where(image[:, :, 3] > 127, 255, 0).astype(np.uint8)

    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if blur > 1:
        k = blur if blur % 2 == 1 else blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
    elif method == "canny":
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        # Fill closed edge loops so the mask covers the object area, not only its outline.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        return mask
    else:  # pragma: no cover - guarded by Literal + CLI/agent validation
        raise InputValidationError(f"Unknown mask method: {method}")
    return mask


def _auto_invert(mask: np.ndarray) -> np.ndarray:
    """Make the object white: if the image border is mostly white the
    background was thresholded as foreground (dark object on light paper)."""
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return cv2.bitwise_not(mask) if border.mean() > 127 else mask


def _contours_to_polygons(
    mask: np.ndarray, min_area_px: float, epsilon_ratio: float
) -> list[Polygon2D]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    polygons: list[Polygon2D] = []

    def simplify(contour: np.ndarray) -> list[list[float]]:
        eps = epsilon_ratio * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True) if eps > 0 else contour
        return [[float(x), float(y)] for x, y in approx.reshape(-1, 2)]

    for idx, contour in enumerate(contours):
        # RETR_CCOMP: top-level contours have parent == -1, their children are holes.
        if hierarchy[idx][3] != -1:
            continue
        area = float(cv2.contourArea(contour))
        if area < min_area_px:
            continue
        exterior = simplify(contour)
        if len(exterior) < 3:
            continue
        holes: list[list[list[float]]] = []
        hole_area = 0.0
        child = hierarchy[idx][2]
        while child != -1:
            child_area = float(cv2.contourArea(contours[child]))
            if child_area >= min_area_px:
                hole = simplify(contours[child])
                if len(hole) >= 3:
                    holes.append(hole)
                    hole_area += child_area
            child = hierarchy[child][0]
        polygons.append(Polygon2D(exterior=exterior, holes=holes, area_px=area - hole_area))

    polygons.sort(key=lambda p: p.area_px, reverse=True)
    return polygons


def _write_svg(path: Path, polygons: list[Polygon2D], width: int, height: int) -> None:
    def ring(points: list[list[float]]) -> str:
        head, *tail = points
        return (
            f"M{head[0]:.2f},{head[1]:.2f} " + " ".join(f"L{x:.2f},{y:.2f}" for x, y in tail) + " Z"
        )

    paths = []
    for poly in polygons:
        d = " ".join([ring(poly.exterior), *(ring(h) for h in poly.holes)])
        paths.append(
            f'  <path d="{d}" fill="#1f2937" fill-rule="evenodd" '
            'stroke="#ef4444" stroke-width="1"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n' + "\n".join(paths) + "\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def extract_mask(
    image_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    *,
    method: MaskMethod = "otsu",
    invert: bool | None = None,
    blur: int = 5,
    min_area_ratio: float = 0.001,
    simplify_ratio: float = 0.002,
    units_per_pixel: float | None = None,
    overwrite: bool = False,
) -> MaskResult:
    """Extract a binary mask and vector contours from an image.

    Args:
        image_path: photo, plan or stencil image.
        output_dir: directory that receives ``<stem>_mask.png``,
            ``<stem>_contours.json`` and ``<stem>_contours.svg``.
        settings: runtime limits.
        method: ``otsu`` (plain backgrounds), ``adaptive`` (uneven lighting,
            line drawings), ``canny`` (edge-based, textured objects) or
            ``alpha`` (use the PNG alpha channel).
        invert: force inversion; ``None`` auto-detects from the image border.
        blur: Gaussian blur kernel applied before thresholding (0 disables).
        min_area_ratio: discard regions smaller than this fraction of the image.
        simplify_ratio: Douglas-Peucker tolerance as a fraction of the perimeter.
        units_per_pixel: optional real-world scale (e.g. mm/px) stored with the
            contours and used later by :func:`silhouette_to_mesh`.
        overwrite: replace existing outputs.

    Raises:
        InputValidationError: bad arguments or image.
        ProcessingError: no region survived the filters.
    """
    if not 0 <= min_area_ratio < 1:
        raise InputValidationError("min_area_ratio must be in [0, 1).")
    if not 0 <= simplify_ratio <= 0.1:
        raise InputValidationError("simplify_ratio must be in [0, 0.1].")
    if not 0 <= blur <= 51:
        raise InputValidationError("blur must be in [0, 51].")
    if units_per_pixel is not None and units_per_pixel <= 0:
        raise InputValidationError("units_per_pixel must be positive.")

    flags = cv2.IMREAD_UNCHANGED if method == "alpha" else cv2.IMREAD_COLOR
    image = load_image(image_path, settings, flags=flags)
    if image.ndim == 3 and image.shape[2] == 4 and method != "alpha":
        image = image[:, :, :3]
    height, width = image.shape[:2]

    mask = _binary_mask(image, method, blur)
    if invert is None and method != "alpha":
        mask = _auto_invert(mask)
    elif invert:
        mask = cv2.bitwise_not(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    polygons = _contours_to_polygons(mask, min_area_ratio * width * height, simplify_ratio)
    if not polygons:
        raise ProcessingError(
            "No region found in the mask.",
            hint="Try another method (adaptive/canny), invert=true, or a lower min_area_ratio.",
        )

    warnings: list[str] = []
    coverage = sum(p.area_px for p in polygons) / (width * height)
    if coverage > 0.95:
        warnings.append("Mask covers >95% of the image; inversion may be wrong.")

    out_dir = prepare_output_dir(output_dir)
    stem = Path(image_path).stem
    mask_path = save_image(mask, out_dir / f"{stem}_mask.png", overwrite=overwrite)
    contours_path = prepare_output_file(
        out_dir / f"{stem}_contours.json", allowed_extensions=JSON_EXTENSIONS, overwrite=overwrite
    )
    svg_path = prepare_output_file(
        out_dir / f"{stem}_contours.svg", allowed_extensions={".svg"}, overwrite=overwrite
    )

    contours_doc = {
        "schema_version": CONTOURS_SCHEMA_VERSION,
        "source_image": Path(image_path).name,
        "image_size": [width, height],
        "units_per_pixel": units_per_pixel,
        "polygons": [asdict(p) for p in polygons],
    }
    contours_path.write_text(json.dumps(contours_doc, indent=2), encoding="utf-8")
    _write_svg(svg_path, polygons, width, height)

    xs, ys, ws, hs = cv2.boundingRect(mask)
    logger.info("Extracted %d polygon(s) from %s", len(polygons), Path(image_path).name)
    return MaskResult(
        mask_path=str(mask_path),
        contours_path=str(contours_path),
        svg_path=str(svg_path),
        image_size=(width, height),
        polygon_count=len(polygons),
        total_area_px=round(sum(p.area_px for p in polygons), 2),
        bbox_px=(xs, ys, ws, hs) if ws and hs else None,
        units_per_pixel=units_per_pixel,
        warnings=warnings,
    )


def load_contours(path: str | Path, settings: Settings) -> dict[str, Any]:
    """Load and structurally validate a contours JSON produced by :func:`extract_mask`."""
    resolved = validate_input_file(
        path,
        allowed_extensions=JSON_EXTENSIONS,
        max_bytes=min(settings.max_input_bytes, 64 * 1024 * 1024),
    )
    try:
        doc = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(f"Invalid contours JSON: {resolved.name}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != CONTOURS_SCHEMA_VERSION:
        raise InputValidationError("Contours file has an unknown schema version.")
    polygons = doc.get("polygons")
    if not isinstance(polygons, list) or not polygons:
        raise InputValidationError("Contours file has no polygons.")
    for poly in polygons:
        if not isinstance(poly, dict) or not isinstance(poly.get("exterior"), list):
            raise InputValidationError("Malformed polygon in contours file.")
    return doc


def silhouette_to_mesh(
    contours_path: str | Path,
    output_path: str | Path,
    settings: Settings,
    *,
    thickness: float = 10.0,
    units_per_pixel: float | None = None,
    max_polygons: int = 50,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extrude the polygons of a contours JSON into a 3D solid (plan → part).

    Image Y points down; the outline is mirrored so the solid keeps the drawing's
    orientation when viewed from +Z. Coordinates are scaled by
    ``units_per_pixel`` (argument, else the value stored in the JSON, else 1).

    Returns:
        A dict with the output path and basic stats of the extruded mesh.
    """
    import shapely.geometry as sg
    import trimesh

    from reverse3d.mesh_io import save_mesh

    if not 0 < thickness <= 1e6:
        raise InputValidationError("thickness must be in (0, 1e6].")
    doc = load_contours(contours_path, settings)
    scale = units_per_pixel or doc.get("units_per_pixel") or 1.0
    if not isinstance(scale, (int, float)) or scale <= 0:
        raise InputValidationError("units_per_pixel must be positive.")

    meshes = []
    for raw in doc["polygons"][:max_polygons]:

        def to_xy(points: list[list[float]]) -> np.ndarray:
            arr = np.asarray(points, dtype=np.float64).reshape(-1, 2) * float(scale)
            arr[:, 1] *= -1.0
            return arr

        shell = to_xy(raw["exterior"])
        holes = [to_xy(h) for h in raw.get("holes", [])]
        polygon = sg.Polygon(shell, holes).buffer(0)  # buffer(0) repairs self-intersections
        if polygon.is_empty:
            continue
        parts = list(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]
        for part in parts:
            if part.area > 0:
                meshes.append(trimesh.creation.extrude_polygon(part, height=float(thickness)))
    if not meshes:
        raise ProcessingError("No valid polygon could be extruded.")

    mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
    out = save_mesh(mesh, output_path, overwrite=overwrite)
    return {
        "output_path": str(out),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "extents": [round(float(v), 6) for v in mesh.extents],
        "volume": round(float(mesh.volume), 6) if mesh.is_watertight else None,
        "units_per_pixel": float(scale),
    }


def foreground_mask(image: np.ndarray, *, work_side: int = 640, margin: float = 0.06) -> np.ndarray:
    """Separate the main object from a cluttered background (OpenCV GrabCut).

    Assumes the object is roughly centred in the photo. Works on a downscaled
    copy and returns a full-size ``uint8`` mask (255 = object) holding only the
    largest region, slightly eroded so depth edges do not leave spikes.
    """
    h, w = image.shape[:2]
    scale = work_side / max(h, w)
    small = cv2.resize(
        image[:, :, :3],
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    sh, sw = small.shape[:2]
    rect = (
        round(sw * margin),
        round(sh * margin),
        round(sw * (1 - 2 * margin)),
        round(sh * (1 - 2 * margin)),
    )
    mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg: np.ndarray = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(
        np.uint8
    )
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, contours, -1, 255, thickness=cv2.FILLED)  # fill holes (reflections)
    fg = cv2.erode(fg, np.ones((3, 3), np.uint8))
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def mask_score(mask: np.ndarray) -> float:
    """How usable a foreground mask is for a relief: a large, compact object
    that does not touch the photo border scores high (0..1)."""
    area = float((mask > 0).mean())
    if area < 0.03:
        return 0.0
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    solidity = float((mask > 0).sum()) / max(float(cv2.contourArea(hull)), 1.0)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    touches = float((border > 0).mean())
    size = min(area / 0.35, 1.0)  # the object filling ~35 % of the frame is plenty
    return size * min(solidity, 1.0) * (1.0 - touches)
