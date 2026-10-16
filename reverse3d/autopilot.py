"""Autopilot: give it anything, it works out what it is and how to process it.

:func:`auto_run` is the engine behind ``reverse3d auto``, the ``auto`` recipe
and the watched inbox (:mod:`reverse3d.watch`). It is deterministic (plain
rules, no AI model), so it works the same offline and on any machine:

1. :func:`detect_input` classifies the input: a folder of photos, a drawing or
   silhouette, a single photo, a mesh or a point cloud.
2. It runs the matching recipe and retries only when another setting can fix
   the failure (another mask method, higher photogrammetry quality, stronger
   mesh repair). Problems that need the user (a missing tool, an unreadable
   input) stop immediately with a clear hint.
   When a photo set cannot be reconstructed (photos that do not overlap,
   shiny or changing surfaces...), it still delivers a model: an AI-depth
   relief of the best photo, flagged as approximate, plus capture advice.
3. It writes ``RESUMEN.md``: what was detected, sizes, files and next steps.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from reverse3d.config import Settings
from reverse3d.errors import (
    CaptureQualityError,
    DependencyMissingError,
    ExternalToolError,
    InputValidationError,
    ProcessingError,
    Reverse3DError,
)
from reverse3d.imaging import prepare_heic_input
from reverse3d.recipes import MaskMethod, RecipeOptions, run_recipe
from reverse3d.security import (
    HEIC_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MESH_EXTENSIONS,
    POINT_CLOUD_EXTENSIONS,
    list_images,
)

logger = logging.getLogger(__name__)

InputKind = Literal["photos", "plan", "photo", "mesh", "point_cloud", "unknown"]

_RECIPE_FOR: dict[str, str | None] = {
    "photos": "photos_to_model",
    "plan": "plan_to_part",
    "photo": "photo_to_relief",
    "mesh": "scan_to_cad",
    "point_cloud": "scan_to_cad",
    "unknown": None,
}

KIND_LABEL_ES: dict[str, str] = {
    "photos": "Fotos de un objeto (fotogrametría)",
    "plan": "Plano, careta o silueta",
    "photo": "Una foto (relieve con IA)",
    "mesh": "Malla o escaneo 3D",
    "point_cloud": "Nube de puntos",
    "unknown": "No reconocido",
}

_MIN_PHOTOS = 3

#: Capture guide in one sentence (docs/captura.md has the full guide; Studio shows it too).
CAPTURE_ADVICE_ES = (
    "Para un modelo 3D completo: 40-60 fotos (mínimo 20) en 4 perfiles: anillo bajo a 10-20° "
    "(16 fotos), medio a 35-45° (16), alto a 60-70° (12) y 4 desde arriba; con 60-80 % de "
    "solape, sin mover el objeto, luz pareja y sin reflejos (luces RGB apagadas; en superficies "
    "brillantes, spray mate o talco). Guía completa: docs/captura.md."
)
_ANALYSIS_SIDE = 256


@dataclass(frozen=True)
class Detection:
    """What :func:`detect_input` found."""

    kind: InputKind
    recipe: str | None
    reason: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _visible(path: Path) -> bool:
    return not path.name.startswith((".", "_", "~$"))


def _has_mesh_faces(ply: Path) -> bool:
    """Read the PLY header only: a PLY with faces is a mesh, without is a point cloud."""
    try:
        with ply.open("rb") as fh:
            header = fh.read(4096).decode("latin-1", errors="ignore")
    except OSError:
        return True
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "face"]:
            return parts[2] != "0"
        if line.strip() == "end_header":
            break
    return False


def looks_like_plan(image_path: Path) -> tuple[bool, str]:
    """Heuristic: drawings, stencils and silhouettes are near-binary images or
    have a transparent background; photos have many tones and colors."""
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return False, "unreadable image"
    if image.ndim == 3 and image.shape[2] == 4:
        transparent = float(np.mean(image[:, :, 3] < 16))
        if transparent > 0.05:
            return True, f"transparent background ({transparent:.0%})"
        image = image[:, :, :3]
    scale = _ANALYSIS_SIDE / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        gray, saturation = image, 0.0
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        saturation = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1]))
    extremes = float(np.mean((gray < 60) | (gray > 195)))
    if extremes > 0.85 and saturation < 45:
        return True, f"high-contrast drawing ({extremes:.0%} black/white pixels)"
    return False, f"photo-like image ({extremes:.0%} black/white pixels)"


def detect_input(path: str | Path) -> Detection:
    """Classify an input file or folder. Never raises for unknown content."""
    src = Path(path).expanduser()
    if src.is_dir():
        files = [p for p in sorted(src.iterdir()) if p.is_file() and _visible(p)]
        images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS | HEIC_EXTENSIONS]
        meshes = [p for p in files if p.suffix.lower() in MESH_EXTENSIONS | POINT_CLOUD_EXTENSIONS]
        if len(images) >= _MIN_PHOTOS:
            return Detection(
                "photos", "photos_to_model", f"folder with {len(images)} photos", str(src)
            )
        if len(images) == 1 and not meshes:
            return detect_input(images[0])
        if len(meshes) == 1 and not images:
            return detect_input(meshes[0])
        if images:
            return Detection(
                "unknown",
                None,
                f"only {len(images)} photos: photogrammetry needs at least {_MIN_PHOTOS} "
                "(ideally 40-60)",
                str(src),
            )
        return Detection("unknown", None, "no photos, drawings or 3D files in the folder", str(src))

    if not src.is_file():
        return Detection("unknown", None, "the path does not exist", str(src))
    ext = src.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        is_plan, why = looks_like_plan(src)
        kind: InputKind = "plan" if is_plan else "photo"
        return Detection(kind, _RECIPE_FOR[kind], why, str(src))
    if ext == ".ply":
        kind = "mesh" if _has_mesh_faces(src) else "point_cloud"
        return Detection(
            kind, "scan_to_cad", f"PLY {'mesh' if kind == 'mesh' else 'point cloud'}", str(src)
        )
    if ext in MESH_EXTENSIONS:
        return Detection("mesh", "scan_to_cad", f"{ext[1:].upper()} mesh", str(src))
    if ext in POINT_CLOUD_EXTENSIONS:
        return Detection("point_cloud", "scan_to_cad", f"{ext[1:].upper()} point cloud", str(src))
    return Detection("unknown", None, f"unsupported file type '{ext or src.name}'", str(src))


@dataclass
class _Attempt:
    label: str
    options: RecipeOptions


def _attempt_plan(detection: Detection, base: RecipeOptions) -> list[_Attempt]:
    if detection.kind == "plan":
        _, why = looks_like_plan(Path(detection.source))
        methods: list[MaskMethod] = (
            ["alpha", "otsu"] if why.startswith("transparent") else ["otsu", "adaptive", "canny"]
        )
        return [
            _Attempt(f"mask={m}", replace(base, mask_method=m, overwrite=base.overwrite or i > 0))
            for i, m in enumerate(methods)
        ]
    if detection.kind == "photos" and base.quality != "high":
        return [
            _Attempt("profile quality", base),
            _Attempt("quality=high", replace(base, quality="high", overwrite=True)),
        ]
    return [_Attempt("default", base)]


def _mesh_from_points(detection: Detection, out: Path, settings: Settings, overwrite: bool) -> Path:
    from reverse3d.reconstruction.pointcloud import poisson_mesh

    target = out / "mesh_from_points.ply"
    poisson_mesh(detection.source, target, settings, overwrite=overwrite)
    return target


def _nearer_part(mask: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    """Keep the part of ``mask`` that stands out from the surface (the object,
    not the desk, pad or light glow around it): Otsu split of its height."""
    inside = mask > 0
    if inside.sum() < 100 or (~inside).sum() < 100:
        return mask
    from reverse3d.reconstruction.depth_estimation import height_above_support

    height = height_above_support(mask, disparity)
    values = height[inside]
    lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    if hi - lo < 1e-6:
        return mask
    scaled = ((height - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(
        scaled[inside].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    near: np.ndarray = np.where(inside & (scaled > threshold), 255, 0).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(near)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    near = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(near, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(near)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    # A tiny blob means the split failed (a flat object): keep the first mask.
    return filled if filled.sum() >= 0.2 * inside.sum() * 255 else mask


def _best_photo(folder: Path, out: Path, settings: Settings) -> tuple[Path, Path | None]:
    """Photo for the fallback relief, and a mask that isolates the object in it.

    Every photo gets a quick GrabCut; the one where the object looks largest,
    whole and sharp wins. Its mask is then refined at higher resolution and
    trimmed with the AI depth so the desk and background stay out of the relief.
    """
    from reverse3d.hardware import torch_device
    from reverse3d.imaging import downscale, load_image, save_image
    from reverse3d.reconstruction.depth_estimation import predict_depth
    from reverse3d.vision.mask_extractor import foreground_mask, mask_score

    images = list_images(folder, max_images=settings.max_images, max_bytes=settings.max_input_bytes)
    best: tuple[float, Path] | None = None
    for path in images:
        small, _ = downscale(load_image(path, settings), 320)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharp = min(float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 300.0, 1.0)
        try:
            score = mask_score(foreground_mask(small, work_side=320)) * (0.5 + 0.5 * sharp)
        except cv2.error:
            score = 0.1 * sharp
        if best is None or score > best[0]:
            best = (score, path)
    if best is None:  # pragma: no cover - detect_input guarantees photos
        raise ProcessingError("No photo available for the fallback relief.")
    score, photo = best
    if score < 0.15:
        return photo, None  # no clear object: keep the whole photo

    full = load_image(photo, settings)
    work, _ = downscale(full, 800)
    mask = foreground_mask(work, work_side=800)
    try:
        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        mask = _nearer_part(
            mask, predict_depth(rgb, device=torch_device(settings), settings=settings)
        )
    except Reverse3DError as exc:  # depth model unavailable: the GrabCut mask still helps
        logger.info("Depth refinement of the fallback mask skipped: %s", exc.message)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask = cv2.resize(mask, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST)
    return photo, save_image(mask, out / "fallback_mask.png", overwrite=True)


def _relief_fallback(
    source: Path,
    out: Path,
    settings: Settings,
    base: RecipeOptions,
    reason: str,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Photogrammetry was not possible: build an approximate model from the best photo."""
    photo, mask = _best_photo(source, out, settings)
    if base.scene in ("architecture", "landscape"):
        mask = None  # an open scene: keep the whole view as a relief
    logger.info("Photogrammetry not possible; building an AI-depth relief from %s", photo.name)
    summary = run_recipe(
        "photo_to_relief",
        photo,
        out,
        settings,
        replace(base, overwrite=True, mask_path=str(mask) if mask else None),
    )
    if mask:
        summary["outputs"]["mask"] = str(mask)
    history.append({"attempt": f"fallback=photo_to_relief ({photo.name})", "ok": True})
    summary["warnings"] = [
        "A full 3D model could not be reconstructed from these photos "
        f"({reason}). This is an approximate relief from the best photo, {photo.name}.",
        *summary.get("warnings", []),
    ]
    summary["fallback"] = {"from": "photos_to_model", "photo": str(photo), "reason": reason}
    return summary


def auto_run(
    input_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    options: RecipeOptions | None = None,
) -> dict[str, Any]:
    """Detect the input, run the right recipe (with smart retries) and write RESUMEN.md.

    Returns the recipe summary plus an ``auto`` block (detection and attempts).

    Raises:
        InputValidationError: the input is not something reverse3d can process.
        CaptureQualityError: the photos must be retaken (with recommendations).
        DependencyMissingError: a required tool or model is not available.
        Reverse3DError: every attempt failed (the last error is raised).
    """
    base = options or RecipeOptions()
    # iPhone photos (HEIC) are converted to JPEG first; other inputs pass through.
    input_path = prepare_heic_input(input_path, output_dir)
    detection = detect_input(input_path)
    if detection.recipe is None:
        raise InputValidationError(
            f"Cannot process this input: {detection.reason}.",
            hint="Use a folder with 20-60 photos, a drawing or photo, or an STL/OBJ/PLY/GLB file.",
        )

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(detection.source)
    if detection.kind == "point_cloud":
        source = _mesh_from_points(detection, out, settings, base.overwrite)

    history: list[dict[str, Any]] = []
    scene_info: dict[str, Any] | None = None
    if detection.kind == "photos":
        from reverse3d.scenes import SCENE_PRESETS, detect_scene

        if base.scene == "auto":
            predicted = detect_scene(source, settings)
            scene_info = predicted.to_dict()
            base = replace(base, scene=predicted.kind)
        elif base.scene in SCENE_PRESETS:
            scene_info = {"kind": base.scene, "confidence": 1.0, "reasons": ["chosen"]}
    summary: dict[str, Any] | None = None
    last_error: Reverse3DError | None = None
    capture_advice: list[str] = []
    for attempt in _attempt_plan(detection, base):
        try:
            summary = run_recipe(detection.recipe, source, out, settings, attempt.options)
        except CaptureQualityError as exc:
            history.append({"attempt": attempt.label, "ok": False, "error": exc.message})
            if detection.kind != "photos":
                raise
            last_error = exc
            break  # better settings will not fix the capture: go to the fallback
        except (DependencyMissingError, InputValidationError) as exc:
            history.append({"attempt": attempt.label, "ok": False, "error": exc.message})
            raise  # the user has to act: retrying with other settings will not help
        except (ProcessingError, ExternalToolError) as exc:
            history.append({"attempt": attempt.label, "ok": False, "error": exc.message})
            last_error = exc
            logger.info("Autopilot attempt '%s' failed: %s", attempt.label, exc.message)
            continue
        history.append({"attempt": attempt.label, "ok": True})
        if (
            detection.kind in ("mesh", "point_cloud")
            and summary["dimensions"].get("watertight") is False
            and attempt.options.repair == "normal"
        ):
            try:
                stronger = run_recipe(
                    detection.recipe,
                    source,
                    out,
                    settings,
                    replace(attempt.options, repair="strong", overwrite=True),
                )
            except Reverse3DError as exc:
                history.append({"attempt": "repair=strong", "ok": False, "error": exc.message})
            else:
                history.append({"attempt": "repair=strong", "ok": True})
                if stronger["dimensions"].get("watertight"):
                    summary = stronger
        break

    if summary is None and detection.kind == "photos" and last_error is not None:
        reason = last_error.message.split(":", 1)[0].rstrip(".")
        from reverse3d.scenes import scene_advice

        photos = len(
            list_images(source, max_images=settings.max_images, max_bytes=settings.max_input_bytes)
        )
        capture_advice = (
            scene_advice(base.scene, photos) if base.scene != "auto" else [CAPTURE_ADVICE_ES]
        )
        try:
            summary = _relief_fallback(source, out, settings, base, reason, history)
        except Reverse3DError as exc:
            logger.warning("Relief fallback failed: %s", exc.message)
            history.append(
                {"attempt": "fallback=photo_to_relief", "ok": False, "error": exc.message}
            )
            raise last_error from exc
    if summary is None:
        raise last_error or ProcessingError("No processing attempt could run.")

    summary["auto"] = {"detected": detection.to_dict(), "attempts": history}
    if scene_info is not None:
        summary["auto"]["scene"] = scene_info
    summary["next_steps"] = capture_advice + next_steps(detection, summary, base)
    summary["outputs"]["summary"] = str(write_summary(out, detection, summary))
    return summary


def next_steps(detection: Detection, summary: dict[str, Any], options: RecipeOptions) -> list[str]:
    """Plain-language suggestions (Spanish) for what to do with the results."""
    outputs = summary.get("outputs", {})
    steps: list[str] = []
    if "cad_openscad" in outputs:
        steps.append(
            "Edita cad.scad en OpenSCAD (o cad.py con CadQuery) para ajustar medidas "
            "y exportar STEP."
        )
    if "model" in outputs:
        steps.append(
            "Abre model.stl en tu laminador (Cura, PrusaSlicer, Bambu Studio…) para imprimirlo."
        )
    if detection.kind == "plan" and options.units_per_pixel is None:
        steps.append(
            "Las medidas están en píxeles: indica la escala con --units-per-pixel "
            "para tener medidas reales."
        )
    relief = detection.kind == "photo" or "fallback" in summary
    if relief:
        steps.append(
            "La profundidad con IA es relativa: usa --scale si necesitas un tamaño concreto."
        )
    if relief:
        steps.append(
            "Un relieve es una superficie abierta: dale grosor en tu programa 3D "
            "si lo vas a imprimir."
        )
    elif summary.get("dimensions", {}).get("watertight") is False:
        steps.append("La malla no quedó cerrada (no estanca): revisa los avisos antes de imprimir.")
    if "render" not in outputs:
        steps.append(
            "Instala Blender (o el extra render) para obtener una vista previa render.png."
        )
    return steps


def _fmt(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " × ".join(f"{float(v):.2f}" for v in value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_summary(out: Path, detection: Detection, summary: dict[str, Any]) -> Path:
    """Write ``RESUMEN.md``: a human-readable report next to the results."""
    dims = summary.get("dimensions", {})
    lines = [
        f"# reverse3d · {Path(detection.source).name}",
        "",
        f"- **Detectado:** {KIND_LABEL_ES[detection.kind]}",
        f"- **Flujo:** `{summary.get('recipe')}` · {str(summary.get('device', 'cpu')).upper()}"
        f" · perfil {summary.get('profile')}",
        f"- **Tiempo:** {summary.get('seconds')} s",
        "",
        "## Medidas (unidades del modelo)",
        "",
        f"- Tamaño (caja orientada): {_fmt(dims.get('size_obb', '—'))}",
        f"- Volumen: {_fmt(dims.get('volume', '—'))}",
        f"- Superficie: {_fmt(dims.get('area', '—'))}",
        f"- Cerrada (estanca): {'sí' if dims.get('watertight') else 'no'}",
        "",
        "## Archivos",
        "",
    ]
    lines += [
        f"- {key}: `{Path(path).name}`" for key, path in summary.get("outputs", {}).items() if path
    ]
    lines.append("- summary: `RESUMEN.md`")
    if summary.get("primitives"):
        lines += ["", "## Geometría detectada", ""]
        lines += [
            f"- {p.get('kind')}: "
            + ", ".join(f"{k}={_fmt(v)}" for k, v in p.items() if k != "kind")
            for p in summary["primitives"]
        ]
    if summary.get("warnings"):
        lines += ["", "## Avisos", ""] + [f"- {w}" for w in summary["warnings"]]
    if summary.get("next_steps"):
        lines += ["", "## Siguiente paso", ""] + [f"- {s}" for s in summary["next_steps"]]
    path = out / "RESUMEN.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
