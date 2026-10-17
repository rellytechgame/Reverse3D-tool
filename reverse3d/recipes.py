"""One-call end-to-end workflows ("recipes").

A recipe chains the individual tools into a complete job and returns a short
summary. For AI agents this is the main token saver: one tool call and one
small result instead of five or six round-trips with full intermediate
reports. The complete details are written to ``<output_dir>/report.json``.

Recipes
-------
photos_to_model   photo set   → analysis → photogrammetry → clean/align → measure → render
plan_to_part      plan/stencil → mask → extrusion → clean → measure → render
scan_to_cad       mesh/scan   → clean/align → measure → primitives → OpenSCAD + CadQuery → render
photo_to_relief   one photo   → depth → 2.5D mesh → clean → render
mesh_report       mesh        → measure → render
auto              anything    → detects the input and runs the right recipe, retrying
                                with other parameters when that can help (see autopilot)

Device (CPU/GPU) and speed/quality come from ``Settings.device`` and
``Settings.profile`` unless explicitly overridden.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from reverse3d.config import Settings
from reverse3d.errors import (
    CaptureQualityError,
    DependencyMissingError,
    InputValidationError,
    ProcessingError,
    Reverse3DError,
)
from reverse3d.hardware import torch_device, wants_gpu
from reverse3d.profiles import profile_values
from reverse3d.reverse_eng.measure import MeshStats
from reverse3d.security import prepare_output_dir

logger = logging.getLogger(__name__)

RecipeName = Literal[
    "photos_to_model", "plan_to_part", "scan_to_cad", "photo_to_relief", "mesh_report", "auto"
]
RECIPES: tuple[RecipeName, ...] = (
    "photos_to_model",
    "plan_to_part",
    "scan_to_cad",
    "photo_to_relief",
    "mesh_report",
    "auto",
)
MaskMethod = Literal["otsu", "adaptive", "canny", "alpha"]
_T = TypeVar("_T")


@dataclass
class RecipeOptions:
    """Optional knobs shared by recipes. ``None`` means "use the profile"."""

    thickness: float = 5.0
    units_per_pixel: float | None = None
    scale: float = 1.0
    render: bool = True
    quality: Literal["low", "medium", "high"] | None = None
    force_analysis: bool = False
    overwrite: bool = False
    #: plan_to_part: how the drawing is separated from the background.
    mask_method: MaskMethod = "otsu"
    #: photos_to_model: scene profile (object, character, architecture, landscape) or "auto".
    scene: str = "auto"
    #: what was photographed (AI classifier) and the render look; None = from the scene.
    subject: str | None = None
    render_preset: str | None = None
    #: photo_to_relief: object mask (white = keep), e.g. from the autopilot fallback.
    mask_path: str | None = None
    #: mesh repair strength; "strong" closes much larger holes.
    repair: Literal["normal", "strong"] = "normal"


@dataclass
class _Run:
    recipe: str
    settings: Settings
    out: Path
    outputs: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def step(self, name: str, fn: Callable[[], _T]) -> _T:
        start = time.perf_counter()
        try:
            value = fn()
        except Exception:
            self.steps.append(
                {"step": name, "ok": False, "s": round(time.perf_counter() - start, 2)}
            )
            raise
        self.steps.append({"step": name, "ok": True, "s": round(time.perf_counter() - start, 2)})
        return value


def _summary_dims(stats: MeshStats) -> dict[str, Any]:
    return {
        "size_obb": stats.obb_extents,
        "size_aabb": stats.aabb_extents,
        "volume": stats.volume,
        "area": stats.surface_area,
        "watertight": stats.watertight,
        "faces": stats.faces,
    }


def _clean(
    run: _Run,
    src: Path,
    dst: Path,
    *,
    scale: float,
    overwrite: bool,
    fill_holes: bool = True,
    align: Literal["none", "obb"] = "obb",
    keep_largest: bool = False,
    repair: Literal["normal", "strong"] = "normal",
) -> MeshStats:
    """Repair + align. ``keep_largest`` only for photogrammetry/relief output,
    where floating fragments are noise; plans and scans may have several real
    parts, and tiny debris is already dropped by the area filter."""
    from reverse3d.reverse_eng.mesh_cleaner import CleanOptions, clean_mesh_file

    report = run.step(
        "clean",
        lambda: clean_mesh_file(
            src,
            dst,
            run.settings,
            CleanOptions(
                keep_largest_only=keep_largest,
                fill_holes=fill_holes,
                align=align,
                ground=align != "none",
                scale=scale,
                max_hole_edges=2000 if repair == "strong" else 200,
            ),
            overwrite=overwrite,
        ),
    )
    run.warnings.extend(report.warnings)
    run.outputs["model"] = str(dst)
    run.details["clean"] = report.to_dict()
    return report.after


def _render(run: _Run, mesh: Path, overwrite: bool) -> None:
    from reverse3d.rendering.blender_render import RenderOptions, preset_options, render_mesh

    prof = profile_values(run.settings)
    look = str(run.details.get("scene", {}).get("render") or "studio")
    options = preset_options(
        look,
        RenderOptions(
            width=prof.render_width,
            height=prof.render_height,
            samples=prof.render_samples,
            device="gpu" if wants_gpu(run.settings) else "cpu",
        ),
    )
    run.details["render_look"] = look
    try:
        result = run.step(
            "render",
            lambda: render_mesh(
                mesh, run.out / "render.png", run.settings, options, overwrite=overwrite
            ),
        )
    except DependencyMissingError as exc:
        run.warnings.append(f"Render skipped: {exc.message} {exc.hint or ''}".strip())
        return
    run.outputs["render"] = result.files[0]
    run.details["render"] = result.to_dict()


def _measure(run: _Run, mesh: Path) -> MeshStats:
    from reverse3d.reverse_eng.measure import measure_mesh_file

    stats = run.step("measure", lambda: measure_mesh_file(mesh, run.settings))
    run.details["measure"] = stats.to_dict()
    return stats


# ─── recipes ──────────────────────────────────────────────────────────────────


# Smallest overlapping group worth reconstructing when the rest of a set does not match.
_MIN_GROUP = 8


_DEFAULT_LOOK = {
    "object": "studio",
    "character": "portrait",
    "architecture": "architecture",
    "landscape": "aerial",
}


def _color_outputs(run: _Run, colored: Path, model: Path, scale: float) -> None:
    """Give the final model the photo colours of the carved hull: model.glb (games,
    web, viewers) and model_color.ply (renders)."""
    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    source: Any = trimesh.load(colored, process=False, force="mesh")
    target: Any = trimesh.load(model, process=False, force="mesh")
    colors = getattr(source.visual, "vertex_colors", None)
    if colors is None or len(colors) != len(source.vertices):
        return
    _, nearest = cKDTree(np.asarray(source.vertices) * scale).query(np.asarray(target.vertices))
    target.visual = trimesh.visual.ColorVisuals(target, vertex_colors=np.asarray(colors)[nearest])
    glb = run.out / "model.glb"
    ply = run.out / "model_color.ply"
    target.export(glb)
    target.export(ply)
    run.outputs["model_glb"] = str(glb)
    run.outputs["model_color"] = str(ply)


def _photos_to_model(run: _Run, src: Path, opt: RecipeOptions) -> MeshStats:
    from reverse3d.reconstruction.photogrammetry import (
        PhotogrammetryOptions,
        PhotogrammetryResult,
        run_photogrammetry,
    )
    from reverse3d.scenes import SCENE_LABEL_ES, SCENE_PRESETS, detect_scene
    from reverse3d.vision.feature_matcher import analyze_image_set

    scene: str
    subject, look = opt.subject, opt.render_preset
    if opt.scene == "auto":
        detection = run.step("scene", lambda: detect_scene(src, run.settings))
        scene, confidence, predicted = detection.kind, detection.confidence, True
        subject, look = detection.subject, detection.render
    elif opt.scene in SCENE_PRESETS:
        scene, confidence, predicted = opt.scene, 1.0, False
    else:
        raise InputValidationError(
            f"Unknown scene '{opt.scene}'.", hint=f"Use auto or one of: {', '.join(SCENE_PRESETS)}"
        )
    preset = SCENE_PRESETS[scene]
    run.details["scene"] = {
        "kind": scene,
        "label": SCENE_LABEL_ES[scene],
        "confidence": confidence,
        "predicted": predicted,
        "subject": subject,
        "render": look or _DEFAULT_LOOK[scene],
    }
    analysis = run.step("analyze", lambda: analyze_image_set(src, run.settings))
    run.details["analysis"] = analysis.to_dict()
    if not analysis.suitable_for_photogrammetry and analysis.connected_images >= _MIN_GROUP:
        # COLMAP places cameras far better than this quick check (it often uses
        # the whole room), so every photo goes in; the check is only advice.
        run.warnings.extend(analysis.recommendations)
    elif not analysis.suitable_for_photogrammetry:
        if not opt.force_analysis:
            raise CaptureQualityError(
                "The photo set is not suitable for photogrammetry: "
                + " ".join(analysis.recommendations),
                hint="Improve the capture, or pass force_analysis=true to try anyway.",
            )
        run.warnings.extend(analysis.recommendations)

    quality = opt.quality or preset.quality or profile_values(run.settings).photo_quality
    # Objects and characters: COLMAP only places the cameras (fast, no dense stage)
    # and the shape is carved from the silhouettes, which also works on plain,
    # white or glossy surfaces. Open scenes keep classic dense reconstruction.
    silhouette = scene in ("object", "character")

    def photogrammetry(classic: bool) -> PhotogrammetryResult:
        return run_photogrammetry(
            src,
            run.out / "reconstruction",
            run.settings,
            PhotogrammetryOptions(
                quality=quality,  # type: ignore[arg-type]
                matcher=preset.matcher,
                dense="auto" if classic else "off",
                use_gpu=wants_gpu(run.settings),
                build_mesh=classic,
                poisson_depth=preset.poisson_depth,
                density_quantile=preset.density_quantile,
            ),
            overwrite=opt.overwrite or not classic,
        )

    photo = run.step("photogrammetry", lambda: photogrammetry(classic=not silhouette))
    mesh_path: str | None = photo.mesh_path
    upright = False
    if silhouette:
        from reverse3d.reconstruction.silhouette import visual_hull
        from reverse3d.vision.segmenter import ObjectSegmenter

        def load_segmenter() -> ObjectSegmenter | None:
            # The AI segmenter keeps dark caps, handles and thin parts; without it
            # (no AI extra, or offline before the first download) GrabCut is used.
            try:
                return ObjectSegmenter(run.settings, device=torch_device(run.settings))
            except Reverse3DError as exc:
                logger.info("AI segmentation unavailable (%s); using GrabCut", exc.message)
                run.warnings.append(
                    "AI segmentation unavailable: thin or dark parts may be missing."
                )
                return None

        try:
            hull = run.step(
                "silhouette",
                lambda: visual_hull(
                    Path(photo.images_dir or ""),
                    Path(photo.sparse_text_dir or ""),
                    run.out / "reconstruction" / "object_hull.ply",
                    segmenter=load_segmenter,
                ),
            )
        except ProcessingError as exc:
            logger.info("Silhouette carving failed (%s); using dense reconstruction", exc.message)
            run.warnings.append(
                f"Silhouette carving failed ({exc.message}); used dense reconstruction."
            )
            photo = run.step("photogrammetry", lambda: photogrammetry(classic=True))
            mesh_path = photo.mesh_path
        else:
            run.details["silhouette"] = hull.to_dict()
            mesh_path = hull.mesh_path
            upright = True
    if photo.registered_images >= 0.9 * max(photo.total_images, 1):
        # COLMAP placed (almost) every photo: the quick pre-check's capture advice
        # would only confuse.
        run.warnings[:] = [w for w in run.warnings if w not in analysis.recommendations]
    run.warnings.extend(photo.warnings)
    run.details["photogrammetry"] = photo.to_dict()
    run.outputs["point_cloud"] = photo.dense_points_path or photo.sparse_points_path
    if not mesh_path:
        raise ProcessingError(
            "Point cloud created but no mesh could be built.",
            hint='Install the meshing extra: pip install "reverse3d[recon]" (Python ≤ 3.12).',
        )
    model = run.out / "model.stl"
    _clean(
        run,
        Path(mesh_path),
        model,
        scale=opt.scale,
        overwrite=opt.overwrite,
        fill_holes=preset.fill_holes,
        align="none" if upright else "obb",
        keep_largest=preset.keep_largest,
        repair=opt.repair,
    )
    if upright:
        try:
            _color_outputs(run, Path(mesh_path), model, opt.scale)
        except Exception as exc:  # colour is a bonus: never fail the model for it
            logger.info("Colour transfer skipped: %s", exc)
    return _measure(run, model)


def _plan_to_part(run: _Run, src: Path, opt: RecipeOptions) -> MeshStats:
    from reverse3d.vision.mask_extractor import extract_mask, silhouette_to_mesh

    mask = run.step(
        "mask",
        lambda: extract_mask(
            src,
            run.out / "mask",
            run.settings,
            method=opt.mask_method,
            units_per_pixel=opt.units_per_pixel,
            overwrite=opt.overwrite,
        ),
    )
    run.warnings.extend(mask.warnings)
    run.outputs["contours_svg"] = mask.svg_path
    if opt.units_per_pixel is None:
        run.warnings.append("No units_per_pixel given: sizes are in pixels.")
    raw = run.out / "extruded.stl"
    run.step(
        "extrude",
        lambda: silhouette_to_mesh(
            mask.contours_path, raw, run.settings, thickness=opt.thickness, overwrite=opt.overwrite
        ),
    )
    model = run.out / "model.stl"
    _clean(run, raw, model, scale=opt.scale, overwrite=opt.overwrite)
    return _measure(run, model)


def _scan_to_cad(run: _Run, src: Path, opt: RecipeOptions) -> MeshStats:
    from reverse3d.reverse_eng.geometry_fit import fit_primitives_file
    from reverse3d.reverse_eng.parametric import export_parametric

    model = run.out / "model.stl"
    _clean(run, src, model, scale=opt.scale, overwrite=opt.overwrite, repair=opt.repair)
    stats = _measure(run, model)
    samples = profile_values(run.settings).fit_samples
    fit = run.step("fit", lambda: fit_primitives_file(model, run.settings, sample_count=samples))
    run.details["primitives"] = fit.to_dict()
    run.details["primitive_summary"] = [
        {
            "kind": p.kind,
            **{k: v for k, v in p.params.items() if k in ("radius", "height", "normal")},
        }
        for p in fit.primitives
    ]
    if fit.primitives:
        run.outputs["cad_openscad"] = str(
            export_parametric(fit, run.out / "cad.scad", overwrite=opt.overwrite)
        )
        run.outputs["cad_cadquery"] = str(
            export_parametric(fit, run.out / "cad.py", overwrite=opt.overwrite)
        )
    else:
        run.warnings.append("No primitives detected; no CAD code generated.")
    return stats


# A relief mesh has one vertex per sampled pixel: keep phone photos (12+ MP) light.
_RELIEF_MAX_VERTICES = 500_000


def _relief_stride(src: Path, minimum: int) -> int:
    """Pixel step so the relief stays under ``_RELIEF_MAX_VERTICES`` vertices."""
    from PIL import Image

    with Image.open(src) as image:
        width, height = image.size
    return max(minimum, math.ceil(math.sqrt(width * height / _RELIEF_MAX_VERTICES)))


def _photo_to_relief(run: _Run, src: Path, opt: RecipeOptions) -> MeshStats:
    from reverse3d.reconstruction.depth_estimation import estimate_depth

    stride = _relief_stride(src, profile_values(run.settings).depth_stride)

    depth = run.step(
        "depth",
        lambda: estimate_depth(
            src,
            run.out / "depth",
            run.settings,
            device=torch_device(run.settings),
            stride=stride,
            mask_path=opt.mask_path,
            overwrite=opt.overwrite,
        ),
    )
    run.details["depth"] = depth.to_dict()
    model = run.out / "model.stl"
    if opt.mask_path:
        # The object is isolated: build a closed solid standing on its surface.
        import numpy as np

        from reverse3d.imaging import load_image
        from reverse3d.mesh_io import save_mesh
        from reverse3d.reconstruction.depth_estimation import object_solid

        disparity = np.load(depth.depth_npy_path, allow_pickle=False)
        mask = load_image(opt.mask_path, run.settings, flags=0)
        solid = run.step("solid", lambda: object_solid(disparity, mask, stride=stride))
        solid_path = save_mesh(solid, run.out / "depth" / "object_solid.ply", overwrite=True)
        run.warnings.append(
            "Single-photo model: the hidden side is flat and the height is estimated, "
            "so proportions are approximate."
        )
        _clean(
            run,
            Path(solid_path),
            model,
            scale=opt.scale,
            overwrite=opt.overwrite,
            fill_holes=True,
            align="obb",
            keep_largest=True,
        )
        return _measure(run, model)
    run.warnings.append("Monocular depth is relative: shape is approximate, not metric.")
    if depth.mesh_path is None:  # pragma: no cover - make_mesh is always on here
        raise ProcessingError("Depth estimation produced no mesh.")
    _clean(
        run,
        Path(depth.mesh_path),
        model,
        scale=opt.scale,
        overwrite=opt.overwrite,
        fill_holes=False,
        align="none",
        keep_largest=True,
    )
    return _measure(run, model)


def _mesh_report(run: _Run, src: Path, opt: RecipeOptions) -> MeshStats:
    run.outputs["model"] = str(src)
    return _measure(run, src)


_RECIPES: dict[str, Callable[[_Run, Path, RecipeOptions], MeshStats]] = {
    "photos_to_model": _photos_to_model,
    "plan_to_part": _plan_to_part,
    "scan_to_cad": _scan_to_cad,
    "photo_to_relief": _photo_to_relief,
    "mesh_report": _mesh_report,
}


def run_recipe(
    recipe: str,
    input_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    options: RecipeOptions | None = None,
) -> dict[str, Any]:
    """Run a recipe and return a compact summary (full report in report.json).

    Raises:
        InputValidationError: unknown recipe or invalid options.
        Reverse3DError: any step failure (the error names the failing step).
    """
    if recipe == "auto":
        from reverse3d.autopilot import auto_run

        return auto_run(input_path, output_dir, settings, options)
    if recipe not in _RECIPES:
        raise InputValidationError(
            f"Unknown recipe '{recipe}'.", hint=f"Use one of: {', '.join(RECIPES)}"
        )
    opt = options or RecipeOptions()
    if not 0 < opt.thickness <= 1e6:
        raise InputValidationError("thickness must be in (0, 1e6].")
    if not 1e-9 < opt.scale < 1e9:
        raise InputValidationError("scale must be a positive factor.")

    out = prepare_output_dir(output_dir)
    run = _Run(recipe=recipe, settings=settings, out=out)
    start = time.perf_counter()
    source = Path(input_path)
    if recipe != "scan_to_cad" and recipe != "mesh_report":
        from reverse3d.imaging import prepare_heic_input

        source = prepare_heic_input(source, out)  # iPhone HEIC photos -> JPEG
    stats = _RECIPES[recipe](run, source, opt)
    if opt.render:
        _render(run, Path(run.outputs.get("model_color") or run.outputs["model"]), opt.overwrite)

    summary: dict[str, Any] = {
        "recipe": recipe,
        "outputs": run.outputs,
        "dimensions": _summary_dims(stats),
        "device": "gpu" if wants_gpu(settings) else "cpu",
        "profile": settings.profile,
        "seconds": round(time.perf_counter() - start, 1),
        "steps": [f"{s['step']}:{s['s']}s" for s in run.steps],
        "warnings": run.warnings,
    }
    if "scene" in run.details:
        summary["scene"] = run.details["scene"]
    if "primitive_summary" in run.details:
        summary["primitives"] = run.details["primitive_summary"]

    report_path = out / "report.json"
    if report_path.exists() and not opt.overwrite:
        report_path = out / f"report_{int(time.time())}.json"
    report_path.write_text(
        json.dumps({**summary, "details": run.details}, indent=2, default=str), encoding="utf-8"
    )
    summary["outputs"]["report"] = str(report_path)
    logger.info("Recipe %s finished in %.1fs", recipe, summary["seconds"])
    return summary
