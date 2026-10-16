"""Tool definitions and a sandboxed dispatcher for AI agents.

One registry feeds every integration:

* :func:`tool_definitions` exports JSON-Schema tool specs for Anthropic (Claude
  tool use), OpenAI-compatible function calling and Google Gemini.
* :func:`execute_tool` validates the arguments an agent produced and runs the
  tool. It is used by the MCP server, by ``reverse3d agent call`` and by
  in-process agent loops.

Security controls (OWASP LLM05/LLM06/LLM10, Agentic ASI02):

* Arguments are parsed with strict Pydantic models (``extra="forbid"``, bounds
  on every number, enums for every mode). Agent output is untrusted input.
* Every path is resolved inside ``Settings.workspace``; traversal and symlink
  escapes are rejected. Existing files are never overwritten unless the agent
  explicitly sets ``overwrite: true``.
* Tools only read/write files and launch COLMAP/Blender with fixed argument
  lists; there is no generic shell, eval or network tool.
* Errors are returned as structured ``{"ok": false, "error": {...}}`` objects;
  unexpected exceptions are logged server-side and reported generically.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from reverse3d.config import Settings
from reverse3d.errors import Reverse3DError
from reverse3d.security import resolve_in_workspace

logger = logging.getLogger(__name__)

SchemaDialect = Literal["anthropic", "openai", "gemini"]


_STR_MAX = 4096


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid", str_max_length=_STR_MAX)


_OVERWRITE = Field(False, description="Replace existing output files. Default false.")
_ALL_KINDS: tuple[Literal["plane", "cylinder", "sphere"], ...] = ("plane", "cylinder", "sphere")


# ─── argument models ──────────────────────────────────────────────────────────


class EnvironmentArgs(_Args):
    pass


class AnalyzeImagesArgs(_Args):
    image_dir: str = Field(description="Folder (inside the workspace) with the capture photos.")
    detector: Literal["sift", "orb"] = Field("sift", description="Feature detector.")
    max_pairs: int = Field(60, ge=1, le=1000, description="Max consecutive pairs to match.")


class MatchImagesArgs(_Args):
    image_a: str = Field(description="First image path.")
    image_b: str = Field(description="Second image path.")
    detector: Literal["sift", "orb"] = "sift"
    ratio: float = Field(0.75, ge=0.5, le=0.95, description="Lowe ratio test threshold.")
    visualization_path: str | None = Field(
        None, description="Optional .png/.jpg to save a drawing of the inlier matches."
    )
    overwrite: bool = _OVERWRITE


class ExtractMaskArgs(_Args):
    image_path: str = Field(description="Photo, technical plan, stencil or RGBA image.")
    output_dir: str = Field(
        description="Folder for <stem>_mask.png, _contours.json, _contours.svg."
    )
    method: Literal["otsu", "adaptive", "canny", "alpha"] = Field(
        "otsu",
        description="otsu: plain background; adaptive: uneven light / line drawings; "
        "canny: edge-based; alpha: use PNG transparency.",
    )
    invert: bool | None = Field(None, description="Force inversion; null = auto-detect.")
    blur: int = Field(5, ge=0, le=51)
    min_area_ratio: float = Field(
        0.001, ge=0, lt=1, description="Drop regions smaller than this image fraction."
    )
    simplify_ratio: float = Field(
        0.002, ge=0, le=0.1, description="Contour simplification tolerance."
    )
    units_per_pixel: float | None = Field(
        None, gt=0, description="Real-world scale, e.g. mm per pixel from a known dimension."
    )
    overwrite: bool = _OVERWRITE


class SilhouetteToMeshArgs(_Args):
    contours_path: str = Field(description="*_contours.json produced by reverse3d_extract_mask.")
    output_path: str = Field(description="Output mesh (.stl/.obj/.ply/.glb/.gltf/.off).")
    thickness: float = Field(10.0, gt=0, le=1e6, description="Extrusion height in output units.")
    units_per_pixel: float | None = Field(
        None, gt=0, description="Overrides the scale stored in the JSON."
    )
    overwrite: bool = _OVERWRITE


class PhotogrammetryArgs(_Args):
    image_dir: str = Field(description="Folder with >=3 overlapping photos of the object.")
    output_dir: str = Field(description="Folder for the COLMAP workspace and outputs.")
    quality: Literal["low", "medium", "high"] | None = Field(
        None, description="null = from the performance profile."
    )
    matcher: Literal["auto", "exhaustive", "sequential"] = "auto"
    dense: Literal["auto", "on", "off"] = Field(
        "auto", description="Dense MVS needs a CUDA build of COLMAP. auto = only if GPU."
    )
    use_gpu: bool | None = Field(None, description="null = from the device setting.")
    camera_model: Literal[
        "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "FULL_OPENCV"
    ] = "SIMPLE_RADIAL"
    build_mesh: bool = Field(True, description="Poisson-mesh the cloud (needs the [recon] extra).")
    overwrite: bool = _OVERWRITE


class EstimateDepthArgs(_Args):
    image_path: str = Field(description="Single image to estimate depth from.")
    output_dir: str
    model: Literal[
        "depth-anything/Depth-Anything-V2-Small-hf",
        "depth-anything/Depth-Anything-V2-Base-hf",
        "depth-anything/Depth-Anything-V2-Large-hf",
    ] = Field("depth-anything/Depth-Anything-V2-Small-hf", description="Allowlisted model id.")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto", description="auto = from the device setting."
    )
    make_mesh: bool = True
    fov_deg: float = Field(
        60.0, ge=10, le=170, description="Horizontal field of view of the camera."
    )
    stride: int | None = Field(
        None, ge=1, le=32, description="Pixel step for the mesh grid; null = profile."
    )
    mask_path: str | None = Field(None, description="Optional mask PNG to mesh only the object.")
    overwrite: bool = _OVERWRITE


class PointCloudMeshArgs(_Args):
    cloud_path: str = Field(description="Point cloud (.ply/.xyz/.pcd).")
    output_path: str = Field(description="Output mesh path.")
    depth: int = Field(9, ge=5, le=12, description="Poisson octree depth (detail).")
    overwrite: bool = _OVERWRITE


class CleanMeshArgs(_Args):
    input_path: str = Field(description="Input mesh.")
    output_path: str = Field(description="Output mesh (.stl/.obj/.ply/.glb/.gltf/.off).")
    keep_largest_only: bool = False
    min_component_ratio: float = Field(0.01, ge=0, lt=1)
    fill_holes: bool = True
    max_hole_edges: int = Field(200, ge=3, le=100_000)
    fix_normals: bool = True
    smooth_iterations: int = Field(0, ge=0, le=200)
    target_faces: int | None = Field(None, ge=4, description="Decimate to this many faces.")
    align: Literal["none", "center", "obb", "pca"] = "none"
    ground: bool = Field(False, description="Translate so min Z = 0.")
    scale: float = Field(1.0, gt=1e-9, lt=1e9, description="Uniform scale (e.g. 1000 for m→mm).")
    overwrite: bool = _OVERWRITE


class MeasureMeshArgs(_Args):
    input_path: str = Field(description="Mesh to measure.")


class FitPrimitivesArgs(_Args):
    input_path: str = Field(description="Mesh or point cloud.")
    kinds: list[Literal["plane", "cylinder", "sphere"]] = Field(
        default_factory=lambda: list(_ALL_KINDS), min_length=1, max_length=3
    )
    sample_count: int | None = Field(None, ge=100, le=500_000, description="null = profile.")
    threshold: float | None = Field(
        None, gt=0, description="Inlier distance; null = 1% of bbox diagonal."
    )
    max_primitives: int = Field(8, ge=1, le=50)
    min_inlier_ratio: float = Field(0.05, gt=0, lt=1)
    parametric_output: str | None = Field(
        None, description="Optional .scad (OpenSCAD) or .py (CadQuery) file to generate."
    )
    overwrite: bool = _OVERWRITE


class RenderArgs(_Args):
    mesh_path: str = Field(description="Mesh to render.")
    output_path: str = Field(description="Output .png (turntables add _0000 suffixes).")
    backend: Literal["auto", "blender", "pyvista"] = "auto"
    width: int | None = Field(None, ge=64, le=8192, description="null = profile.")
    height: int | None = Field(None, ge=64, le=8192, description="null = profile.")
    engine: Literal["cycles", "eevee", "workbench"] = "cycles"
    device: Literal["auto", "cpu", "gpu"] = Field("auto", description="auto = device setting.")
    samples: int | None = Field(None, ge=1, le=4096, description="null = profile.")
    color: str = Field("#9aa4b2", pattern=r"^#[0-9a-fA-F]{6}$")
    metallic: float = Field(0.0, ge=0, le=1)
    roughness: float = Field(0.45, ge=0, le=1)
    background: str = Field("#f4f5f7", pattern=r"^#[0-9a-fA-F]{6}$")
    transparent: bool = False
    elevation_deg: float = Field(25.0, ge=-89, le=89)
    azimuth_deg: float = Field(35.0, ge=0, le=360)
    turntable_frames: int = Field(0, ge=0, le=360)
    ground: bool = True
    shading: Literal["auto", "smooth", "flat"] = "auto"
    overwrite: bool = _OVERWRITE


class RunRecipeArgs(_Args):
    recipe: Literal[
        "photos_to_model", "plan_to_part", "scan_to_cad", "photo_to_relief", "mesh_report", "auto"
    ] = Field(
        description="auto: detect the input type and pick/retry the right workflow (use it when "
        "unsure). photos_to_model: photo folder→measured 3D model+render. "
        "plan_to_part: plan/stencil image→extruded part. "
        "scan_to_cad: mesh→clean+measure+primitives+OpenSCAD/CadQuery. "
        "photo_to_relief: one photo→2.5D relief. mesh_report: mesh→dimensions+render."
    )
    input_path: str = Field(description="Photo folder (photos_to_model), image, or mesh.")
    output_dir: str = Field(description="Folder for all results (model.stl, render.png...).")
    thickness: float = Field(5.0, gt=0, le=1e6, description="plan_to_part extrusion height.")
    units_per_pixel: float | None = Field(
        None, gt=0, description="plan_to_part scale, e.g. mm per pixel."
    )
    scale: float = Field(1.0, gt=1e-9, lt=1e9, description="Final uniform scale (m→mm: 1000).")
    render: bool = Field(True, description="Render a PNG preview at the end.")
    quality: Literal["low", "medium", "high"] | None = Field(
        None, description="photos_to_model quality; null = profile."
    )
    force_analysis: bool = Field(
        False, description="photos_to_model: continue even if the photo check fails."
    )
    scene: Literal["auto", "object", "character", "architecture", "landscape"] = Field(
        "auto",
        description="Photo sets: auto predicts it. object=part/product, character=person "
        "for games, architecture=building/interior, landscape=terrain/drone flight.",
    )
    overwrite: bool = _OVERWRITE


class ConfigureArgs(_Args):
    device: Literal["auto", "cpu", "gpu"] | None = Field(
        None, description="Compute device preference. auto picks GPU when available."
    )
    profile: Literal["fast", "balanced", "quality"] | None = Field(
        None, description="Speed/quality trade-off for every tool."
    )
    verbosity: Literal["compact", "full"] | None = Field(
        None, description="compact saves tokens; full returns every detail."
    )
    threads: int | None = Field(None, ge=0, le=1024, description="CPU threads; 0 = automatic.")


# ─── handlers ─────────────────────────────────────────────────────────────────


def _p(path: str, settings: Settings) -> Path:
    return resolve_in_workspace(path, settings.workspace)


def _opt(path: str | None, settings: Settings) -> Path | None:
    return _p(path, settings) if path else None


def _environment(_: EnvironmentArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.doctor import environment_report

    return environment_report(s)


def _analyze_images(a: AnalyzeImagesArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.vision.feature_matcher import analyze_image_set

    return analyze_image_set(
        _p(a.image_dir, s), s, detector=a.detector, max_pairs=a.max_pairs
    ).to_dict()


def _match_images(a: MatchImagesArgs, s: Settings) -> dict[str, Any]:
    from dataclasses import asdict

    from reverse3d.vision.feature_matcher import match_pair

    return asdict(
        match_pair(
            _p(a.image_a, s),
            _p(a.image_b, s),
            s,
            detector=a.detector,
            ratio=a.ratio,
            visualization_path=_opt(a.visualization_path, s),
            overwrite=a.overwrite,
        )
    )


def _extract_mask(a: ExtractMaskArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.vision.mask_extractor import extract_mask

    return extract_mask(
        _p(a.image_path, s),
        _p(a.output_dir, s),
        s,
        method=a.method,
        invert=a.invert,
        blur=a.blur,
        min_area_ratio=a.min_area_ratio,
        simplify_ratio=a.simplify_ratio,
        units_per_pixel=a.units_per_pixel,
        overwrite=a.overwrite,
    ).to_dict()


def _silhouette_to_mesh(a: SilhouetteToMeshArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.vision.mask_extractor import silhouette_to_mesh

    return silhouette_to_mesh(
        _p(a.contours_path, s),
        _p(a.output_path, s),
        s,
        thickness=a.thickness,
        units_per_pixel=a.units_per_pixel,
        overwrite=a.overwrite,
    )


def _photogrammetry(a: PhotogrammetryArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.hardware import wants_gpu
    from reverse3d.profiles import profile_values
    from reverse3d.reconstruction.photogrammetry import PhotogrammetryOptions, run_photogrammetry

    options = PhotogrammetryOptions(
        quality=a.quality or profile_values(s).photo_quality,  # type: ignore[arg-type]
        matcher=a.matcher,
        dense=a.dense,
        use_gpu=wants_gpu(s) if a.use_gpu is None else a.use_gpu,
        camera_model=a.camera_model,
        build_mesh=a.build_mesh,
    )
    return run_photogrammetry(
        _p(a.image_dir, s), _p(a.output_dir, s), s, options, overwrite=a.overwrite
    ).to_dict()


def _estimate_depth(a: EstimateDepthArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.hardware import torch_device
    from reverse3d.profiles import profile_values
    from reverse3d.reconstruction.depth_estimation import estimate_depth

    return estimate_depth(
        _p(a.image_path, s),
        _p(a.output_dir, s),
        s,
        model=a.model,
        device=torch_device(s) if a.device == "auto" else a.device,
        make_mesh=a.make_mesh,
        fov_deg=a.fov_deg,
        stride=a.stride or profile_values(s).depth_stride,
        mask_path=_opt(a.mask_path, s),
        overwrite=a.overwrite,
    ).to_dict()


def _point_cloud_mesh(a: PointCloudMeshArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.reconstruction.pointcloud import poisson_mesh

    return poisson_mesh(
        _p(a.cloud_path, s), _p(a.output_path, s), s, depth=a.depth, overwrite=a.overwrite
    ).to_dict()


def _clean_mesh(a: CleanMeshArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.reverse_eng.mesh_cleaner import CleanOptions, clean_mesh_file

    options = CleanOptions(
        keep_largest_only=a.keep_largest_only,
        min_component_ratio=a.min_component_ratio,
        fix_normals=a.fix_normals,
        fill_holes=a.fill_holes,
        max_hole_edges=a.max_hole_edges,
        smooth_iterations=a.smooth_iterations,
        target_faces=a.target_faces,
        align=a.align,
        ground=a.ground,
        scale=a.scale,
    )
    return clean_mesh_file(
        _p(a.input_path, s), _p(a.output_path, s), s, options, overwrite=a.overwrite
    ).to_dict()


def _measure_mesh(a: MeasureMeshArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.reverse_eng.measure import measure_mesh_file

    return measure_mesh_file(_p(a.input_path, s), s).to_dict()


def _fit_primitives(a: FitPrimitivesArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.profiles import profile_values
    from reverse3d.reverse_eng.geometry_fit import fit_primitives_file
    from reverse3d.reverse_eng.parametric import export_parametric

    report = fit_primitives_file(
        _p(a.input_path, s),
        s,
        kinds=tuple(dict.fromkeys(a.kinds)),
        sample_count=a.sample_count or profile_values(s).fit_samples,
        threshold=a.threshold,
        max_primitives=a.max_primitives,
        min_inlier_ratio=a.min_inlier_ratio,
    )
    result = report.to_dict()
    if a.parametric_output and report.primitives:
        result["parametric_path"] = str(
            export_parametric(report, _p(a.parametric_output, s), overwrite=a.overwrite)
        )
    return result


def _render(a: RenderArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.hardware import wants_gpu
    from reverse3d.profiles import profile_values
    from reverse3d.rendering.blender_render import RenderOptions, render_mesh

    prof = profile_values(s)
    device = ("gpu" if wants_gpu(s) else "cpu") if a.device == "auto" else a.device
    options = RenderOptions(
        width=a.width or prof.render_width,
        height=a.height or prof.render_height,
        engine=a.engine,
        device=device,
        samples=a.samples or prof.render_samples,
        color=a.color,
        metallic=a.metallic,
        roughness=a.roughness,
        background=a.background,
        transparent=a.transparent,
        elevation_deg=a.elevation_deg,
        azimuth_deg=a.azimuth_deg,
        turntable_frames=a.turntable_frames,
        ground=a.ground,
        shading=a.shading,
    )
    return render_mesh(
        _p(a.mesh_path, s),
        _p(a.output_path, s),
        s,
        options,
        backend=a.backend,
        overwrite=a.overwrite,
    ).to_dict()


def _run_recipe(a: RunRecipeArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.recipes import RecipeOptions, run_recipe

    return run_recipe(
        a.recipe,
        _p(a.input_path, s),
        _p(a.output_dir, s),
        s,
        RecipeOptions(
            thickness=a.thickness,
            units_per_pixel=a.units_per_pixel,
            scale=a.scale,
            render=a.render,
            quality=a.quality,
            force_analysis=a.force_analysis,
            scene=a.scene,
            overwrite=a.overwrite,
        ),
    )


def _configure(a: ConfigureArgs, s: Settings) -> dict[str, Any]:
    from reverse3d.hardware import hardware_report
    from reverse3d.userconfig import AGENT_WRITABLE, config_path, env_overrides, update_config

    updates = {k: v for k, v in a.model_dump().items() if v is not None}
    if updates:
        # Agents may only touch harmless keys; never the workspace, limits or binaries.
        update_config(updates, allowed=AGENT_WRITABLE)
    effective = Settings.load().with_workspace(s.workspace)
    overridden = [k for k in env_overrides() if k in updates]
    result: dict[str, Any] = {
        "changed": updates,
        "effective": {
            "device": effective.device,
            "profile": effective.profile,
            "verbosity": effective.verbosity,
            "threads": effective.threads,
            "toolset": effective.toolset,
        },
        "hardware": hardware_report(effective),
        "config_file": str(config_path()),
    }
    if overridden:
        result["warning"] = (
            f"{overridden} are fixed by environment variables in the MCP/client config; "
            "remove them there for this change to take effect."
        )
    return result


# ─── registry ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[_Args]
    handler: Callable[[Any, Settings], dict[str, Any]]


TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "reverse3d_environment_check",
            "Report which reverse3d capabilities are available on this machine (COLMAP, Blender, "
            "Open3D, depth models, PyVista, GPU), the sandbox workspace and resource limits. Call "
            "this first to plan a pipeline.",
            EnvironmentArgs,
            _environment,
        ),
        ToolSpec(
            "reverse3d_run_recipe",
            "PREFERRED entry point: run a complete 3D workflow in ONE call and get a short summary "
            "(output files, dimensions, warnings); full details are saved to report.json. Use the "
            "individual tools only for custom steps.",
            RunRecipeArgs,
            _run_recipe,
        ),
        ToolSpec(
            "reverse3d_configure",
            "View or change persistent settings: device (auto/cpu/gpu), profile "
            "(fast/balanced/quality), verbosity (compact/full), CPU threads. Call with no "
            "arguments to see the current settings and detected GPU.",
            ConfigureArgs,
            _configure,
        ),
        ToolSpec(
            "reverse3d_analyze_images",
            "Assess a folder of multi-view photos before photogrammetry: sharpness, exposure, "
            "texture, keypoints and feature overlap between consecutive images. Returns per-image "
            "issues, weak links and recommendations.",
            AnalyzeImagesArgs,
            _analyze_images,
        ),
        ToolSpec(
            "reverse3d_match_images",
            "Match features between two images (ratio test + RANSAC) and report how many "
            "geometrically consistent correspondences they share. Optionally draws the matches.",
            MatchImagesArgs,
            _match_images,
        ),
        ToolSpec(
            "reverse3d_extract_mask",
            "Extract a binary mask and vector contours (outer outline + holes) from a photo, "
            "technical plan, stencil or RGBA image. Writes a mask PNG, contours JSON and SVG.",
            ExtractMaskArgs,
            _extract_mask,
        ),
        ToolSpec(
            "reverse3d_silhouette_to_mesh",
            "Extrude the contours JSON from reverse3d_extract_mask into a 3D solid (plan to part). "
            "Use units_per_pixel for real-world scale.",
            SilhouetteToMeshArgs,
            _silhouette_to_mesh,
        ),
        ToolSpec(
            "reverse3d_photogrammetry",
            "Reconstruct a 3D point cloud (and optionally a mesh) from overlapping photos using "
            "COLMAP. Long-running (minutes to hours). Run reverse3d_analyze_images first.",
            PhotogrammetryArgs,
            _photogrammetry,
        ),
        ToolSpec(
            "reverse3d_estimate_depth",
            "Estimate relative depth from a single image with an allowlisted Depth Anything V2 "
            "model and optionally build a 2.5D mesh. Shape is approximate, not metric.",
            EstimateDepthArgs,
            _estimate_depth,
        ),
        ToolSpec(
            "reverse3d_mesh_from_point_cloud",
            "Build a surface mesh from a point cloud with screened Poisson reconstruction "
            "(requires Open3D).",
            PointCloudMeshArgs,
            _point_cloud_mesh,
        ),
        ToolSpec(
            "reverse3d_clean_mesh",
            "Repair a mesh: remove degenerate faces and debris, fix normals, fill holes, smooth, "
            "decimate, align (OBB/PCA), place on ground and rescale. Returns before/after stats.",
            CleanMeshArgs,
            _clean_mesh,
        ),
        ToolSpec(
            "reverse3d_measure_mesh",
            "Measure a mesh: bounding boxes (axis-aligned and minimum oriented), surface area, "
            "volume, watertightness, components and open boundary edges.",
            MeasureMeshArgs,
            _measure_mesh,
        ),
        ToolSpec(
            "reverse3d_fit_primitives",
            "Detect planes, cylinders and spheres (RANSAC) on a mesh or point cloud and report "
            "their parameters (radii, axes, normals). Optionally writes OpenSCAD/CadQuery code.",
            FitPrimitivesArgs,
            _fit_primitives,
        ),
        ToolSpec(
            "reverse3d_render",
            "Render a mesh to PNG headlessly with Blender (Cycles/Eevee/Workbench) or PyVista, "
            "with studio lighting, material color and optional turntable frames.",
            RenderArgs,
            _render,
        ),
    )
}


#: Minimal tool set for ``toolset=lean``: recipes cover the common workflows, so
#: agents only pay the schema tokens of 5 tools instead of all of them.
LEAN_TOOLS = (
    "reverse3d_environment_check",
    "reverse3d_run_recipe",
    "reverse3d_configure",
    "reverse3d_measure_mesh",
    "reverse3d_render",
)


# ─── schema export ────────────────────────────────────────────────────────────


# Gemini function declarations accept an OpenAPI 3 subset of JSON Schema.
_GEMINI_KEYS = frozenset(
    {
        "type",
        "format",
        "description",
        "nullable",
        "enum",
        "properties",
        "required",
        "items",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "anyOf",
    }
)


def _clean_schema(node: Any, dialect: SchemaDialect, *, in_properties: bool = False) -> Any:
    """Strip noise (titles) and adapt the schema to Gemini's OpenAPI subset."""
    if isinstance(node, dict):
        if in_properties:  # keys are property names, not schema keywords
            return {k: _clean_schema(v, dialect) for k, v in node.items()}
        node = {
            k: _clean_schema(v, dialect, in_properties=(k == "properties"))
            for k, v in node.items()
            # The global 4096-char string cap is enforced server-side; repeating it
            # on every field only costs schema tokens.
            if k != "title" and not (k == "maxLength" and v == _STR_MAX)
        }
        if dialect == "gemini":
            if "exclusiveMinimum" in node:
                node.setdefault("minimum", node["exclusiveMinimum"])
            if "exclusiveMaximum" in node:
                node.setdefault("maximum", node["exclusiveMaximum"])
            if "anyOf" in node:
                variants = [v for v in node["anyOf"] if v.get("type") != "null"]
                if len(variants) == 1 and len(node["anyOf"]) == 2:
                    merged = {k: v for k, v in node.items() if k != "anyOf"}
                    merged.update(variants[0])
                    merged["nullable"] = True
                    node = merged
            node = {k: v for k, v in node.items() if k in _GEMINI_KEYS}
        return node
    if isinstance(node, list):
        return [_clean_schema(v, dialect) for v in node]
    return node


def input_schema(spec: ToolSpec, dialect: SchemaDialect = "anthropic") -> dict[str, Any]:
    """JSON Schema of a tool's arguments."""
    schema = copy.deepcopy(spec.model.model_json_schema())
    schema.setdefault("properties", {})
    schema["additionalProperties"] = False
    cleaned: dict[str, Any] = _clean_schema(schema, dialect)
    if dialect == "gemini":
        cleaned.pop("additionalProperties", None)
    return cleaned


def visible_tools(toolset: str = "full") -> list[ToolSpec]:
    """Tools exposed to agents for a toolset (``full`` or ``lean``)."""
    if toolset == "lean":
        return [TOOLS[name] for name in LEAN_TOOLS]
    return list(TOOLS.values())


def tool_definitions(
    dialect: SchemaDialect = "anthropic", toolset: str = "full"
) -> list[dict[str, Any]]:
    """Return the tool list in the format expected by each provider.

    * ``anthropic``: ``[{"name", "description", "input_schema"}]``
    * ``openai``: ``[{"type": "function", "function": {"name", "description", "parameters"}}]``
    * ``gemini``: ``[{"name", "description", "parameters"}]`` (function declarations)
    """
    tools: list[dict[str, Any]] = []
    for spec in visible_tools(toolset):
        schema = input_schema(spec, dialect)
        if dialect == "anthropic":
            tools.append(
                {"name": spec.name, "description": spec.description, "input_schema": schema}
            )
        elif dialect == "openai":
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": schema,
                    },
                }
            )
        elif dialect == "gemini":
            tools.append({"name": spec.name, "description": spec.description, "parameters": schema})
        else:
            raise ValueError(f"Unknown dialect: {dialect}")
    return tools


# ─── dispatcher ───────────────────────────────────────────────────────────────


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors(include_url=False, include_input=False)[:10]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg')}")
    return "; ".join(parts)


def execute_tool(name: str, arguments: dict[str, Any] | None, settings: Settings) -> dict[str, Any]:
    """Validate ``arguments`` and run tool ``name`` inside the sandbox.

    Always returns a JSON-serialisable dict: ``{"ok": true, "tool", "result"}``
    or ``{"ok": false, "tool", "error": {"code", "message", ...}}``. Never raises
    for tool-level failures, so agent loops can feed the error back to the model.
    With ``settings.verbosity == "compact"`` results are shrunk for token economy
    (see :mod:`reverse3d.agent.compact`).
    """
    spec = TOOLS.get(name)
    if spec is None:
        return {
            "ok": False,
            "tool": name,
            "error": {
                "code": "unknown_tool",
                "message": f"Unknown tool. Available: {', '.join(TOOLS)}",
            },
        }
    if arguments is not None and not isinstance(arguments, dict):
        return {
            "ok": False,
            "tool": name,
            "error": {"code": "invalid_input", "message": "arguments must be an object"},
        }
    try:
        args = spec.model.model_validate(arguments or {})
    except ValidationError as exc:
        return {
            "ok": False,
            "tool": name,
            "error": {"code": "invalid_input", "message": _validation_message(exc)},
        }

    logger.info("Tool call: %s", name)
    try:
        result = spec.handler(args, settings)
    except Reverse3DError as exc:
        logger.warning("Tool %s failed: %s", name, exc.code)
        return {"ok": False, "tool": name, "error": exc.to_dict()}
    except Exception:
        logger.exception("Unexpected error in tool %s", name)
        return {
            "ok": False,
            "tool": name,
            "error": {"code": "internal_error", "message": "Unexpected error; see reverse3d logs."},
        }
    if settings.verbosity == "compact":
        from reverse3d.agent.compact import compact_result

        result = compact_result(name, result, settings.workspace)
    return {"ok": True, "tool": name, "result": result}
