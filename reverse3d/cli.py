"""Command-line interface: ``reverse3d <group> <command> [options]``.

Every command prints a JSON document on stdout (pretty in a terminal, compact
with ``--json``) and logs to stderr, so both humans and shell-driven AI agents
(Claude Code, Gemini CLI, OpenCode...) can consume it. Exit codes:

====  ==============================================
0     success
1     unexpected error / tool reported failure
2     invalid input or security violation
3     missing dependency (extra or external binary)
4     external tool (COLMAP/Blender) failed
====  ==============================================

Direct commands act on the paths you pass (you are the trusted operator).
``reverse3d agent call`` and ``reverse3d agent mcp`` enforce the workspace
sandbox, because there the arguments come from a model.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from reverse3d._version import __version__
from reverse3d.config import Settings
from reverse3d.errors import (
    DependencyMissingError,
    ExternalToolError,
    InputValidationError,
    Reverse3DError,
    SecurityViolationError,
)
from reverse3d.hardware import apply_threads
from reverse3d.logging_utils import configure_logging

_err = Console(stderr=True)

app = typer.Typer(
    name="reverse3d",
    help="AI-agent-ready toolkit for 3D reverse engineering — by Voxel Enterprise (voxel.lat).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
vision_app = typer.Typer(
    help="2D analysis: image quality, matching, masks and silhouettes.", no_args_is_help=True
)
recon_app = typer.Typer(
    help="3D reconstruction: photogrammetry, monocular depth, meshing.", no_args_is_help=True
)
mesh_app = typer.Typer(
    help="Reverse engineering: clean, measure and fit primitives.", no_args_is_help=True
)
agent_app = typer.Typer(
    help="AI agent integration: tool schemas, sandboxed calls, MCP server.", no_args_is_help=True
)
app.add_typer(vision_app, name="vision")
app.add_typer(recon_app, name="recon")
app.add_typer(mesh_app, name="mesh")
app.add_typer(agent_app, name="agent")


@dataclass
class _State:
    settings: Settings
    json_output: bool
    workspace_override: Path | None = None
    device_override: str | None = None


# Choice types as str-Enums: the stable, version-independent way to declare
# choices in Typer (values are passed to the library as plain strings).
class Backend(str, Enum):
    auto = "auto"
    blender = "blender"
    pyvista = "pyvista"


class Engine(str, Enum):
    cycles = "cycles"
    eevee = "eevee"
    workbench = "workbench"


class Shading(str, Enum):
    auto = "auto"
    smooth = "smooth"
    flat = "flat"


class Detector(str, Enum):
    sift = "sift"
    orb = "orb"


class MaskMethod(str, Enum):
    otsu = "otsu"
    adaptive = "adaptive"
    canny = "canny"
    alpha = "alpha"


class Quality(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class Matcher(str, Enum):
    auto = "auto"
    exhaustive = "exhaustive"
    sequential = "sequential"


class Dense(str, Enum):
    auto = "auto"
    on = "on"
    off = "off"


class Align(str, Enum):
    # Upper-case member names: "center" would shadow str.center().
    NONE = "none"
    CENTER = "center"
    OBB = "obb"
    PCA = "pca"


class DeviceChoice(str, Enum):
    auto = "auto"
    cpu = "cpu"
    gpu = "gpu"


class ProfileChoice(str, Enum):
    fast = "fast"
    balanced = "balanced"
    quality = "quality"


class RecipeChoice(str, Enum):
    photos_to_model = "photos_to_model"
    plan_to_part = "plan_to_part"
    scan_to_cad = "scan_to_cad"
    photo_to_relief = "photo_to_relief"
    mesh_report = "mesh_report"


class ToolsetChoice(str, Enum):
    full = "full"
    lean = "lean"


class ToolFormat(str, Enum):
    anthropic = "anthropic"
    openai = "openai"
    gemini = "gemini"


def _state(ctx: typer.Context) -> _State:
    state = ctx.find_root().obj
    if not isinstance(state, _State):  # pragma: no cover - callback always sets it
        raise typer.Exit(1)
    return state


def _emit(state: _State, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, default=str)
    if state.json_output or not sys.stdout.isatty():
        sys.stdout.write(text + "\n")
    else:
        Console().print_json(text)


def _exit_code(exc: Reverse3DError) -> int:
    if isinstance(exc, (InputValidationError, SecurityViolationError)):
        return 2
    if isinstance(exc, DependencyMissingError):
        return 3
    if isinstance(exc, ExternalToolError):
        return 4
    return 1


def _run(ctx: typer.Context, fn: Callable[[Settings], Any]) -> None:
    state = _state(ctx)
    try:
        result = fn(state.settings)
    except Reverse3DError as exc:
        payload = {"ok": False, "error": exc.to_dict()}
        if state.json_output:
            sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        else:
            _err.print(f"[bold red]Error ({exc.code}):[/] {exc.message}")
            if exc.hint:
                _err.print(f"[yellow]Hint:[/] {exc.hint}")
            if isinstance(exc, ExternalToolError) and exc.stderr_tail:
                _err.print(exc.stderr_tail, markup=False, highlight=False)
        raise typer.Exit(_exit_code(exc)) from None
    data = result.to_dict() if hasattr(result, "to_dict") else result
    _emit(state, {"ok": True, "result": data})


def _version_callback(value: bool) -> None:
    if value:
        from reverse3d._about import AUTHOR, AUTHOR_URL, LICENSE, OFFICIAL_REPOSITORY

        typer.echo(f"reverse3d {__version__} — © {AUTHOR} ({AUTHOR_URL}) · {LICENSE}")
        typer.echo(f"Official source: {OFFICIAL_REPOSITORY}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False, "--json", help="Compact JSON on stdout (for agents/scripts)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging on stderr."),
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Sandbox root for agent commands (env: REVERSE3D_WORKSPACE).",
    ),
    device: DeviceChoice | None = typer.Option(
        None, "--device", "-d", help="Override for this run: auto | cpu | gpu."
    ),
    profile: ProfileChoice | None = typer.Option(
        None, "--profile", "-p", help="Override for this run: fast | balanced | quality."
    ),
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    try:
        settings = Settings.load()
    except Reverse3DError as exc:
        _err.print(f"[bold red]Configuration error:[/] {exc.message}")
        if exc.hint:
            _err.print(f"[yellow]Hint:[/] {exc.hint}")
        raise typer.Exit(2) from None
    if workspace is not None:
        settings = settings.with_workspace(workspace)
    if device is not None:
        settings = settings.with_device(device.value)
    if profile is not None:
        settings = replace(settings, profile=profile.value)
    configure_logging("DEBUG" if verbose else settings.log_level, json_lines=settings.log_json)
    apply_threads(settings)
    ctx.obj = _State(
        settings=settings,
        json_output=json_output,
        workspace_override=workspace,
        device_override=device.value if device else None,
    )


# ─── top-level ────────────────────────────────────────────────────────────────


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Show available capabilities, optional extras, external tools and limits."""
    from reverse3d.doctor import environment_report

    _run(ctx, environment_report)


@app.command()
def render(
    ctx: typer.Context,
    mesh: Path = typer.Argument(..., help="Mesh to render."),
    output: Path = typer.Argument(..., help="Output .png"),
    backend: Backend = typer.Option(Backend.auto),
    width: int | None = typer.Option(None, min=64, max=8192, help="Default: from profile."),
    height: int | None = typer.Option(None, min=64, max=8192, help="Default: from profile."),
    engine: Engine = typer.Option(Engine.cycles),
    samples: int | None = typer.Option(None, min=1, max=4096, help="Default: from profile."),
    color: str = typer.Option("#9aa4b2", help="Base color #RRGGBB."),
    metallic: float = typer.Option(0.0, min=0, max=1),
    roughness: float = typer.Option(0.45, min=0, max=1),
    background: str = typer.Option("#f4f5f7", help="Background #RRGGBB."),
    transparent: bool = typer.Option(False, help="Transparent PNG background."),
    elevation: float = typer.Option(25.0, min=-89, max=89),
    azimuth: float = typer.Option(35.0, min=0, max=360),
    turntable: int = typer.Option(
        0, min=0, max=360, help="Number of turntable frames (0 = still)."
    ),
    ground: bool = typer.Option(True, help="Shadow-catcher ground plane (Cycles)."),
    shading: Shading = typer.Option(Shading.auto),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite outputs."),
) -> None:
    """Render a mesh headlessly with Blender (or PyVista as fallback). GPU: use --device gpu."""
    from reverse3d.hardware import wants_gpu
    from reverse3d.profiles import profile_values
    from reverse3d.rendering.blender_render import RenderOptions, render_mesh

    settings = _state(ctx).settings
    prof = profile_values(settings)
    options = RenderOptions(
        width=width or prof.render_width,
        height=height or prof.render_height,
        engine=engine.value,
        device="gpu" if wants_gpu(settings) else "cpu",
        samples=samples or prof.render_samples,
        color=color,
        metallic=metallic,
        roughness=roughness,
        background=background,
        transparent=transparent,
        elevation_deg=elevation,
        azimuth_deg=azimuth,
        turntable_frames=turntable,
        ground=ground,
        shading=shading.value,
    )
    _run(
        ctx,
        lambda s: render_mesh(
            mesh,
            output,
            s,
            options,
            backend=backend.value,
            overwrite=force,
        ),
    )


# ─── vision ───────────────────────────────────────────────────────────────────


@vision_app.command("analyze")
def vision_analyze(
    ctx: typer.Context,
    image_dir: Path = typer.Argument(..., help="Folder with capture photos."),
    detector: Detector = typer.Option(Detector.sift),
    max_pairs: int = typer.Option(60, min=1, max=1000),
) -> None:
    """Assess photo quality and overlap before photogrammetry."""
    from reverse3d.vision.feature_matcher import analyze_image_set

    _run(
        ctx,
        lambda s: analyze_image_set(
            image_dir,
            s,
            detector=detector.value,
            max_pairs=max_pairs,
        ),
    )


@vision_app.command("match")
def vision_match(
    ctx: typer.Context,
    image_a: Path = typer.Argument(...),
    image_b: Path = typer.Argument(...),
    detector: Detector = typer.Option(Detector.sift),
    ratio: float = typer.Option(0.75, min=0.5, max=0.95),
    viz: Path | None = typer.Option(None, help="Save a PNG/JPG with the inlier matches."),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Match features between two images (ratio test + RANSAC)."""
    from dataclasses import asdict

    from reverse3d.vision.feature_matcher import match_pair

    _run(
        ctx,
        lambda s: asdict(
            match_pair(
                image_a,
                image_b,
                s,
                detector=detector.value,
                ratio=ratio,
                visualization_path=viz,
                overwrite=force,
            )
        ),
    )


@vision_app.command("mask")
def vision_mask(
    ctx: typer.Context,
    image: Path = typer.Argument(..., help="Photo, plan, stencil or RGBA image."),
    output_dir: Path = typer.Argument(...),
    method: MaskMethod = typer.Option(MaskMethod.otsu),
    invert: bool | None = typer.Option(None, "--invert/--no-invert", help="Default: auto-detect."),
    blur: int = typer.Option(5, min=0, max=51),
    min_area: float = typer.Option(0.001, min=0, max=0.99, help="Min region as image fraction."),
    simplify: float = typer.Option(0.002, min=0, max=0.1),
    units_per_pixel: float | None = typer.Option(
        None, min=0, help="Real-world scale (e.g. mm/px)."
    ),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Extract mask + vector contours (JSON/SVG) from an image."""
    from reverse3d.vision.mask_extractor import extract_mask

    _run(
        ctx,
        lambda s: extract_mask(
            image,
            output_dir,
            s,
            method=method.value,
            invert=invert,
            blur=blur,
            min_area_ratio=min_area,
            simplify_ratio=simplify,
            units_per_pixel=units_per_pixel,
            overwrite=force,
        ),
    )


@vision_app.command("extrude")
def vision_extrude(
    ctx: typer.Context,
    contours: Path = typer.Argument(..., help="*_contours.json from 'vision mask'."),
    output: Path = typer.Argument(..., help="Output mesh."),
    thickness: float = typer.Option(10.0, min=0),
    units_per_pixel: float | None = typer.Option(None, min=0),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Extrude contours into a 3D solid (plan → part)."""
    from reverse3d.vision.mask_extractor import silhouette_to_mesh

    _run(
        ctx,
        lambda s: silhouette_to_mesh(
            contours,
            output,
            s,
            thickness=thickness,
            units_per_pixel=units_per_pixel,
            overwrite=force,
        ),
    )


# ─── reconstruction ───────────────────────────────────────────────────────────


@recon_app.command("photo")
def recon_photo(
    ctx: typer.Context,
    image_dir: Path = typer.Argument(..., help="Folder with overlapping photos."),
    output_dir: Path = typer.Argument(..., help="COLMAP workspace / output folder."),
    quality: Quality | None = typer.Option(None, help="Default: from profile."),
    matcher: Matcher = typer.Option(Matcher.auto),
    dense: Dense = typer.Option(Dense.auto),
    camera_model: str = typer.Option("SIMPLE_RADIAL"),
    mesh: bool = typer.Option(True, help="Poisson-mesh the result (needs [recon])."),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Photogrammetry with COLMAP: photos → point cloud → mesh. GPU: use --device gpu."""
    from reverse3d.hardware import wants_gpu
    from reverse3d.profiles import profile_values
    from reverse3d.reconstruction.photogrammetry import PhotogrammetryOptions, run_photogrammetry

    settings = _state(ctx).settings
    options = PhotogrammetryOptions(
        quality=quality.value if quality else profile_values(settings).photo_quality,  # type: ignore[arg-type]
        matcher=matcher.value,
        dense=dense.value,
        use_gpu=wants_gpu(settings),
        camera_model=camera_model.upper(),
        build_mesh=mesh,
    )
    _run(ctx, lambda s: run_photogrammetry(image_dir, output_dir, s, options, overwrite=force))


@recon_app.command("depth")
def recon_depth(
    ctx: typer.Context,
    image: Path = typer.Argument(...),
    output_dir: Path = typer.Argument(...),
    model: str = typer.Option(
        "depth-anything/Depth-Anything-V2-Small-hf", help="Allowlisted model id."
    ),
    mesh: bool = typer.Option(True),
    fov: float = typer.Option(60.0, min=10, max=170),
    stride: int | None = typer.Option(None, min=1, max=32, help="Default: from profile."),
    mask: Path | None = typer.Option(None, help="Mask PNG to mesh only the object."),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Monocular depth estimation (+ optional 2.5D mesh). GPU: use --device gpu."""
    from reverse3d.hardware import torch_device
    from reverse3d.profiles import profile_values
    from reverse3d.reconstruction.depth_estimation import estimate_depth

    _run(
        ctx,
        lambda s: estimate_depth(
            image,
            output_dir,
            s,
            model=model,
            device=torch_device(s),
            make_mesh=mesh,
            fov_deg=fov,
            stride=stride or profile_values(s).depth_stride,
            mask_path=mask,
            overwrite=force,
        ),
    )


@recon_app.command("mesh")
def recon_mesh(
    ctx: typer.Context,
    cloud: Path = typer.Argument(..., help="Point cloud .ply/.xyz/.pcd"),
    output: Path = typer.Argument(...),
    depth: int = typer.Option(9, min=5, max=12, help="Poisson octree depth."),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Point cloud → surface mesh (screened Poisson, Open3D)."""
    from reverse3d.reconstruction.pointcloud import poisson_mesh

    _run(ctx, lambda s: poisson_mesh(cloud, output, s, depth=depth, overwrite=force))


# ─── mesh ─────────────────────────────────────────────────────────────────────


@mesh_app.command("clean")
def mesh_clean(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., metavar="INPUT"),
    output: Path = typer.Argument(...),
    keep_largest: bool = typer.Option(False, help="Keep only the largest component."),
    min_component: float = typer.Option(0.01, min=0, max=0.99),
    fill_holes: bool = typer.Option(True),
    max_hole_edges: int = typer.Option(200, min=3, max=100_000),
    fix_normals: bool = typer.Option(True),
    smooth: int = typer.Option(0, min=0, max=200, help="Taubin smoothing iterations."),
    target_faces: int | None = typer.Option(None, min=4, help="Decimate to N faces."),
    align: Align = typer.Option(Align.NONE),
    ground: bool = typer.Option(False, help="Move min Z to 0."),
    scale: float = typer.Option(1.0, min=1e-9, help="Uniform scale (unit conversion)."),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Repair, simplify and align a mesh."""
    from reverse3d.reverse_eng.mesh_cleaner import CleanOptions, clean_mesh_file

    options = CleanOptions(
        keep_largest_only=keep_largest,
        min_component_ratio=min_component,
        fix_normals=fix_normals,
        fill_holes=fill_holes,
        max_hole_edges=max_hole_edges,
        smooth_iterations=smooth,
        target_faces=target_faces,
        align=align.value,
        ground=ground,
        scale=scale,
    )
    _run(ctx, lambda s: clean_mesh_file(input_path, output, s, options, overwrite=force))


@mesh_app.command("measure")
def mesh_measure(
    ctx: typer.Context, input_path: Path = typer.Argument(..., metavar="INPUT")
) -> None:
    """Dimensions (AABB/OBB), area, volume and topology of a mesh."""
    from reverse3d.reverse_eng.measure import measure_mesh_file

    _run(ctx, lambda s: measure_mesh_file(input_path, s))


@mesh_app.command("fit")
def mesh_fit(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., metavar="INPUT"),
    kinds: str = typer.Option("plane,cylinder,sphere", help="Comma-separated primitive kinds."),
    samples: int | None = typer.Option(None, min=100, max=500_000, help="Default: from profile."),
    threshold: float | None = typer.Option(
        None, min=0, help="Inlier distance (default 1% of diagonal)."
    ),
    max_primitives: int = typer.Option(8, min=1, max=50),
    min_inlier_ratio: float = typer.Option(0.05, min=0.0001, max=0.99),
    parametric: Path | None = typer.Option(None, help="Write OpenSCAD (.scad) or CadQuery (.py)."),
    seed: int = typer.Option(0),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Detect planes/cylinders/spheres and optionally emit parametric CAD code."""
    from reverse3d.profiles import profile_values
    from reverse3d.reverse_eng.geometry_fit import ALL_KINDS, fit_primitives_file
    from reverse3d.reverse_eng.parametric import export_parametric

    requested = dict.fromkeys(k.strip().lower() for k in kinds.split(",") if k.strip())
    parsed = tuple(k for k in ALL_KINDS if k in requested)
    if not parsed or len(parsed) != len(requested):
        raise typer.BadParameter(f"kinds must be a subset of {','.join(ALL_KINDS)}")

    def job(s: Settings) -> dict[str, Any]:
        report = fit_primitives_file(
            input_path,
            s,
            kinds=parsed,
            sample_count=samples or profile_values(s).fit_samples,
            threshold=threshold,
            max_primitives=max_primitives,
            min_inlier_ratio=min_inlier_ratio,
            seed=seed,
        )
        data = report.to_dict()
        if parametric and report.primitives:
            data["parametric_path"] = str(export_parametric(report, parametric, overwrite=force))
        return data

    _run(ctx, job)


# ─── agent ────────────────────────────────────────────────────────────────────


@agent_app.command("tools")
def agent_tools(
    ctx: typer.Context,
    fmt: ToolFormat = typer.Option(ToolFormat.anthropic, "--format"),
    toolset: ToolsetChoice = typer.Option(
        ToolsetChoice.full, help="lean = 5 tools only (fewest schema tokens)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write to a file instead of stdout."
    ),
) -> None:
    """Print the tool definitions (JSON Schema) for Claude, OpenAI-compatible or Gemini."""
    from reverse3d.agent.tools import tool_definitions

    definitions = tool_definitions(fmt.value, toolset.value)
    text = json.dumps(definitions, indent=2, ensure_ascii=False) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        _err.print(f"Wrote {output}")
    else:
        sys.stdout.write(text)


@agent_app.command("call")
def agent_call(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Tool name, e.g. reverse3d_measure_mesh"),
    args: str | None = typer.Option(None, "--args", help="Tool arguments as a JSON object."),
    args_file: Path | None = typer.Option(
        None, "--args-file", help="JSON file with the arguments."
    ),
) -> None:
    """Run one tool through the sandboxed dispatcher (paths confined to the workspace)."""
    from reverse3d.agent.tools import execute_tool

    state = _state(ctx)
    if args and args_file:
        raise typer.BadParameter("Use either --args or --args-file, not both.")
    try:
        raw = args_file.read_text(encoding="utf-8") if args_file else (args or "{}")
        arguments = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"Invalid JSON arguments: {exc}") from None
    result = execute_tool(name, arguments, state.settings)
    _emit(state, result)
    if not result.get("ok"):
        raise typer.Exit(1)


@agent_app.command("mcp")
def agent_mcp(
    ctx: typer.Context,
    max_concurrent: int = typer.Option(1, min=1, max=4, help="Concurrent tool executions."),
) -> None:
    """Start the MCP server on stdio (for Claude Desktop/Code, Cursor, Gemini CLI...)."""
    from reverse3d.agent.mcp_server import serve

    state = _state(ctx)

    def provider() -> Settings:
        # Re-read config on every call so reverse3d_configure applies immediately;
        # command-line overrides given at server start keep priority.
        try:
            fresh = Settings.load()
        except Reverse3DError:
            return state.settings
        if state.workspace_override is not None:
            fresh = fresh.with_workspace(state.workspace_override)
        if state.device_override is not None:
            fresh = fresh.with_device(state.device_override)  # type: ignore[arg-type]
        return fresh

    try:
        serve(provider, max_concurrent=max_concurrent)
    except DependencyMissingError as exc:
        _err.print(f"[bold red]{exc.message}[/] {exc.hint or ''}")
        raise typer.Exit(3) from None


# ─── recipes ──────────────────────────────────────────────────────────────────


@app.command("run")
def run_cmd(
    ctx: typer.Context,
    recipe: RecipeChoice = typer.Argument(..., help="Workflow to run."),
    input_path: Path = typer.Argument(..., metavar="INPUT", help="Photo folder, image or mesh."),
    output_dir: Path = typer.Argument(..., help="Folder for all results."),
    thickness: float = typer.Option(5.0, min=0.000001, help="plan_to_part extrusion height."),
    units_per_pixel: float | None = typer.Option(None, min=0, help="plan_to_part scale (mm/px)."),
    scale: float = typer.Option(1.0, min=1e-9, help="Final uniform scale (m→mm: 1000)."),
    render: bool = typer.Option(True, help="Render a PNG preview at the end."),
    quality: Quality | None = typer.Option(
        None, help="photos_to_model quality (default: profile)."
    ),
    force_analysis: bool = typer.Option(False, help="Continue even if the photo check fails."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite outputs."),
) -> None:
    """One-command workflows: photos/plan/scan/photo/mesh → model, dimensions, CAD, render."""
    from reverse3d.recipes import RecipeOptions, run_recipe

    options = RecipeOptions(
        thickness=thickness,
        units_per_pixel=units_per_pixel,
        scale=scale,
        render=render,
        quality=quality.value if quality else None,
        force_analysis=force_analysis,
        overwrite=force,
    )
    _run(ctx, lambda s: run_recipe(recipe.value, input_path, output_dir, s, options))


# ─── configuration ────────────────────────────────────────────────────────────


config_app = typer.Typer(
    help="Persistent settings (~/.reverse3d/config.json): device, profile...", no_args_is_help=True
)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(ctx: typer.Context) -> None:
    """Show effective settings, the config file and the detected hardware."""
    from reverse3d.hardware import hardware_report
    from reverse3d.userconfig import config_path, env_overrides, read_config

    def job(s: Settings) -> dict[str, Any]:
        return {
            "config_file": str(config_path()),
            "stored": read_config(),
            "env_overrides": env_overrides(),
            "effective": {
                "workspace": str(s.workspace),
                "device": s.device,
                "profile": s.profile,
                "verbosity": s.verbosity,
                "toolset": s.toolset,
                "threads": s.threads,
                "timeout_s": s.subprocess_timeout_s,
            },
            "hardware": hardware_report(s),
        }

    _run(ctx, job)


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="device | profile | verbosity | toolset | threads | ..."),
    value: str = typer.Argument(...),
) -> None:
    """Set a persistent value, e.g. `reverse3d config set device gpu`."""
    from reverse3d.userconfig import update_config

    _run(ctx, lambda s: {"config": update_config({key: value})})


@config_app.command("unset")
def config_unset(ctx: typer.Context, key: str = typer.Argument(...)) -> None:
    """Remove a persistent value (back to the default)."""
    from reverse3d.userconfig import update_config

    _run(ctx, lambda s: {"config": update_config({key: None})})


@config_app.command("path")
def config_path_cmd() -> None:
    """Print the config file location."""
    from reverse3d.userconfig import config_path

    typer.echo(str(config_path()))


# ─── agent setup ──────────────────────────────────────────────────────────────


@app.command()
def setup(
    ctx: typer.Context,
    agents: str = typer.Option(
        "auto",
        "--agents",
        "-a",
        help="auto (detected) | all | comma list: claude-code,claude-desktop,cursor,windsurf,"
        "vscode,gemini,opencode,codex",
    ),
    workspace: Path | None = typer.Option(
        None, "--workspace", help="Sandbox folder for agents (default: ~/reverse3d-workspace)."
    ),
    device: DeviceChoice | None = typer.Option(None, help="Also store the device preference."),
    profile: ProfileChoice | None = typer.Option(None, help="Also store the performance profile."),
    toolset: ToolsetChoice | None = typer.Option(
        None, help="lean = 5 tools (fewest tokens), full = all tools."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change."),
    uninstall: bool = typer.Option(False, "--uninstall", help="Remove reverse3d from agents."),
    skill: bool = typer.Option(True, help="Install the Claude Code skill."),
    show_snippets: bool = typer.Option(False, "--snippets", help="Print manual config snippets."),
) -> None:
    """Register the reverse3d MCP server in your AI agents (one command, idempotent)."""
    from reverse3d.setup_agents import manual_snippets, setup_agents
    from reverse3d.userconfig import update_config

    def job(s: Settings) -> dict[str, Any]:
        ws = workspace or Path.home() / "reverse3d-workspace"
        if show_snippets:
            return manual_snippets(ws)
        updates = {
            "workspace": None if uninstall else str(ws.expanduser().resolve()),
            "device": device.value if device else None,
            "profile": profile.value if profile else None,
            "toolset": toolset.value if toolset else None,
        }
        stored: dict[str, str] | None = None
        if not dry_run and not uninstall:
            stored = update_config({k: v for k, v in updates.items() if v is not None})
        results = setup_agents(
            agents, workspace=ws, remove=uninstall, dry_run=dry_run, install_skill=skill
        )
        configured = [r.agent for r in results if r.status in ("configured", "unchanged")]
        if dry_run:
            planned = [r.agent for r in results if r.status == "dry_run"]
            return {
                "workspace": str(ws.expanduser().resolve()),
                "agents": [r.to_dict() for r in results],
                "next_step": f"Dry run only. Would configure: {', '.join(planned) or 'nothing'}. "
                "Re-run without --dry-run to apply.",
            }
        return {
            "workspace": str(ws.expanduser().resolve()),
            "agents": [r.to_dict() for r in results],
            "config": stored,
            "next_step": (
                "Restart your AI agent(s) and ask: 'usa reverse3d_environment_check'."
                if configured and not uninstall
                else "No agent was configured: use --agents <name> or --snippets."
                if not uninstall
                else "Removed. Restart your agents."
            ),
        }

    _run(ctx, job)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
