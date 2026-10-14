"""MCP prompts: ready-made commands that clients show as slash commands.

Example in Claude Code: ``/mcp__reverse3d__fotos-a-3d capturas/taza``. Each
prompt expands into a short instruction that makes the agent call the right
tool in one step, so users don't need to know tool names or arguments.

Prompt arguments come from the user and are only interpolated into text sent
to the model (never executed); they are length-capped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ARG_MAX = 300

#: Short guidance sent to MCP clients that support server instructions.
SERVER_INSTRUCTIONS = (
    "reverse3d: local 3D reverse-engineering tools. Prefer reverse3d_run_recipe (one call per "
    "workflow). Paths are relative to the workspace. Use reverse3d_configure to switch "
    "device (cpu/gpu) or profile (fast/balanced/quality). If a result has ok=false, read "
    "error.hint and retry. Report sizes with units and show the render path to the user."
)


@dataclass(frozen=True)
class PromptArg:
    name: str
    description: str
    required: bool = True


@dataclass(frozen=True)
class PromptSpec:
    name: str
    description: str
    template: str
    arguments: tuple[PromptArg, ...] = field(default_factory=tuple)


PROMPTS: dict[str, PromptSpec] = {
    p.name: p
    for p in (
        PromptSpec(
            "fotos-a-3d",
            "Fotos de un objeto → modelo 3D medido y renderizado (fotogrametría).",
            "Run reverse3d_run_recipe with recipe=photos_to_model, input_path='{carpeta}', "
            "output_dir='{salida}'. If the photo check fails, explain the recommendations to the "
            "user before retrying. Then report the dimensions and the render path.",
            (
                PromptArg("carpeta", "Carpeta con las fotos, dentro del workspace"),
                PromptArg("salida", "Carpeta de resultados", required=False),
            ),
        ),
        PromptSpec(
            "plano-a-pieza",
            "Plano, careta o silueta → pieza 3D extruida con escala real.",
            "Run reverse3d_run_recipe with recipe=plan_to_part, input_path='{imagen}', "
            "output_dir='{salida}', thickness={espesor}{escala}. Report the dimensions and the "
            "render path.",
            (
                PromptArg("imagen", "Imagen del plano o careta"),
                PromptArg("espesor", "Espesor de extrusión (p. ej. 5)", required=False),
                PromptArg("mm_por_pixel", "Escala, milímetros por píxel", required=False),
                PromptArg("salida", "Carpeta de resultados", required=False),
            ),
        ),
        PromptSpec(
            "escaneo-a-cad",
            "Malla o escaneo → limpieza, cotas, primitivas y código CAD (OpenSCAD/CadQuery).",
            "Run reverse3d_run_recipe with recipe=scan_to_cad, input_path='{malla}', "
            "output_dir='{salida}'. Summarise the dimensions, the detected primitives "
            "(radii, planes) and the generated CAD files.",
            (
                PromptArg("malla", "Archivo STL/OBJ/PLY/GLB"),
                PromptArg("salida", "Carpeta de resultados", required=False),
            ),
        ),
        PromptSpec(
            "foto-a-relieve",
            "Una sola foto → relieve 2.5D aproximado (IA de profundidad).",
            "Run reverse3d_run_recipe with recipe=photo_to_relief, input_path='{imagen}', "
            "output_dir='{salida}'. Remind the user that the depth is relative, not metric.",
            (
                PromptArg("imagen", "Foto de entrada"),
                PromptArg("salida", "Carpeta de resultados", required=False),
            ),
        ),
        PromptSpec(
            "medir",
            "Cotas, volumen y render de una malla existente.",
            "Run reverse3d_run_recipe with recipe=mesh_report, input_path='{malla}', "
            "output_dir='{salida}'. Report the dimensions in a small table.",
            (
                PromptArg("malla", "Archivo de malla"),
                PromptArg("salida", "Carpeta de resultados", required=False),
            ),
        ),
        PromptSpec(
            "usar-gpu",
            "Usar la GPU (CUDA/Apple) en todas las operaciones.",
            "Call reverse3d_configure with device='gpu'. Then tell the user which GPU was "
            "detected, or that it will fall back to CPU if none is usable.",
        ),
        PromptSpec(
            "usar-cpu",
            "Forzar CPU en todas las operaciones.",
            "Call reverse3d_configure with device='cpu' and confirm the new setting.",
        ),
        PromptSpec(
            "rendimiento",
            "Elegir perfil: fast (rápido), balanced (equilibrado) o quality (máxima calidad).",
            "Call reverse3d_configure with profile='{perfil}' and confirm the new setting.",
            (PromptArg("perfil", "fast | balanced | quality"),),
        ),
        PromptSpec(
            "estado",
            "Ver capacidades instaladas, GPU detectada y configuración actual.",
            "Call reverse3d_environment_check and reverse3d_configure (no arguments). Summarise "
            "in 5 lines what is available and what is missing, with how to install it.",
        ),
    )
}


def _clip(value: str) -> str:
    return value.replace("'", "").replace("\n", " ").strip()[:_ARG_MAX]


def render_prompt(name: str, arguments: dict[str, str] | None) -> str:
    """Expand a prompt template with user arguments and sensible defaults.

    Raises:
        KeyError: unknown prompt.
        ValueError: missing required argument.
    """
    spec = PROMPTS[name]
    args = {k: _clip(str(v)) for k, v in (arguments or {}).items() if v not in (None, "")}
    for arg in spec.arguments:
        if arg.required and arg.name not in args:
            raise ValueError(f"Missing argument: {arg.name}")
    source = args.get("carpeta") or args.get("imagen") or args.get("malla") or "input"
    stem = source.rstrip("/\\").replace("\\", "/").split("/")[-1].rsplit(".", 1)[0] or "result"
    values = {
        "salida": args.get("salida", f"out/{stem}"),
        "espesor": args.get("espesor", "5"),
        "escala": f", units_per_pixel={args['mm_por_pixel']}" if "mm_por_pixel" in args else "",
        **args,
    }
    return spec.template.format_map(_Defaults(values))


class _Defaults(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""
