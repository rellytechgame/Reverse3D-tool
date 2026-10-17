# Integración con agentes de IA (Function Calling / Tool Use)

Un agente de IA puede usar reverse3d de tres formas, de mayor a menor aislamiento:

| Modo | Para quién | Sandbox de workspace |
|------|-----------|----------------------|
| **Servidor MCP** (`reverse3d agent mcp`) | Claude Desktop/Code, Cursor, Gemini CLI, OpenCode | ✅ |
| **Function calling in-process** (`execute_tool`) | Tus propios bucles de agente con el SDK de Anthropic, OpenAI o Gemini | ✅ |
| **CLI por shell** (`reverse3d --json agent call ...`) | Agentes con terminal (Claude Code, OpenCode, Aider…) | ✅ |
| CLI directa (`reverse3d mesh clean ...`) | Humanos y scripts de confianza | ❌ (el operador es de confianza) |

## Catálogo de herramientas

| Herramienta | Qué hace | Requiere |
|-------------|----------|----------|
| **`reverse3d_run_recipe`** | **Flujo completo en 1 llamada** (`photos_to_model`, `plan_to_part`, `scan_to_cad`, `photo_to_relief`, `mesh_report`) | según la receta |
| **`reverse3d_configure`** | Ver o cambiar `device` (auto/cpu/gpu), `profile` (fast/balanced/quality), `verbosity` y `threads` | — |
| `reverse3d_environment_check` | Capacidades, GPU, límites y workspace | — |
| `reverse3d_analyze_images` | Calidad y solape de fotos multivista | — |
| `reverse3d_match_images` | Correspondencias entre dos imágenes | — |
| `reverse3d_extract_mask` | Máscara + contornos JSON/SVG desde foto, plano o careta | — |
| `reverse3d_silhouette_to_mesh` | Contornos → sólido extruido (plano → pieza) | — |
| `reverse3d_photogrammetry` | Fotos → nube de puntos → malla | COLMAP (+ `[recon]`) |
| `reverse3d_estimate_depth` | Profundidad monocular → malla 2.5D | `[depth]` |
| `reverse3d_mesh_from_point_cloud` | Nube → malla (Poisson) | `[recon]` |
| `reverse3d_clean_mesh` | Reparación, decimación, alineación y escala | — |
| `reverse3d_measure_mesh` | Cotas AABB/OBB, área, volumen, topología | — |
| `reverse3d_fit_primitives` | Planos, cilindros y esferas + código OpenSCAD/CadQuery | — |
| `reverse3d_render` | Render PNG / turntable | Blender o `[render]` |

Los esquemas JSON completos están en:

- [`docs/tools/anthropic.json`](tools/anthropic.json): formato `tools` de la Messages API de Claude.
- [`docs/tools/openai.json`](tools/openai.json): formato `tools` de APIs compatibles con OpenAI.
- [`docs/tools/gemini.json`](tools/gemini.json): `function_declarations` de Gemini (subconjunto OpenAPI).

También se generan en tiempo de ejecución:

```bash
reverse3d agent tools --format anthropic   # | openai | gemini
```

```python
from reverse3d.agent import tool_definitions

tools = tool_definitions("anthropic")
```

## Economía de tokens

| Mecanismo | Uso |
|-----------|-----|
| Recetas | `reverse3d_run_recipe` sustituye 5 o 6 llamadas por una. El detalle completo queda en `<output_dir>/report.json`. |
| Salida compacta | Por defecto (`verbosity=compact`): rutas relativas al workspace, floats a 6 cifras significativas, sin nulos, listas recortadas con `{"_omitted": n}` y reductores por herramienta. |
| Toolset `lean` | `tool_definitions(dialect, "lean")` o `REVERSE3D_TOOLSET=lean`: 5 herramientas (environment_check, run_recipe, configure, measure_mesh, render). |
| Esquemas limpios | Sin `title` ni límites repetidos; Gemini recibe solo el subconjunto OpenAPI. |

## Contrato de respuesta

Todas las herramientas devuelven un único objeto JSON:

```json
{ "ok": true,  "tool": "reverse3d_measure_mesh", "result": { "...": "..." } }
{ "ok": false, "tool": "reverse3d_measure_mesh",
  "error": { "code": "security_violation", "message": "Path is outside the allowed workspace.",
             "hint": "Use paths inside REVERSE3D_WORKSPACE." } }
```

Códigos de error estables: `invalid_input`, `security_violation`, `dependency_missing`, `external_tool_failed`, `processing_failed`, `unknown_tool`, `internal_error`. Están pensados para que el modelo **lea el `hint` y corrija** su siguiente llamada.

## Cómo pasa el agente las imágenes y los archivos

reverse3d trabaja con **rutas de archivo**, no con contenido en base64:

1. El usuario (o el agente, con su herramienta de archivos) coloca las fotos, planos o mallas dentro del workspace, por ejemplo `workspace/capturas/pieza01/*.jpg`.
2. El agente pasa **rutas relativas al workspace** en los argumentos: `{"image_dir": "capturas/pieza01"}`.
3. Las salidas también se escriben en rutas del workspace: `{"output_path": "salidas/pieza01.stl"}`.
4. Para *ver* un resultado, por ejemplo un render, el agente lee el PNG generado con su propia capacidad de visión o lo adjunta al usuario.

Si el usuario adjunta una imagen en el chat, el cliente, o un paso previo del agente, debe guardarla primero en el workspace. reverse3d nunca descarga URLs: no hay herramientas de red.

## Ejemplo: bucle de tool use con Claude (Python)

El ejemplo completo y ejecutable está en [`examples/claude_agent.py`](../examples/claude_agent.py). Esta es la parte esencial:

```python
import json
import anthropic
from reverse3d import Settings
from reverse3d.agent import execute_tool, tool_definitions

client = anthropic.Anthropic()
settings = Settings.from_env()  # REVERSE3D_WORKSPACE define el sandbox
tools = tool_definitions("anthropic")
messages = [{"role": "user", "content": "Mide piezas/soporte.stl y genera un render."}]

for _ in range(20):  # límite duro de turnos (OWASP LLM10)
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        tools=tools,
        messages=messages,
    )
    if response.stop_reason != "tool_use":
        break
    messages.append({"role": "assistant", "content": response.content})
    results = []
    for block in response.content:
        if block.type == "tool_use":
            out = execute_tool(block.name, block.input, settings)  # valida + sandbox
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(out),
                    "is_error": not out["ok"],
                }
            )
    messages.append({"role": "user", "content": results})
```

## Ejemplo: OpenAI-compatible

```python
from reverse3d.agent import execute_tool, tool_definitions

tools = tool_definitions("openai")
# ... para cada tool_call del modelo:
result = execute_tool(call.function.name, json.loads(call.function.arguments), settings)
```

## Ejemplo: Gemini

```python
from reverse3d.agent import execute_tool, tool_definitions

declarations = tool_definitions("gemini")  # function_declarations
# ... para cada function_call del modelo:
result = execute_tool(fc.name, dict(fc.args), settings)
```

## Ejemplo: agente con terminal

Un agente con shell, como Claude Code u OpenCode, puede usar la CLI con sandbox:

```bash
export REVERSE3D_WORKSPACE="$PWD/workspace"
reverse3d --json agent call reverse3d_analyze_images --args '{"image_dir": "capturas/pieza01"}'
reverse3d --json agent call reverse3d_clean_mesh \
  --args '{"input_path": "scan.ply", "output_path": "out/scan_clean.stl", "align": "obb"}'
```

Con argumentos largos, conviene `--args-file args.json` para evitar problemas de escapado en la shell.

## Flujos recomendados para el agente

**Primera opción: una receta.** Por ejemplo, `reverse3d_run_recipe {"recipe": "photos_to_model", "input_path": "capturas/pieza01", "output_dir": "out/pieza01", "scale": 1000}`. Los pasos manuales de abajo solo hacen falta para flujos a medida.

**Fotos → modelo 3D medido y renderizado (manual)**

1. `reverse3d_environment_check`
2. `reverse3d_analyze_images`: si `suitable_for_photogrammetry` es `false`, transmite las `recommendations` al usuario y para.
3. `reverse3d_photogrammetry` (`quality: "medium"`)
4. `reverse3d_clean_mesh` (`keep_largest_only: true`, `align: "obb"`, `ground: true`)
5. `reverse3d_measure_mesh`
6. `reverse3d_render`

**Plano o careta → pieza extruida**

1. `reverse3d_extract_mask` (con `units_per_pixel` si se conoce una cota del plano)
2. `reverse3d_silhouette_to_mesh` (`thickness` = espesor real)
3. `reverse3d_render`

**Escaneo → CAD paramétrico**

1. `reverse3d_clean_mesh`
2. `reverse3d_fit_primitives` (`parametric_output: "cad/pieza.scad"`)
3. Revisión humana del código generado

## Buenas prácticas de seguridad para quien integra

- Usa un workspace dedicado y con permisos mínimos.
- Limita turnos y tokens en tu bucle de agente; reverse3d limita recursos por llamada, no por conversación.
- Trata como datos no confiables el texto que el agente obtiene de los resultados, por ejemplo nombres de archivo de terceros.
- No pongas credenciales en el prompt de sistema. reverse3d no necesita ninguna.
- Las herramientas con efectos costosos (`reverse3d_photogrammetry`, `reverse3d_render` con muchas muestras) pueden pasar por aprobación humana en tu cliente.
