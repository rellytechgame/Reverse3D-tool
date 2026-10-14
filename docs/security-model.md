# Modelo de seguridad y amenazas

**Clasificación** (`policy.yaml`): criticidad 1, datos públicos, sin PII y sin superficie de red. El threat model se exige igualmente porque las herramientas las invocan agentes de IA autónomos.

## Activos

| Activo | Por qué importa |
|--------|-----------------|
| Archivos del usuario fuera del workspace | Confidencialidad e integridad del equipo del operador |
| Variables de entorno, como API keys de otros sistemas | Podrían filtrarse a procesos hijos o a respuestas de herramientas |
| CPU, RAM, GPU y disco | Disponibilidad del equipo |
| Integridad de las salidas (mallas, cotas, CAD) | Decisiones de fabricación basadas en ellas |

## Actores y fronteras de confianza

```
 [Operador humano] ──confía──► CLI directa
 [LLM / agente] ──NO confiable──► argumentos de herramientas ──► agent/tools.py (frontera)
 [Archivos de terceros: fotos, STL, glTF, JSON] ──NO confiable──► loaders
 [COLMAP / Blender / modelos HF] ──supply chain──► procesos y pesos externos
```

## Amenazas y controles

| # | Amenaza | Referencia | Control | Test |
|---|---------|------------|---------|------|
| T1 | El agente lee o escribe fuera del workspace (`../`, absolutas, symlinks) | CWE-22, ASI02 | `resolve_in_workspace` resuelve symlinks y comprueba la contención con `is_relative_to` | `test_security.py::TestWorkspaceSandbox`, `test_agent_tools.py::test_*traversal*` |
| T2 | Un glTF u OBJ referencia `../../secreto` como buffer, textura o mtl | CWE-22 | `SandboxedResolver` solo sirve archivos del directorio del modelo, también en variantes URL-encoded | `test_mesh.py::test_resolver_blocks_parent_traversal` |
| T3 | Inyección de comandos en COLMAP o Blender | CWE-78, LLM05 | `run_command`: lista de argumentos, `shell=False`, ejecutable resuelto, enums para todos los modos | `test_security.py::test_arguments_are_not_shell_interpreted` |
| T4 | Inyección de argumentos vía `.bat` en Windows (BatBadBut) | CVE-2024-24576 | Se rechazan metacaracteres de cmd cuando el ejecutable es `.bat`/`.cmd` | `test_batch_metacharacters_rejected` |
| T5 | Inyección de código en Blender | CWE-94 | Script estático y parámetros por JSON; `--factory-startup` y `--disable-autoexec` | `test_render.py::test_blender_launch_is_hardened` |
| T6 | Inyección en el código CAD generado | CWE-94 | Solo se emiten valores con `float()` | `test_geometry_fit.py::test_mixed_part_and_parametric` |
| T7 | Ejecución de código al cargar pesos de modelos | LLM03, ASI04 | Lista blanca de modelos, `trust_remote_code=False`, `use_safetensors=True` (sin pickle) | `test_model_allowlist_enforced_before_download` |
| T8 | Deserialización insegura | CWE-502 | `np.save(..., allow_pickle=False)`; JSON con validación estructural; sin pickle | `test_malformed_contours_rejected` |
| T9 | Bombas de descompresión e imágenes gigantes | CWE-400, LLM10 | Comprobación de cabecera con Pillow antes de decodificar; límite de píxeles | `test_pixel_limit` |
| T10 | Mallas o lotes gigantes y procesos colgados | CWE-400, LLM10 | Límites de tamaño, caras e imágenes; timeout duro por proceso; semáforo MCP | `test_face_limit`, `test_timeout` |
| T11 | Fuga de secretos a procesos hijos | CWE-526 | Entorno sin variables `*KEY*`, `*TOKEN*`, `*SECRET*`, `*PASSWORD*`, etc. | `test_secrets_are_scrubbed_from_child_env` |
| T12 | Fuga de detalles internos en errores | CWE-209 | Errores estructurados; las excepciones inesperadas se reportan como `internal_error` genérico | `test_unexpected_errors_are_not_leaked` |
| T13 | El agente destruye datos sobrescribiendo archivos | LLM06 | No se sobrescribe sin `overwrite=true`; el overwrite de fotogrametría solo borra artefactos propios y nunca la carpeta de origen | `test_refuses_to_overwrite`, `test_source_inside_artifacts_is_refused` |
| T14 | Argumentos inesperados o fuera de rango del LLM | LLM05 | Pydantic `extra="forbid"`, límites numéricos, enums, `str_max_length` | `test_extra_arguments_rejected`, `test_bounds_enforced` |
| T15 | Acceso remoto al servidor MCP | ASI07 | Solo transporte stdio; sin puertos de red | Revisión de diseño |
| T16 | Dependencias comprometidas | Supply chain | Actions fijadas por SHA, Dependabot, pip-audit, CodeQL, SBOM CycloneDX | CI |
| T17 | El agente usa `reverse3d_configure` para escapar del sandbox o ejecutar binarios (`workspace`, `blender`, `colmap`, límites) | LLM06, ASI02 | Lista blanca `AGENT_WRITABLE` = device, profile, verbosity, threads; validación completa antes de escribir, de forma atómica | `test_automation.py::test_configure_tool`, `TestUserConfig::test_agent_allowlist` |
| T18 | `reverse3d setup` corrompe o borra configuraciones de otros agentes | Integridad | Solo toca la entrada `reverse3d`; hace backup previo; no escribe archivos que no puede parsear; respeta secciones manuales en Codex; escritura atómica; `--dry-run` | `TestSetupAgents::*` |
| T19 | Instalador remoto (`curl \| bash`, `irm \| iex`) | Supply chain | Sin `sudo` ni admin, venv aislado, validación de `--with` y `--ref`, versión fijable con `--ref vX.Y.Z` y opción de revisar el script antes de ejecutarlo | Job `installer` de CI (3 SO) |
| T20 | Inyección mediante argumentos de los prompts MCP | LLM01 | Los argumentos solo se insertan como texto para el modelo; se eliminan comillas y saltos de línea y se limitan a 300 caracteres; las acciones resultantes pasan por la validación estricta de las herramientas | `test_automation.py::test_prompts` |

## Riesgos residuales y deuda técnica

| Riesgo | Severidad | Mitigación pendiente |
|--------|-----------|----------------------|
| Vulnerabilidades en los parsers nativos de OpenCV, trimesh u Open3D ante archivos maliciosos | MEDIUM | Ejecutar el servidor MCP en un contenedor o usuario sin privilegios. Seguir los avisos de Dependabot. |
| Los modelos de Hugging Face se descargan por la revisión `main` (sin fijar commit) | MEDIUM | Fijar `revision=<sha>` por modelo de la lista blanca (issue de seguimiento recomendado). |
| COLMAP y Blender se ejecutan con los permisos del usuario | LOW | Documentado. Para entornos multiusuario: contenedor con el workspace montado. |
| Sin lockfile con hashes en el repositorio | LOW | `pip-compile --generate-hashes` en el pipeline de release. |
| La CLI directa no aplica el sandbox | Aceptado | Por diseño: el operador es de confianza. Los agentes deben usar `agent call` o MCP. |

## Recomendaciones de despliegue

- Workspace dedicado, sin datos sensibles y con cuota de disco.
- Para servidores compartidos: contenedor no-root, sistema de archivos de solo lectura salvo el workspace y sin red si no se usa `[depth]`.
- Al publicar en PyPI: Trusted Publishing (OIDC) y attestations; nunca tokens de larga duración.
