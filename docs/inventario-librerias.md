# Inventario de librerías — decisiones

El inventario completo y auditable es el **SBOM CycloneDX** que genera CI (artefacto `sbom`). Este documento registra solo las **decisiones**.

## Dependencias críticas (núcleo)

| Paquete | Uso | Licencia | Notas |
|---------|-----|----------|-------|
| numpy, scipy | Álgebra lineal, RANSAC | BSD-3 | — |
| opencv-python-headless | Visión: umbrales, contornos, SIFT/ORB, RANSAC F-matrix | Apache-2.0 | Variante *headless*, sin dependencias de GUI. AKAZE no se usa porque OpenCV 5 lo sacó del paquete base. |
| pillow | Lectura de cabeceras (protección contra bombas de descompresión) | MIT-CMU | — |
| trimesh | E/S y operaciones de mallas | MIT | Carga con `SandboxedResolver` propio |
| shapely, mapbox-earcut | Polígonos y triangulación para la extrusión | BSD-3 / ISC | — |
| networkx | Requisito de grafos de trimesh (componentes) | BSD-3 | — |
| fast-simplification | Decimación cuádrica | MIT | — |
| pydantic | Validación estricta de los argumentos del agente | MIT | Control de seguridad (LLM05) |
| typer, rich | CLI | MIT | Typer ≥ 0.2x vendoriza Click: no importar `click` directamente |

## Dependencias opcionales (extras)

| Extra | Paquetes | Licencia | Decisión |
|-------|----------|----------|----------|
| `recon` | open3d | MIT | Wheels hasta Python 3.12 en el momento de escribir esto; aislado como extra |
| `depth` | torch, transformers | BSD-3 / Apache-2.0 | Lista blanca de modelos y solo safetensors |
| `render` | pyvista | MIT | Alternativa cuando no hay Blender |
| `mcp` | mcp, anyio | MIT | Servidor stdio |
| `agents` | anthropic | MIT | Solo para el ejemplo |

## Binarios que descarga el instalador

Versiones fijadas en `scripts/install.sh` y `scripts/install.ps1`. Cada archivo se verifica con su **SHA-256 fijado en el script**, copiado de la fuente oficial: el digest de GitHub Releases o el `.sha256` de download.blender.org. Para actualizar una versión hay que cambiar la versión y todas sus huellas en el mismo PR.

| Componente | Versión | Fuente | Licencia | Decisión |
|------------|---------|--------|----------|----------|
| uv | 0.12.19 | github.com/astral-sh/uv/releases | MIT / Apache-2.0 | Instala un Python 3.12 privado sin administrador ni git. Se elige 3.12 porque Open3D no tiene wheels para versiones más nuevas. |
| Blender | 5.2.2 LTS | download.blender.org | GPL-2.0+ | Portable en `~/.reverse3d/tools`. Se invoca como proceso separado; no se enlaza ni se redistribuye. |
| COLMAP | 4.2.1 | github.com/colmap/colmap/releases | BSD-3 | Build CUDA o sin CUDA según la GPU. Linux no tiene binario oficial: se usa el gestor de paquetes solo si no requiere contraseña. |
| PyTorch | la última compatible | download.pytorch.org (`cu126` o `cpu`) | BSD-3 | El índice CPU evita bajar ~2 GB de librerías CUDA en equipos sin NVIDIA. |

## Licencias restringidas

| Componente | Licencia | Restricción |
|------------|----------|-------------|
| Depth Anything V2 Base/Large (pesos) | CC-BY-NC-4.0 | **Uso no comercial.** El valor por defecto es Small (Apache-2.0). La herramienta devuelve `model_license` en el resultado. |
| Blender (binario externo) | GPL-2.0+ | Se invoca como proceso separado; no se enlaza ni se redistribuye (el instalador lo descarga de la fuente oficial). |
| COLMAP (binario externo) | BSD-3 | — |

## Excepciones de CVE activas

Ninguna (ver `policy.yaml → active_exceptions`).
