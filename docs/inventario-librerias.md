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

## Licencias restringidas

| Componente | Licencia | Restricción |
|------------|----------|-------------|
| Depth Anything V2 Base/Large (pesos) | CC-BY-NC-4.0 | **Uso no comercial.** El valor por defecto es Small (Apache-2.0). La herramienta devuelve `model_license` en el resultado. |
| Blender (binario externo) | GPL-2.0+ | Se invoca como proceso separado; no se enlaza ni se distribuye. |
| COLMAP (binario externo) | BSD-3 | — |

## Excepciones de CVE activas

Ninguna (ver `policy.yaml → active_exceptions`).
