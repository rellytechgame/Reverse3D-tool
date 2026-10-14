# Changelog

Todos los cambios notables se documentan aquí. Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado [SemVer](https://semver.org/lang/es/).

## [0.2.0] — 2026-10-14

### Añadido
- **Instalación de un comando** con `scripts/install.sh` (Linux/macOS) y `scripts/install.ps1` (Windows). Crean un venv aislado en `~/.reverse3d`, no requieren `sudo` ni administrador, detectan la GPU, instalan PyTorch CUDA en Windows y registran los agentes.
- **`AGENTS.md`**, más `CLAUDE.md` y `GEMINI.md`: protocolo para que cualquier agente se instale solo con la URL del repo.
- **`reverse3d setup`**: registro automático e idempotente del servidor MCP en Claude Code (con Skill), Claude Desktop, Cursor, Windsurf, VS Code/Copilot, Gemini CLI, OpenCode y Codex. Hace backups, admite `--dry-run`, `--uninstall` y `--snippets`.
- **GPU/CPU configurable**: `device` (`auto`/`cpu`/`gpu`) con detección de NVIDIA CUDA y Apple MPS, y vuelta a CPU sin fallos (incluido COLMAP sin CUDA).
- **Perfiles de rendimiento** `fast`, `balanced` y `quality`, aplicados a fotogrametría, render, profundidad y primitivas.
- **Configuración persistente** `~/.reverse3d/config.json` con `reverse3d config show/set/unset` y las opciones globales `--device` y `--profile`.
- **Recetas** de una llamada (`reverse3d run` / `reverse3d_run_recipe`): `photos_to_model`, `plan_to_part`, `scan_to_cad`, `photo_to_relief` y `mesh_report`. Guardan el informe completo en `report.json`.
- **Herramienta `reverse3d_configure`** para que el agente cambie dispositivo, perfil, verbosidad e hilos. Lista blanca: no puede tocar el workspace, los límites ni los binarios.
- **Prompts MCP** que aparecen como comandos `/`: `fotos-a-3d`, `plano-a-pieza`, `escaneo-a-cad`, `foto-a-relieve`, `medir`, `usar-gpu`, `usar-cpu`, `rendimiento` y `estado`. También instrucciones de servidor.
- **Ahorro de tokens**: salida compacta por defecto (−44 %), recetas (−82 % y 1 sola ida y vuelta), toolset `lean` (esquemas −65 %) y esquemas sin ruido.
- El servidor MCP **relee la configuración en cada llamada**: los cambios se aplican sin reiniciar. Soporta el SDK MCP 1.x y 2.x.
- `doctor` informa de la GPU detectada y de la configuración activa.

### Cambiado
- Las herramientas toman la calidad, las muestras y el dispositivo del perfil y de la configuración cuando no se indican.
- Las recetas de planos y escaneos conservan todas las piezas reales; `keep_largest` solo se aplica a fotogrametría y relieve.

## [0.1.0] — 2026-10-14

### Añadido
- **Visión**: extracción de máscaras y contornos vectoriales (JSON/SVG) desde fotos, planos, caretas y PNG con transparencia. Extrusión de siluetas a sólidos 3D con escala real.
- **Análisis de capturas**: nitidez, exposición, textura, keypoints y solape entre vistas consecutivas, con recomendaciones accionables.
- **Fotogrametría**: pipeline COLMAP completo (features → matching → SfM → MVS denso opcional) con detección automática de nombres de opciones entre versiones de COLMAP.
- **Profundidad monocular**: Depth Anything V2 (lista blanca, solo safetensors) y conversión profundidad → malla 2.5D.
- **Mallado**: Poisson apantallado sobre nubes de puntos (Open3D).
- **Ingeniería inversa**: limpieza y reparación de mallas, relleno de agujeros, suavizado Taubin, decimación, alineación OBB/PCA, medición de cotas (AABB/OBB, área, volumen) y ajuste RANSAC de planos, cilindros y esferas.
- **CAD paramétrico**: generación de código OpenSCAD y CadQuery desde las primitivas detectadas.
- **Render**: Blender headless (Cycles/Eevee/Workbench, turntable) con fallback a PyVista.
- **Agentes de IA**: 12 herramientas con esquemas para Claude, OpenAI y Gemini, despachador con sandbox, CLI `agent call` y servidor MCP por stdio.
- **Seguridad**: sandbox de workspace, límites de recursos, procesos sin shell con entorno sin secretos, resolver sandboxed para glTF/OBJ, CI con bandit, CodeQL, pip-audit, gitleaks y SBOM.
