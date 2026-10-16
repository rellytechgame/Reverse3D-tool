# Changelog

Todos los cambios notables se documentan aquí. Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado [SemVer](https://semver.org/lang/es/).

## [0.4.0] — 2026-10-16

### Funciona sin internet
- **Cambio automático a modo offline:** si el equipo no detecta conexión, reverse3d deja de usar la red por sí solo. La comprobación es una sonda TCP rápida, no envía datos y se guarda en caché un minuto.
- **`reverse3d offline status`:** muestra si hay internet y qué funciona sin él. `reverse3d doctor` incluye ese informe.
- **`reverse3d offline prepare`:** descarga los modelos de IA de profundidad en `~/.reverse3d/models`. Solo baja `safetensors` y configuración, en una **revisión fijada por commit**, de modo que un cambio en el repositorio de origen no altera lo que se ejecuta. El instalador lo hace solo.
- La IA de profundidad usa siempre la copia local; una máquina preparada no toca la red.
- Nuevo ajuste `offline` (`auto` · `on` · `off`), con `reverse3d config set offline` o `REVERSE3D_OFFLINE`.

### Piloto automático
- **`reverse3d auto <entrada>`** y la receta **`auto`**: detectan si la entrada es una carpeta de fotos, un plano o silueta, una foto, una malla o una nube de puntos, y ejecutan el flujo adecuado.
- **Reintentos que sirven:**
  - otro método de máscara para planos;
  - calidad `high` si la fotogrametría falla;
  - reparación fuerte si la malla no queda cerrada.
- Lo que necesita al usuario (fotos borrosas, una herramienta que falta) se explica en vez de reintentar.
- **`RESUMEN.md`** en cada resultado: qué se detectó, medidas, archivos y siguiente paso, en lenguaje claro.
- **`reverse3d watch`:**
  - procesa sola lo que dejes en `<workspace>/bandeja`;
  - espera a que termine de copiarse antes de procesarlo;
  - guarda el resultado en `resultados/<nombre>/` y mueve el original a `_procesado/`;
  - si algo falla, mueve el original a `_revisar/` con un `LEEME.txt`.
- Nuevo comando `/procesar` (prompt MCP) para los agentes.

### Corregido
- El sandbox trata `\` como separador también en Linux y macOS: una ruta de estilo Windows como `..\..\x` ya no se acepta como nombre de archivo.
- `install.sh` cumple ShellCheck en las versiones de los runners de GitHub, y pip-audit ya no falla por el paquete editable.
## [0.3.0] — 2026-10-15

### Instalación sin preguntas
- **El instalador nunca pregunta nada** y no necesita administrador, Python ni git:
  - consigue un **Python 3.12 privado con uv**;
  - instala desde el ZIP de GitHub;
  - nunca usa `sudo` interactivo.
- **Todo por defecto:** todos los módulos (render, mallado Open3D, IA de profundidad) y las herramientas externas **Blender 5.2.2** y **COLMAP 4.2.1** en versión portable en `~/.reverse3d/tools`, con descargas verificadas por **SHA-256 fijado en el script**.
  - `--lite` instala todo menos PyTorch.
  - `--no-tools` omite Blender y COLMAP.
- **PyTorch adecuado a la máquina:** CUDA si hay NVIDIA; si no, la versión CPU, hasta 10 veces más ligera.
- **Resistente:** un módulo opcional que falla no detiene la instalación. Si un agente corta el instalador por tiempo, volver a ejecutarlo **continúa donde se quedó**; una reejecución completa tarda unos segundos.
- **Resultado legible por agentes:** `~/.reverse3d/install-status.json` (`ok`, `missing`, `command`, `workspace`), `~/.reverse3d/install.log` y una línea final `RESULT: OK|CHECK`.
- **Herramientas aprobadas de antemano:** `reverse3d setup` añade `mcp__reverse3d` al allow-list de Claude Code y `trust` a la entrada de Gemini CLI. Así el agente no pide permiso en cada uso; se desactiva con `--no-auto-approve`.
- `AGENTS.md` pide a los agentes un único comando, sin preguntas, en segundo plano y reintentándolo si se corta. El README incluye una frase para pegar que ya da el permiso.

### Corregido
- La GPU NVIDIA no se detectaba en el instalador de Windows.
- Render de Blender descentrado o recortado: la cámara apunta al centro real de la pieza y encuadra con el ángulo de visión más estrecho. Verificado con Blender 5.2.2.

## [0.2.0] — 2026-10-14

### Autoría y procedencia
- **Guardián de autoría** (`scripts/check_authorship.py`): huellas SHA-256 de LICENSE y NOTICE y verificación de la atribución en metadatos, código y README. Se ejecuta en CI, en pre-commit y en los tests, que incluyen 12 escenarios de ataque.
- **`reverse3d/_about.py`**: fuente única de la autoría. `reverse3d --version`, `doctor` y el servidor MCP muestran el autor y el repo oficial.
- **`release.yml`**: releases solo desde tags firmados y verificados, opcionalmente limitados a `RELEASE_MAINTAINERS`. Añade atestaciones Sigstore de procedencia, `SHA256SUMS` y PyPI con Trusted Publishing tras la aprobación del propietario.
- **`AUTHORS.md`**, **`TRADEMARKS.md`**, un `NOTICE` reforzado y **`docs/authenticity.md`** (verificación, reporte de suplantaciones y lista de blindaje del propietario).
- CODEOWNERS del propietario para todos los archivos de autoría, y regla explícita para agentes de IA en `AGENTS.md §4`.

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
