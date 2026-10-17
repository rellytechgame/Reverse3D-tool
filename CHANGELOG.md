# Changelog

Todos los cambios notables se documentan aquí. Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y versionado [SemVer](https://semver.org/lang/es/).

## [0.6.0] — 2026-10-16

### Objetos y personajes: reconstrucción por silueta (mucho más rápida y fiel)
- **Nuevo método para objetos y personajes** (`reverse3d.reconstruction.silhouette`): COLMAP solo ubica las cámaras (sin la fase densa) y la forma se esculpe con las siluetas del objeto en todas las fotos (visual hull). Funciona con superficies lisas, blancas o brillantes (termos, botellas, jarrones), que antes salían como una masa abstracta porque el motor reconstruía la habitación en lugar del objeto.
- **Mucho más rápido:** un set de 40 fotos de móvil pasa de cerca de 1 hora a unos 6 minutos en una GPU de 4 GB.
- Pasos: el objeto está donde convergen los ejes de las cámaras; GrabCut alrededor de su proyección; tallado en vóxeles; 3 rondas en las que la silueta proyectada guía un nuevo recorte; se elimina el plano de la mesa (RANSAC con los puntos dispersos); marching cubes, suavizado de Taubin y el modelo queda de pie sobre su base.
- Si el tallado falla, vuelve al método clásico (densa + Poisson).
- COLMAP exporta también el modelo en texto (`sparse_txt`) y la carpeta de fotos usada.
- Nueva dependencia: `scikit-image`.

### Menos ruido y menos fotos perdidas
- El análisis previo ya **no descarta fotos**: COLMAP ubica las cámaras mucho mejor (usa toda la escena). Sus avisos se ocultan si COLMAP ubica casi todas las fotos.

## [0.5.2] — 2026-10-16

### Perfiles de escena: de piezas a personajes, arquitectura y paisajes
- **Predicción automática de la escena** (`reverse3d scene detect <carpeta>`, `reverse3d.scenes.detect_scene`): objeto o pieza, personaje, arquitectura o paisaje. Usa una muestra de las fotos: cámara y GPS del EXIF (drones, área recorrida), inclinación del gimbal (XMP de DJI), qué tan aislado está el sujeto, si parece una persona, líneas rectas y cielo. Devuelve el perfil, la confianza, los motivos, las fotos mínimas y recomendadas y consejos de captura. Es determinista y funciona sin conexión.
- **Cada perfil ajusta el proceso** (`SCENE_PRESETS`): calidad, emparejamiento, profundidad de Poisson, recorte de superficie poco soportada, y si el resultado es un sólido cerrado (objetos, personajes) o una superficie abierta que conserva todas las partes (edificios, terrenos).
- **Emparejamiento por GPS** (`matcher="spatial"`) para paisajes con fotos geolocalizadas; sin GPS, o si falla, usa el secuencial.
- `reverse3d auto --scene auto|object|character|architecture|landscape`, `RecipeOptions.scene` y el parámetro `scene` de la herramienta `run_recipe` para agentes. El resumen incluye `scene` y `auto.scene`.
- El piloto automático detecta la escena una vez, la pasa a la receta y adapta los consejos y el plan B: en arquitectura y paisajes el modelo aproximado es un relieve de la vista completa.
- `docs/captura.md` incluye la guía de captura de cada perfil.

## [0.5.1] — 2026-10-16

### Guía de captura
- **[docs/captura.md](docs/captura.md):** cuántas fotos hacen falta (mínimo 20, recomendadas 40-60, 80-150 para objetos grandes) y los 4 perfiles de foto (anillos a 10-20°, 35-45° y 60-70°, más 4 cenitales), con las reglas de luz, solape, encuadre y superficies. Las recomendaciones del análisis y del piloto automático usan las mismas cifras.

## [0.5.0] — 2026-10-16

### Fotos difíciles: siempre hay un resultado
- **El análisis de fotos ya no depende del orden.** Antes solo comparaba cada foto con la siguiente y rechazaba sets tomados sin orden. Ahora compara todas contra todas (hasta 80 fotos; las características se calculan una sola vez por foto) y mide qué parte del set forma un grupo que se solapa, que es lo que necesita COLMAP. El informe incluye `connected_images`.
- **Reconstrucción con el grupo que sí funciona.** Si solo una parte de las fotos se solapa (8 o más), `photos_to_model` reconstruye con esas y avisa de cuántas omitió, en lugar de detenerse.
- **Plan B del piloto automático.** Si con las fotos no se puede hacer fotogrametría (no se solapan, superficies brillantes, luces que cambian, objeto movido), `reverse3d auto` entrega igualmente un modelo 3D: un modelo aproximado hecho con IA a partir de la mejor foto, marcado como tal (`fallback` en el resumen), y explica cómo tomar las fotos para obtener el modelo completo. Para elegir la foto, recorta el objeto en cada una (GrabCut) y se queda con la que lo muestra más grande, entero y nítido. Después separa el objeto de la mesa o el mousepad midiendo su altura sobre el plano de apoyo (la profundidad con IA menos el plano ajustado alrededor) y genera un **sólido cerrado e imprimible**: superficie superior suavizada, base plana y bordes redondeados (`object_solid`). La máscara se guarda como `fallback_mask.png`. Si también falla, se mantiene el error original con sus recomendaciones.

- **Control de calidad antes de la fase densa.** Si COLMAP ubica menos de 5 fotos o menos de 1.000 puntos 3D, se detiene antes de la reconstrucción densa (la etapa lenta) en lugar de producir una malla inservible; el piloto automático pasa al plan B.
- **Relieves livianos con fotos de móvil.** La densidad del relieve se ajusta al tamaño de la imagen (como máximo unos 500.000 vértices): una foto de 12 MP ya no genera una malla de 18 millones de caras.

### Fotos de iPhone (HEIC)
- **HEIC y HEIF se aceptan directamente**, sueltas o en carpetas. Se convierten a JPEG conservando el EXIF (la distancia focal ayuda a COLMAP). Nueva dependencia: `pillow-heif`.

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
