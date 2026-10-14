# Integración MCP (Model Context Protocol)

`reverse3d` incluye un servidor MCP con **14 herramientas** y **9 prompts**. Los clientes muestran los prompts como comandos `/`. Funciona con Claude Desktop, Claude Code, Cursor, Windsurf, VS Code/Copilot, Gemini CLI, Codex, OpenCode y cualquier cliente MCP.

## Configuración automática (recomendada)

```bash
reverse3d setup                    # detecta tus agentes y registra el servidor en todos
reverse3d setup --dry-run          # muestra qué cambiaría, sin escribir nada
reverse3d setup --agents cursor,vscode --workspace ~/mis-piezas
reverse3d setup --snippets         # imprime el JSON/TOML para configurarlo a mano
reverse3d setup --uninstall        # elimina reverse3d de los agentes
```

Los instaladores (`scripts/install.sh`, `scripts/install.ps1`) ya ejecutan `reverse3d setup`. El comando registrado es la **ruta absoluta del Python del venv** (`python -m reverse3d agent mcp`), así que no depende del PATH del cliente. Cada archivo se respalda antes de modificarlo, y los que no se pueden leer con seguridad (por ejemplo, JSON con comentarios) no se tocan.

## Prompts (comandos `/`)

| Prompt | Argumentos | Acción |
|--------|-----------|--------|
| `fotos-a-3d` | `carpeta`, `salida?` | Receta `photos_to_model` |
| `plano-a-pieza` | `imagen`, `espesor?`, `mm_por_pixel?`, `salida?` | Receta `plan_to_part` |
| `escaneo-a-cad` | `malla`, `salida?` | Receta `scan_to_cad` |
| `foto-a-relieve` | `imagen`, `salida?` | Receta `photo_to_relief` |
| `medir` | `malla`, `salida?` | Receta `mesh_report` |
| `usar-gpu` / `usar-cpu` | — | `reverse3d_configure` con el dispositivo |
| `rendimiento` | `perfil` | `reverse3d_configure` con el perfil |
| `estado` | — | Capacidades, GPU y configuración |

En Claude Code aparecen como `/mcp__reverse3d__fotos-a-3d`, etc. El servidor también envía **instrucciones** breves que orientan al modelo hacia las recetas para ahorrar tokens.

## Sin instalar: `uvx`

```json
{
  "mcpServers": {
    "reverse3d": {
      "command": "uvx",
      "args": ["--from", "reverse3d[mcp,render] @ git+https://github.com/voxelenterprise/reverse3d",
               "reverse3d", "agent", "mcp"],
      "env": { "REVERSE3D_WORKSPACE": "/ruta/a/mi-workspace" }
    }
  }
}
```

## Configuración manual

```bash
pip install -e ".[mcp]"          # o: pip install "reverse3d[mcp]"
reverse3d agent mcp              # servidor por stdio (lo lanza el cliente, no tú)
```

## Propiedades de seguridad del servidor

| Propiedad | Detalle |
|-----------|---------|
| Transporte | **Solo stdio**. No abre puertos de red; solo el proceso cliente que lo lanza puede hablarle. |
| Sandbox | Toda ruta se resuelve dentro de `REVERSE3D_WORKSPACE`. Se rechazan `../`, rutas absolutas externas y symlinks que escapan. |
| Sobrescritura | Nunca se sobrescriben archivos salvo con `"overwrite": true` explícito. |
| Validación | Los argumentos se validan con modelos Pydantic estrictos: no se aceptan campos extra, los números tienen límites y los modos son enums. |
| Concurrencia | Una herramienta a la vez por defecto (`--max-concurrent`, máximo 4). |
| Configuración en vivo | Relee `~/.reverse3d/config.json` en cada llamada: `usar-gpu` se aplica al instante. El *toolset* se fija al iniciar la sesión. |
| Configuración por el agente | `reverse3d_configure` solo puede cambiar `device`, `profile`, `verbosity` y `threads`. |
| Límites | Tamaño de archivo, píxeles, número de imágenes, caras de malla y timeout por proceso (ver `.env.example`). |
| Errores | Respuestas estructuradas `{"ok": false, "error": {"code", "message", "hint"}}`, sin trazas internas. |
| Logs | Salen por stderr y no incluyen contenido de archivos ni secretos. |

> **Recomendación:** dedica una carpeta exclusiva como workspace (por ejemplo `~/reverse3d-workspace`). No uses tu carpeta de usuario completa.

## Claude Desktop

Edita `claude_desktop_config.json` (en Windows: `%APPDATA%\Claude\`; en macOS: `~/Library/Application Support/Claude/`):

```json
{
  "mcpServers": {
    "reverse3d": {
      "command": "reverse3d",
      "args": ["agent", "mcp"],
      "env": {
        "REVERSE3D_WORKSPACE": "/home/tu-usuario/reverse3d-workspace",
        "REVERSE3D_TIMEOUT_S": "3600"
      }
    }
  }
}
```

Si `reverse3d` no está en el `PATH` del sistema, usa la ruta absoluta del ejecutable del entorno virtual, por ejemplo `.../.venv/bin/reverse3d` o `...\.venv\Scripts\reverse3d.exe`.

## Claude Code

```bash
claude mcp add reverse3d \
  -e REVERSE3D_WORKSPACE="$PWD/workspace" \
  -- reverse3d agent mcp
```

Claude Code también puede usar la CLI directamente con su herramienta de terminal. En ese caso, conviene usar `reverse3d --json agent call ...`, que aplica el mismo sandbox (ver [agent-integration.md](agent-integration.md)).

## Cursor

`.cursor/mcp.json` en el proyecto, o `~/.cursor/mcp.json` para todos los proyectos:

```json
{
  "mcpServers": {
    "reverse3d": {
      "command": "reverse3d",
      "args": ["agent", "mcp"],
      "env": { "REVERSE3D_WORKSPACE": "${workspaceFolder}/workspace" }
    }
  }
}
```

## Gemini CLI

`~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "reverse3d": {
      "command": "reverse3d",
      "args": ["agent", "mcp"],
      "env": { "REVERSE3D_WORKSPACE": "/home/tu-usuario/reverse3d-workspace" }
    }
  }
}
```

## OpenCode

`opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "reverse3d": {
      "type": "local",
      "command": ["reverse3d", "agent", "mcp"],
      "enabled": true,
      "environment": { "REVERSE3D_WORKSPACE": "/home/tu-usuario/reverse3d-workspace" }
    }
  }
}
```

## Verificación

1. Reinicia el cliente MCP.
2. Pide al agente: *"Usa reverse3d_environment_check y dime qué capacidades están disponibles"*.
3. Copia tus archivos dentro del workspace y pide, por ejemplo: *"Mide `piezas/soporte.stl` y repara la malla en `salidas/soporte_limpio.stl`"*.

## Solución de problemas

| Síntoma | Causa probable |
|---------|----------------|
| El cliente no lista herramientas | `reverse3d` no está en el PATH del cliente: usa la ruta absoluta. Falta el extra `[mcp]`. |
| `security_violation` | La ruta está fuera de `REVERSE3D_WORKSPACE`. Copia el archivo dentro o cambia el workspace. |
| `dependency_missing` | Falta COLMAP o Blender en el PATH, o un extra opcional. Usa `reverse3d_environment_check` para verlo. |
| `external_tool_failed` con timeout | Sube `REVERSE3D_TIMEOUT_S` o usa `quality: "low"`. |
