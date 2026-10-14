# AGENTS.md — reverse3d

Instructions for AI coding agents (Claude Code, Codex, Cursor, Gemini CLI, OpenCode, Copilot, Windsurf, Aider…).
reverse3d is a local toolkit by Voxel Enterprise for 3D modeling and reverse engineering: photos, plans or scans become measured 3D models, CAD code and renders.

---

## 1. "Install reverse3d from <repo URL>" — installation protocol

Follow these steps without asking unless something fails. Each command is idempotent and safe to re-run.

**Step 1 — run the installer for the user's OS.** Use one command, with no sudo or admin rights:

| OS | Command |
|----|---------|
| Linux / macOS | `curl -fsSL https://raw.githubusercontent.com/voxelenterprise/reverse3d/main/scripts/install.sh \| bash` |
| Windows (PowerShell) | `powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/voxelenterprise/reverse3d/main/scripts/install.ps1 \| iex"` |
| Repo already cloned | Linux/macOS: `bash scripts/install.sh --source .` · Windows: `powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -Source .` |

Add options only when the user asked for them:

| User says | bash flag | PowerShell flag |
|-----------|-----------|-----------------|
| "everything" / "AI depth" / "full" | `--full` | `-Full` |
| "use GPU" / "use CPU" | `--device gpu` / `--device cpu` | `-Device gpu` / `-Device cpu` |
| "fast" / "max quality" | `--profile fast` / `--profile quality` | `-Performance fast` / `-Performance quality` |
| "save tokens" | `--toolset lean` | `-Toolset lean` |
| A specific workspace folder | `--workspace DIR` | `-Workspace DIR` |
| Only some agents | `--agents claude-code,cursor` | `-Agents claude-code,cursor` |

With curl, pass flags like this: `curl -fsSL …/install.sh | bash -s -- --full --device gpu`.
For the PowerShell one-liner with flags: `& ([scriptblock]::Create((irm …/install.ps1))) -Full -Device gpu`.

**What the installer does:**
- creates `~/.reverse3d/venv`;
- installs the package;
- detects NVIDIA or Apple GPUs;
- links the `reverse3d` command;
- runs `reverse3d setup`, which registers the MCP server in every agent found: Claude Code (plus a skill), Claude Desktop, Cursor, Windsurf, VS Code/Copilot, Gemini CLI, OpenCode and Codex. It backs up each config file it modifies.

**Step 2 — verify:** `reverse3d doctor` (or `~/.reverse3d/venv/bin/reverse3d doctor`; on Windows `%USERPROFILE%\.reverse3d\venv\Scripts\reverse3d.exe doctor`).

**Step 3 — tell the user:** "Restart your agent to load the reverse3d tools. Your workspace is `~/reverse3d-workspace`; put your photos, plans and models there."
If you cannot restart yourself, you can already use the CLI in this session (see §3).

If the installer fails:
1. Show the last error lines.
2. Python missing → ask the user to install Python 3.10+.
3. If you cannot edit an agent config, run `reverse3d setup --snippets`. It prints the exact JSON or TOML to paste by hand.

Uninstall: `install.sh --uninstall` / `install.ps1 -Uninstall`. This keeps the workspace and settings.

---

## 2. Using reverse3d (MCP tools `reverse3d_*`)

- **Prefer `reverse3d_run_recipe`:** one call runs a whole workflow and returns a short summary. Full details go to `report.json`.
  - `photos_to_model` — photo folder → measured model + render
  - `plan_to_part` — plan, stencil or silhouette image → extruded part (`thickness`, `units_per_pixel`)
  - `scan_to_cad` — mesh → clean + dimensions + primitives + OpenSCAD/CadQuery
  - `photo_to_relief` — one photo → 2.5D relief (approximate)
  - `mesh_report` — mesh → dimensions + render
- **Paths are relative to the workspace.** Save user attachments into the workspace first.
- **Device and speed:** `reverse3d_configure {"device":"gpu"|"cpu"|"auto"}`, `{"profile":"fast"|"balanced"|"quality"}`. Call it with `{}` to show settings and the detected GPU.
- **Errors:** on `ok:false`, read `error.hint`, fix and retry once.
- **Slash commands (MCP prompts):** `fotos-a-3d`, `plano-a-pieza`, `escaneo-a-cad`, `foto-a-relieve`, `medir`, `usar-gpu`, `usar-cpu`, `rendimiento`, `estado`.

## 3. CLI fallback (no MCP, or before restarting)

Same sandbox, same JSON:

```bash
reverse3d --json run plan_to_part in/plan.png out/plan --thickness 3 --units-per-pixel 0.5
reverse3d --json --device gpu run photos_to_model in/fotos out/modelo
reverse3d --json agent call reverse3d_measure_mesh --args '{"input_path":"out/plan/model.stl"}'
reverse3d config set device gpu        # persistent: auto | cpu | gpu
reverse3d config set profile fast      # fast | balanced | quality
```

---

## 4. Working on this repository (contributors)

- Setup: `pip install -e ".[dev]"`.
- Before finishing:
  - `ruff format reverse3d tests`
  - `ruff check reverse3d tests`
  - `mypy reverse3d`
  - `pytest`
  - `bandit -c pyproject.toml -r reverse3d`
- If you change agent tools, regenerate the schemas: `reverse3d agent tools --format <anthropic|openai|gemini> -o docs/tools/<fmt>.json`. A test checks they are in sync.
- Security rules (mandatory):
  - Never use `shell=True`, `eval`, `pickle` or `trust_remote_code=True`.
  - External processes go only through `reverse3d.security.run_command`.
  - Every agent-facing path goes through `resolve_in_workspace`.
  - Agents may only change `device`, `profile`, `verbosity` and `threads` via `reverse3d_configure`.
  - See [SECURITY.md](SECURITY.md) and [docs/security-model.md](docs/security-model.md).
- Code style: type hints everywhere; docstrings on public API with `Raises:`; errors are `Reverse3DError` subclasses with an actionable `hint`.
