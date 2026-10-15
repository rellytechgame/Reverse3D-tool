"""Register the reverse3d MCP server in the user's AI agents (``reverse3d setup``).

Supported targets and the files they use:

=============== ==========================================================
claude-code     ``claude mcp add --scope user`` (official CLI) + Agent Skill
claude-desktop  ``<appdata>/Claude/claude_desktop_config.json``
cursor          ``~/.cursor/mcp.json``
windsurf        ``~/.codeium/windsurf/mcp_config.json``
vscode          ``<appdata>/Code/User/mcp.json`` (GitHub Copilot agent mode)
gemini          ``~/.gemini/settings.json``
opencode        ``~/.config/opencode/opencode.json``
codex           ``~/.codex/config.toml``
=============== ==========================================================

Safety rules (configuration files belong to the user and to other tools):

* Only the ``reverse3d`` entry is added, replaced or removed; everything else in
  the file is preserved.
* Every modified file is backed up first (``<file>.reverse3d-backup-<ts>``).
* Files that cannot be parsed (e.g. JSON with comments) are **not** modified;
  the manual snippet is reported instead.
* ``auto`` only touches agents whose configuration folder already exists.
* The server command is the absolute path of the current Python interpreter,
  so it works without PATH changes and always uses this installation.

Auto-approval (``auto_approve=True``, the default) pre-authorises **only the
reverse3d tools**, so non-technical users are not asked to confirm every call:

* Claude Code: ``mcp__reverse3d`` in ``permissions.allow`` of
  ``~/.claude/settings.json``.
* Gemini CLI: ``"trust": true`` on the reverse3d server entry.

This is safe because every tool is confined to the workspace sandbox, never
overwrites without ``overwrite=true`` and can only change harmless settings
(see ``docs/security-model.md``). Other agents have no per-server switch; they
ask once and offer "always allow".
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reverse3d.errors import InputValidationError, Reverse3DError
from reverse3d.security import resolve_executable, run_command

SERVER_NAME = "reverse3d"
CLAUDE_PERMISSION = f"mcp__{SERVER_NAME}"  # matches every tool of the server
_CODEX_BEGIN = "# >>> reverse3d (managed by `reverse3d setup`) >>>"
_CODEX_END = "# <<< reverse3d <<<"


@dataclass
class SetupResult:
    agent: str
    status: str  # configured | removed | unchanged | not_detected | skipped | error | dry_run
    detail: str = ""
    path: str | None = None
    backup: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "")}


# ─── locations (functions so tests can monkeypatch them) ──────────────────────


def _home() -> Path:
    return Path.home()


def _appdata() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path(os.environ.get("APPDATA", _home() / "AppData" / "Roaming"))
    if system == "Darwin":
        return _home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME", _home() / ".config"))


def _xdg_config() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", _home() / ".config"))


@dataclass(frozen=True)
class _Target:
    key: str
    label: str
    path: Callable[[], Path]
    detect: Callable[[], Path]
    style: str  # mcpServers | vscode | opencode | codex


TARGETS: dict[str, _Target] = {
    t.key: t
    for t in (
        _Target(
            "claude-desktop",
            "Claude Desktop",
            lambda: _appdata() / "Claude" / "claude_desktop_config.json",
            lambda: _appdata() / "Claude",
            "mcpServers",
        ),
        _Target(
            "cursor",
            "Cursor",
            lambda: _home() / ".cursor" / "mcp.json",
            lambda: _home() / ".cursor",
            "mcpServers",
        ),
        _Target(
            "windsurf",
            "Windsurf",
            lambda: _home() / ".codeium" / "windsurf" / "mcp_config.json",
            lambda: _home() / ".codeium" / "windsurf",
            "mcpServers",
        ),
        _Target(
            "vscode",
            "VS Code (Copilot)",
            lambda: _appdata() / "Code" / "User" / "mcp.json",
            lambda: _appdata() / "Code" / "User",
            "vscode",
        ),
        _Target(
            "gemini",
            "Gemini CLI",
            lambda: _home() / ".gemini" / "settings.json",
            lambda: _home() / ".gemini",
            "mcpServers",
        ),
        _Target(
            "opencode",
            "OpenCode",
            lambda: _xdg_config() / "opencode" / "opencode.json",
            lambda: _xdg_config() / "opencode",
            "opencode",
        ),
        _Target(
            "codex",
            "Codex CLI",
            lambda: _home() / ".codex" / "config.toml",
            lambda: _home() / ".codex",
            "codex",
        ),
    )
}
ALL_AGENTS = ("claude-code", *TARGETS)


# ─── server entry ─────────────────────────────────────────────────────────────


def server_command() -> tuple[str, list[str]]:
    """Absolute interpreter + module invocation: independent of PATH."""
    return sys.executable, ["-m", "reverse3d", "agent", "mcp"]


def _entry(style: str, env: dict[str, str]) -> dict[str, Any]:
    command, args = server_command()
    if style == "vscode":
        return {"type": "stdio", "command": command, "args": args, "env": env}
    if style == "opencode":
        return {"type": "local", "command": [command, *args], "enabled": True, "environment": env}
    return {"command": command, "args": args, "env": env}


def _container_key(style: str) -> str:
    return {"vscode": "servers", "opencode": "mcp"}.get(style, "mcpServers")


# ─── JSON targets ─────────────────────────────────────────────────────────────


def _backup(path: Path) -> Path:
    backup = path.with_name(f"{path.name}.reverse3d-backup-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    return backup


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".reverse3d-tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_json_object(path: Path) -> dict[str, Any] | str:
    """The file's top-level object (``{}`` if missing), or a reason it can't be used."""
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "Could not parse the file (comments?)"
    if not isinstance(loaded, dict):
        return "Top-level JSON is not an object"
    return loaded


def _setup_json(
    target: _Target, env: dict[str, str], *, remove: bool, dry_run: bool, auto_approve: bool
) -> SetupResult:
    path = target.path()
    trust = auto_approve and target.key == "gemini"
    loaded = _read_json_object(path)
    if isinstance(loaded, str):
        entry = _entry(target.style, env) | ({"trust": True} if trust else {})
        snippet = json.dumps({_container_key(target.style): {SERVER_NAME: entry}}, indent=2)
        return SetupResult(
            target.key, "skipped", f"{loaded}. Add this manually:\n{snippet}", str(path)
        )
    data = loaded

    key = _container_key(target.style)
    container = data.get(key)
    if container is not None and not isinstance(container, dict):
        return SetupResult(target.key, "skipped", f"'{key}' is not an object", str(path))
    container = dict(container or {})

    if remove:
        if SERVER_NAME not in container:
            return SetupResult(target.key, "unchanged", "not registered", str(path))
        container.pop(SERVER_NAME)
    else:
        new = _entry(target.style, env)
        if trust:
            # Gemini CLI: skip the confirmation prompt for this server's tools only.
            new["trust"] = True
        if container.get(SERVER_NAME) == new:
            return SetupResult(target.key, "unchanged", "already up to date", str(path))
        container[SERVER_NAME] = new
    data[key] = container

    if dry_run:
        return SetupResult(target.key, "dry_run", "would update", str(path))
    backup = _backup(path) if path.exists() else None
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return SetupResult(
        target.key,
        "removed" if remove else "configured",
        target.label,
        str(path),
        str(backup) if backup else None,
    )


# ─── Codex (TOML) ─────────────────────────────────────────────────────────────


def _toml_str(value: str) -> str:
    # JSON string escaping is a valid TOML basic string.
    return json.dumps(value, ensure_ascii=False)


def _codex_block(env: dict[str, str]) -> str:
    command, args = server_command()
    lines = [
        _CODEX_BEGIN,
        f"[mcp_servers.{SERVER_NAME}]",
        f"command = {_toml_str(command)}",
        "args = [" + ", ".join(_toml_str(a) for a in args) + "]",
        "",
        f"[mcp_servers.{SERVER_NAME}.env]",
        *(f"{k} = {_toml_str(v)}" for k, v in env.items()),
        _CODEX_END,
    ]
    return "\n".join(lines) + "\n"


def _setup_codex(
    target: _Target, env: dict[str, str], *, remove: bool, dry_run: bool
) -> SetupResult:
    path = target.path()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    begin, end = text.find(_CODEX_BEGIN), text.find(_CODEX_END)
    managed = begin != -1 and end > begin
    outside = text if not managed else text[:begin] + text[end + len(_CODEX_END) :]
    if f"[mcp_servers.{SERVER_NAME}]" in outside:
        return SetupResult(
            target.key,
            "skipped",
            "A manual [mcp_servers.reverse3d] section exists; left as is.",
            str(path),
        )
    if remove:
        if not managed:
            return SetupResult(target.key, "unchanged", "not registered", str(path))
        new_text = outside.rstrip() + "\n" if outside.strip() else ""
    else:
        block = _codex_block(env)
        if managed and text[begin : end + len(_CODEX_END)] + "\n" == block:
            return SetupResult(target.key, "unchanged", "already up to date", str(path))
        base = outside.rstrip()
        new_text = (base + "\n\n" if base else "") + block
    if dry_run:
        return SetupResult(target.key, "dry_run", "would update", str(path))
    backup = _backup(path) if path.exists() else None
    _write_text(path, new_text)
    return SetupResult(
        target.key,
        "removed" if remove else "configured",
        target.label,
        str(path),
        str(backup) if backup else None,
    )


# ─── Claude Code (official CLI) ───────────────────────────────────────────────


def _claude_cli() -> Path | None:
    try:
        return resolve_executable(["claude", "claude.cmd", "claude.exe"], None, tool="claude")
    except Reverse3DError:
        return None


def skill_source() -> Path:
    return Path(__file__).parent / "agent" / "skill" / "SKILL.md"


def _skill_dest() -> Path:
    return _home() / ".claude" / "skills" / "reverse3d" / "SKILL.md"


def _claude_settings() -> Path:
    return _home() / ".claude" / "settings.json"


def _claude_permissions(*, remove: bool) -> str:
    """Add/remove ``mcp__reverse3d`` in Claude Code's allow list; returns a note."""
    path = _claude_settings()
    data = _read_json_object(path)
    if isinstance(data, str):
        return f'{data}: add "{CLAUDE_PERMISSION}" to permissions.allow in {path} by hand'
    permissions = data.get("permissions")
    if permissions is not None and not isinstance(permissions, dict):
        return f"'permissions' in {path} is not an object; auto-approval skipped"
    permissions = dict(permissions or {})
    allow = permissions.get("allow")
    if allow is not None and not isinstance(allow, list):
        return f"'permissions.allow' in {path} is not a list; auto-approval skipped"
    allow = list(allow or [])

    if remove:
        if CLAUDE_PERMISSION not in allow:
            return ""
        allow = [rule for rule in allow if rule != CLAUDE_PERMISSION]
    else:
        if CLAUDE_PERMISSION in allow:
            return "tools already auto-approved"
        allow.append(CLAUDE_PERMISSION)
    permissions["allow"] = allow
    data["permissions"] = permissions
    if path.exists():
        _backup(path)
    _write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return "auto-approval removed" if remove else "reverse3d tools auto-approved"


def _setup_claude_code(
    env: dict[str, str], *, remove: bool, dry_run: bool, install_skill: bool, auto_approve: bool
) -> SetupResult:
    cli = _claude_cli()
    detected = cli is not None or (_home() / ".claude").is_dir()
    if not detected:
        return SetupResult("claude-code", "not_detected")
    notes: list[str] = []
    if dry_run:
        return SetupResult("claude-code", "dry_run", "would run `claude mcp add` + install skill")

    if cli is not None:
        # Remove first so re-running setup updates the entry (idempotent).
        run_command(
            cli,
            ["mcp", "remove", SERVER_NAME, "--scope", "user"],
            timeout_s=60,
            tool="claude",
            allow_nonzero=True,
        )
        if not remove:
            command, args = server_command()
            env_args = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
            run_command(
                cli,
                ["mcp", "add", "--scope", "user", *env_args, SERVER_NAME, "--", command, *args],
                timeout_s=60,
                tool="claude",
            )
            notes.append("MCP registered (user scope)")
    else:
        notes.append("`claude` CLI not on PATH: run `claude mcp add` manually (see README)")

    dest = _skill_dest()
    if remove:
        if dest.exists():
            dest.unlink()
            notes.append("skill removed")
    elif install_skill:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(skill_source(), dest)
        notes.append(f"skill installed at {dest}")
    if remove or auto_approve:
        note = _claude_permissions(remove=remove)
        if note:
            notes.append(note)
    status = "removed" if remove else ("configured" if cli else "skipped")
    return SetupResult("claude-code", status, "; ".join(notes))


# ─── public API ───────────────────────────────────────────────────────────────


def parse_agents(spec: str) -> tuple[list[str], bool]:
    """``auto`` | ``all`` | comma list → (agents, only_detected)."""
    spec = spec.strip().lower()
    if spec in ("", "auto"):
        return list(ALL_AGENTS), True
    if spec == "all":
        return list(ALL_AGENTS), False
    agents = [a.strip() for a in spec.split(",") if a.strip()]
    unknown = [a for a in agents if a not in ALL_AGENTS]
    if unknown:
        raise InputValidationError(
            f"Unknown agent(s): {', '.join(unknown)}", hint=f"Valid: {', '.join(ALL_AGENTS)}"
        )
    return agents, False


def setup_agents(
    agents: str = "auto",
    *,
    workspace: Path,
    remove: bool = False,
    dry_run: bool = False,
    install_skill: bool = True,
    auto_approve: bool = True,
    extra_env: dict[str, str] | None = None,
) -> list[SetupResult]:
    """Register (or remove) the MCP server in the selected agents.

    Args:
        agents: ``auto`` (detected agents), ``all`` or a comma list of agent keys.
        workspace: Sandbox folder the server is pinned to.
        remove: Unregister instead of register (also removes auto-approval).
        dry_run: Report what would change without writing anything.
        install_skill: Copy the Claude Code skill.
        auto_approve: Pre-authorise the reverse3d tools where the agent supports
            it per server (Claude Code, Gemini CLI).
        extra_env: Additional environment for the server entry.

    Raises:
        InputValidationError: ``agents`` names an unknown agent.
    """
    selected, only_detected = parse_agents(agents)
    workspace = workspace.expanduser().resolve()
    if not remove and not dry_run:
        workspace.mkdir(parents=True, exist_ok=True)
    env = {"REVERSE3D_WORKSPACE": str(workspace), **(extra_env or {})}

    results: list[SetupResult] = []
    for key in selected:
        try:
            if key == "claude-code":
                results.append(
                    _setup_claude_code(
                        env,
                        remove=remove,
                        dry_run=dry_run,
                        install_skill=install_skill,
                        auto_approve=auto_approve,
                    )
                )
                continue
            target = TARGETS[key]
            if only_detected and not target.detect().is_dir():
                results.append(SetupResult(key, "not_detected"))
                continue
            if target.style == "codex":
                results.append(_setup_codex(target, env, remove=remove, dry_run=dry_run))
            else:
                results.append(
                    _setup_json(
                        target, env, remove=remove, dry_run=dry_run, auto_approve=auto_approve
                    )
                )
        except (Reverse3DError, OSError) as exc:
            message = exc.message if isinstance(exc, Reverse3DError) else str(exc)
            results.append(SetupResult(key, "error", message))
    return results


def manual_snippets(workspace: Path) -> dict[str, Any]:
    """Configuration snippets for agents the user wants to set up by hand."""
    env = {"REVERSE3D_WORKSPACE": str(workspace.expanduser().resolve())}
    command, args = server_command()
    return {
        "mcpServers": {"mcpServers": {SERVER_NAME: _entry("mcpServers", env)}},
        "vscode": {"servers": {SERVER_NAME: _entry("vscode", env)}},
        "opencode": {"mcp": {SERVER_NAME: _entry("opencode", env)}},
        "codex_toml": _codex_block(env),
        "claude_code": (
            "claude mcp add --scope user "
            f'-e REVERSE3D_WORKSPACE="{env["REVERSE3D_WORKSPACE"]}" '
            f'{SERVER_NAME} -- "{command}" {" ".join(args)}'
        ),
    }
