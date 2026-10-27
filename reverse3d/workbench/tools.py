"""The hands of a model: files and commands inside the working folder.

Every path goes through resolve_in_workspace (no escaping the folder, links
included) and is_secret (keys and credentials are out of reach). Every change
keeps the previous content in <folder>/.reverse3d/undo/, so `reverse3d work
undo` puts it back. Changes are described before they happen (a diff), so the
person can see exactly what they are approving.
"""

from __future__ import annotations

import difflib
import fnmatch
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reverse3d.security import SecurityViolationError, resolve_in_workspace
from reverse3d.workbench.permissions import Kind, Permissions, is_secret

MAX_READ = 60_000
MAX_OUTPUT = 12_000
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".next",
    "dist",
    "build",
    ".reverse3d",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


class ToolError(Exception):
    """A tool could not do what was asked (the model reads the reason)."""


@dataclass
class Tool:
    name: str
    kind: Kind
    description: str
    params: dict[str, Any]
    required: list[str]
    run: Callable[[Workspace, dict[str, Any]], str]
    preview: Callable[[Workspace, dict[str, Any]], str] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.params,
                    "required": self.required,
                },
            },
        }


class Workspace:
    def __init__(self, permissions: Permissions) -> None:
        self.perms = permissions
        self.root = permissions.folder
        self.undo_dir = self.root / ".reverse3d" / "undo"

    def path(self, raw: str) -> Path:
        last: Exception | None = None
        for root in self.perms.roots():
            try:
                p = resolve_in_workspace(raw, root)
            except SecurityViolationError as exc:
                last = exc
                continue
            if is_secret(p):
                raise ToolError(f"{raw}: es un secreto (claves o credenciales); no se toca")
            return p
        raise ToolError(f"{raw}: está fuera de la carpeta de trabajo") from last

    def rel(self, p: Path) -> str:
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    def backup(self, p: Path, action: str) -> None:
        """Keep what was there before a change (for undo)."""
        stamp = f"{time.time():.6f}"
        entry = self.undo_dir / stamp
        entry.mkdir(parents=True, exist_ok=True)
        existed = p.is_file()
        if existed:
            shutil.copy2(p, entry / "before")
        (entry / "change.json").write_text(
            json.dumps({"path": str(p), "action": action, "existed": existed}), encoding="utf-8"
        )

    def undo(self) -> str | None:
        """Put back the last change; returns what was undone."""
        entries = sorted(self.undo_dir.glob("*/change.json")) if self.undo_dir.is_dir() else []
        if not entries:
            return None
        change_file = entries[-1]
        change = json.loads(change_file.read_text(encoding="utf-8"))
        target = Path(change["path"])
        if change["existed"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(change_file.parent / "before", target)
        elif target.exists():
            target.unlink()
        shutil.rmtree(change_file.parent)
        return f"{change['action']} {self.rel(target)}"


def _diff(path: str, old: str, new: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"{path} (antes)",
        tofile=f"{path} (después)",
        n=2,
    )
    text = "".join(lines)
    return text[:8000] + ("\n… (diferencia recortada)" if len(text) > 8000 else "")


def _flag(value: Any) -> bool:
    """Models send booleans as true, "true" or "False": only a real yes counts."""
    return value is True or str(value).strip().lower() in ("true", "1", "yes", "sí", "si")


def _read_text(p: Path) -> str:
    data = p.read_bytes()
    if b"\0" in data[:4096]:
        raise ToolError(f"{p.name} es un archivo binario")
    return data.decode("utf-8", errors="replace")


# ─── reading ─────────────────────────────────────────────────────────────────


def list_files(ws: Workspace, a: dict[str, Any]) -> str:
    base = ws.path(a.get("path") or ".")
    pattern = a.get("pattern") or "*"
    if not base.is_dir():
        raise ToolError(f"{a.get('path')} no es una carpeta")
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            p = Path(dirpath) / name
            if fnmatch.fnmatch(name, pattern) and not is_secret(p):
                out.append(f"{ws.rel(p)}  ({p.stat().st_size} B)")
            if len(out) >= 300:
                return "\n".join(out) + "\n… (más de 300 archivos: usa pattern o path)"
    return "\n".join(out) or "(sin archivos)"


def read_file(ws: Workspace, a: dict[str, Any]) -> str:
    p = ws.path(a["path"])
    if not p.is_file():
        raise ToolError(f"{a['path']} no existe")
    lines = _read_text(p).splitlines()
    start = max(1, int(a.get("start") or 1))
    count = int(a.get("lines") or 400)
    chunk = lines[start - 1 : start - 1 + count]
    if _flag(a.get("numbers")):
        text = "\n".join(f"{i:>5}  {line}" for i, line in enumerate(chunk, start))
    else:  # plain by default: small models copy the numbers into their edits
        text = "\n".join(chunk)
    more = len(lines) - (start - 1 + len(chunk))
    if more > 0:
        text += f"\n… ({more} líneas más: usa start={start + len(chunk)})"
    return text[:MAX_READ]


def search(ws: Workspace, a: dict[str, Any]) -> str:
    base = ws.path(a.get("path") or ".")
    try:
        rx = re.compile(a["pattern"], re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"patrón inválido: {exc}") from exc
    glob = a.get("glob") or "*"
    hits: list[str] = []
    files = [base] if base.is_file() else []
    if base.is_dir():
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            files += [Path(dirpath) / f for f in filenames if fnmatch.fnmatch(f, glob)]
    for p in files:
        if is_secret(p) or p.stat().st_size > 2_000_000:
            continue
        try:
            text = _read_text(p)
        except (ToolError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(f"{ws.rel(p)}:{i}: {line.strip()[:200]}")
                if len(hits) >= 200:
                    return "\n".join(hits) + "\n… (más de 200 coincidencias)"
    return "\n".join(hits) or "(sin coincidencias)"


# ─── changing ────────────────────────────────────────────────────────────────


def _write_preview(ws: Workspace, a: dict[str, Any]) -> str:
    p = ws.path(a["path"])
    old = _read_text(p) if p.is_file() else ""
    return _diff(ws.rel(p), old, a["content"]) or "(sin cambios)"


def write_file(ws: Workspace, a: dict[str, Any]) -> str:
    p = ws.path(a["path"])
    ws.backup(p, "escribir")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(a["content"], encoding="utf-8", newline="")
    return f"escrito {ws.rel(p)} ({len(a['content'])} caracteres)"


def _edit_target(ws: Workspace, a: dict[str, Any]) -> tuple[Path, str, str]:
    p = ws.path(a["path"])
    if not p.is_file():
        raise ToolError(f"{a['path']} no existe")
    old = _read_text(p)
    n = old.count(a["old"])
    if n == 0:
        shown = "\n".join(f"{i:>4}  {x}" for i, x in enumerate(old.splitlines()[:60], 1))
        raise ToolError(
            "el texto a reemplazar no está en el archivo. Su contenido real es:\n" + shown
        )
    every = _flag(a.get("all"))
    if n > 1 and not every:
        raise ToolError(f"el texto aparece {n} veces: añade más contexto o all=true")
    return (
        p,
        old,
        old.replace(a["old"], a["new"]) if every else old.replace(a["old"], a["new"], 1),
    )


def edit_file(ws: Workspace, a: dict[str, Any]) -> str:
    p, _, new = _edit_target(ws, a)
    ws.backup(p, "editar")
    p.write_text(new, encoding="utf-8", newline="")  # keep its own line endings
    return f"editado {ws.rel(p)}"


def _edit_preview(ws: Workspace, a: dict[str, Any]) -> str:
    p, old, new = _edit_target(ws, a)
    return _diff(ws.rel(p), old, new)


def delete_file(ws: Workspace, a: dict[str, Any]) -> str:
    p = ws.path(a["path"])
    if not p.is_file():
        raise ToolError(f"{a['path']} no es un archivo")
    ws.backup(p, "borrar")
    p.unlink()
    return f"borrado {ws.rel(p)} (se puede deshacer)"


def run_command(ws: Workspace, a: dict[str, Any]) -> str:
    timeout = min(int(a.get("timeout") or 120), 1800)
    try:
        res = subprocess.run(  # noqa: S602 - the person approved this exact command
            a["command"],
            shell=True,
            cwd=ws.root,
            capture_output=True,
            timeout=timeout,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return f"(se detuvo: pasó de {timeout} s)"
    out = (res.stdout or "") + (("\n[stderr]\n" + res.stderr) if res.stderr else "")
    if len(out) > MAX_OUTPUT:
        out = out[:3000] + f"\n… ({len(out) - MAX_OUTPUT} caracteres omitidos) …\n" + out[-9000:]
    return f"código de salida {res.returncode}\n{out}".strip()


STR = {"type": "string"}
INT = {"type": "integer"}
TOOLS: dict[str, Tool] = {
    t.name: t
    for t in (
        Tool(
            "list_files",
            "read",
            "Lista archivos de una carpeta (recursivo).",
            {"path": STR, "pattern": {**STR, "description": "p. ej. *.py"}},
            [],
            list_files,
        ),
        Tool(
            "read_file",
            "read",
            "Lee un archivo de texto con números de línea.",
            {"path": STR, "start": INT, "lines": INT, "numbers": {"type": "boolean"}},
            ["path"],
            read_file,
        ),
        Tool(
            "search",
            "read",
            "Busca una expresión regular en los archivos.",
            {"pattern": STR, "path": STR, "glob": STR},
            ["pattern"],
            search,
        ),
        Tool(
            "write_file",
            "write",
            "Crea o reemplaza un archivo con este contenido.",
            {"path": STR, "content": STR},
            ["path", "content"],
            write_file,
            _write_preview,
        ),
        Tool(
            "edit_file",
            "write",
            "Reemplaza un texto exacto de un archivo por otro (old debe ser único).",
            {"path": STR, "old": STR, "new": STR, "all": {"type": "boolean"}},
            ["path", "old", "new"],
            edit_file,
            _edit_preview,
        ),
        Tool(
            "delete_file",
            "write",
            "Borra un archivo (se puede deshacer).",
            {"path": STR},
            ["path"],
            delete_file,
            lambda ws, a: f"borrar {ws.rel(ws.path(a['path']))}",
        ),
        Tool(
            "run_command",
            "run",
            "Ejecuta un comando en la carpeta de trabajo y devuelve su salida.",
            {"command": STR, "timeout": INT},
            ["command"],
            run_command,
            lambda ws, a: f"$ {a['command']}",
        ),
    )
}
