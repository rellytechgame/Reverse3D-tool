"""Persistent user configuration (``~/.reverse3d/config.json``).

Managed with ``reverse3d config set/unset/show`` and, for a small allowlist of
harmless keys, by AI agents through the ``reverse3d_configure`` tool.

Security notes:

* Agents may only change :data:`AGENT_WRITABLE` keys (device, profile,
  verbosity, threads). They can **not** move the workspace sandbox, raise
  resource limits or point ``blender``/``colmap`` to another executable, which
  would amount to arbitrary code execution.
* Every value is validated by building a :class:`~reverse3d.config.Settings`
  from it before the file is written; the write is atomic.
* The file holds no secrets; reverse3d needs no credentials.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reverse3d.errors import InputValidationError

#: config key → environment variable it maps to.
CONFIG_KEYS: dict[str, str] = {
    "workspace": "REVERSE3D_WORKSPACE",
    "device": "REVERSE3D_DEVICE",
    "profile": "REVERSE3D_PROFILE",
    "verbosity": "REVERSE3D_VERBOSITY",
    "toolset": "REVERSE3D_TOOLSET",
    "offline": "REVERSE3D_OFFLINE",
    "threads": "REVERSE3D_THREADS",
    "timeout_s": "REVERSE3D_TIMEOUT_S",
    "max_input_mb": "REVERSE3D_MAX_INPUT_MB",
    "max_image_pixels": "REVERSE3D_MAX_IMAGE_PIXELS",
    "max_images": "REVERSE3D_MAX_IMAGES",
    "max_mesh_faces": "REVERSE3D_MAX_MESH_FACES",
    "blender": "REVERSE3D_BLENDER",
    "colmap": "REVERSE3D_COLMAP",
    "log_level": "REVERSE3D_LOG_LEVEL",
}

#: Keys an AI agent may change through the reverse3d_configure tool.
AGENT_WRITABLE = frozenset({"device", "profile", "verbosity", "threads"})

_MAX_CONFIG_BYTES = 64 * 1024


def home_dir() -> Path:
    """Folder holding config.json (``REVERSE3D_HOME`` or ``~/.reverse3d``)."""
    raw = os.environ.get("REVERSE3D_HOME")
    return Path(raw).expanduser() if raw else Path.home() / ".reverse3d"


def config_path() -> Path:
    return home_dir() / "config.json"


def read_config(path: Path | None = None) -> dict[str, str]:
    """Read the config file. A missing file is an empty config.

    Raises:
        InputValidationError: malformed file or unknown keys.
    """
    path = path or config_path()
    if not path.is_file():
        return {}
    if path.stat().st_size > _MAX_CONFIG_BYTES:
        raise InputValidationError(f"Config file too large: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputValidationError(
            f"Invalid JSON in {path}", hint="Fix it or delete it to restore defaults."
        ) from exc
    if not isinstance(data, dict):
        raise InputValidationError(f"{path} must contain a JSON object")
    unknown = set(data) - set(CONFIG_KEYS)
    if unknown:
        raise InputValidationError(
            f"Unknown config keys in {path}: {sorted(unknown)}",
            hint=f"Valid keys: {', '.join(CONFIG_KEYS)}",
        )
    return {k: str(v) for k, v in data.items() if v is not None}


def config_as_env(config: Mapping[str, str]) -> dict[str, str]:
    """Translate config keys into the environment variables they stand for."""
    return {CONFIG_KEYS[k]: v for k, v in config.items() if k in CONFIG_KEYS}


def _validate(config: Mapping[str, str]) -> None:
    from reverse3d.config import Settings

    Settings.from_env(config_as_env(config))


def _write_atomic(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_config(
    updates: Mapping[str, Any],
    *,
    path: Path | None = None,
    allowed: frozenset[str] | None = None,
) -> dict[str, str]:
    """Validate and persist ``updates``. A value of ``None`` removes the key.

    Args:
        allowed: restrict which keys may be written (used for agents).

    Returns:
        The new config dict.

    Raises:
        InputValidationError: unknown/forbidden key or invalid value.
    """
    path = path or config_path()
    config = read_config(path)
    for key, value in updates.items():
        if key not in CONFIG_KEYS:
            raise InputValidationError(
                f"Unknown config key '{key}'", hint=f"Valid keys: {', '.join(CONFIG_KEYS)}"
            )
        if allowed is not None and key not in allowed:
            raise InputValidationError(
                f"Config key '{key}' cannot be changed from here.",
                hint=f"Allowed: {', '.join(sorted(allowed))}. Use the CLI for the others.",
            )
        if value is None or str(value).strip() == "":
            config.pop(key, None)
        else:
            config[key] = str(value).strip()
    _validate(config)
    _write_atomic(path, config)
    return config


def env_overrides(env: Mapping[str, str] | None = None) -> list[str]:
    """Config keys currently overridden by environment variables."""
    env = os.environ if env is None else env
    return sorted(k for k, var in CONFIG_KEYS.items() if env.get(var))
