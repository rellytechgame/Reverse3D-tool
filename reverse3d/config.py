"""Runtime settings.

Precedence (lowest → highest): built-in defaults → user config file
(``~/.reverse3d/config.json``, managed with ``reverse3d config``) → environment
variables. :meth:`Settings.load` applies all three; :meth:`Settings.from_env`
reads only the given mapping (useful in tests).

All resource limits exist to bound the work an AI agent (or a malicious input
file) can trigger: OWASP LLM10 "Unbounded Consumption". Every value is parsed
and range-checked; invalid values fail loudly instead of silently falling back
to "unlimited".

==============================  =====================================  ============
Variable                        Meaning                                Default
==============================  =====================================  ============
REVERSE3D_WORKSPACE             Sandbox root for the agent tool layer  current dir
REVERSE3D_DEVICE                auto | cpu | gpu                       auto
REVERSE3D_PROFILE               fast | balanced | quality              balanced
REVERSE3D_VERBOSITY             compact | full (agent tool output)     compact
REVERSE3D_TOOLSET               full | lean (tools exposed to agents)  full
REVERSE3D_OFFLINE               auto | on | off (internet use)         auto
REVERSE3D_THREADS               CPU threads, 0 = automatic             0
REVERSE3D_MAX_INPUT_MB          Max size of any input file             512
REVERSE3D_MAX_IMAGE_PIXELS      Max pixels per decoded image           100000000
REVERSE3D_MAX_IMAGES            Max images per photogrammetry job      500
REVERSE3D_MAX_MESH_FACES        Max faces of a loaded mesh             5000000
REVERSE3D_TIMEOUT_S             Timeout per external process (s)       3600
REVERSE3D_BLENDER               Explicit path to the Blender binary    (PATH)
REVERSE3D_COLMAP                Explicit path to the COLMAP binary     (PATH)
REVERSE3D_LOG_LEVEL             DEBUG | INFO | WARNING | ERROR         INFO
REVERSE3D_LOG_JSON              1 to emit JSON log lines on stderr     0
REVERSE3D_HOME                  Folder of config.json                  ~/.reverse3d
==============================  =====================================  ============
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from reverse3d.errors import InputValidationError

_MB = 1024 * 1024
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}

Device = Literal["auto", "cpu", "gpu"]
Profile = Literal["fast", "balanced", "quality"]
Verbosity = Literal["compact", "full"]
Toolset = Literal["full", "lean"]
OfflineMode = Literal["auto", "on", "off"]

_CHOICES: dict[str, tuple[str, ...]] = {
    "REVERSE3D_DEVICE": ("auto", "cpu", "gpu"),
    "REVERSE3D_PROFILE": ("fast", "balanced", "quality"),
    "REVERSE3D_VERBOSITY": ("compact", "full"),
    "REVERSE3D_TOOLSET": ("full", "lean"),
    "REVERSE3D_OFFLINE": ("auto", "on", "off"),
}


def _int_env(env: Mapping[str, str], name: str, default: int, lo: int, hi: int) -> int:
    raw = env.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        value = int(str(raw).strip())
    except ValueError as exc:
        raise InputValidationError(f"{name} must be an integer, got {raw!r}") from exc
    if not lo <= value <= hi:
        raise InputValidationError(f"{name} must be between {lo} and {hi}, got {value}")
    return value


def _choice_env(env: Mapping[str, str], name: str, default: str) -> str:
    raw = str(env.get(name, "") or "").strip().lower()
    if not raw:
        return default
    if raw not in _CHOICES[name]:
        raise InputValidationError(f"{name} must be one of {list(_CHOICES[name])}, got {raw!r}")
    return raw


def _optional_path(env: Mapping[str, str], name: str) -> Path | None:
    raw = env.get(name)
    return Path(raw).expanduser() if raw and str(raw).strip() else None


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration. Build it with :meth:`load`."""

    workspace: Path = field(default_factory=Path.cwd)
    device: Device = "auto"
    profile: Profile = "balanced"
    verbosity: Verbosity = "compact"
    toolset: Toolset = "full"
    offline: OfflineMode = "auto"
    threads: int = 0
    max_input_bytes: int = 512 * _MB
    max_image_pixels: int = 100_000_000
    max_images: int = 500
    max_mesh_faces: int = 5_000_000
    subprocess_timeout_s: int = 3600
    blender_path: Path | None = None
    colmap_path: Path | None = None
    log_level: str = "INFO"
    log_json: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``env`` only (defaults to ``os.environ``).

        Raises:
            InputValidationError: if a variable is malformed or out of range.
        """
        env = os.environ if env is None else env
        workspace_raw = env.get("REVERSE3D_WORKSPACE")
        workspace = Path(workspace_raw).expanduser() if workspace_raw else Path.cwd()

        log_level = str(env.get("REVERSE3D_LOG_LEVEL", "INFO")).strip().upper()
        if log_level not in _LOG_LEVELS:
            raise InputValidationError(f"REVERSE3D_LOG_LEVEL must be one of {sorted(_LOG_LEVELS)}")

        return cls(
            workspace=workspace.resolve(),
            device=_choice_env(env, "REVERSE3D_DEVICE", "auto"),  # type: ignore[arg-type]
            profile=_choice_env(env, "REVERSE3D_PROFILE", "balanced"),  # type: ignore[arg-type]
            verbosity=_choice_env(env, "REVERSE3D_VERBOSITY", "compact"),  # type: ignore[arg-type]
            toolset=_choice_env(env, "REVERSE3D_TOOLSET", "full"),  # type: ignore[arg-type]
            offline=_choice_env(env, "REVERSE3D_OFFLINE", "auto"),  # type: ignore[arg-type]
            threads=_int_env(env, "REVERSE3D_THREADS", 0, 0, 1024),
            max_input_bytes=_int_env(env, "REVERSE3D_MAX_INPUT_MB", 512, 1, 16_384) * _MB,
            max_image_pixels=_int_env(
                env, "REVERSE3D_MAX_IMAGE_PIXELS", 100_000_000, 1_000, 1_000_000_000
            ),
            max_images=_int_env(env, "REVERSE3D_MAX_IMAGES", 500, 1, 10_000),
            max_mesh_faces=_int_env(env, "REVERSE3D_MAX_MESH_FACES", 5_000_000, 100, 200_000_000),
            subprocess_timeout_s=_int_env(env, "REVERSE3D_TIMEOUT_S", 3600, 10, 172_800),
            blender_path=_optional_path(env, "REVERSE3D_BLENDER"),
            colmap_path=_optional_path(env, "REVERSE3D_COLMAP"),
            log_level=log_level,
            log_json=str(env.get("REVERSE3D_LOG_JSON", "0")).strip() in {"1", "true", "yes"},
        )

    @classmethod
    def load(
        cls, env: Mapping[str, str] | None = None, *, config_file: Path | None = None
    ) -> Settings:
        """Defaults ← user config file ← environment variables."""
        from reverse3d.userconfig import config_as_env, read_config

        env = os.environ if env is None else env
        merged = {**config_as_env(read_config(config_file)), **env}
        return cls.from_env(merged)

    def with_workspace(self, workspace: Path) -> Settings:
        """Return a copy with a different (resolved) sandbox root."""
        return replace(self, workspace=workspace.expanduser().resolve())

    def with_device(self, device: Device) -> Settings:
        """Return a copy with another compute device preference."""
        if device not in ("auto", "cpu", "gpu"):
            raise InputValidationError("device must be auto, cpu or gpu")
        return replace(self, device=device)
