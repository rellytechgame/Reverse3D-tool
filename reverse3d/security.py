"""Security primitives shared by every module.

Centralising these controls keeps the attack surface reviewable in one place:

* **Path sandboxing** — agent-supplied paths are resolved (symlinks included)
  and must stay inside the configured workspace (CWE-22 path traversal).
* **Input validation** — extension allowlists and size limits for every file
  that is parsed (CWE-400 resource exhaustion, decompression bombs).
* **Safe process execution** — external tools are launched with an argument
  list, never through a shell (CWE-78 OS command injection), with a hard
  timeout and a scrubbed environment (no credentials leak to child processes).
"""

from __future__ import annotations

import logging
import os
import re
import shutil

# Justification for B404: subprocess is only used through run_command(), never with a shell.
import subprocess  # nosec B404
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

from reverse3d.errors import (
    DependencyMissingError,
    ExternalToolError,
    InputValidationError,
    SecurityViolationError,
)

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"})
MESH_EXTENSIONS = frozenset({".stl", ".obj", ".ply", ".glb", ".gltf", ".off"})
POINT_CLOUD_EXTENSIONS = frozenset({".ply", ".xyz", ".pcd"})
JSON_EXTENSIONS = frozenset({".json"})

_STDERR_TAIL_CHARS = 2000

# Environment variables matching this pattern are removed before spawning
# external tools: COLMAP/Blender never need credentials, so a compromised or
# buggy child process must not be able to read them.
_SECRET_ENV_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE)", re.IGNORECASE
)

# cmd.exe re-parses the command line of .bat/.cmd files, which makes argument
# lists injectable on Windows ("BatBadBut", CVE-2024-24576 class). We refuse
# arguments with cmd metacharacters when the target is a batch file.
_CMD_METACHARS = re.compile(r'[&|<>^%!"\r\n]')


# ─── Paths ────────────────────────────────────────────────────────────────────


def resolve_in_workspace(path: str | os.PathLike[str], workspace: Path) -> Path:
    """Resolve ``path`` (relative paths are relative to ``workspace``) and make
    sure the result lives inside ``workspace``.

    Symlinks are resolved before the containment check, so a link pointing
    outside the sandbox is rejected as well.

    Raises:
        SecurityViolationError: if the resolved path escapes the workspace.
    """
    root = workspace.resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise SecurityViolationError(
            "Path is outside the allowed workspace.",
            hint="Use paths inside REVERSE3D_WORKSPACE.",
        )
    return resolved


def validate_input_file(
    path: str | os.PathLike[str],
    *,
    allowed_extensions: Iterable[str],
    max_bytes: int,
) -> Path:
    """Check that ``path`` is an existing regular file with an allowed extension
    and a size below ``max_bytes``. Returns the resolved path.

    Raises:
        InputValidationError: on any violation.
    """
    resolved = Path(path).expanduser().resolve()
    allowed = {e.lower() for e in allowed_extensions}
    if not resolved.is_file():
        raise InputValidationError(f"Input file not found: {resolved.name}")
    if resolved.suffix.lower() not in allowed:
        raise InputValidationError(
            f"Unsupported file type '{resolved.suffix}'.",
            hint=f"Allowed: {', '.join(sorted(allowed))}",
        )
    size = resolved.stat().st_size
    if size == 0:
        raise InputValidationError(f"Input file is empty: {resolved.name}")
    if size > max_bytes:
        raise InputValidationError(
            f"Input file too large ({size / 1e6:.1f} MB > {max_bytes / 1e6:.1f} MB).",
            hint="Raise REVERSE3D_MAX_INPUT_MB if this is expected.",
        )
    return resolved


def validate_input_dir(path: str | os.PathLike[str]) -> Path:
    """Check that ``path`` is an existing directory and return it resolved."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise InputValidationError(f"Input directory not found: {resolved.name}")
    return resolved


def list_images(directory: Path, *, max_images: int, max_bytes: int) -> list[Path]:
    """Return the validated, sorted image files directly inside ``directory``.

    Raises:
        InputValidationError: if there are no images or more than ``max_images``.
    """
    images = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise InputValidationError(f"No images found in '{directory.name}'.")
    if len(images) > max_images:
        raise InputValidationError(
            f"Too many images ({len(images)} > {max_images}).",
            hint="Raise REVERSE3D_MAX_IMAGES if this is expected.",
        )
    return [
        validate_input_file(p, allowed_extensions=IMAGE_EXTENSIONS, max_bytes=max_bytes)
        for p in images
    ]


def prepare_output_file(
    path: str | os.PathLike[str],
    *,
    allowed_extensions: Iterable[str],
    overwrite: bool = False,
) -> Path:
    """Validate an output file path, create its parent directory and refuse to
    clobber an existing file unless ``overwrite`` is set.

    Raises:
        InputValidationError: wrong extension, existing file, or path is a dir.
    """
    resolved = Path(path).expanduser().resolve()
    allowed = {e.lower() for e in allowed_extensions}
    if resolved.suffix.lower() not in allowed:
        raise InputValidationError(
            f"Unsupported output type '{resolved.suffix}'.",
            hint=f"Allowed: {', '.join(sorted(allowed))}",
        )
    if resolved.is_dir():
        raise InputValidationError(f"Output path is a directory: {resolved.name}")
    if resolved.exists() and not overwrite:
        raise InputValidationError(
            f"Output file already exists: {resolved.name}",
            hint="Pass overwrite=true (CLI: --force) to replace it.",
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def prepare_output_dir(path: str | os.PathLike[str]) -> Path:
    """Create (if needed) and return a resolved output directory."""
    resolved = Path(path).expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise InputValidationError(f"Output path is not a directory: {resolved.name}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


# ─── External processes ───────────────────────────────────────────────────────


def resolve_executable(names: Sequence[str], override: Path | None, *, tool: str) -> Path:
    """Locate an external executable.

    ``override`` (from settings) wins and must be an existing file. Otherwise each
    name in ``names`` is searched on ``PATH``.

    Raises:
        DependencyMissingError: if the binary cannot be found.
    """
    if override is not None:
        candidate = override.expanduser().resolve()
        if not candidate.is_file():
            raise DependencyMissingError(
                f"{tool} executable configured but not found: {candidate}",
                hint=f"Fix REVERSE3D_{tool.upper()} or unset it to search PATH.",
            )
        return candidate
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found).resolve()
    raise DependencyMissingError(
        f"{tool} was not found on PATH.",
        hint=f"Install {tool} and add it to PATH, or set REVERSE3D_{tool.upper()}.",
    )


def _scrubbed_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV_PATTERN.search(k)}
    if extra:
        env.update(extra)
    return env


def _check_batch_arguments(executable: Path, args: Sequence[str]) -> None:
    if sys.platform != "win32" or executable.suffix.lower() not in {".bat", ".cmd"}:
        return
    for arg in args:
        if _CMD_METACHARS.search(arg):
            raise SecurityViolationError(
                "Argument contains characters that are unsafe for a Windows batch launcher.",
                hint="Rename files/folders to avoid & | < > ^ % ! and quotes, "
                "or point REVERSE3D_COLMAP to the real .exe.",
            )


def run_command(
    executable: Path,
    args: Sequence[str | os.PathLike[str]],
    *,
    timeout_s: int,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    tool: str = "external tool",
    allow_nonzero: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run ``executable`` with ``args`` safely and return the completed process.

    * No shell is involved; every argument is passed verbatim.
    * The child inherits a scrubbed environment (no credential-like variables).
    * A hard timeout kills the process tree root on expiry.

    Args:
        allow_nonzero: return instead of raising on a non-zero exit code
            (used to read ``--help`` output of tools that exit with 1).

    Raises:
        ExternalToolError: on non-zero exit code (unless allowed) or timeout.
    """
    str_args = [os.fspath(a) for a in args]
    _check_batch_arguments(executable, str_args)
    cmd = [os.fspath(executable), *str_args]
    logger.debug("Running %s with %d arguments", executable.name, len(str_args))
    try:
        # Justification for B603: argument list, shell=False, executable resolved by
        # resolve_executable() and batch-file arguments checked above.
        proc = subprocess.run(  # nosec B603
            cmd,
            cwd=os.fspath(cwd) if cwd else None,
            env=_scrubbed_env(env),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalToolError(
            f"{tool} timed out after {timeout_s}s.",
            hint="Use a lower quality preset or raise REVERSE3D_TIMEOUT_S.",
        ) from exc
    except OSError as exc:
        raise ExternalToolError(f"Could not start {tool}: {exc.strerror}") from exc

    if proc.returncode != 0 and not allow_nonzero:
        tail = (proc.stderr or proc.stdout or "")[-_STDERR_TAIL_CHARS:]
        raise ExternalToolError(
            f"{tool} exited with code {proc.returncode}.",
            returncode=proc.returncode,
            stderr_tail=tail,
        )
    return proc
