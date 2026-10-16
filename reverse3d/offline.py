"""Offline mode: keep working without internet and switch to it automatically.

Everything reverse3d computes runs locally (OpenCV, COLMAP, Open3D, trimesh,
Blender). The only part that ever needs the network is the AI depth model,
downloaded from Hugging Face the first time it is used. This module removes
that dependency:

* :func:`has_internet` is a cheap, cached TCP probe (no HTTP, no data sent).
* Models live in ``~/.reverse3d/models``. :func:`prepare` downloads the
  allowlisted ones in advance (``reverse3d offline prepare``; the installer runs
  it too), fetching only ``*.safetensors`` weights and small config files.
* :func:`resolve_model` always prefers the local copy, so a prepared machine
  never touches the network, and fails with a clear hint when a model is
  missing and there is no connection.
* :func:`offline_status` tells the user, in one call, what works offline.

``Settings.offline`` (``REVERSE3D_OFFLINE`` / ``reverse3d config set offline``):
``auto`` (default) detects the connection, ``on`` never uses the network,
``off`` skips the probe and assumes the network is available.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError, InputValidationError, Reverse3DError
from reverse3d.userconfig import home_dir

logger = logging.getLogger(__name__)

#: Hosts probed to decide whether the machine is online. Hugging Face is the
#: only service reverse3d downloads from at run time; 1.1.1.1 covers networks
#: where it is blocked but the internet works.
_PROBE_HOSTS: tuple[tuple[str, int], ...] = (("huggingface.co", 443), ("1.1.1.1", 443))
_PROBE_TIMEOUT_S = 1.5
_CACHE_TTL_S = 60.0
_cache: tuple[float, bool] | None = None

#: Only these files are downloaded: safetensors weights and small configs.
#: Pickled ``*.bin`` checkpoints (code execution on load) are never fetched.
MODEL_FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt"]

#: Pinned Hugging Face commit of every allowlisted model: only this exact
#: revision is downloaded or accepted from disk, so a changed upstream
#: repository cannot alter what runs (bump together with a review).
MODEL_REVISIONS: dict[str, str] = {
    "depth-anything/Depth-Anything-V2-Small-hf": "5426e4f0f36572d16453bbda7a8389317b1bef99",
    "depth-anything/Depth-Anything-V2-Base-hf": "b1958afc87fb45a9e3746cb387596094de553ed8",
    "depth-anything/Depth-Anything-V2-Large-hf": "7581137eff8d4e94f6e796d3baea0e9fa79b22d2",
}


def models_dir() -> Path:
    """Folder that holds the downloaded AI models (``~/.reverse3d/models``)."""
    return home_dir() / "models"


def _reachable(hosts: tuple[tuple[str, int], ...]) -> bool:
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_S):
                return True
        except OSError:
            continue
    return False


def _probe() -> bool:
    return _reachable(_PROBE_HOSTS)


def has_internet(*, refresh: bool = False) -> bool:
    """Return True if the internet is reachable. The answer is cached for a minute."""
    global _cache
    now = time.monotonic()
    if not refresh and _cache is not None and now - _cache[0] < _CACHE_TTL_S:
        return _cache[1]
    online = _probe()
    _cache = (now, online)
    if not online:
        logger.info("No internet connection detected: working offline.")
    return online


def is_offline(settings: Settings) -> bool:
    """Whether reverse3d must avoid the network right now."""
    if settings.offline == "on":
        return True
    if settings.offline == "off":
        return False
    return not has_internet()


def _allowed_models() -> dict[str, tuple[str, str]]:
    from reverse3d.reconstruction.depth_estimation import ALLOWED_MODELS

    return dict(ALLOWED_MODELS)


def _default_model() -> str:
    from reverse3d.reconstruction.depth_estimation import DEFAULT_MODEL

    return DEFAULT_MODEL


def _cache_folder(model_id: str) -> Path:
    return models_dir() / ("models--" + model_id.replace("/", "--"))


def local_model_path(model_id: str) -> Path | None:
    """Return the local snapshot folder of ``model_id`` if its weights are on disk.

    Pure filesystem check: it never imports Hugging Face nor touches the network.
    """
    revision = MODEL_REVISIONS.get(model_id)
    if revision is None:
        return None
    snapshot = _cache_folder(model_id) / "snapshots" / revision
    return snapshot if snapshot.is_dir() and any(snapshot.glob("*.safetensors")) else None


def _folder_mb(path: Path) -> float:
    total = 0
    for file in path.rglob("*"):
        try:
            if file.is_file():
                total += file.stat().st_size  # resolves the symlinked blobs
        except OSError:
            continue
    return round(total / (1024 * 1024), 1)


def _depth_stack_installed() -> bool:
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "transformers", "huggingface_hub")
    )


def _download(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        model_id,
        revision=MODEL_REVISIONS[model_id],
        cache_dir=str(models_dir()),
        allow_patterns=MODEL_FILE_PATTERNS,
    )
    return Path(path)


def resolve_model(model_id: str, settings: Settings) -> Path:
    """Return a local folder with the files of an allowlisted model.

    Uses the local copy when there is one (no network at all). Otherwise it
    downloads the model if the machine is online.

    Raises:
        InputValidationError: the model is not in the allowlist.
        DependencyMissingError: the model is not on disk and cannot be downloaded
            (offline, or the download failed).
    """
    if model_id not in _allowed_models() or model_id not in MODEL_REVISIONS:
        raise InputValidationError(
            f"Model '{model_id}' is not in the allowlist.",
            hint=f"Allowed: {', '.join(_allowed_models())}",
        )
    local = local_model_path(model_id)
    if local is not None:
        return local
    if is_offline(settings):
        raise DependencyMissingError(
            "The AI depth model is not on this computer and there is no internet connection.",
            hint="Connect to the internet once and run: reverse3d offline prepare",
        )
    try:
        return _download(model_id)
    except ImportError as exc:
        raise DependencyMissingError(
            "huggingface_hub is required to download AI models.",
            hint='pip install "reverse3d[depth]"',
        ) from exc
    except Exception as exc:  # network errors come in many shapes; report them plainly
        has_internet(refresh=True)
        raise DependencyMissingError(
            f"Could not download the AI depth model ({exc.__class__.__name__}).",
            hint="Check your connection and run: reverse3d offline prepare",
        ) from exc


def prepare(settings: Settings, *, include_non_commercial: bool = False) -> dict[str, Any]:
    """Download the allowlisted AI models so reverse3d works without internet.

    Safe to run any time: models already on disk are skipped. It never raises
    for a missing optional stack; that is reported in the result instead.
    """
    models = [_default_model()]
    if include_non_commercial:
        models += [m for m in _allowed_models() if m not in models]

    if not _depth_stack_installed():
        return {
            "ready": True,
            "models_dir": str(models_dir()),
            "models": [
                {"model": m, "status": "skipped", "detail": "AI depth extra not installed"}
                for m in models
            ],
            "message": "Nothing to download: AI depth is not installed and everything else "
            "already works offline.",
        }

    results: list[dict[str, Any]] = []
    online = not is_offline(settings)
    for model_id in models:
        local = local_model_path(model_id)
        if local is not None:
            results.append(
                {
                    "model": model_id,
                    "status": "ready",
                    "path": str(local),
                    "size_mb": _folder_mb(local),
                }
            )
            continue
        if not online:
            results.append(
                {
                    "model": model_id,
                    "status": "missing",
                    "detail": "No internet connection: connect and run this again.",
                }
            )
            continue
        try:
            path = resolve_model(model_id, settings)
        except Reverse3DError as exc:
            results.append({"model": model_id, "status": "failed", "detail": exc.message})
            continue
        results.append(
            {
                "model": model_id,
                "status": "downloaded",
                "path": str(path),
                "size_mb": _folder_mb(path),
            }
        )

    ready = all(r["status"] in ("ready", "downloaded") for r in results)
    return {
        "ready": ready,
        "models_dir": str(models_dir()),
        "models": results,
        "message": (
            "Ready: reverse3d now works without internet."
            if ready
            else "Some models are missing: connect to the internet and run "
            "`reverse3d offline prepare`."
        ),
    }


def offline_status(settings: Settings) -> dict[str, Any]:
    """Report what works right now and what would work without internet."""
    from reverse3d.doctor import environment_report

    env = environment_report(settings, include_offline=False)
    caps: dict[str, bool] = env["capabilities"]  # type: ignore[assignment]
    depth_installed = bool(caps.get("depth_estimation"))
    model_on_disk = local_model_path(_default_model()) is not None
    internet = None if settings.offline != "auto" else has_internet()

    components = {
        "measure_repair_cad": {"available": True, "offline": True},
        "plan_to_part": {"available": True, "offline": True},
        "photogrammetry": {"available": caps.get("photogrammetry", False), "offline": True},
        "dense_meshing": {"available": caps.get("dense_meshing", False), "offline": True},
        "render": {
            "available": caps.get("render_blender", False) or caps.get("render_pyvista", False),
            "offline": True,
        },
        "ai_depth": {"available": depth_installed, "offline": depth_installed and model_on_disk},
    }
    missing = []
    if depth_installed and not model_on_disk:
        missing.append(
            "AI depth model not downloaded yet: run `reverse3d offline prepare` while online."
        )
    working_offline = is_offline(settings)
    return {
        "mode": settings.offline,
        "internet": internet,
        "working_offline": working_offline,
        "ready_for_offline": not missing,
        "models_dir": str(models_dir()),
        "components": components,
        "missing": missing,
        "message": ("Everything installed works without internet." if not missing else missing[0]),
    }


def force_offline_env() -> None:
    """Tell Hugging Face libraries not to reach the network in this process."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
