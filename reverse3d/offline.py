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
    "facebook/sam-vit-base": "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
    "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M": "95ec8197b3f2fe7f747865c61ca556cf0768b2f7",
    # Studio's assistant: understands requests, Optiverse and the Core act (Apache-2.0).
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": (
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    ),
    # Studio's assistant listens: speech to text on this computer (Apache-2.0).
    "openai/whisper-base": "e37978b90ca9030d5170a5c07aadb050351a65bb",
}


#: Learned feature models COLMAP runs through ONNX (ALIKED, BSD-3; LightGlue,
#: Apache-2.0): (download URL, pinned SHA-256). Used to rescue hard photo sets.
COLMAP_MODELS: dict[str, tuple[str, str]] = {
    "aliked-n16rot.onnx": (
        "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
        "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    ),
    "aliked-lightglue.onnx": (
        "https://github.com/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
        "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    ),
}


def colmap_model_path(name: str) -> Path | None:
    """Local copy of a COLMAP ONNX model, if downloaded (filesystem check only)."""
    path = models_dir() / "colmap" / name
    return path if path.is_file() else None


def download_colmap_models() -> dict[str, str]:
    """Download the learned-feature models, verifying their pinned SHA-256."""
    import hashlib
    import urllib.request

    status: dict[str, str] = {}
    folder = models_dir() / "colmap"
    folder.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in COLMAP_MODELS.items():
        target = folder / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == sha:
            status[name] = "ready"
            continue
        try:
            with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310  # nosec B310 - pinned https URL, SHA-256 checked
                data = response.read()
        except Exception as exc:  # network errors come in many shapes
            status[name] = f"failed ({exc.__class__.__name__})"
            continue
        if hashlib.sha256(data).hexdigest() != sha:
            status[name] = "failed (checksum mismatch)"
            continue
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        status[name] = "downloaded"
    return status


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
    from reverse3d.assistant.brain import ALLOWED_ASSISTANTS
    from reverse3d.assistant.ears import ALLOWED_LISTENERS
    from reverse3d.reconstruction.depth_estimation import ALLOWED_MODELS
    from reverse3d.vision.classifier import ALLOWED_CLASSIFIERS
    from reverse3d.vision.segmenter import ALLOWED_SEGMENTERS

    return {
        **ALLOWED_MODELS,
        **ALLOWED_SEGMENTERS,
        **ALLOWED_CLASSIFIERS,
        **ALLOWED_ASSISTANTS,
        **ALLOWED_LISTENERS,
    }


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
            "The AI model is not on this computer and there is no internet connection.",
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
            f"Could not download the AI model ({exc.__class__.__name__}).",
            hint="Check your connection and run: reverse3d offline prepare",
        ) from exc


def prepare(settings: Settings, *, include_non_commercial: bool = False) -> dict[str, Any]:
    """Download the allowlisted AI models so reverse3d works without internet.

    Safe to run any time: models already on disk are skipped. It never raises
    for a missing optional stack; that is reported in the result instead.
    """
    from reverse3d.assistant.brain import ASSISTANT_MODEL
    from reverse3d.assistant.ears import SPEECH_MODEL
    from reverse3d.vision.classifier import DEFAULT_CLASSIFIER
    from reverse3d.vision.segmenter import DEFAULT_SEGMENTER

    # Depth (reliefs), segmentation (silhouettes), the subject classifier and
    # Studio's assistant, all commercial-friendly.
    models = [
        _default_model(),
        DEFAULT_SEGMENTER,
        DEFAULT_CLASSIFIER,
        ASSISTANT_MODEL,
        SPEECH_MODEL,
    ]
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

    if online or all(colmap_model_path(n) for n in COLMAP_MODELS):
        for name, state in download_colmap_models().items():
            results.append({"model": f"colmap/{name}", "status": state.split(" ")[0]})
    else:
        results += [
            {"model": f"colmap/{n}", "status": "missing", "detail": "No internet connection."}
            for n in COLMAP_MODELS
            if not colmap_model_path(n)
        ]
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
    from reverse3d.vision.classifier import DEFAULT_CLASSIFIER
    from reverse3d.vision.segmenter import DEFAULT_SEGMENTER

    depth_installed = bool(caps.get("depth_estimation"))
    model_on_disk = local_model_path(_default_model()) is not None
    # The local AI team: what each model does, and whether it is on this computer.
    ai_models = {
        "ai_subject": DEFAULT_CLASSIFIER,
        "ai_segmentation": DEFAULT_SEGMENTER,
        "ai_depth": _default_model(),
    }
    features_ready = all(colmap_model_path(n) for n in COLMAP_MODELS)
    on_disk = {key: local_model_path(model) is not None for key, model in ai_models.items()}
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
        **{
            key: {"available": depth_installed, "offline": depth_installed and on_disk[key]}
            for key in ai_models
        },
        "ai_features": {"available": True, "offline": features_ready},
    }
    missing = []
    names = {
        "ai_subject": "AI subject classifier",
        "ai_segmentation": "AI segmentation",
        "ai_depth": "AI depth",
    }
    if depth_installed and not model_on_disk:
        missing.append(
            "AI depth model not downloaded yet: run `reverse3d offline prepare` while online."
        )
    if not features_ready:
        missing.append(
            "AI feature models (ALIKED + LightGlue) not downloaded yet: "
            "run `reverse3d offline prepare` while online."
        )
    for key in ("ai_subject", "ai_segmentation"):
        if depth_installed and not on_disk[key]:
            missing.append(
                f"{names[key]} model not downloaded yet: "
                "run `reverse3d offline prepare` while online."
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
