"""Compute-device detection and resolution (CPU / NVIDIA CUDA / Apple MPS).

``Settings.device`` expresses a *preference* (auto | cpu | gpu). These helpers
turn it into concrete choices per backend, falling back to CPU when the
requested accelerator is not usable, so a GPU preference never breaks a job.
"""

from __future__ import annotations

import logging
import platform
import shutil
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from reverse3d.config import Settings
from reverse3d.errors import Reverse3DError
from reverse3d.security import run_command

logger = logging.getLogger(__name__)

TorchDevice = Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True)
class GpuInfo:
    """What accelerators this machine exposes."""

    cuda: bool
    name: str | None
    vram_mb: int | None
    driver: str | None
    mps: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=1)
def detect_gpu() -> GpuInfo:
    """Detect NVIDIA GPUs (via ``nvidia-smi``) and Apple Silicon (MPS). Cached."""
    mps = platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}
    smi = shutil.which("nvidia-smi")
    if not smi:
        return GpuInfo(cuda=False, name=None, vram_mb=None, driver=None, mps=mps)
    try:
        proc = run_command(
            Path(smi),
            ["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            timeout_s=15,
            tool="nvidia-smi",
        )
    except Reverse3DError:
        return GpuInfo(cuda=False, name=None, vram_mb=None, driver=None, mps=mps)
    first = (proc.stdout or "").strip().splitlines()[:1]
    if not first:
        return GpuInfo(cuda=False, name=None, vram_mb=None, driver=None, mps=mps)
    parts = [p.strip() for p in first[0].split(",")]
    vram = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    return GpuInfo(
        cuda=True,
        name=parts[0] or None,
        vram_mb=vram,
        driver=parts[2] if len(parts) > 2 else None,
        mps=mps,
    )


def wants_gpu(settings: Settings) -> bool:
    """Should GPU-capable external tools (COLMAP, Blender Cycles) use the GPU?"""
    if settings.device == "cpu":
        return False
    if settings.device == "gpu":
        return True
    return detect_gpu().cuda


def torch_device(settings: Settings) -> TorchDevice:
    """Best PyTorch device for the preference; CPU if torch/accelerator is missing."""
    if settings.device == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    if settings.device == "gpu":
        logger.warning("GPU requested but PyTorch has no CUDA/MPS support; using CPU")
    return "cpu"


def apply_threads(settings: Settings) -> None:
    """Apply the CPU thread budget to in-process libraries (OpenCV; PyTorch lazily)."""
    if settings.threads <= 0:
        return
    import cv2

    cv2.setNumThreads(settings.threads)


def hardware_report(settings: Settings) -> dict[str, object]:
    """JSON-friendly summary used by doctor / environment_check / configure."""
    gpu = detect_gpu()
    return {
        "device_preference": settings.device,
        "external_tools_use_gpu": wants_gpu(settings),
        "gpu": gpu.to_dict(),
        "threads": settings.threads or "auto",
    }
