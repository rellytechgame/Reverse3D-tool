"""Monocular depth estimation (feed-forward, single image) and depth → mesh.

Inference uses Hugging Face ``transformers`` (optional ``[depth]`` extra).

Supply-chain controls (OWASP LLM03 / ASI04):

* Only models in :data:`ALLOWED_MODELS` can be loaded — an agent cannot make the
  toolkit download arbitrary repositories.
* ``trust_remote_code`` is never enabled, so no repository Python code runs.
* Weights are loaded with ``use_safetensors=True``: pickle-based ``.bin``
  checkpoints (arbitrary code execution on load) are refused.
* Models are loaded from ``~/.reverse3d/models`` (see :mod:`reverse3d.offline`):
  once downloaded, depth estimation works without internet.

The geometric part (:func:`depth_to_mesh`) is pure NumPy/trimesh and works
without the extra, e.g. with depth maps from other sources.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import trimesh

from reverse3d.config import Settings
from reverse3d.errors import DependencyMissingError, InputValidationError, ProcessingError
from reverse3d.imaging import load_image, save_image
from reverse3d.mesh_io import save_mesh
from reverse3d.security import prepare_output_dir, prepare_output_file

logger = logging.getLogger(__name__)

DepthMode = Literal["disparity", "depth"]

#: model id → (output semantics, license note). Depth Anything V2 predicts
#: relative inverse depth (disparity).
ALLOWED_MODELS: dict[str, tuple[DepthMode, str]] = {
    "depth-anything/Depth-Anything-V2-Small-hf": ("disparity", "Apache-2.0"),
    "depth-anything/Depth-Anything-V2-Base-hf": ("disparity", "CC-BY-NC-4.0 (non-commercial)"),
    "depth-anything/Depth-Anything-V2-Large-hf": ("disparity", "CC-BY-NC-4.0 (non-commercial)"),
}
DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass
class DepthResult:
    """Outputs of :func:`estimate_depth`."""

    depth_png_path: str
    depth_npy_path: str
    mesh_path: str | None
    model: str
    model_license: str
    width: int
    height: int
    mesh_faces: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _load_pipeline(model: str, device: str, settings: Settings | None = None) -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise DependencyMissingError(
            "transformers/torch are required for depth estimation.",
            hint='pip install "reverse3d[depth]"',
        ) from exc
    from reverse3d.offline import force_offline_env, is_offline, resolve_model

    settings = settings or Settings.load()
    local_path = resolve_model(model, settings)
    if is_offline(settings):
        force_offline_env()
    try:  # keep CLI/agent output clean: no download/loading progress bars
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except ImportError:  # pragma: no cover - very old transformers
        pass
    return pipeline(
        task="depth-estimation",
        model=str(local_path),
        device=device,
        trust_remote_code=False,
        model_kwargs={"use_safetensors": True},
    )


def predict_depth(
    rgb: np.ndarray,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    settings: Settings | None = None,
) -> np.ndarray:
    """Run the depth model on an RGB uint8 image and return a float32 map of the
    same height/width (relative units, semantics per :data:`ALLOWED_MODELS`)."""
    if model not in ALLOWED_MODELS:
        raise InputValidationError(
            f"Model '{model}' is not in the allowlist.",
            hint=f"Allowed: {', '.join(ALLOWED_MODELS)}",
        )
    if device not in {"cpu", "cuda", "mps"}:
        raise InputValidationError("device must be cpu, cuda or mps.")
    from PIL import Image

    pipe = _load_pipeline(model, device, settings)
    output = pipe(Image.fromarray(rgb))
    predicted = output["predicted_depth"]
    depth = np.asarray(predicted.squeeze().detach().cpu().numpy(), dtype=np.float32)
    h, w = rgb.shape[:2]
    if depth.shape != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    return depth


def _normalize(values: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(values, [1, 99])
    if hi - lo < 1e-9:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def depth_to_mesh(
    depth: np.ndarray,
    rgb: np.ndarray | None = None,
    *,
    mode: DepthMode = "disparity",
    fov_deg: float = 60.0,
    near: float = 1.0,
    far: float = 5.0,
    stride: int = 2,
    max_depth_jump: float = 0.08,
    mask: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Back-project a (relative) depth map into a triangulated height-field mesh.

    Relative predictions are mapped to the metric-like range ``[near, far]``.
    Triangles spanning a depth discontinuity larger than ``max_depth_jump``
    (relative to the range) are dropped to avoid "rubber sheet" artifacts
    between foreground and background. Camera looks down -Z (OpenGL convention).

    Args:
        depth: HxW array.
        rgb: optional HxWx3 uint8 colors (RGB order).
        mask: optional HxW boolean/uint8 array; only masked pixels are meshed.
    """
    if depth.ndim != 2:
        raise InputValidationError("depth must be a 2D array.")
    if not 10 <= fov_deg <= 170:
        raise InputValidationError("fov_deg must be in [10, 170].")
    if not 0 < near < far:
        raise InputValidationError("Require 0 < near < far.")
    if not 1 <= stride <= 32:
        raise InputValidationError("stride must be in [1, 32].")

    d = _normalize(depth[::stride, ::stride].astype(np.float64))
    if mode == "disparity":
        # Interpolate in inverse depth: disparity 1 → near, 0 → far.
        z = 1.0 / (d * (1.0 / near - 1.0 / far) + 1.0 / far)
    else:
        z = near + d * (far - near)

    h, w = z.shape
    full_w = depth.shape[1]
    f = (full_w / 2.0) / math.tan(math.radians(fov_deg) / 2.0) / stride
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    x = (u - w / 2.0) * z / f
    y = -(v - h / 2.0) * z / f
    vertices = np.stack([x, y, -z], axis=-1).reshape(-1, 3)

    idx = np.arange(h * w).reshape(h, w)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, e = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, e], 1)])

    zf = z.ravel()[faces]
    keep = (zf.max(axis=1) - zf.min(axis=1)) <= max_depth_jump * (far - near)
    if mask is not None:
        m = np.asarray(mask)[::stride, ::stride].astype(bool).ravel()
        keep &= m[faces].all(axis=1)
    faces = faces[keep]
    if len(faces) == 0:
        raise InputValidationError("No triangles left after filtering; relax max_depth_jump.")

    colors = None
    if rgb is not None:
        small = rgb[::stride, ::stride][:h, :w]
        colors = np.concatenate(
            [small.reshape(-1, 3), np.full((h * w, 1), 255, dtype=np.uint8)], axis=1
        )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def estimate_depth(
    image_path: str | Path,
    output_dir: str | Path,
    settings: Settings,
    *,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    make_mesh: bool = True,
    fov_deg: float = 60.0,
    stride: int = 2,
    mask_path: str | Path | None = None,
    overwrite: bool = False,
) -> DepthResult:
    """Estimate depth for one image and optionally build a 2.5D mesh.

    Outputs in ``output_dir``: ``<stem>_depth.png`` (16-bit visualisation),
    ``<stem>_depth.npy`` (raw float32) and ``<stem>_depth_mesh.ply``.
    """
    bgr = load_image(image_path, settings)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = None
    if mask_path is not None:
        mask = load_image(mask_path, settings, flags=cv2.IMREAD_GRAYSCALE) > 127
        if mask.shape != rgb.shape[:2]:
            raise InputValidationError("Mask size must match the image size.")

    depth = predict_depth(rgb, model=model, device=device, settings=settings)
    out_dir = prepare_output_dir(output_dir)
    stem = Path(image_path).stem

    png = save_image(
        (_normalize(depth) * 65535).astype(np.uint16),
        out_dir / f"{stem}_depth.png",
        overwrite=overwrite,
    )
    npy = prepare_output_file(
        out_dir / f"{stem}_depth.npy", allowed_extensions={".npy"}, overwrite=overwrite
    )
    np.save(npy, depth.astype(np.float32), allow_pickle=False)

    mesh_path: str | None = None
    faces: int | None = None
    if make_mesh:
        mode, _ = ALLOWED_MODELS[model]
        mesh = depth_to_mesh(depth, rgb, mode=mode, fov_deg=fov_deg, stride=stride, mask=mask)
        mesh_path = str(save_mesh(mesh, out_dir / f"{stem}_depth_mesh.ply", overwrite=overwrite))
        faces = len(mesh.faces)

    h, w = depth.shape
    return DepthResult(
        depth_png_path=str(png),
        depth_npy_path=str(npy),
        mesh_path=mesh_path,
        model=model,
        model_license=ALLOWED_MODELS[model][1],
        width=w,
        height=h,
        mesh_faces=faces,
    )


def height_above_support(mask: np.ndarray, disparity: np.ndarray) -> np.ndarray:
    """Disparity minus the plane fitted to the surface around the object.

    Photos look down at an angle, so the desk itself gets nearer towards the
    bottom of the frame; removing that plane leaves how much each pixel stands
    out from the surface the object rests on.
    """
    h, w = disparity.shape
    ring = cv2.dilate(mask, np.ones((41, 41), np.uint8)) > 0
    around = ring & (mask == 0)
    if around.sum() < 200:
        around = mask == 0
    ys, xs = np.nonzero(around)
    step = max(1, len(xs) // 20000)
    xs, ys = xs[::step], ys[::step]
    design = np.column_stack([xs, ys, np.ones_like(xs)]).astype(np.float64)
    coeffs, *_ = np.linalg.lstsq(design, disparity[ys, xs].astype(np.float64), rcond=None)
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    plane = coeffs[0] * grid_x + coeffs[1] * grid_y + coeffs[2]
    return disparity - plane


def object_solid(
    disparity: np.ndarray,
    mask: np.ndarray,
    *,
    stride: int = 4,
    height_ratio: float = 0.35,
    smooth_px: int = 9,
) -> trimesh.Trimesh:
    """Closed, printable solid of the object in a photo: smoothed height above
    the surface it rests on, a flat base and side walls.

    ``height_ratio`` sets the tallest point relative to the object's longest
    side (single-photo depth has no scale; 0.35 suits most hand-held objects).

    Raises:
        ProcessingError: the mask is empty or the object has no height.
    """
    inside = mask > 0
    if inside.sum() < 100:
        raise ProcessingError("The object mask is empty.")
    k = smooth_px | 1
    height = height_above_support(np.where(inside, 255, 0).astype(np.uint8), disparity)
    weights = inside.astype(np.float32)
    num = cv2.GaussianBlur(np.where(inside, height, 0).astype(np.float32), (k, k), 0)
    den = cv2.GaussianBlur(weights, (k, k), 0)
    height = np.where(inside, num / np.maximum(den, 1e-6), 0)
    # Smooth outline: no jagged pixel stairs on the side walls.
    outline = cv2.GaussianBlur(weights * 255, (2 * k + 1, 2 * k + 1), 0) > 127
    h = height[::stride, ::stride]
    keep = outline[::stride, ::stride]
    if keep.sum() < 16:
        raise ProcessingError("The object is too small in the photo.")
    h = np.clip(h - np.percentile(h[keep], 2), 0, None)
    peak = float(h[keep].max())
    if peak <= 0:
        raise ProcessingError("The object shows no height above its surface.")
    ys, xs = np.nonzero(keep)
    extent = float(max(np.ptp(xs), np.ptp(ys)) + 1)
    z = h / peak * height_ratio * extent
    # Round the outline: height eases to a low rim instead of a vertical cliff.
    rim = cv2.distanceTransform(keep.astype(np.uint8), cv2.DIST_L2, 5)
    z = z * (0.15 + 0.85 * np.sqrt(np.clip(rim / (0.08 * extent), 0.0, 1.0)))

    index = -np.ones(keep.shape, dtype=np.int64)
    index[keep] = np.arange(int(keep.sum()))
    # A thin base slab keeps the top and bottom apart everywhere (closed solid).
    base = 0.03 * extent
    top = np.column_stack([xs, -ys, z[ys, xs] + base]).astype(np.float64)
    bottom = top.copy()
    bottom[:, 2] = 0.0
    a, b = index[:-1, :-1], index[:-1, 1:]
    c, d_ = index[1:, :-1], index[1:, 1:]
    ok = (a >= 0) & (b >= 0) & (c >= 0) & (d_ >= 0)
    a, b, c, d_ = a[ok], b[ok], c[ok], d_[ok]
    top_faces = np.vstack([np.column_stack([a, c, b]), np.column_stack([b, c, d_])])
    n = len(top)
    bottom_faces = top_faces[:, ::-1] + n
    edges = np.vstack([top_faces[:, [0, 1]], top_faces[:, [1, 2]], top_faces[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    border = edges[counts[inverse.ravel()] == 1]
    walls = np.vstack(
        [
            np.column_stack([border[:, 1], border[:, 0], border[:, 0] + n]),
            np.column_stack([border[:, 1], border[:, 0] + n, border[:, 1] + n]),
        ]
    )
    solid = trimesh.Trimesh(
        np.vstack([top, bottom]), np.vstack([top_faces, bottom_faces, walls]), process=True
    )
    solid.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(solid)
    return solid
