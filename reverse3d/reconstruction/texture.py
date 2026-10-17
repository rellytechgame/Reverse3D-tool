"""Photo textures: a UV-mapped texture baked from the photos (detail beyond vertices).

Per-vertex colour can only be as detailed as the mesh. A texture is not: every
texel gets its own colour, so logos, lettering and fine patterns survive on a
light mesh. Steps:

1. unwrap the mesh into a UV atlas (xatlas, MIT);
2. rasterise each triangle into the atlas to know the 3D point and normal of
   every texel;
3. colour each texel from the photos that see it head-on and unoccluded
   (weights favour frontal views, so the result stays sharp);
4. bleed colour into the empty gutters so mip-mapping shows no seams.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from reverse3d.errors import DependencyMissingError, ProcessingError
from reverse3d.reconstruction.silhouette import _project, _View, read_colmap_text

logger = logging.getLogger(__name__)


def _unwrap(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import xatlas
    except ImportError as exc:  # pragma: no cover - core dependency
        raise DependencyMissingError(
            "xatlas is needed for textures.", hint="pip install xatlas"
        ) from exc
    vmapping, indices, uvs = xatlas.parametrize(
        np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.uint32)
    )
    return np.asarray(vmapping), np.asarray(indices, np.int64), np.asarray(uvs, np.float64)


def _texel_points(
    verts: np.ndarray, faces: np.ndarray, uvs: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For every covered texel: (row, col), its 3D point and the face normal."""
    face_id = np.full((size, size), -1, np.int32)
    px = uvs * (size - 1)
    px[:, 1] = (size - 1) - px[:, 1]  # glTF v goes up, image rows go down
    for f, tri in enumerate(faces):
        cv2.fillConvexPoly(face_id, np.round(px[tri]).astype(np.int32), f)
    rows, cols = np.nonzero(face_id >= 0)
    fids = face_id[rows, cols]
    tri = faces[fids]
    a, b, c = px[tri[:, 0]], px[tri[:, 1]], px[tri[:, 2]]
    p = np.column_stack([cols, rows]).astype(np.float64)
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(1)
    d01 = (v0 * v1).sum(1)
    d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1)
    d21 = (v2 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom[np.abs(denom) < 1e-12] = 1e-12
    w1 = np.clip((d11 * d20 - d01 * d21) / denom, 0, 1)
    w2 = np.clip((d00 * d21 - d01 * d20) / denom, 0, 1)
    w0 = np.clip(1 - w1 - w2, 0, 1)
    points = (
        w0[:, None] * verts[tri[:, 0]]
        + w1[:, None] * verts[tri[:, 1]]
        + w2[:, None] * verts[tri[:, 2]]
    )
    e1 = verts[tri[:, 1]] - verts[tri[:, 0]]
    e2 = verts[tri[:, 2]] - verts[tri[:, 0]]
    normals = np.cross(e1, e2)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return np.column_stack([rows, cols]), points, normals


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear colour samples at sub-pixel positions (no size limits, unlike remap)."""
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    x1 = np.minimum(x0 + 1, img.shape[1] - 1)
    y1 = np.minimum(y0 + 1, img.shape[0] - 1)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bottom = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return np.asarray(top * (1 - fy) + bottom * fy, np.float64)


def _bake(
    points: np.ndarray,
    normals: np.ndarray,
    views: list[_View],
    images: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Colour per texel and whether any photo saw it."""
    total = np.zeros((len(points), 3))
    weight = np.zeros(len(points))
    for (img, scale), view in zip(images, views, strict=True):
        center = -view.rotation.T @ view.translation
        to_cam = center - points
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, to_cam)
        uv, z = _project(points, view)
        u, v = uv[:, 0] * scale, uv[:, 1] * scale
        h, w = img.shape[:2]
        ok = (facing > 0.15) & (z > 0) & (u >= 0) & (v >= 0) & (u < w - 1) & (v < h - 1)
        if not ok.any():
            continue
        cell = 3
        cu = (u[ok] / cell).astype(np.int64)
        cv_ = (v[ok] / cell).astype(np.int64)
        zbuf = np.full((h // cell + 1, w // cell + 1), np.inf)
        np.minimum.at(zbuf, (cv_, cu), z[ok])
        span = float(np.ptp(z[ok])) or 1.0
        visible = z[ok] <= zbuf[cv_, cu] + 0.02 * span
        idx = np.flatnonzero(ok)[visible]
        sample = _bilinear(img, u[idx], v[idx])[:, ::-1]  # BGR -> RGB
        wgt = facing[idx] ** 6  # strongly prefer frontal views: sharp, not averaged mush
        total[idx] += sample * wgt[:, None]
        weight[idx] += wgt
    seen = weight > 0
    colours = np.zeros((len(points), 3))
    colours[seen] = total[seen] / weight[seen, None]
    return colours, seen


def bake_texture(
    mesh: trimesh.Trimesh,
    image_dir: Path,
    model_text_dir: Path,
    *,
    to_world: np.ndarray | None = None,
    size: int = 2048,
    max_side: int = 2048,
) -> trimesh.Trimesh:
    """A copy of ``mesh`` with UVs and a texture baked from the COLMAP photos.

    ``to_world`` maps the mesh's coordinates to the COLMAP world (4x4), when the
    mesh was moved or scaled after reconstruction.

    Raises:
        ProcessingError: photos cannot be read or nothing could be baked.
    """
    from PIL import Image

    views, _ = read_colmap_text(model_text_dir)
    images: list[tuple[np.ndarray, float]] = []
    for view in views:
        bgr = cv2.imread(str(image_dir / view.name))
        if bgr is None:
            raise ProcessingError(f"Could not read {view.name} for the texture.")
        h, w = bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            bgr = cv2.resize(
                bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA
            )
        images.append((bgr, scale))

    vmapping, faces, uvs = _unwrap(mesh)
    verts_model = np.asarray(mesh.vertices)[vmapping]
    verts_world = verts_model
    if to_world is not None:
        verts_world = trimesh.transform_points(verts_model, to_world)
    texel_rc, points, normals = _texel_points(verts_world, faces, uvs, size)
    colours, seen = _bake(points, normals, views, images)
    if not seen.any():
        raise ProcessingError("No photo sees the model: the texture could not be baked.")

    texture: np.ndarray = np.zeros((size, size, 3), np.uint8)
    covered = np.zeros((size, size), np.uint8)
    fill = np.median(colours[seen], axis=0)
    colours[~seen] = fill
    rr, cc = texel_rc[:, 0], texel_rc[:, 1]
    texture[rr, cc] = np.clip(colours, 0, 255).astype(np.uint8)
    covered[rr, cc] = 255
    # Isolated wrong texels (a grazing view catching background) look like salt and
    # pepper; a 3x3 median removes them and keeps lettering and edges.
    texture = cv2.medianBlur(texture, 3)
    # Bleed colours into the gutters between UV islands (no seams when filtered).
    texture = cv2.inpaint(texture, 255 - covered, 3, cv2.INPAINT_TELEA) if size <= 2048 else texture

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(texture), metallicFactor=0.0, roughnessFactor=0.7
    )
    textured = trimesh.Trimesh(verts_model, faces, process=False)
    visual: Any = trimesh.visual.TextureVisuals(uv=uvs, material=material)
    textured.visual = visual
    logger.info("Texture baked: %dx%d from %d photos", size, size, len(views))
    return textured
