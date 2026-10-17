from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

pytest.importorskip("xatlas")
sys.path.insert(0, str(Path(__file__).parent))

from test_silhouette import HEIGHT, RADIUS, _scene

from reverse3d.reconstruction.texture import bake_texture


def test_texture_is_baked_from_the_photos(tmp_path: Path) -> None:
    images, text = _scene(tmp_path)
    mesh = trimesh.creation.cylinder(radius=RADIUS * 0.98, height=HEIGHT * 0.98, sections=48)
    mesh = mesh.subdivide()
    mesh.apply_translation([0, 0, HEIGHT / 2])
    # The model is usually moved after reconstruction: give the baker the way back.
    moved = mesh.copy()
    moved.apply_translation([5.0, 0, 0])
    to_world = trimesh.transformations.translation_matrix([-5.0, 0, 0])
    textured = bake_texture(moved, images, text, to_world=to_world, size=256)
    image = np.asarray(textured.visual.material.baseColorTexture)
    assert image.shape == (256, 256, 3)
    assert textured.visual.uv.shape == (len(textured.vertices), 2)
    assert np.allclose(textured.bounds, moved.bounds, atol=1e-6)  # geometry unchanged
    # The cylinder is white in every photo: most of the texture is bright.
    assert np.median(image) > 200
    glb = tmp_path / "model.glb"
    textured.export(glb)
    assert glb.stat().st_size > 1000
