"""Meshes to STEP for CAD programs: a solid when closed, a surface when not.

Validated by hand with OpenCascade (the kernel FreeCAD reads STEP with): a box
gives one valid solid of volume 1000 and a 327 680-face sphere one valid solid
with the mesh's volume. These tests check the file itself.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from reverse3d.cad_export import _num, save_step, step_text
from reverse3d.mesh_io import save_mesh


def _entities(text: str) -> dict[int, str]:
    return {int(m.group(1)): m.group(2) for m in re.finditer(r"^#(\d+)=(.*);$", text, re.M)}


def test_a_closed_mesh_is_a_faceted_solid_in_millimetres() -> None:
    box = trimesh.creation.box(extents=(20, 10, 5))
    text = step_text(box, "pieza")
    assert text.startswith("ISO-10303-21;") and text.rstrip().endswith("END-ISO-10303-21;")
    assert "AUTOMOTIVE_DESIGN" in text and "SI_UNIT(.MILLI.,.METRE.)" in text
    assert "CLOSED_SHELL" in text and "FACETED_BREP_SHAPE_REPRESENTATION('pieza'" in text
    ents = _entities(text)
    assert sum(v.startswith("FACE_SURFACE") for v in ents.values()) == len(box.faces)
    assert sum(v.startswith("CARTESIAN_POINT") for v in ents.values()) == len(box.vertices) + 1
    # every reference points to an entity that exists
    refs = {int(r) for v in ents.values() for r in re.findall(r"#(\d+)", v)}
    assert refs <= set(ents)


def test_a_mesh_with_holes_is_a_surface_not_a_false_solid() -> None:
    mesh = trimesh.creation.box(extents=(10, 10, 10))
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    text = step_text(mesh)
    assert "OPEN_SHELL" in text and "SHELL_BASED_SURFACE_MODEL" in text
    assert "CLOSED_SHELL" not in text and "FACETED_BREP(" not in text


def test_numbers_are_step_reals() -> None:
    assert _num(1) == "1." and _num(0.5) == "0.5" and _num(-2) == "-2."
    assert _num(1e-7) == "1.E-7" and _num(12345678) == "1.23457E7"


def test_saving_by_extension_and_refusing_empty(tmp_path: Path) -> None:
    box = trimesh.creation.box(extents=(1, 1, 1))
    out = save_mesh(box, tmp_path / "pieza.stp")
    assert out.read_text(encoding="ascii").startswith("ISO-10303-21;")
    with pytest.raises(ValueError):
        step_text(trimesh.Trimesh())
    assert save_step(box, tmp_path / "otra.step").stat().st_size > 1000
