"""Meshes to industrial CAD exchange formats.

STEP (ISO 10303-21, AP214), written directly from the mesh, no CAD kernel needed:

* a watertight mesh becomes a faceted solid (FACETED_BREP in a closed shell),
  which SolidWorks, Fusion, Inventor, CATIA or FreeCAD open as a solid;
* a mesh with holes becomes a surface model (OPEN_SHELL): calling it a solid
  would be false, and CAD programs would reject it.

Units are millimetres (reverse3d's real-size models are in millimetres).
Vertices are shared between faces, so the file grows with the mesh, not with
three copies of every corner.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import trimesh

from reverse3d.security import prepare_output_file

STEP_EXTENSIONS = frozenset({".step", ".stp"})


def _num(x: float) -> str:
    """A STEP real: always with a decimal point (1 -> '1.', 0.5 -> '0.5')."""
    text = f"{x:.6g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return (mantissa if "." in mantissa else mantissa + ".") + "E" + str(int(exponent))
    return text if "." in text else text + "."


def _text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def step_text(mesh: trimesh.Trimesh, name: str = "reverse3d") -> str:
    """The STEP file for ``mesh`` (a solid if it is watertight, else a surface)."""
    if len(mesh.faces) == 0:
        raise ValueError("The mesh has no faces.")
    name = _text(name)
    solid = bool(mesh.is_watertight)
    lines: list[str] = []
    counter = 0

    def add(entity: str) -> int:
        nonlocal counter
        counter += 1
        lines.append(f"#{counter}={entity};")
        return counter

    app = add("APPLICATION_CONTEXT('automotive design')")
    add(
        f"APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#{app})"
    )
    pctx = add(f"PRODUCT_CONTEXT('',#{app},'mechanical')")
    product = add(f"PRODUCT('{name}','{name}','',(#{pctx}))")
    add(f"PRODUCT_RELATED_PRODUCT_CATEGORY('part',$,(#{product}))")
    formation = add(f"PRODUCT_DEFINITION_FORMATION('','',#{product})")
    dctx = add(f"PRODUCT_DEFINITION_CONTEXT('part definition',#{app},'design')")
    definition = add(f"PRODUCT_DEFINITION('design','',#{formation},#{dctx})")
    shape = add(f"PRODUCT_DEFINITION_SHAPE('','',#{definition})")
    length = add("( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )")
    angle = add("( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) )")
    solid_angle = add("( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() )")
    tolerance = add(
        f"UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-06),#{length},'distance_accuracy_value','confusion accuracy')"
    )
    context = add(
        f"( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#{tolerance})) "
        f"GLOBAL_UNIT_ASSIGNED_CONTEXT((#{length},#{angle},#{solid_angle})) REPRESENTATION_CONTEXT('','') )"
    )
    origin = add("CARTESIAN_POINT('',(0.,0.,0.))")
    z = add("DIRECTION('',(0.,0.,1.))")
    x = add("DIRECTION('',(1.,0.,0.))")
    axes = add(f"AXIS2_PLACEMENT_3D('',#{origin},#{z},#{x})")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    first_point = counter + 1
    for v in vertices:
        add(f"CARTESIAN_POINT('',({_num(v[0])},{_num(v[1])},{_num(v[2])}))")
    faces = []
    tris = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    for tri, normal in zip(tris, normals, strict=True):
        a, b, c = (first_point + int(i) for i in tri)
        # Each triangle lies on its plane: FACE_SURFACE + PLANE is what every CAD reader converts.
        edge = vertices[tri[1]] - vertices[tri[0]]
        edge = edge / (np.linalg.norm(edge) or 1.0)
        n_dir = add(f"DIRECTION('',({_num(normal[0])},{_num(normal[1])},{_num(normal[2])}))")
        r_dir = add(f"DIRECTION('',({_num(edge[0])},{_num(edge[1])},{_num(edge[2])}))")
        place = add(f"AXIS2_PLACEMENT_3D('',#{a},#{n_dir},#{r_dir})")
        plane = add(f"PLANE('',#{place})")
        loop = add(f"POLY_LOOP('',(#{a},#{b},#{c}))")
        bound = add(f"FACE_OUTER_BOUND('',#{loop},.T.)")
        faces.append(add(f"FACE_SURFACE('',(#{bound}),#{plane},.T.)"))
    face_list = ",".join(f"#{f}" for f in faces)
    if solid:
        shell = add(f"CLOSED_SHELL('',({face_list}))")
        body = add(f"FACETED_BREP('{name}',#{shell})")
        rep = add(f"FACETED_BREP_SHAPE_REPRESENTATION('{name}',(#{axes},#{body}),#{context})")
    else:
        shell = add(f"OPEN_SHELL('',({face_list}))")
        body = add(f"SHELL_BASED_SURFACE_MODEL('{name}',(#{shell}))")
        rep = add(f"MANIFOLD_SURFACE_SHAPE_REPRESENTATION('{name}',(#{axes},#{body}),#{context})")
    add(f"SHAPE_DEFINITION_REPRESENTATION(#{shape},#{rep})")

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    kind = "faceted solid" if solid else "surface model"
    header = (
        "ISO-10303-21;\nHEADER;\n"
        f"FILE_DESCRIPTION(('reverse3d {kind}'),'2;1');\n"
        f"FILE_NAME('{name}.step','{stamp}',('Voxel Enterprise'),('Reverse3D'),'reverse3d','reverse3d','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }'));\nENDSEC;\nDATA;\n"
    )
    return header + "\n".join(lines) + "\nENDSEC;\nEND-ISO-10303-21;\n"


def save_step(
    mesh: trimesh.Trimesh, path: str | Path, *, name: str | None = None, overwrite: bool = False
) -> Path:
    """Write ``mesh`` as STEP; returns the file path."""
    out = prepare_output_file(path, allowed_extensions=STEP_EXTENSIONS, overwrite=overwrite)
    out.write_text(step_text(mesh, name or out.stem), encoding="ascii", errors="replace")
    return out
