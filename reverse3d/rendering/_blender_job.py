"""Headless render job executed *inside* Blender's embedded Python.

Invoked by :mod:`reverse3d.rendering.blender_render` as::

    blender --background --factory-startup --disable-autoexec \
            --python-exit-code 1 --python _blender_job.py -- job.json

This file must not import reverse3d (Blender ships its own interpreter). It only
reads a JSON job written by the launcher, so no code or expression coming from
users/agents is ever evaluated here. Compatible with Blender 3.6 LTS → 4.x.
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import bpy
from mathutils import Vector

RESULT_MARKER = "REVERSE3D_RESULT "


def _read_job() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Missing job file argument")
    job_path = Path(argv[argv.index("--") + 1])
    return json.loads(job_path.read_text(encoding="utf-8"))


def _hex_to_rgba(value: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) / 255.0 for i in (0, 2, 4))

    # Convert sRGB to linear for Blender's scene-referred colour inputs.
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return lin(r), lin(g), lin(b), alpha


def _call_first(*candidates: tuple[str, str], **kwargs: object) -> None:
    """Call the first operator that exists (operators were renamed in 4.x)."""
    last_error: Exception | None = None
    for module, name in candidates:
        try:
            getattr(getattr(bpy.ops, module), name)(**kwargs)
            return
        except (AttributeError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"No importer available: {last_error}")


def import_mesh(path: str) -> bpy.types.Object:
    ext = Path(path).suffix.lower()
    before = set(bpy.data.objects)
    if ext == ".obj":
        _call_first(("wm", "obj_import"), ("import_scene", "obj"), filepath=path)
    elif ext == ".stl":
        _call_first(("wm", "stl_import"), ("import_mesh", "stl"), filepath=path)
    elif ext == ".ply":
        _call_first(("wm", "ply_import"), ("import_mesh", "ply"), filepath=path)
    elif ext in {".glb", ".gltf"}:
        _call_first(("import_scene", "gltf"), filepath=path)
    else:
        raise ValueError(f"Unsupported format for Blender: {ext}")

    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not meshes:
        raise RuntimeError("Imported file contains no mesh objects")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def normalize(obj: bpy.types.Object, ground: bool) -> tuple[float, float]:
    """Center the object and scale its largest dimension to 2 units.

    Returns the bounding-sphere radius and the height of the object's centre
    (non-zero when it rests on the ground), which is where the camera aims.
    """
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    size = max((hi - lo).x, (hi - lo).y, (hi - lo).z) or 1.0
    scale = 2.0 / size
    center = (lo + hi) / 2
    obj.location = -center * scale
    if ground:
        obj.location.z = -lo.z * scale
    obj.scale = (scale, scale, scale)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    center_z = (hi.z - lo.z) * scale / 2 if ground else 0.0
    return ((hi - lo) * scale).length / 2, center_z


def shade(obj: bpy.types.Object, mode: str) -> None:
    if mode == "flat":
        bpy.ops.object.shade_flat()
        return
    if mode == "smooth":
        bpy.ops.object.shade_smooth()
        return
    try:  # Blender >= 4.1
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
    except (AttributeError, RuntimeError, TypeError):
        bpy.ops.object.shade_smooth()
        if hasattr(obj.data, "use_auto_smooth"):  # Blender <= 4.0
            obj.data.use_auto_smooth = True
            obj.data.auto_smooth_angle = math.radians(30)


def make_material(obj: bpy.types.Object, job: dict) -> None:
    mat = bpy.data.materials.new("reverse3d_material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = _hex_to_rgba(job["color"])
    bsdf.inputs["Metallic"].default_value = float(job["metallic"])
    bsdf.inputs["Roughness"].default_value = float(job["roughness"])
    attrs = getattr(obj.data, "color_attributes", None)
    if job.get("vertex_colors", True) and attrs is not None and len(attrs) > 0:
        node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        node.layer_name = attrs[0].name
        mat.node_tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def add_light(name: str, location: tuple[float, float, float], energy: float, size: float) -> None:
    data = bpy.data.lights.new(name, type="AREA")
    data.energy = energy
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    direction = Vector((0, 0, 0.5)) - light.location
    light.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_world(job: dict) -> None:
    world = bpy.data.worlds.new("reverse3d_world")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = _hex_to_rgba(job["background"])
    bg.inputs["Strength"].default_value = 0.6


def setup_camera(radius: float, center_z: float, job: dict) -> bpy.types.Object:
    rig = bpy.data.objects.new("reverse3d_rig", None)
    bpy.context.collection.objects.link(rig)
    rig.location = (0, 0, center_z)

    cam_data = bpy.data.cameras.new("reverse3d_camera")
    cam_data.lens = 50
    cam = bpy.data.objects.new("reverse3d_camera", cam_data)
    bpy.context.collection.objects.link(cam)
    # camera.angle is the FOV of the longer image side; fit the sphere in the shorter one.
    width, height = int(job["width"]), int(job["height"])
    fov = cam_data.angle
    if width != height:
        fov = 2 * math.atan(math.tan(fov / 2) * min(width, height) / max(width, height))
    distance = radius / math.sin(fov / 2) * 1.1
    elev, azim = math.radians(job["elevation_deg"]), math.radians(job["azimuth_deg"])
    # Parented with an identity parent-inverse: this location is relative to the rig.
    cam.location = (
        distance * math.cos(elev) * math.sin(azim),
        -distance * math.cos(elev) * math.cos(azim),
        distance * math.sin(elev),
    )
    cam.parent = rig
    track = cam.constraints.new(type="TRACK_TO")
    track.target = rig
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    bpy.context.scene.camera = cam
    cam_data.clip_end = distance * 10
    return rig


def setup_ground(job: dict) -> None:
    if not job["ground"] or job["engine"] != "cycles":
        return
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.is_shadow_catcher = True


def setup_render(job: dict) -> None:
    scene = bpy.context.scene
    engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
    if job["engine"] == "cycles":
        scene.render.engine = "CYCLES"
        scene.cycles.samples = int(job["samples"])
        if job["device"] == "gpu":
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
                    try:
                        prefs.compute_device_type = backend
                        prefs.get_devices()
                        if any(d.type == backend for d in prefs.devices):
                            for d in prefs.devices:
                                d.use = True
                            scene.cycles.device = "GPU"
                            break
                    except TypeError:
                        continue
            except KeyError:
                pass
    elif job["engine"] == "eevee":
        scene.render.engine = (
            "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
        )
        scene.eevee.taa_render_samples = int(job["samples"])
    else:
        scene.render.engine = "BLENDER_WORKBENCH"

    scene.render.resolution_x = int(job["width"])
    scene.render.resolution_y = int(job["height"])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = bool(job["transparent"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA" if job["transparent"] else "RGB"
    for transform in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = transform
            break
        except TypeError:
            continue


def main() -> None:
    job = _read_job()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    obj = import_mesh(job["mesh"])
    radius, center_z = normalize(obj, job["ground"])
    shade(obj, job["shading"])
    make_material(obj, job)
    setup_world(job)
    r = max(radius, 0.1)
    add_light("key", (r * 3, -r * 3, r * 4), 800 * r * r, r * 2)
    add_light("fill", (-r * 4, -r * 2, r * 2), 300 * r * r, r * 3)
    add_light("rim", (0, r * 4, r * 3), 500 * r * r, r * 2)
    rig = setup_camera(radius, center_z, job)
    setup_ground(job)
    setup_render(job)

    output = Path(job["output"])
    files: list[str] = []
    frames = int(job["turntable_frames"])
    if frames <= 0:
        bpy.context.scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        files.append(str(output))
    else:
        for i in range(frames):
            rig.rotation_euler.z = 2 * math.pi * i / frames
            path = output.with_name(f"{output.stem}_{i:04d}{output.suffix}")
            bpy.context.scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            files.append(str(path))

    print(
        RESULT_MARKER + json.dumps({"files": files, "blender": bpy.app.version_string}), flush=True
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
