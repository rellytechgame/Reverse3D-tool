---
name: reverse3d
description: 3D modeling and reverse engineering with the local reverse3d toolkit. Use when the user wants to turn photos, a plan/stencil/silhouette, or an existing mesh (STL/OBJ/PLY/GLB) into a 3D model, measure dimensions, repair a mesh, get CAD code (OpenSCAD/CadQuery), render a preview, or switch reverse3d between CPU and GPU.
---

# reverse3d

Local toolkit by Voxel Enterprise. Exposed as MCP tools named `reverse3d_*`. If those tools are not available, use the CLI: `reverse3d --json agent call <tool> --args '<json>'`. It has the same sandbox and the same output.

## Golden rules (save tokens)

1. **Use `reverse3d_run_recipe` first.** One call runs a whole workflow and returns a short summary. Full details go to `<output_dir>/report.json`; read that file only if the user asks for details.
2. Paths are **relative to the workspace** (`REVERSE3D_WORKSPACE`). If the user attached a file, save it inside the workspace first.
3. On `ok:false`, read `error.hint`, fix the arguments and retry once. Do not loop.
4. Report sizes with units and give the user the `render` path. Say clearly when values are estimates.

## Recipes

| User wants | recipe | input_path |
|---|---|---|
| Photos of an object → 3D model | `photos_to_model` | folder of photos |
| Plan, stencil or silhouette → part | `plan_to_part` (+ `thickness`, `units_per_pixel`) | image |
| Scan or mesh → clean + dimensions + CAD | `scan_to_cad` | mesh |
| A single photo → approximate relief | `photo_to_relief` | image |
| Dimensions + preview of a mesh | `mesh_report` | mesh |

Example: `reverse3d_run_recipe {"recipe":"plan_to_part","input_path":"in/bracket.png","output_dir":"out/bracket","thickness":3,"units_per_pixel":0.5}`.

Scale tips: if the user knows one real dimension (e.g. "the plate is 100 mm wide and spans 200 px"), then `units_per_pixel = 100/200`. To convert meters to millimeters, use `scale: 1000`.

## Device and performance

- "use the GPU" → `reverse3d_configure {"device":"gpu"}`. It falls back to CPU if no usable GPU.
- "use CPU" → `{"device":"cpu"}`. Faster / better quality → `{"profile":"fast"}` or `{"profile":"quality"}`.
- Call `reverse3d_configure {}` to show the current settings and the detected GPU.
- Call `reverse3d_environment_check` when something is missing (COLMAP, Blender, extras). It reports how to install it.

## Individual tools (only for custom steps)

`analyze_images`, `match_images`, `extract_mask`, `silhouette_to_mesh`, `photogrammetry`, `estimate_depth`, `mesh_from_point_cloud`, `clean_mesh`, `measure_mesh`, `fit_primitives`, `render`. All of them are prefixed with `reverse3d_`.

Never overwrite user files unless the user asks. Pass `overwrite:true` only then.
