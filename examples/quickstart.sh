#!/usr/bin/env bash
# End-to-end tour of reverse3d using generated sample data (no photos needed).
set -euo pipefail

WS="${REVERSE3D_WORKSPACE:-./workspace}"
mkdir -p "$WS/in" "$WS/out"

# 1. Synthetic "technical plan": a 200x120 px plate with a hole (1 px = 0.5 mm).
python - "$WS/in/plan.png" <<'PY'
import sys, cv2, numpy as np
img = np.full((300, 400, 3), 255, np.uint8)
cv2.rectangle(img, (100, 90), (299, 209), (20, 20, 20), -1)
cv2.circle(img, (200, 150), 30, (255, 255, 255), -1)
cv2.imwrite(sys.argv[1], img)
PY

# 2. Plan → contours → extruded part (3 mm thick).
reverse3d --json vision mask "$WS/in/plan.png" "$WS/out" --units-per-pixel 0.5 --force
reverse3d --json vision extrude "$WS/out/plan_contours.json" "$WS/out/plate.stl" --thickness 3 --force

# 3. Repair / align / measure.
reverse3d --json mesh clean "$WS/out/plate.stl" "$WS/out/plate_clean.stl" --align obb --ground --force
reverse3d --json mesh measure "$WS/out/plate_clean.stl"

# 4. Primitive fitting + parametric CAD.
reverse3d --json mesh fit "$WS/out/plate_clean.stl" --parametric "$WS/out/plate.scad" --force

# 5. Render (Blender if available, else PyVista if installed).
reverse3d --json render "$WS/out/plate_clean.stl" "$WS/out/plate.png" --samples 32 --force \
  || echo "Render skipped: install Blender or pip install 'reverse3d[render]'"

# 6. Same thing through the sandboxed agent interface.
REVERSE3D_WORKSPACE="$WS" reverse3d --json agent call reverse3d_measure_mesh \
  --args '{"input_path": "out/plate_clean.stl"}'
