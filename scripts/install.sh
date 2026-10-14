#!/usr/bin/env bash
# reverse3d installer for Linux and macOS — Voxel Enterprise (https://voxel.lat)
#
#   curl -fsSL https://raw.githubusercontent.com/voxelenterprise/reverse3d/main/scripts/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --full --device gpu
#
# What it does (nothing else):
#   1. finds Python >= 3.10
#   2. creates an isolated virtualenv in ~/.reverse3d/venv (no system packages touched, no sudo)
#   3. installs reverse3d with the selected extras (GPU-aware)
#   4. links the `reverse3d` command into ~/.local/bin
#   5. runs `reverse3d setup` to register the MCP server in the AI agents found on this machine
#      (each modified config file is backed up first)
#
# Options:
#   --full                 also install 3D meshing (Open3D) and AI depth (PyTorch, ~2 GB)
#   --minimal              only the core + MCP server
#   --with EXTRAS          comma list of extras: recon,depth,render,mcp,agents
#   --device auto|cpu|gpu  default compute device            (default: auto)
#   --profile fast|balanced|quality                           (default: balanced)
#   --toolset full|lean    lean = fewer tools, fewer tokens   (default: full)
#   --agents LIST          auto | all | claude-code,cursor,... (default: auto)
#   --workspace DIR        folder the agents may use          (default: ~/reverse3d-workspace)
#   --ref REF              git branch/tag to install          (default: main)
#   --source PATH          install from a local clone instead of GitHub
#   --no-setup             do not register agents
#   --uninstall            remove agents' entries, the venv and the command link
set -euo pipefail

REPO_URL="${REVERSE3D_REPO:-https://github.com/voxelenterprise/reverse3d}"
REF="${REVERSE3D_REF:-main}"
R3D_HOME="${REVERSE3D_HOME:-$HOME/.reverse3d}"
VENV="$R3D_HOME/venv"
WORKSPACE="${REVERSE3D_WORKSPACE:-$HOME/reverse3d-workspace}"
EXTRAS="mcp,render"
AGENTS="auto"
DEVICE=""; PROFILE=""; TOOLSET=""; SOURCE=""
DO_SETUP=1; UNINSTALL=0

say()  { printf '\033[1;36m[reverse3d]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[reverse3d]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[reverse3d]\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --full) EXTRAS="mcp,render,recon,depth" ;;
    --minimal) EXTRAS="mcp" ;;
    --with) EXTRAS="mcp,$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --profile) PROFILE="$2"; shift ;;
    --toolset) TOOLSET="$2"; shift ;;
    --agents) AGENTS="$2"; shift ;;
    --workspace) WORKSPACE="$2"; shift ;;
    --ref) REF="$2"; shift ;;
    --source) SOURCE="$2"; shift ;;
    --no-setup) DO_SETUP=0 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
  shift
done

# Values are passed to pip/reverse3d as single arguments; still refuse anything odd.
case "$EXTRAS" in *[!a-z0-9,]*) die "Invalid --with value" ;; esac
case "$REF" in *[!A-Za-z0-9._/-]*) die "Invalid --ref value" ;; esac

if [ "$UNINSTALL" -eq 1 ]; then
  if [ -x "$VENV/bin/reverse3d" ]; then
    "$VENV/bin/reverse3d" setup --uninstall --agents all || true
  fi
  rm -rf "$VENV"
  rm -f "$HOME/.local/bin/reverse3d"
  say "Uninstalled. Your workspace ($WORKSPACE) and $R3D_HOME/config.json were kept."
  exit 0
fi

# 1. Python ---------------------------------------------------------------------------------
pick_python() {
  local wants_open3d="$1" c
  # Open3D wheels lag behind new Python releases: prefer 3.12/3.11 when meshing is requested.
  local candidates="python3.12 python3.11 python3.10 python3.13 python3 python"
  [ "$wants_open3d" = 0 ] && candidates="python3 python3.13 python3.12 python3.11 python3.10 python"
  for c in $candidates; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}
case ",$EXTRAS," in *,recon,*) WANT_O3D=1 ;; *) WANT_O3D=0 ;; esac
PY="$(pick_python "$WANT_O3D")" || die "Python 3.10+ not found. Install it from https://www.python.org/downloads/"
PYVER="$("$PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
say "Using $PY (Python $PYVER)"
if [ "$WANT_O3D" = 1 ] && [ "$PYVER" = "3.13" ]; then
  warn "Open3D has no wheels for Python 3.13 yet: skipping the 'recon' extra (install Python 3.12 to enable it)."
  EXTRAS="$(echo "$EXTRAS" | sed 's/,recon//; s/recon,//')"
fi

# 2. GPU ------------------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  say "NVIDIA GPU detected: ${GPU_NAME:-yes}"
  [ -z "$DEVICE" ] && DEVICE="auto"
elif [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
  say "Apple Silicon detected (MPS available for AI depth)."
else
  say "No GPU detected: everything will run on CPU."
fi

# 3. Virtualenv + install -------------------------------------------------------------------
mkdir -p "$R3D_HOME"
if [ ! -x "$VENV/bin/python" ]; then
  say "Creating isolated environment in $VENV"
  "$PY" -m venv "$VENV" || die "Could not create a virtualenv (Debian/Ubuntu: sudo apt install python3-venv)"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip

if [ -n "$SOURCE" ]; then
  SPEC="$(cd "$SOURCE" && pwd)[$EXTRAS]"
else
  SPEC="reverse3d[$EXTRAS] @ git+$REPO_URL@$REF"
fi
say "Installing reverse3d [$EXTRAS] (this can take a few minutes)…"
"$VENV/bin/python" -m pip install --upgrade "$SPEC" || die "pip install failed"

# 4. Command link ---------------------------------------------------------------------------
mkdir -p "$HOME/.local/bin"
ln -sf "$VENV/bin/reverse3d" "$HOME/.local/bin/reverse3d"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) warn "Add ~/.local/bin to your PATH to use the 'reverse3d' command:  export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

# 5. Agents ---------------------------------------------------------------------------------
if [ "$DO_SETUP" -eq 1 ]; then
  set -- setup --agents "$AGENTS" --workspace "$WORKSPACE"
  [ -n "$DEVICE" ] && set -- "$@" --device "$DEVICE"
  [ -n "$PROFILE" ] && set -- "$@" --profile "$PROFILE"
  [ -n "$TOOLSET" ] && set -- "$@" --toolset "$TOOLSET"
  say "Registering reverse3d in your AI agents…"
  "$VENV/bin/reverse3d" "$@" || warn "Agent setup reported a problem; run 'reverse3d setup --snippets' for manual config."
fi

# 6. Summary --------------------------------------------------------------------------------
"$VENV/bin/reverse3d" doctor >/dev/null 2>&1 && say "Installation verified (reverse3d doctor OK)."
cat <<EOF

  reverse3d is ready.   Workspace: $WORKSPACE

  Next:
    1. Restart your AI agent (Claude, Cursor, VS Code, Gemini, Codex, OpenCode…).
    2. Ask it:  "usa reverse3d para medir y renderizar mi pieza.stl"

  Optional external tools (more power):
    Blender (photoreal renders):   https://www.blender.org/download/
    COLMAP  (photogrammetry):      Ubuntu: sudo apt install colmap   ·   macOS: brew install colmap
  Check anytime:  reverse3d doctor     ·    GPU:  reverse3d config set device gpu

EOF
