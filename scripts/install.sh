#!/usr/bin/env bash
# reverse3d installer for Linux and macOS — Voxel Enterprise (https://voxel.lat)
#
#   curl -fsSL https://raw.githubusercontent.com/rellytechgame/Reverse3D-tool/main/scripts/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --lite --device cpu
#
# Zero-question installer: it never prompts, never uses sudo interactively and
# installs everything by default. Safe to re-run: finished steps are skipped, so
# if an AI agent stops it on a timeout, running it again continues.
#
#   1. gets Python 3.12 through uv (downloaded and SHA-256 verified, no root);
#      falls back to an existing Python 3.10+
#   2. creates an isolated environment in ~/.reverse3d/venv
#   3. installs reverse3d with all modules (an optional module that fails never
#      stops the install; PyTorch is the CUDA build with an NVIDIA GPU, else CPU)
#   4. installs Blender and COLMAP into ~/.reverse3d/tools (pinned + verified;
#      COLMAP on Linux only through a package manager that needs no password)
#   5. registers the MCP server in every AI agent found and pre-approves the
#      reverse3d tools (each modified config file is backed up first)
#   6. writes ~/.reverse3d/install-status.json and ~/.reverse3d/install.log
#
# Options:
#   --lite                 skip AI depth (PyTorch); everything else is installed
#   --minimal              only the core + MCP server, no external tools
#   --with EXTRAS          comma list of extras: recon,depth,render,mcp,agents
#   --no-tools             do not install Blender and COLMAP
#   --no-auto-approve      let agents ask before each reverse3d tool call
#   --device auto|cpu|gpu  default compute device            (default: auto)
#   --profile fast|balanced|quality                           (default: balanced)
#   --toolset full|lean    lean = fewer tools, fewer tokens   (default: full)
#   --agents LIST          auto | all | claude-code,cursor,... (default: auto)
#   --workspace DIR        folder the agents may use          (default: ~/reverse3d-workspace)
#   --ref REF              git branch/tag to install          (default: main)
#   --source PATH          install from a local clone instead of GitHub
#   --no-setup             do not register agents
#   --uninstall            remove agents' entries, the venv, tools and the command link
set -euo pipefail

# ─── Pinned downloads (bump version + SHA-256 together) ───────────────────────
UV_VERSION="0.12.19"
# shellcheck disable=SC2034  # read indirectly through ${!UV_KEY}
UV_SHA_x86_64_linux="23bf5552d220e0842b65c862097b2ebaeba0064b74eda5e565e77fd25969d8c8"
# shellcheck disable=SC2034
UV_SHA_aarch64_linux="0804e9b164c64b6914182d5920c08551958a095986f10a3731056df701126436"
# shellcheck disable=SC2034
UV_SHA_x86_64_darwin="cb5fa57bafe68fc0fb94b17f06bee0b0b9a7feb94ccbd110445afa0696e39273"
# shellcheck disable=SC2034
UV_SHA_aarch64_darwin="a9a8df1eedeb192f2e47e40e2faabfb387db4b850209118786d42f89dde3e0ba"
BLENDER_VERSION="5.2.2"
BLENDER_SHA_linux_x64="84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168"
BLENDER_SHA_macos_arm64="dc4125399b8bfefe283cc1624d6cfc7809d1cac20ace51072127eb371f31f210"
COLMAP_VERSION="4.2.1"
COLMAP_SHA_macos_arm64="1fe9d65bddd61e0598f3087641cb14c4d4d4c74529e9f58911e8a5238a140a04"

REPO_URL="${REVERSE3D_REPO:-https://github.com/rellytechgame/Reverse3D-tool}"
REF="${REVERSE3D_REF:-main}"
R3D_HOME="${REVERSE3D_HOME:-$HOME/.reverse3d}"
VENV="$R3D_HOME/venv"
VPY="$VENV/bin/python"
EXE="$VENV/bin/reverse3d"
TOOLS="$R3D_HOME/tools"
DOWNLOADS="$R3D_HOME/downloads"
STATUS_FILE="$R3D_HOME/install-status.json"
LOG_FILE="$R3D_HOME/install.log"
WORKSPACE="${REVERSE3D_WORKSPACE:-$HOME/reverse3d-workspace}"
EXTRAS="mcp,render,recon,depth"
AGENTS="auto"
DEVICE=""; PROFILE=""; TOOLSET=""; SOURCE=""
DO_SETUP=1; DO_TOOLS=1; AUTO_APPROVE=1; UNINSTALL=0
MISSING=""
OS="$(uname -s)"; ARCH="$(uname -m)"
[ "$ARCH" = "arm64" ] && ARCH="aarch64"
[ "$ARCH" = "amd64" ] && ARCH="x86_64"

export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 UV_NO_PROGRESS=1
export DEBIAN_FRONTEND=noninteractive HOMEBREW_NO_AUTO_UPDATE=1 NONINTERACTIVE=1

say()  { printf '\033[1;36m[reverse3d]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[reverse3d]\033[0m %s\n' "$*" >&2; }
add_missing() { MISSING="${MISSING:+$MISSING; }$1"; }
json_str() { printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g')"; }
write_status() {
  mkdir -p "$R3D_HOME"
  {
    printf '{\n  "ok": %s,\n  "error": %s,\n' "$1" "$(json_str "${2:-}")"
    printf '  "finished": %s,\n' "$(json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
    printf '  "command": %s,\n  "workspace": %s,\n' "$(json_str "$EXE")" "$(json_str "$WORKSPACE")"
    printf '  "extras": %s,\n  "missing": %s,\n' "$(json_str "$EXTRAS")" "$(json_str "$MISSING")"
    printf '  "log": %s\n}\n' "$(json_str "$LOG_FILE")"
  } > "$STATUS_FILE" 2>/dev/null || true
}
die() { printf '\033[1;31m[reverse3d]\033[0m %s\n' "$*" >&2; write_status false "$*"; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --full) EXTRAS="mcp,render,recon,depth" ;;   # kept for compatibility (now the default)
    --lite) EXTRAS="mcp,render,recon" ;;
    --minimal) EXTRAS="mcp"; DO_TOOLS=0 ;;
    --with) EXTRAS="mcp,$2"; shift ;;
    --no-tools) DO_TOOLS=0 ;;
    --no-auto-approve) AUTO_APPROVE=0 ;;
    --device) DEVICE="$2"; shift ;;
    --profile) PROFILE="$2"; shift ;;
    --toolset) TOOLSET="$2"; shift ;;
    --agents) AGENTS="$2"; shift ;;
    --workspace) WORKSPACE="$2"; shift ;;
    --ref) REF="$2"; shift ;;
    --source) SOURCE="$2"; shift ;;
    --no-setup) DO_SETUP=0 ;;
    --uninstall) UNINSTALL=1 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
  shift
done

# Values are passed to pip/reverse3d as single arguments; still refuse anything odd.
case "$EXTRAS" in *[!a-z0-9,]*) die "Invalid --with value" ;; esac
case "$REF" in *[!A-Za-z0-9._/-]*) die "Invalid --ref value" ;; esac

if [ "$UNINSTALL" -eq 1 ]; then
  if [ -x "$EXE" ]; then
    "$EXE" setup --uninstall --agents all || true
  fi
  rm -rf "$VENV" "$TOOLS" "$DOWNLOADS" "${R3D_HOME:?}/bin" "${R3D_HOME:?}/python"
  rm -f "$HOME/.local/bin/reverse3d"
  say "Uninstalled. Your workspace ($WORKSPACE) and $R3D_HOME/config.json were kept."
  exit 0
fi

mkdir -p "$R3D_HOME" "$DOWNLOADS"
exec > >(tee -a "$LOG_FILE") 2>&1
say "Installing reverse3d. No questions will be asked; this can take 10-20 minutes."

# ─── helpers ──────────────────────────────────────────────────────────────────
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

fetch() {
  if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 -o "$2" "$1"
  else wget -q -O "$2" "$1"; fi
}

# Download to a temp name, verify SHA-256, then move into place. Re-runs reuse a verified file.
get_verified() {
  local url="$1" sha="$2" dest="$3" i
  if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$sha" ]; then return 0; fi
  for i in 1 2 3; do
    if fetch "$url" "$dest.part" && [ "$(sha256_of "$dest.part")" = "$sha" ]; then
      mv -f "$dest.part" "$dest"; return 0
    fi
    warn "Download or checksum failed for $url (attempt $i)"
    rm -f "$dest.part"; sleep $((3 * i))
  done
  return 1
}

find_file() {  # find_file ROOT NAME... → first match (depth ≤ 6)
  local root="$1" n hit; shift
  [ -d "$root" ] || return 1
  for n in "$@"; do
    hit="$(find "$root" -maxdepth 6 -name "$n" -type f 2>/dev/null | head -n1)"
    [ -n "$hit" ] && { echo "$hit"; return 0; }
  done
  return 1
}

can_sudo() { command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; }

# ─── 1. Python (uv-managed 3.12, or an existing 3.10+) ────────────────────────
UV="$R3D_HOME/bin/uv"
if [ ! -x "$UV" ]; then
  case "$OS" in
    Linux)  UV_TRIPLE="$ARCH-unknown-linux-gnu"; UV_KEY="UV_SHA_${ARCH}_linux" ;;
    Darwin) UV_TRIPLE="$ARCH-apple-darwin";      UV_KEY="UV_SHA_${ARCH}_darwin" ;;
    *)      UV_TRIPLE=""; UV_KEY="" ;;
  esac
  UV_SHA=""
  [ -n "$UV_KEY" ] && UV_SHA="${!UV_KEY:-}"
  if [ -n "$UV_SHA" ]; then
    say "Fetching uv $UV_VERSION (Python manager, no root needed)"
    archive="$DOWNLOADS/uv-$UV_VERSION-$UV_TRIPLE.tar.gz"
    if get_verified "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TRIPLE.tar.gz" "$UV_SHA" "$archive"; then
      tmp="$DOWNLOADS/uv-extract"; rm -rf "$tmp"; mkdir -p "$tmp" "$R3D_HOME/bin"
      tar -xzf "$archive" -C "$tmp" && found="$(find_file "$tmp" uv)" && cp -f "$found" "$UV" && chmod +x "$UV"
      rm -rf "$tmp"
    fi
  fi
fi
if [ -x "$UV" ]; then
  export UV_PYTHON_INSTALL_DIR="$R3D_HOME/python"
else
  UV=""
  warn "uv unavailable: falling back to a Python already installed on this computer."
fi

venv_version() {
  if [ -x "$VPY" ]; then
    "$VPY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true
  fi
}

case ",$EXTRAS," in *,recon,*) WANT_O3D=1 ;; *) WANT_O3D=0 ;; esac
CURRENT="$(venv_version)"
if [ -n "$CURRENT" ] && [ -n "$UV" ] && [ "$WANT_O3D" = 1 ]; then
  case "$CURRENT" in 3.10|3.11|3.12) ;; *)
    # Open3D has no wheels for newer Pythons: rebuild the venv on 3.12.
    say "Rebuilding the environment on Python 3.12 (was $CURRENT) so 3D meshing can be installed"
    rm -rf "$VENV"; CURRENT="" ;;
  esac
fi

if [ -z "$CURRENT" ]; then
  if [ -n "$UV" ]; then
    say "Preparing Python 3.12 (isolated, in $R3D_HOME)"
    if ! { "$UV" python install 3.12 --quiet && "$UV" venv --python 3.12 --seed --clear --quiet "$VENV"; }; then
      warn "uv could not create the environment; trying an installed Python."; UV=""
    fi
  fi
  if [ ! -x "$VPY" ]; then
    PY=""
    for c in python3.12 python3.11 python3.10 python3.13 python3 python; do
      if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PY="$c"; break
      fi
    done
    [ -n "$PY" ] || die "Could not get Python. Check the internet connection and run this installer again."
    "$PY" -m venv "$VENV" || die "Could not create the Python environment (Debian/Ubuntu: the python3-venv package is missing)."
  fi
fi
PYVER="$(venv_version)"
say "Python $PYVER ready"
if [ "$WANT_O3D" = 1 ]; then
  case "$PYVER" in 3.10|3.11|3.12) ;; *)
    warn "Open3D has no wheels for Python $PYVER: 3D meshing from point clouds is skipped."
    add_missing "recon (Open3D needs Python 3.10-3.12)"
    EXTRAS="$(echo "$EXTRAS" | sed 's/,recon//; s/recon,//')" ;;
  esac
fi

install_pkgs() {
  local i
  for i in 1 2; do
    if [ -n "$UV" ]; then "$UV" pip install --python "$VPY" --quiet "$@" && return 0
    else "$VPY" -m pip install --quiet --upgrade "$@" && return 0; fi
    sleep 3
  done
  return 1
}

# ─── 2. GPU ───────────────────────────────────────────────────────────────────
HAS_NVIDIA=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  HAS_NVIDIA=1
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)"
  say "NVIDIA GPU detected: ${GPU_NAME:-yes}"
  [ -z "$DEVICE" ] && DEVICE="auto"
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "aarch64" ]; then
  say "Apple Silicon detected (MPS available for AI depth)."
else
  say "No GPU detected: everything will run on CPU."
fi

# ─── 3. reverse3d + modules ───────────────────────────────────────────────────
# A GitHub archive needs no git on the user's machine.
spec() {
  if [ -n "$SOURCE" ]; then echo "$(cd "$SOURCE" && pwd)[$1]"
  else echo "reverse3d[$1] @ $REPO_URL/archive/$REF.zip"; fi
}
REFRESH=()
[ -n "$UV" ] && REFRESH=(--reinstall-package reverse3d)

CORE=""; OPTIONAL=""
for ex in ${EXTRAS//,/ }; do
  case "$ex" in mcp|render) CORE="${CORE:+$CORE,}$ex" ;; *) OPTIONAL="$OPTIONAL $ex" ;; esac
done
[ -z "$CORE" ] && CORE="mcp"

case " $OPTIONAL " in *" depth "*)
  if [ -n "${REVERSE3D_TORCH_INDEX:-}" ]; then TORCH_INDEX="$REVERSE3D_TORCH_INDEX"
  elif [ "$OS" = "Darwin" ]; then TORCH_INDEX=""
  elif [ "$HAS_NVIDIA" = 1 ]; then TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  else TORCH_INDEX="https://download.pytorch.org/whl/cpu"; fi   # ~10x smaller than the default
  if ! "$VPY" -c 'import torch' 2>/dev/null; then
    say "Installing PyTorch (large download, please wait)"
    if [ -n "$TORCH_INDEX" ]; then
      install_pkgs torch --index-url "$TORCH_INDEX" \
        || { warn "PyTorch from $TORCH_INDEX failed; trying the CPU build."; install_pkgs torch --index-url https://download.pytorch.org/whl/cpu || true; }
    else
      install_pkgs torch || true
    fi
  fi ;;
esac

say "Installing reverse3d [$CORE]"
# ${arr[@]+...}: macOS ships bash 3.2, where an empty array trips `set -u`.
install_pkgs "$(spec "$CORE")" ${REFRESH[@]+"${REFRESH[@]}"} || die "Could not install reverse3d. Check the internet connection and run this installer again."
INSTALLED="$CORE"
for ex in $OPTIONAL; do
  say "Installing module '$ex'"
  if install_pkgs "$(spec "$CORE,$ex")"; then INSTALLED="$INSTALLED,$ex"
  else warn "Module '$ex' could not be installed; reverse3d works without it."; add_missing "module $ex"; fi
done
EXTRAS="$INSTALLED"

# ─── 4. Command link ──────────────────────────────────────────────────────────
mkdir -p "$HOME/.local/bin"
ln -sf "$EXE" "$HOME/.local/bin/reverse3d"

# ─── 5. Blender + COLMAP ──────────────────────────────────────────────────────
set_tool() { "$EXE" config set "$1" "$2" >/dev/null 2>&1 || warn "Could not store the $1 path."; }

install_blender() {
  local dir="$TOOLS/blender-$BLENDER_VERSION" series="${BLENDER_VERSION%.*}" bin="" archive mnt
  if [ "$OS" = "Darwin" ]; then bin="$dir/Blender.app/Contents/MacOS/Blender"; [ -x "$bin" ] || bin=""
  else bin="$(find_file "$dir" blender || true)"; fi
  if [ -z "$bin" ] && command -v blender >/dev/null 2>&1; then say "Blender already installed: $(command -v blender)"; return 0; fi
  if [ -z "$bin" ] && [ "$OS" = "Darwin" ] && [ -x "/Applications/Blender.app/Contents/MacOS/Blender" ]; then
    bin="/Applications/Blender.app/Contents/MacOS/Blender"
  fi
  if [ -z "$bin" ]; then
    if [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
      archive="$DOWNLOADS/blender-$BLENDER_VERSION-linux-x64.tar.xz"
      say "Downloading Blender $BLENDER_VERSION (~380 MB, photoreal renders)"
      if get_verified "https://download.blender.org/release/Blender$series/blender-$BLENDER_VERSION-linux-x64.tar.xz" "$BLENDER_SHA_linux_x64" "$archive"; then
        mkdir -p "$dir" && tar -xJf "$archive" -C "$dir" && rm -f "$archive"
        bin="$(find_file "$dir" blender || true)"
      fi
    elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "aarch64" ]; then
      archive="$DOWNLOADS/blender-$BLENDER_VERSION-macos-arm64.dmg"
      say "Downloading Blender $BLENDER_VERSION (~350 MB, photoreal renders)"
      if get_verified "https://download.blender.org/release/Blender$series/blender-$BLENDER_VERSION-macos-arm64.dmg" "$BLENDER_SHA_macos_arm64" "$archive"; then
        mnt="$DOWNLOADS/blender-mnt"; mkdir -p "$mnt" "$dir"
        if hdiutil attach -nobrowse -readonly -quiet -mountpoint "$mnt" "$archive"; then
          cp -R "$mnt/Blender.app" "$dir/" || true
          hdiutil detach -quiet "$mnt" || true
        fi
        rm -f "$archive"
        [ -x "$dir/Blender.app/Contents/MacOS/Blender" ] && bin="$dir/Blender.app/Contents/MacOS/Blender"
      fi
    elif command -v brew >/dev/null 2>&1; then
      say "Installing Blender with Homebrew"
      brew install --cask blender >/dev/null 2>&1 || true
      command -v blender >/dev/null 2>&1 && return 0
    fi
  fi
  if [ -n "$bin" ]; then set_tool blender "$bin"; say "Blender ready"
  else warn "Blender could not be installed; quick renders still work."; add_missing "Blender"; fi
}

install_colmap() {
  local dir="$TOOLS/colmap-$COLMAP_VERSION" bin="" archive
  command -v colmap >/dev/null 2>&1 && { say "COLMAP already installed: $(command -v colmap)"; return 0; }
  bin="$(find_file "$dir" colmap || true)"
  if [ -z "$bin" ] && [ "$OS" = "Darwin" ] && [ "$ARCH" = "aarch64" ]; then
    archive="$DOWNLOADS/colmap-$COLMAP_VERSION-macos-arm64.zip"
    say "Downloading COLMAP $COLMAP_VERSION (photogrammetry)"
    if get_verified "https://github.com/colmap/colmap/releases/download/$COLMAP_VERSION/colmap-arm64-macos.zip" "$COLMAP_SHA_macos_arm64" "$archive"; then
      mkdir -p "$dir" && unzip -q -o "$archive" -d "$dir" && rm -f "$archive"
      bin="$(find_file "$dir" colmap || true)"; [ -n "$bin" ] && chmod +x "$bin"
    fi
  fi
  if [ -z "$bin" ]; then
    # No portable Linux build exists: use a package manager, but only without a password prompt.
    if command -v brew >/dev/null 2>&1; then
      say "Installing COLMAP with Homebrew"; brew install colmap >/dev/null 2>&1 || true
    elif command -v apt-get >/dev/null 2>&1 && can_sudo; then
      say "Installing COLMAP with apt"; sudo -n apt-get install -y -qq colmap >/dev/null 2>&1 || true
    elif command -v dnf >/dev/null 2>&1 && can_sudo; then
      say "Installing COLMAP with dnf"; sudo -n dnf install -y -q colmap >/dev/null 2>&1 || true
    fi
    command -v colmap >/dev/null 2>&1 && { say "COLMAP ready"; return 0; }
  fi
  if [ -n "$bin" ]; then set_tool colmap "$bin"; say "COLMAP ready"
  else
    warn "COLMAP needs administrator rights here (Ubuntu: sudo apt install colmap). Everything else works."
    add_missing "COLMAP"
  fi
}

if [ "$DO_TOOLS" -eq 1 ]; then
  install_blender || { warn "Blender step failed"; add_missing "Blender"; }
  install_colmap || { warn "COLMAP step failed"; add_missing "COLMAP"; }
fi

# ─── 6. Agents ────────────────────────────────────────────────────────────────
if [ "$DO_SETUP" -eq 1 ]; then
  set -- setup --agents "$AGENTS" --workspace "$WORKSPACE"
  [ -n "$DEVICE" ] && set -- "$@" --device "$DEVICE"
  [ -n "$PROFILE" ] && set -- "$@" --profile "$PROFILE"
  [ -n "$TOOLSET" ] && set -- "$@" --toolset "$TOOLSET"
  [ "$AUTO_APPROVE" -eq 0 ] && set -- "$@" --no-auto-approve
  say "Connecting reverse3d to your AI agents"
  "$EXE" "$@" || { warn "Agent setup reported a problem; run 'reverse3d setup --snippets' for manual config."; add_missing "agent registration"; }
fi

# ─── 7. Offline readiness ─────────────────────────────────────────────────────
# Downloads the pinned AI depth model once, so reverse3d keeps working without
# internet (no-op when the AI depth module is not installed).
say "Preparing offline mode (downloads the AI depth model once)"
OFFLINE_OUT="$("$EXE" --json offline prepare 2>/dev/null || true)"
case "$OFFLINE_OUT" in
  *'"ready": true'*) say "Offline mode ready: reverse3d works without internet" ;;
  *) warn "AI depth model not downloaded yet; run 'reverse3d offline prepare' when online."
     add_missing "offline AI depth model" ;;
esac

# ─── 8. Summary ───────────────────────────────────────────────────────────────
if "$EXE" doctor >/dev/null 2>&1; then DOCTOR_OK=true; RESULT=OK; else DOCTOR_OK=false; RESULT=CHECK; fi
write_status "$DOCTOR_OK"
if [ -n "$MISSING" ]; then NOTE="Not installed: $MISSING"; else NOTE="Everything installed."; fi
cat <<EOF

  reverse3d is ready.   Workspace: $WORKSPACE
  $NOTE

  Next:
    1. Restart your AI agent (Claude, Cursor, VS Code, Gemini, Codex, OpenCode…).
    2. Put photos, plans or models in the workspace and ask, for example:
       "usa reverse3d para convertir mis fotos en un modelo 3D"
    3. Or use the autopilot, no agent needed:  $EXE watch
       and drop photos, drawings or STL files in $WORKSPACE/bandeja

  Works offline.  Check anytime:  $EXE doctor
RESULT: $RESULT
EOF
