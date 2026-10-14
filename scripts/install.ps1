<#
.SYNOPSIS
  reverse3d installer for Windows — Voxel Enterprise (https://voxel.lat)

.DESCRIPTION
  1. Finds Python >= 3.10 (py launcher or python on PATH)
  2. Creates an isolated virtualenv in %USERPROFILE%\.reverse3d\venv (no admin rights needed)
  3. Installs reverse3d with the selected extras (installs CUDA PyTorch when an NVIDIA GPU is found)
  4. Adds the venv Scripts folder to the *user* PATH
  5. Runs `reverse3d setup` to register the MCP server in the AI agents found on this machine
     (each modified config file is backed up first)

.PARAMETER Performance
  fast | balanced | quality (stored as the default profile).

.PARAMETER NoPath
  Do not add the venv to the user PATH.

.EXAMPLE
  irm https://raw.githubusercontent.com/rellytechgame/Reverse3D-tool/main/scripts/install.ps1 | iex

.EXAMPLE
  & ([scriptblock]::Create((irm https://raw.githubusercontent.com/rellytechgame/Reverse3D-tool/main/scripts/install.ps1))) -Full -Device gpu
#>
[CmdletBinding()]
param(
    [switch]$Full,
    [switch]$Minimal,
    [string]$With = "",
    [ValidateSet("", "auto", "cpu", "gpu")][string]$Device = "",
    [ValidateSet("", "fast", "balanced", "quality")][string]$Performance = "",
    [ValidateSet("", "full", "lean")][string]$Toolset = "",
    [string]$Agents = "auto",
    [string]$Workspace = "",
    [string]$Ref = "main",
    [string]$Source = "",
    [switch]$NoSetup,
    [switch]$NoPath,
    [switch]$Uninstall
)
$ErrorActionPreference = "Stop"

$RepoUrl = if ($env:REVERSE3D_REPO) { $env:REVERSE3D_REPO } else { "https://github.com/rellytechgame/Reverse3D-tool" }
$R3dHome = if ($env:REVERSE3D_HOME) { $env:REVERSE3D_HOME } else { Join-Path $env:USERPROFILE ".reverse3d" }
$Venv = Join-Path $R3dHome "venv"
$Scripts = Join-Path $Venv "Scripts"
$Exe = Join-Path $Scripts "reverse3d.exe"
if (-not $Workspace) { $Workspace = Join-Path $env:USERPROFILE "reverse3d-workspace" }

function Say($m) { Write-Host "[reverse3d] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[reverse3d] $m" -ForegroundColor Yellow }
function Die($m) { Write-Host "[reverse3d] $m" -ForegroundColor Red; exit 1 }

$Extras = "mcp,render"
if ($Full) { $Extras = "mcp,render,recon,depth" }
if ($Minimal) { $Extras = "mcp" }
if ($With) { $Extras = "mcp,$With" }
if ($Extras -notmatch '^[a-z0-9,]+$') { Die "Invalid -With value" }
if ($Ref -notmatch '^[A-Za-z0-9._/-]+$') { Die "Invalid -Ref value" }

function Remove-FromUserPath($dir) {
    $p = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($p) {
        $parts = $p.Split(";") | Where-Object { $_ -and ($_ -ne $dir) }
        [Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "User")
    }
}

if ($Uninstall) {
    if (Test-Path $Exe) { & $Exe setup --uninstall --agents all }
    if (Test-Path $Venv) { Remove-Item -Recurse -Force $Venv }
    Remove-FromUserPath $Scripts
    Say "Uninstalled. Your workspace ($Workspace) and $R3dHome\config.json were kept."
    exit 0
}

# 1. Python ---------------------------------------------------------------------------------
$wantO3d = $Extras -match "recon"
$candidates = if ($wantO3d) { @("-3.12", "-3.11", "-3.10", "-3.13") } else { @("-3.13", "-3.12", "-3.11", "-3.10") }
$Py = $null; $PyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in $candidates) {
        & py $v -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $Py = "py"; $PyArgs = @($v); break }
    }
}
if (-not $Py -and (Get-Command python -ErrorAction SilentlyContinue)) {
    & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
    if ($LASTEXITCODE -eq 0) { $Py = "python" }
}
if (-not $Py) { Die "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH')." }
$PyVer = & $Py @PyArgs -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
Say "Using Python $PyVer"
if ($wantO3d -and $PyVer -eq "3.13") {
    Warn "Open3D has no wheels for Python 3.13 yet: skipping 'recon' (install Python 3.12 to enable it)."
    $Extras = ($Extras.Split(",") | Where-Object { $_ -ne "recon" }) -join ","
}

# 2. GPU ------------------------------------------------------------------------------------
$HasNvidia = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $gpu = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1
    if ($LASTEXITCODE -eq 0 -and $gpu) { $HasNvidia = $true; Say "NVIDIA GPU detected: $gpu" }
}
if (-not $HasNvidia) { Say "No NVIDIA GPU detected: everything will run on CPU." }

# 3. Virtualenv + install -------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $R3dHome | Out-Null
if (-not (Test-Path (Join-Path $Scripts "python.exe"))) {
    Say "Creating isolated environment in $Venv"
    & $Py @PyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Die "Could not create the virtualenv." }
}
$VPy = Join-Path $Scripts "python.exe"
& $VPy -m pip install --quiet --upgrade pip

# PyPI ships CPU-only PyTorch on Windows: fetch the CUDA build when a GPU is present.
if ($HasNvidia -and $Extras -match "depth") {
    $TorchIndex = if ($env:REVERSE3D_TORCH_INDEX) { $env:REVERSE3D_TORCH_INDEX } else { "https://download.pytorch.org/whl/cu126" }
    Say "Installing CUDA PyTorch from $TorchIndex"
    & $VPy -m pip install torch --index-url $TorchIndex
}

if ($Source) { $Spec = "$((Resolve-Path $Source).Path)[$Extras]" }
else { $Spec = "reverse3d[$Extras] @ git+$RepoUrl@$Ref" }
Say "Installing reverse3d [$Extras] (this can take a few minutes)..."
& $VPy -m pip install --upgrade $Spec
if ($LASTEXITCODE -ne 0) { Die "pip install failed" }

# 4. PATH (user scope) ----------------------------------------------------------------------
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $NoPath -and (-not $userPath -or -not ($userPath.Split(";") -contains $Scripts))) {
    [Environment]::SetEnvironmentVariable("Path", (@($userPath, $Scripts) | Where-Object { $_ }) -join ";", "User")
    Say "Added $Scripts to your user PATH (open a new terminal to use 'reverse3d')."
}

# 5. Agents ---------------------------------------------------------------------------------
if (-not $NoSetup) {
    $setupArgs = @("setup", "--agents", $Agents, "--workspace", $Workspace)
    if ($Device) { $setupArgs += @("--device", $Device) } elseif ($HasNvidia) { $setupArgs += @("--device", "auto") }
    if ($Performance) { $setupArgs += @("--profile", $Performance) }
    if ($Toolset) { $setupArgs += @("--toolset", $Toolset) }
    Say "Registering reverse3d in your AI agents..."
    & $Exe @setupArgs
    if ($LASTEXITCODE -ne 0) { Warn "Agent setup reported a problem; run 'reverse3d setup --snippets' for manual config." }
}

# 6. Summary --------------------------------------------------------------------------------
& $Exe doctor *> $null
if ($LASTEXITCODE -eq 0) { Say "Installation verified (reverse3d doctor OK)." }
Write-Host @"

  reverse3d is ready.   Workspace: $Workspace

  Next:
    1. Restart your AI agent (Claude, Cursor, VS Code, Gemini, Codex, OpenCode...).
    2. Ask it:  "usa reverse3d para medir y renderizar mi pieza.stl"

  Optional external tools (more power):
    Blender (photoreal renders):  winget install BlenderFoundation.Blender
    COLMAP  (photogrammetry):     https://github.com/colmap/colmap/releases  (CUDA build for NVIDIA)
  Check anytime:  reverse3d doctor     GPU:  reverse3d config set device gpu

"@
