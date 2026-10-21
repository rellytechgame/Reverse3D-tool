<#
.SYNOPSIS
  reverse3d installer for Windows — Voxel Enterprise (https://voxel.lat)

.DESCRIPTION
  Zero-question installer: it never prompts, needs no admin rights and installs
  everything by default. Safe to re-run: finished steps are skipped, so if an
  AI agent stops it on a timeout, running it again continues where it stopped.

  1. Gets Python 3.12 through uv (downloaded and SHA-256 verified; no admin,
     nothing installed system-wide). Falls back to an existing Python 3.10+.
  2. Creates an isolated environment in %USERPROFILE%\.reverse3d\venv.
  3. Installs reverse3d with all modules. PyTorch comes from the CUDA index on
     NVIDIA machines, otherwise the small CPU build. An optional module that
     fails never stops the install.
  4. Downloads portable Blender and COLMAP into %USERPROFILE%\.reverse3d\tools
     (pinned versions, SHA-256 verified).
  5. Registers the MCP server in every AI agent found and pre-approves the
     reverse3d tools (each modified config file is backed up first).
  6. Writes %USERPROFILE%\.reverse3d\install-status.json and install.log.

.PARAMETER Lite
  Run the local AI on the CPU: installs the CPU build of PyTorch (~250 MB instead
  of ~2.5 GB with CUDA). The local AI models are always installed and downloaded.

.PARAMETER Minimal
  Only the core, the MCP server and the local AI; no other extras and no
  external tools (Blender, COLMAP).

.PARAMETER NoTools
  Do not download Blender and COLMAP.

.PARAMETER NoAutoApprove
  Register the agents but let them ask before each reverse3d tool call.

.PARAMETER Performance
  fast | balanced | quality (stored as the default profile).

.EXAMPLE
  irm https://raw.githubusercontent.com/rellytechgame/Reverse3D-tool/main/scripts/install.ps1 | iex

.EXAMPLE
  & ([scriptblock]::Create((irm https://raw.githubusercontent.com/rellytechgame/Reverse3D-tool/main/scripts/install.ps1))) -Lite -Device cpu
#>
[CmdletBinding()]
param(
    [switch]$Full,          # kept for compatibility: everything is the default now
    [switch]$Lite,
    [switch]$Minimal,
    [string]$With = "",
    [switch]$NoTools,
    [switch]$NoAutoApprove,
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
# "Continue": Windows PowerShell 5.1 turns any stderr line of a native tool (pip, uv) into a
# terminating error under "Stop". Native steps are checked through $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is 10x slower with the progress bar
$ConfirmPreference = "None"
$env:PIP_NO_INPUT = "1"
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

# ─── Pinned downloads (bump version + SHA-256 together) ───────────────────────────────────
$UvVersion = "0.12.19"
$UvSha = @{
    "x86_64"  = "6dbb02d79e419522f1c500f0adb1cddcff0cda7d59b0d66ea7f5e3b4a1b2f5f0"
    "aarch64" = "115b54cb823bc48260670f5782001add6067ac8d98d18c8263a833704e287de9"
}
$BlenderVersion = "5.2.2"
$BlenderSha = @{
    "x64"   = "3849d17a682cba006075aaa3f3597ecb5c9c30ec31035b2e092c53e40679b535"
    "arm64" = "2e90b53c9443cc56d2be75a61550a38e4aba942c8cded046ceff98ed58ebc67b"
}
$ColmapVersion = "4.2.1"
$ColmapSha = @{
    "cuda"   = "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881"
    "nocuda" = "c493d88cb4a43f21afba42cb55bf786e5434a0420e73db0ba1454c5d8b3686ae"
}

$RepoUrl = if ($env:REVERSE3D_REPO) { $env:REVERSE3D_REPO } else { "https://github.com/rellytechgame/Reverse3D-tool" }
$R3dHome = if ($env:REVERSE3D_HOME) { $env:REVERSE3D_HOME } else { Join-Path $env:USERPROFILE ".reverse3d" }
$Venv = Join-Path $R3dHome "venv"
$Scripts = Join-Path $Venv "Scripts"
$Exe = Join-Path $Scripts "reverse3d.exe"
$VPy = Join-Path $Scripts "python.exe"
$Tools = Join-Path $R3dHome "tools"
$Downloads = Join-Path $R3dHome "downloads"
$StatusFile = Join-Path $R3dHome "install-status.json"
if (-not $Workspace) { $Workspace = Join-Path $env:USERPROFILE "reverse3d-workspace" }
$Arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }

$Missing = New-Object System.Collections.Generic.List[string]
function Say($m) { Write-Host "[reverse3d] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[reverse3d] $m" -ForegroundColor Yellow }
function Die($m) {
    Write-Host "[reverse3d] $m" -ForegroundColor Red
    Write-Status $false $m
    try { Stop-Transcript | Out-Null } catch { }
    exit 1
}
function Write-Status([bool]$ok, [string]$err = "") {
    try {
        New-Item -ItemType Directory -Force -Path $R3dHome | Out-Null
        $status = [ordered]@{
            ok        = $ok
            error     = $err
            finished  = (Get-Date).ToString("s")
            command   = $Exe
            workspace = $Workspace
            extras    = $script:Extras
            missing   = @($Missing)
            log       = (Join-Path $R3dHome "install.log")
        }
        # No BOM: Set-Content -Encoding utf8 adds one in Windows PowerShell and breaks JSON readers.
        [IO.File]::WriteAllText($StatusFile, ($status | ConvertTo-Json), (New-Object System.Text.UTF8Encoding $false))
    } catch { }
}

$Extras = "mcp,render,recon,depth"
# The local AI (extra "depth": PyTorch + transformers) is always installed: every
# mode works offline. -Lite only switches it to the smaller CPU build.
if ($Minimal) { $Extras = "mcp,depth"; $NoTools = $true }
if (($Extras.Split(",")) -notcontains "depth") { $Extras = "$Extras,depth" }
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
    foreach ($d in @($Venv, $Tools, $Downloads, (Join-Path $R3dHome "bin"), (Join-Path $R3dHome "python"))) {
        if (Test-Path $d) { Remove-Item -Recurse -Force $d }
    }
    Remove-FromUserPath $Scripts
    Say "Uninstalled. Your workspace ($Workspace) and $R3dHome\config.json were kept."
    exit 0
}

New-Item -ItemType Directory -Force -Path $R3dHome, $Downloads | Out-Null
try { Start-Transcript -Path (Join-Path $R3dHome "install.log") -Force | Out-Null } catch { }
Say "Installing reverse3d. No questions will be asked; this can take 10-20 minutes."

# ─── helpers ────────────────────────────────────────────────────────────────────────────────

# Download to a temp name, verify SHA-256, then move into place. Re-runs reuse a verified file.
function Get-Verified([string]$Url, [string]$Sha, [string]$Dest) {
    if ((Test-Path $Dest) -and ((Get-FileHash -Algorithm SHA256 $Dest).Hash -eq $Sha.ToUpper())) { return $true }
    $tmp = "$Dest.part"
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $tmp -ErrorAction Stop
            if ((Get-FileHash -Algorithm SHA256 $tmp).Hash -eq $Sha.ToUpper()) {
                Move-Item -Force $tmp $Dest
                return $true
            }
            Warn "Checksum mismatch for $Url (attempt $i)"
        } catch {
            Warn "Download failed (attempt $i): $($_.Exception.Message)"
        }
        Remove-Item -Force -ErrorAction SilentlyContinue $tmp
        Start-Sleep -Seconds (3 * $i)
    }
    return $false
}

# tar.exe (Windows 10+) extracts large zips far faster than Expand-Archive.
function Expand-Zip([string]$Zip, [string]$Dest) {
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $tar = Join-Path $env:SystemRoot "System32\tar.exe"
    if (Test-Path $tar) {
        & $tar -xf $Zip -C $Dest
        if ($LASTEXITCODE -eq 0) { return }
    }
    Expand-Archive -Force -Path $Zip -DestinationPath $Dest
}

function Find-File([string]$Root, [string[]]$Names) {
    foreach ($n in $Names) {
        $hit = Get-ChildItem -Path $Root -Recurse -Depth 3 -File -Filter $n -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

# ─── 1. Python (uv-managed 3.12, or an existing 3.10+) ─────────────────────────────────────
$Uv = $null
$UvArch = if ($Arch -eq "arm64") { "aarch64" } else { "x86_64" }
$UvDir = Join-Path $R3dHome "bin"
$UvExe = Join-Path $UvDir "uv.exe"
if (-not (Test-Path $UvExe)) {
    $zip = Join-Path $Downloads "uv-$UvVersion-$UvArch.zip"
    $url = "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-$UvArch-pc-windows-msvc.zip"
    Say "Fetching uv $UvVersion (Python manager, no admin needed)"
    if (Get-Verified $url $UvSha[$UvArch] $zip) {
        $tmpDir = Join-Path $Downloads "uv-extract"
        Expand-Zip $zip $tmpDir
        $found = Find-File $tmpDir @("uv.exe")
        if ($found) {
            New-Item -ItemType Directory -Force -Path $UvDir | Out-Null
            Copy-Item -Force $found $UvExe
        }
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $tmpDir
    }
}
if (Test-Path $UvExe) {
    $Uv = $UvExe
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $R3dHome "python"
    $env:UV_NO_PROGRESS = "1"
} else {
    Warn "uv unavailable: falling back to a Python already installed on this PC."
}

function Get-VenvVersion {
    if (-not (Test-Path $VPy)) { return $null }
    try { return (& $VPy -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')").Trim() } catch { return $null }
}

$wantO3d = $Extras -match "recon"
$current = Get-VenvVersion
if ($current -and $Uv -and $wantO3d -and @("3.10", "3.11", "3.12") -notcontains $current) {
    # Open3D has no wheels for newer Pythons: rebuild the venv on 3.12 when possible.
    Say "Rebuilding the environment on Python 3.12 (was $current) so 3D meshing can be installed"
    try { Remove-Item -Recurse -Force $Venv -ErrorAction Stop; $current = $null }
    catch { Warn "The environment is in use (close your AI agent to rebuild it). Keeping Python $current." }
}

if (-not $current) {
    if ($Uv) {
        Say "Preparing Python 3.12 (isolated, in $R3dHome)"
        & $Uv python install 3.12 --quiet
        & $Uv venv --python 3.12 --seed --clear --quiet $Venv
        if ($LASTEXITCODE -ne 0) { $Uv = $null; Warn "uv could not create the environment; trying an installed Python." }
    }
    if (-not (Test-Path $VPy)) {
        $Py = $null; $PyArgs = @()
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($v in @("-3.12", "-3.11", "-3.10", "-3.13", "-3.14")) {
                & py $v -c "import sys" 2>$null
                if ($LASTEXITCODE -eq 0) { $Py = "py"; $PyArgs = @($v); break }
            }
        }
        if (-not $Py -and (Get-Command python -ErrorAction SilentlyContinue)) {
            & python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>$null
            if ($LASTEXITCODE -eq 0) { $Py = "python" }
        }
        if (-not $Py) { Die "Could not get Python. Check the internet connection and run this installer again." }
        & $Py @PyArgs -m venv $Venv
        if ($LASTEXITCODE -ne 0) { Die "Could not create the Python environment." }
    }
}
$PyVer = Get-VenvVersion
Say "Python $PyVer ready"
if ($wantO3d -and @("3.10", "3.11", "3.12") -notcontains $PyVer) {
    Warn "Open3D has no wheels for Python ${PyVer}: 3D meshing from point clouds is skipped."
    $Missing.Add("recon (Open3D needs Python 3.10-3.12)")
    $Extras = ($Extras.Split(",") | Where-Object { $_ -ne "recon" }) -join ","
}

function Install-Pkgs([string[]]$PkgArgs) {
    for ($i = 1; $i -le 2; $i++) {
        if ($Uv) { & $Uv pip install --python $VPy --quiet @PkgArgs }
        else { & $VPy -m pip install --quiet --upgrade @PkgArgs }
        if ($LASTEXITCODE -eq 0) { return $true }
        Start-Sleep -Seconds 3
    }
    return $false
}

# ─── 2. GPU ─────────────────────────────────────────────────────────────────────────────────
$HasNvidia = $false
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    # Capture everything first: cutting the pipeline (Select -First) leaves a bogus exit code.
    $smiOut = @(& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null)
    if ($LASTEXITCODE -eq 0 -and $smiOut.Count -gt 0 -and $smiOut[0]) {
        $HasNvidia = $true; Say "NVIDIA GPU detected: $($smiOut[0])"
    }
}
if (-not $HasNvidia) { Say "No NVIDIA GPU detected: everything will run on CPU." }

# ─── 3. reverse3d + modules ─────────────────────────────────────────────────────────────────
# A GitHub archive needs no git on the user's machine.
if ($Source) { $Base = (Resolve-Path $Source).Path } else { $Base = "reverse3d @ $RepoUrl/archive/$Ref.zip" }
function Spec([string]$ex) { if ($Source) { return "$Base[$ex]" } else { return $Base.Replace("reverse3d @", "reverse3d[$ex] @") } }
$Refresh = if ($Uv) { @("--reinstall-package", "reverse3d") } else { @() }

$extraList = @($Extras.Split(",") | Where-Object { $_ })
$core = @(@("mcp", "render") | Where-Object { $extraList -contains $_ })
if (-not $core) { $core = @("mcp") }
$optional = @($extraList | Where-Object { $core -notcontains $_ })

if ($optional -contains "depth") {
    # PyPI only ships CPU PyTorch for Windows; the CPU index build is ~10x smaller than the default.
    $TorchIndex = if ($env:REVERSE3D_TORCH_INDEX) { $env:REVERSE3D_TORCH_INDEX }
                  elseif ($HasNvidia -and -not $Lite) { "https://download.pytorch.org/whl/cu126" }
                  else { "https://download.pytorch.org/whl/cpu" }
    & $VPy -c "import torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Say "Installing PyTorch from $TorchIndex (large download, please wait)"
        if (-not (Install-Pkgs @("torch", "--index-url", $TorchIndex))) {
            Warn "CUDA PyTorch failed; trying the CPU build."
            [void](Install-Pkgs @("torch", "--index-url", "https://download.pytorch.org/whl/cpu"))
        }
    }
}

Say "Installing reverse3d [$($core -join ',')]"
if (-not (Install-Pkgs (@((Spec ($core -join ","))) + $Refresh))) { Die "Could not install reverse3d. Check the internet connection and run this installer again." }
$installed = @($core)
foreach ($ex in $optional) {
    Say "Installing module '$ex'"
    if (Install-Pkgs @((Spec (($core + $ex) -join ",")))) { $installed += $ex }
    else { Warn "Module '$ex' could not be installed; reverse3d works without it."; $Missing.Add("module $ex") }
}
$Extras = $installed -join ","

# Optiverse's own voice runs on llama.cpp (MIT): a prebuilt CPU wheel, no compiler needed.
Say "Installing Optiverse's voice runtime (llama.cpp)"
if (-not (Install-Pkgs @("llama-cpp-python>=0.3.16", "--only-binary", ":all:", "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"))) {
    Warn "llama.cpp could not be installed: Optiverse answers with its written replies."
    $Missing.Add("Optiverse voice runtime (llama.cpp)")
}

# ─── 4. PATH (user scope) ───────────────────────────────────────────────────────────────────
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not $NoPath -and (-not $userPath -or -not ($userPath.Split(";") -contains $Scripts))) {
    [Environment]::SetEnvironmentVariable("Path", (@($userPath, $Scripts) | Where-Object { $_ }) -join ";", "User")
    Say "Added reverse3d to your user PATH."
}

# ─── 5. Blender + COLMAP (portable, verified) ───────────────────────────────────────────────
function Set-ToolPath([string]$Key, [string]$Path) {
    & $Exe config set $Key $Path *> $null
    if ($LASTEXITCODE -ne 0) { Warn "Could not store the $Key path in the configuration." }
}

if (-not $NoTools) {
    $blenderDir = Join-Path $Tools "blender-$BlenderVersion"
    $blender = Find-File $blenderDir @("blender.exe")
    if (-not $blender -and (Get-Command blender -ErrorAction SilentlyContinue)) {
        Say "Blender already installed: $((Get-Command blender).Source)"
    } elseif (-not $blender) {
        $zip = Join-Path $Downloads "blender-$BlenderVersion-windows-$Arch.zip"
        $url = "https://download.blender.org/release/Blender$($BlenderVersion.Substring(0, $BlenderVersion.LastIndexOf('.')))/blender-$BlenderVersion-windows-$Arch.zip"
        Say "Downloading Blender $BlenderVersion (~400 MB, photoreal renders)"
        if (Get-Verified $url $BlenderSha[$Arch] $zip) {
            Expand-Zip $zip $blenderDir
            $blender = Find-File $blenderDir @("blender.exe")
            Remove-Item -Force -ErrorAction SilentlyContinue $zip
        }
        if (-not $blender) { Warn "Blender could not be installed; quick renders still work."; $Missing.Add("Blender") }
    }
    if ($blender) { Set-ToolPath "blender" $blender; Say "Blender ready" }

    $flavor = if ($HasNvidia) { "cuda" } else { "nocuda" }
    # One folder per flavour: a re-run after adding an NVIDIA GPU fetches the CUDA build.
    $colmapDir = Join-Path $Tools "colmap-$ColmapVersion-$flavor"
    $colmap = Find-File $colmapDir @("COLMAP.bat", "colmap.exe")
    if ($Arch -eq "arm64") {
        Warn "COLMAP has no Windows ARM build: photogrammetry is unavailable on this PC."; $Missing.Add("COLMAP")
    } elseif (-not $colmap -and (Get-Command colmap -ErrorAction SilentlyContinue)) {
        Say "COLMAP already installed: $((Get-Command colmap).Source)"
    } elseif (-not $colmap) {
        $zip = Join-Path $Downloads "colmap-$ColmapVersion-$flavor.zip"
        $url = "https://github.com/colmap/colmap/releases/download/$ColmapVersion/colmap-x64-windows-$flavor.zip"
        Say "Downloading COLMAP $ColmapVersion ($flavor, photogrammetry)"
        if (Get-Verified $url $ColmapSha[$flavor] $zip) {
            Expand-Zip $zip $colmapDir
            $colmap = Find-File $colmapDir @("COLMAP.bat", "colmap.exe")
            Remove-Item -Force -ErrorAction SilentlyContinue $zip
        }
        if (-not $colmap) { Warn "COLMAP could not be installed; photo-to-3D needs it."; $Missing.Add("COLMAP") }
    }
    if ($colmap) { Set-ToolPath "colmap" $colmap; Say "COLMAP ready" }
}

# ─── 6. Agents ──────────────────────────────────────────────────────────────────────────────
if (-not $NoSetup) {
    $setupArgs = @("setup", "--agents", $Agents, "--workspace", $Workspace)
    if ($Device) { $setupArgs += @("--device", $Device) } elseif ($HasNvidia) { $setupArgs += @("--device", "auto") }
    if ($Performance) { $setupArgs += @("--profile", $Performance) }
    if ($Toolset) { $setupArgs += @("--toolset", $Toolset) }
    if ($NoAutoApprove) { $setupArgs += "--no-auto-approve" }
    Say "Connecting reverse3d to your AI agents"
    & $Exe @setupArgs
    if ($LASTEXITCODE -ne 0) { Warn "Agent setup reported a problem; run 'reverse3d setup --snippets' for manual config."; $Missing.Add("agent registration") }
}

# ─── 7. Offline readiness ───────────────────────────────────────────────────────────────────
# Downloads the pinned local AI models once (subject classifier, segmentation and
# depth), so every reverse3d feature keeps working without internet.
Say "Downloading the local AI (subject, cut-out and depth models, about 750 MB) so everything works offline"
$offlineOut = (& $Exe --json offline prepare 2>$null) | Out-String
if ($offlineOut -match '"ready":\s*true') {
    Say "Offline mode ready: reverse3d works without internet"
} else {
    Warn "Local AI models not downloaded yet; run 'reverse3d offline prepare' when online."
    $Missing.Add("offline local AI models")
}

# ─── 8. Summary ─────────────────────────────────────────────────────────────────────────────
& $Exe doctor *> $null
$doctorOk = ($LASTEXITCODE -eq 0)
Write-Status $doctorOk
$extraNote = if ($Missing.Count) { "Not installed: $($Missing -join '; ')" } else { "Everything installed." }
Write-Host @"

  reverse3d is ready.   Workspace: $Workspace
  $extraNote

  Next:
    1. Restart your AI agent (Claude, Cursor, VS Code, Gemini, Codex, OpenCode...).
    2. Put photos, plans or models in the workspace and ask, for example:
       "usa reverse3d para convertir mis fotos en un modelo 3D"
    3. Or use the autopilot, no agent needed:  reverse3d watch
       and drop photos, drawings or STL files in $Workspace\bandeja

  Works offline.  Check anytime:  reverse3d doctor
RESULT: $(if ($doctorOk) { "OK" } else { "CHECK" })
"@
try { Stop-Transcript | Out-Null } catch { }
