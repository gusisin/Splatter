<#
  Splatter / 3DGRUT one-shot installer (Windows, NVIDIA RTX required)

  Usage (from a PowerShell terminal in the repo root):

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\install.ps1
    .\install.ps1 -Debug   # test mode: bypass GPU compatibility gate

  This script will:
    - Verify that an RTX-class NVIDIA GPU (compute capability >= 7.0) is available
    - Create a Python virtual environment under .venv-splatter
    - Install GUI dependencies from requirements-gui.txt
    - Clone the 3DGRUT repo into tools\3dgrut if not already present
    - Run 3DGRUT's own install_env_uv.ps1 to set up its CUDA + Torch environment
    - (Optionally) verify ffmpeg/ffprobe and COLMAP presence, and point a shim to them

  The script is intentionally conservative: if core prerequisites are missing,
  it will emit a clear message and exit with a non-zero code instead of guessing.
#>

param(
    [switch]$Debug
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-ErrorLine($msg) { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Fail($msg, [int]$code = 1) {
    Write-ErrorLine $msg
    exit $code
}

# Resolve repo root as the location of this script
$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptRoot
Write-Info "Using repo root: $ScriptRoot"

### 0. Check NVIDIA GPU compatibility (RTX class, cc >= 7.0)

function Get-GpuInfo {
    if (-not (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
        return ,@()
    }

    $raw = & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if (-not $raw) {
        return ,@()
    }

    $list = @()
    foreach ($line in @($raw)) {
        $name = $line.Trim()
        if ($name) {
            $list += [PSCustomObject]@{
                Name = $name
            }
        }
    }
    return ,$list
}

Write-Info "Checking for compatible NVIDIA GPU..."
$gpus = @(Get-GpuInfo)
if ($gpus.Count -eq 0) {
    if ($Debug) {
        Write-Warn "Debug mode enabled: skipping NVIDIA GPU compatibility gate (no GPU detected)."
    } else {
        Fail "No NVIDIA GPU detected via nvidia-smi. 3DGRUT requires an RTX-class NVIDIA GPU (compute capability >= 7.0)."
    }
}

$supported = $false
foreach ($gpu in $gpus) {
    $name = $gpu.Name
    # Heuristic: require RTX in the name (20/30/40/50 series and beyond).
    if ($name -match "RTX") {
        $supported = $true
        Write-Info "Detected compatible GPU: $name"
        break
    }
}

if (($gpus.Count -gt 0) -and (-not $supported)) {
    Write-Warn "Detected NVIDIA GPU(s):"
    foreach ($gpu in $gpus) {
        Write-Warn "  - $($gpu.Name)"
    }
    if ($Debug) {
        Write-Warn "Debug mode enabled: proceeding despite non-RTX GPU(s)."
    } else {
        Fail "No RTX-class NVIDIA GPU detected. 3DGRUT kernels target RTX-generation GPUs only. Please upgrade to an RTX 20/30/40/50 series card and rerun this installer."
    }
}

### 1. Ensure Python is available

Write-Info "Checking for Python 3.10+..."
try {
    $pyVersion = & python -c "import sys; print(sys.version.split()[0])" 2>$null
} catch {
    $pyVersion = $null
}

if (-not $pyVersion) {
    Fail "Python not found on PATH. Please install Python 3.10+ from python.org, ensure 'python' is on PATH, then rerun .\install.ps1."
}

Write-Info "Found Python $pyVersion"

### 2. Create / reuse venv for Splatter UI

$venvPath = Join-Path $ScriptRoot ".venv-splatter"
if (-not (Test-Path $venvPath)) {
    Write-Info "Creating virtual environment at $venvPath..."
    & python -m venv $venvPath
} else {
    Write-Info "Reusing existing virtual environment at $venvPath"
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Fail "Virtual environment seems corrupted (missing Scripts\python.exe). Delete .venv-splatter and rerun .\install.ps1."
}

### 3. Install GUI requirements

Write-Info "Installing Python GUI dependencies into .venv-splatter..."
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -r (Join-Path $ScriptRoot "requirements-gui.txt")

### 4. Clone / update 3dgrut under tools\3dgrut

$toolsDir = Join-Path $ScriptRoot "tools"
if (-not (Test-Path $toolsDir)) {
    New-Item -ItemType Directory -Path $toolsDir | Out-Null
}

$grutDir = Join-Path $toolsDir "3dgrut"
if (-not (Test-Path $grutDir)) {
    Write-Info "Cloning 3dgrut into $grutDir..."
    & git clone --recursive "https://github.com/nv-tlabs/3dgrut.git" $grutDir
} else {
    Write-Info "3dgrut already present at $grutDir"
    Write-Info "You may run 'git pull' manually inside that folder if you want the latest version."
}

### 5. Run 3dgrut's own environment installer

Write-Info "Running 3dgrut install_env_uv.ps1..."
$installScript = Join-Path $grutDir "install_env_uv.ps1"
if (-not (Test-Path $installScript)) {
    Fail "Could not find install_env_uv.ps1 inside 3dgrut at $installScript. Ensure the repo cloned correctly."
}

Push-Location $grutDir
try {
    # install_env_uv.ps1 will manage its own .venv and CUDA config.
    & powershell -ExecutionPolicy Bypass -File $installScript
} finally {
    Pop-Location
}

### 6. (Optional) Check ffmpeg/ffprobe

Write-Info "Checking for ffmpeg/ffprobe..."
if (-not (Get-Command "ffmpeg" -ErrorAction SilentlyContinue)) {
    Write-Warn "ffmpeg not found on PATH. Stills extraction will fail until ffmpeg/ffprobe are installed."
    Write-Warn "Install recommendation (one-time):"
    Write-Warn "  choco install ffmpeg -y   # if you use Chocolatey"
}
if (-not (Get-Command "ffprobe" -ErrorAction SilentlyContinue)) {
    Write-Warn "ffprobe not found on PATH. Video duration probing will fall back to defaults."
}

### 7. (Optional) Check COLMAP

Write-Info "Checking for COLMAP..."
if (-not (Get-Command "colmap" -ErrorAction SilentlyContinue)) {
    Write-Warn "colmap not found on PATH. The 'Build COLMAP Dataset' step will fail until COLMAP is installed."
    Write-Warn "Install recommendation (one-time):"
    Write-Warn "  Download the Windows binaries from https://colmap.github.io/install.html"
    Write-Warn "  and add the extracted folder to your PATH."
}

### 8. Ensure MSVC C++ Build Tools (required by 3DGRUT JIT kernels)

function Find-MsvcCl {
    if (Get-Command "cl.exe" -ErrorAction SilentlyContinue) { return $true }
    $patterns = @(
        "C:\Program Files\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "C:\Program Files\Microsoft Visual Studio\*\Community\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "C:\Program Files\Microsoft Visual Studio\*\Professional\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "C:\Program Files\Microsoft Visual Studio\*\Enterprise\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe",
        "C:\Program Files (x86)\Microsoft Visual Studio\*\BuildTools\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
    )
    foreach ($p in $patterns) {
        if (Get-ChildItem -Path $p -ErrorAction SilentlyContinue) { return $true }
    }
    return $false
}

Write-Info "Checking for Microsoft Visual C++ compiler (cl.exe)..."
if (Find-MsvcCl) {
    Write-Info "  MSVC C++ Build Tools detected."
} else {
    Write-Info "  MSVC C++ Build Tools not detected. 3DGRUT requires them for JIT kernel compilation."
    $winget = Get-Command "winget" -ErrorAction SilentlyContinue
    if ($winget) {
        Write-Info "Attempting to install via winget (will prompt for elevation if needed)..."
        Write-Info "  Package : Microsoft.VisualStudio.2022.BuildTools"
        Write-Info "  Workload: Microsoft.VisualStudio.Workload.VCTools (Desktop development with C++)"
        & winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
            --accept-source-agreements --accept-package-agreements `
            --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --norestart"
        if ($LASTEXITCODE -eq 0 -and (Find-MsvcCl)) {
            Write-Info "  MSVC Build Tools installed successfully."
        } else {
            Write-Warn "winget exited with code $LASTEXITCODE and cl.exe still not found."
            Write-Warn "Manual install: https://visualstudio.microsoft.com/downloads/"
            Write-Warn "  Pick 'Build Tools for Visual Studio 2022' and the 'Desktop development with C++' workload."
        }
    } else {
        Write-Warn "winget not available on this machine, cannot auto-install."
        Write-Warn "Manual install (one-time, ~10 min):"
        Write-Warn "  1. Download 'Build Tools for Visual Studio 2022' from"
        Write-Warn "     https://visualstudio.microsoft.com/downloads/"
        Write-Warn "  2. In the installer, select 'Desktop development with C++' workload."
        Write-Warn "  3. Open a fresh PowerShell and re-run install.ps1 to confirm detection."
    }
}

### 9. Summary

Write-Host ""
Write-Host "================================ INSTALL COMPLETE ================================" -ForegroundColor Green
Write-Host ""
Write-Host "Environment summary:"
Write-Host "  - Splatter venv  : $venvPath"
Write-Host "  - 3dgrut repo    : $grutDir"
Write-Host "  - Python (UI)    : $venvPython"
Write-Host ""
Write-Host "To run the unified Splatter app (recommended):" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy RemoteSigned"
Write-Host "  Set-Location `"$ScriptRoot`""
Write-Host "  .\.venv-splatter\Scripts\Activate.ps1"
Write-Host "  python .\splatter_app.py"
Write-Host "  # Tabs: Produce Stills + COLMAP, Train Splat, Display Splat"
Write-Host ""
Write-Host "Optional advanced usage (individual apps):" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy RemoteSigned"
Write-Host "  Set-Location `"$ScriptRoot`""
Write-Host "  .\.venv-splatter\Scripts\Activate.ps1"
Write-Host "  python .\stills_extractor_app.py"
Write-Host "  python .\gui_wrapper.py"
Write-Host ""
Write-Host "If 3dgrut training fails due to CUDA or compiler issues, rerun:" -ForegroundColor Yellow
Write-Host "  cd `"$grutDir`""
Write-Host "  .\install_env_uv.ps1"
Write-Host ""

