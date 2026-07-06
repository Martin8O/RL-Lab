<#
.SYNOPSIS
    Build the RL Dashboard standalone Windows app (F5) -- one double-click executable, no Python
    needed on the target machine.

.DESCRIPTION
    1. Builds the frontend (`npm run build` -> frontend/dist).
    2. Ensures PyInstaller is installed in the project .venv.
    3. Runs PyInstaller against rl_dashboard.spec (bundles backend + whatever torch is in the build
       venv + gymnasium + pygame-ce + ale-py + minigrid + the built frontend). On the cu128 desktop
       this yields the GPU/universal edition (~4-6 GB): runs everywhere, auto-uses a friend's NVIDIA
       GPU, falls back to CPU otherwise. Build from a CPU-only venv for a smaller ~1-2 GB CPU edition.
    4. Optionally zips the one-folder output for sharing.

    Output: dist\RL-Dashboard\RL-Dashboard.exe  (the whole RL-Dashboard\ folder ships together).

.PARAMETER Zip
    Also produce dist\RL-Dashboard.zip for transfer.

.PARAMETER SkipFrontend
    Skip the `npm run build` step (reuse an existing frontend/dist).

.EXAMPLE
    .\build-standalone.ps1 -Zip
#>
param(
    [switch]$Zip,
    [switch]$SkipFrontend
)

$ErrorActionPreference = "Stop"
$Root     = $PSScriptRoot
$Venv     = "$Root\.venv\Scripts"
$Python   = "$Venv\python.exe"
$Frontend = "$Root\frontend"

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python. Create it first (see CLAUDE.md > Running)."
}

# 1. Frontend build ----------------------------------------------------------------------------
if (-not $SkipFrontend) {
    Write-Host "`n==> Building frontend (npm run build)" -ForegroundColor Cyan
    Push-Location $Frontend
    try {
        npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed ($LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "`n==> Skipping frontend build (-SkipFrontend)" -ForegroundColor Yellow
}
if (-not (Test-Path "$Frontend\dist\index.html")) {
    throw "frontend/dist/index.html missing -- build the frontend first (drop -SkipFrontend)."
}

# 2. PyInstaller present? ----------------------------------------------------------------------
Write-Host "`n==> Ensuring PyInstaller is installed" -ForegroundColor Cyan
& $Python -m pip install -r "$Root\backend\requirements-build.txt"
if ($LASTEXITCODE -ne 0) { throw "pip install of build tooling failed ($LASTEXITCODE)" }

# 3. Package -----------------------------------------------------------------------------------
Write-Host "`n==> Running PyInstaller (this takes a few minutes; ~1-2 GB output)" -ForegroundColor Cyan
& $Python -m PyInstaller "$Root\rl_dashboard.spec" --noconfirm --distpath "$Root\dist" --workpath "$Root\build"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed ($LASTEXITCODE)" }

$OutDir = "$Root\dist\RL-Dashboard"
$Exe    = "$OutDir\RL-Dashboard.exe"
if (-not (Test-Path $Exe)) { throw "Expected $Exe was not produced." }

# 4. Optional zip ------------------------------------------------------------------------------
if ($Zip) {
    Write-Host "`n==> Zipping for transfer" -ForegroundColor Cyan
    $ZipPath = "$Root\dist\RL-Dashboard.zip"
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path $OutDir -DestinationPath $ZipPath
    Write-Host "    -> $ZipPath" -ForegroundColor Green
}

Write-Host "`n==> Done." -ForegroundColor Green
Write-Host "    App:  $Exe"
Write-Host "    Run it, or send the whole RL-Dashboard\ folder (or the .zip) to anyone on Windows."
