<#
.SYNOPSIS
    Dev shortcuts for the RL All-in-One Dashboard monorepo.

USAGE
    .\tasks.ps1 <task>

TASKS
    dev-backend     Activate venv and start the FastAPI dev server (hot-reload).
    dev-frontend    Start the Vite frontend dev server.
    lint            Run ruff + mypy on the backend.
    test            Run pytest on the backend.
    i18n            Check frontend i18n completeness (en/cz parity + every t() key).
    all             Run lint + test.
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev-backend","dev-frontend","lint","test","i18n","all")]
    [string]$Task
)

$Root    = $PSScriptRoot
$Venv    = "$Root\.venv\Scripts"
$Python  = "$Venv\python.exe"
$Ruff    = "$Venv\ruff.exe"
$Mypy    = "$Venv\mypy.exe"
$Pytest  = "$Venv\pytest.exe"
$Backend = "$Root\backend"
$Frontend= "$Root\frontend"

function Invoke-Lint {
    Write-Host "`n==> ruff check backend" -ForegroundColor Cyan
    & $Ruff check $Backend
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n==> mypy backend/app" -ForegroundColor Cyan
    & $Mypy --config-file "$Backend\pyproject.toml" "$Backend\app"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Test {
    Write-Host "`n==> pytest backend" -ForegroundColor Cyan
    & $Pytest --rootdir $Backend -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

switch ($Task) {
    "dev-backend" {
        Write-Host "Starting FastAPI backend (uvicorn --reload) ..." -ForegroundColor Green
        & $Venv\uvicorn.exe app.main:app --app-dir $Backend --host 127.0.0.1 --port 8000 --reload
    }
    "dev-frontend" {
        if (-not (Test-Path $Frontend)) {
            Write-Host "frontend/ not scaffolded yet (arrives in A4)." -ForegroundColor Yellow
            exit 0
        }
        Write-Host "Starting Vite dev server ..." -ForegroundColor Green
        Push-Location $Frontend
        npm run dev
        Pop-Location
    }
    "lint" { Invoke-Lint }
    "test" { Invoke-Test }
    "i18n" {
        Write-Host "`n==> i18n check (frontend)" -ForegroundColor Cyan
        Push-Location $Frontend
        node scripts/check-i18n.mjs
        $code = $LASTEXITCODE
        Pop-Location
        if ($code -ne 0) { exit $code }
    }
    "all"  { Invoke-Lint; Invoke-Test }
}
