<#
.SYNOPSIS
    Dev shortcuts for the RL All-in-One Dashboard monorepo.

USAGE
    .\tasks.ps1 <task>

TASKS
    dev-backend     Activate venv and start the FastAPI dev server (hot-reload).
    dev-frontend    Start the Vite frontend dev server.
    lint            Lint both sides: ruff + mypy (backend) and eslint (frontend).
    test            Run all tests: pytest (backend) and vitest (frontend).
    i18n            Check frontend i18n completeness (en/cz parity + every t() key).
    build           Type-check + production build of the frontend.
    all             Full quality gate: lint + i18n + test + build. The single command for
                    pre-commit / CI — every check both sides must pass.

NOTE
    Prettier is installed but intentionally NOT part of the gate: the frontend is hand-formatted
    (F2 decision). Format a specific new file on demand with:
        cd frontend; npm run format -- src/path/to/NewFile.tsx
#>

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("dev-backend","dev-frontend","lint","test","i18n","build","all")]
    [string]$Task
)

$Root    = $PSScriptRoot
$Venv    = "$Root\.venv\Scripts"
$Ruff    = "$Venv\ruff.exe"
$Mypy    = "$Venv\mypy.exe"
$Pytest  = "$Venv\pytest.exe"
$Backend = "$Root\backend"
$Frontend= "$Root\frontend"

# Run an npm script in the frontend workspace, propagating its exit code.
function Invoke-Npm($scriptName) {
    Push-Location $Frontend
    npm run $scriptName
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { exit $code }
}

function Invoke-Lint {
    Write-Host "`n==> ruff check backend" -ForegroundColor Cyan
    & $Ruff check $Backend
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n==> mypy backend/app" -ForegroundColor Cyan
    & $Mypy --config-file "$Backend\pyproject.toml" "$Backend\app"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n==> eslint frontend" -ForegroundColor Cyan
    Invoke-Npm "lint"
}

function Invoke-Test {
    Write-Host "`n==> pytest backend" -ForegroundColor Cyan
    & $Pytest --rootdir $Backend -q
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n==> vitest frontend" -ForegroundColor Cyan
    Invoke-Npm "test"
}

function Invoke-I18n {
    Write-Host "`n==> i18n check (frontend)" -ForegroundColor Cyan
    Invoke-Npm "i18n:check"
}

function Invoke-Build {
    Write-Host "`n==> frontend build (tsc -b + vite build)" -ForegroundColor Cyan
    Invoke-Npm "build"
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
    "lint"  { Invoke-Lint }
    "test"  { Invoke-Test }
    "i18n"  { Invoke-I18n }
    "build" { Invoke-Build }
    "all"   {
        Invoke-Lint
        Invoke-I18n
        Invoke-Test
        Invoke-Build
        Write-Host "`n==> all checks passed " -NoNewline -ForegroundColor Green
        Write-Host ([char]0x2713) -ForegroundColor Green
    }
}
