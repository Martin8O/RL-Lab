# PyInstaller spec for the RL Dashboard standalone build (F5).
#
# Build (from the repo root, inside the .venv):
#   pyinstaller rl_dashboard.spec --noconfirm
# Output: dist/RL-Dashboard/RL-Dashboard.exe  (one-folder — zip the folder to share).
#
# GPU / universal edition: it bundles whatever torch is in the build venv — on the cu128 desktop that
# is the CUDA build, so the exe is a *superset* that runs everywhere (CUDA torch falls back to CPU when
# no NVIDIA GPU/driver is present) and auto-unlocks the GPU-only games on a friend's NVIDIA machine via
# the registry's `hw_requirement` gating + `torch.cuda.is_available()`. Plus gymnasium + pygame-ce +
# ale-py + minigrid + stable-baselines3 and the built frontend. Expect ~4-6 GB (the CUDA libs dominate;
# build from a CPU-only venv instead for a ~1-2 GB CPU edition).
#
# One-folder (not one-file) on purpose: a 1-2 GB one-file build re-extracts to a temp dir on every
# launch (slow startup, AV friction). The folder starts fast and reliably; zip it for transfer.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

# SPECPATH is injected by PyInstaller and always equals the spec file's own directory (repo root),
# regardless of the shell CWD — use it for all repo-relative paths in this file.
_spec_dir = Path(SPECPATH)  # noqa: F821 — injected by PyInstaller at spec-parse time

# --- The built single-page frontend, served by FastAPI as static files at runtime -------------
# `app.core.paths.frontend_dist_dir()` looks for a `frontend_dist` tree under sys._MEIPASS.
_dist = _spec_dir / "frontend" / "dist"
if not _dist.is_dir():
    raise SystemExit(
        "frontend/dist not found — run `npm run build` in frontend/ before packaging "
        "(see build-standalone.ps1, which does this for you)."
    )

datas = [(str(_dist), "frontend_dist")]
binaries = []
hiddenimports = []

# --- Pull in the heavy/data-bearing ML packages wholesale (data files + native libs + submodules).
# Guarded so an absent optional package never breaks the build. Import names differ from pip names:
# pygame-ce -> pygame, box2d -> Box2D, opencv-python -> cv2.
_COLLECT = [
    "torch",
    "torchvision",
    "gymnasium",
    "ale_py",
    "minigrid",
    "stable_baselines3",
    "pygame",
    "Box2D",
    "cv2",
    "shimmy",
]
for _pkg in _COLLECT:
    try:
        _d, _b, _h = collect_all(_pkg)
        datas += _d
        binaries += _b
        hiddenimports += _h
    except ModuleNotFoundError as exc:  # absent optional package — skip silently
        print(f"[spec] skipping {_pkg} (not installed): {exc}")

# Modules the app reaches indirectly (env registration / lazy imports) that static analysis can miss.
hiddenimports += [
    "app.main",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
]

a = Analysis(
    [str(_spec_dir / "backend" / "launcher.py")],
    pathex=[str(_spec_dir / "backend")],  # so `import app.main` resolves (app == backend/app)
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RL-Dashboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX off: avoids AV false-positives and torch DLL corruption.
    console=True,  # keep the console for now so a clean-machine crash is visible (F5 polish: windowed).
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RL-Dashboard",
)
