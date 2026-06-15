"""Filesystem path resolution that works both in dev and in a packaged (frozen) build.

In normal development the app runs from the source tree, so writable state (checkpoints, runs,
score files) lives under the gitignored repo-root ``data/`` dir and the built frontend (if any)
under ``frontend/dist``. In a **PyInstaller bundle** (``sys.frozen``) that layout disappears: the
package tree is read-only (one-file extracts to a temp dir; one-folder lives in ``_internal``), so
writes must go to a per-user data dir instead, and bundled read-only resources live next to the
executable / in ``sys._MEIPASS``.

This module is the single place that knows the difference (F5). Every path the app *writes* goes
through :func:`data_dir`; the bundled SPA is found via :func:`frontend_dist_dir`. Keeping it here
means the four data services and ``main.py`` stay layout-agnostic.
"""

import os
import sys
from pathlib import Path

__all__ = ["is_frozen", "resource_root", "data_dir", "frontend_dist_dir"]

# Env var that lets a user (or a test) redirect all writable state, e.g. to keep saves next to a
# portable copy of the app. Takes precedence over the frozen/dev defaults.
_DATA_DIR_ENV = "RL_DASHBOARD_DATA_DIR"


def is_frozen() -> bool:
    """True when running inside a PyInstaller (or similar) bundle."""
    return bool(getattr(sys, "frozen", False))


def _repo_root() -> Path:
    """Repo root in the dev layout: ``backend/app/core/paths.py`` → ``parents[3]``."""
    return Path(__file__).resolve().parents[3]


def resource_root() -> Path:
    """Base dir for bundled, read-only resources (e.g. the built frontend).

    Frozen: PyInstaller's extraction dir (``sys._MEIPASS``), falling back to the executable's
    folder. Dev: the repo root.
    """
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        return Path(meipass) if meipass else Path(sys.executable).resolve().parent
    return _repo_root()


def data_dir() -> Path:
    """Directory for all writable app state (checkpoints, runs, highscores, play scores).

    Resolution order:

    1. ``$RL_DASHBOARD_DATA_DIR`` if set (explicit override / portable mode / tests).
    2. Frozen build → ``%LOCALAPPDATA%\\RLDashboard\\data`` (always writable; survives moving the
       app folder; ``~`` fallback if ``LOCALAPPDATA`` is somehow unset, e.g. non-Windows).
    3. Dev → repo-root ``data/`` (unchanged behaviour; gitignored).

    The directory is *not* created here — each store creates its own subtree lazily on first write
    (mirroring the previous module-constant behaviour).
    """
    override = os.environ.get(_DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "RLDashboard" / "data"
    return _repo_root() / "data"


def frontend_dist_dir() -> Path | None:
    """Locate the built single-page frontend to serve as static files, or ``None`` if absent.

    Frozen: the spec bundles ``frontend/dist`` as a ``frontend_dist`` data tree under
    :func:`resource_root`. Dev: the real Vite output at ``frontend/dist`` (present only after
    ``npm run build`` — in the normal Vite-dev workflow it is absent, so the backend serves only
    ``/api`` + ``/ws`` and Vite proxies, exactly as before).
    """
    bundled = resource_root() / "frontend_dist"
    if bundled.is_dir():
        return bundled
    dev = _repo_root() / "frontend" / "dist"
    return dev if dev.is_dir() else None
