"""Standalone launcher for the RL Dashboard (F5) — the PyInstaller entry point.

One process serves everything: FastAPI hosts the API + WebSocket *and* the bundled single-page
frontend (mounted in ``app.main``), so a non-developer just double-clicks one executable. On start
it picks a free local port, launches the browser at it once the server is accepting connections,
and runs uvicorn in the foreground (closing the window stops the app).

Run it from source too, to preview the packaged experience without building::

    .venv/Scripts/python.exe backend/launcher.py
    # (requires a prior `npm run build` so `frontend/dist` exists to be served)
"""

import multiprocessing
import socket
import threading
import time
import webbrowser

import uvicorn

# Importing the app also wires up the static-frontend mount (app.core.paths finds the SPA).
from app.core.config import settings
from app.core.logging import get_logger
from app.core.paths import data_dir, frontend_dist_dir
from app.main import app

logger = get_logger(__name__)

# How long to wait for the server to come up before opening the browser anyway.
_READY_TIMEOUT_S = 30.0


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _choose_port(host: str, preferred: int) -> int:
    """Use the configured port if free, else let the OS hand us an open one."""
    if _port_is_free(host, preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    """Poll the TCP port until uvicorn is accepting, then open the default browser once."""
    deadline = time.monotonic() + _READY_TIMEOUT_S
    ready = False
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((host, port)) == 0:
                ready = True
                break
        time.sleep(0.25)
    if not ready:
        logger.warning("Server did not respond within %ss — not opening browser.", _READY_TIMEOUT_S)
        return
    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover - a missing browser must not crash the server
        logger.warning("Could not open a browser automatically; open %s manually.", url)


def main() -> None:
    host = settings.host
    port = _choose_port(host, settings.port)
    # Bind-all hosts (0.0.0.0 / ::) are valid for uvicorn but not for a browser URL or TCP probe.
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{port}"

    if frontend_dist_dir() is None:
        logger.warning(
            "No built frontend found — the API will run but there is no UI to serve. "
            "Build it with `npm run build` (dev) or use the packaged executable."
        )

    print("=" * 60)
    print("  RL Dashboard")
    print(f"  Open:  {url}")
    print(f"  Data:  {data_dir()}")
    print("  Close this window to stop the app.")
    print("=" * 60, flush=True)

    threading.Thread(
        target=_open_browser_when_ready, args=(url, display_host, port), daemon=True
    ).start()

    # Pass the app object (not an import string) so the frozen build doesn't re-import by name,
    # and keep reload/workers off (single user, single process).
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Must be the first thing in a frozen build: without it, any multiprocessing child (torch
    # DataLoader workers, SB3 SubprocVecEnv) re-runs this whole module and fork-bombs the app.
    multiprocessing.freeze_support()
    main()
