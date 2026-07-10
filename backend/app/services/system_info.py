"""Runtime hardware- / package-capability detection (G4a, R1).

Cached checks for runtime facts the UI gates features on: whether a CUDA GPU is present (gates
**GPU-only training** of image-obs envs like Atari, whose PPO needs a ``CnnPolicy`` on CUDA), and
whether the optional ``ale-py`` package is installed (gates the Atari family entirely — ADR-101
pulled it out of the default install, so a fresh clone has no Atari backend). Both feed the API
(``GET /api/system``) and the backend safety nets. ``torch`` is imported lazily on first call —
kept out of app startup so ``/health`` and the REST surface stay fast to boot, like the trainers.

Results are cached for the process lifetime (neither CUDA presence nor an installed package changes
at runtime), so the probe cost is paid at most once. Any GPU import/probe failure is treated as
"no GPU".
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache

# One canonical message for the missing-ale-py failure, shared by the safety-net import sites so the
# error a bypassed-UI caller sees is identical everywhere (R1 / ADR-101).
_ALE_MISSING_MSG = "Atari support requires the optional ale-py package — pip install ale-py"


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True if a CUDA device is available (cached). False if torch is missing or CUDA is absent."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — any probe failure ⇒ treat as CPU-only
        return False


@lru_cache(maxsize=1)
def atari_available() -> bool:
    """True if the optional ``ale-py`` package is importable (cached, R1).

    Probes with ``find_spec`` and deliberately does **not** import ``ale_py`` — its import has a
    heavy Gymnasium-namespace-registration side effect we don't want to pay just to answer the
    capability question the UI asks at startup.
    """
    return importlib.util.find_spec("ale_py") is not None


def require_ale_py() -> None:
    """Import ``ale_py`` (registering the ``ALE/*`` namespace) or raise a clean typed error (R1).

    The lazy Atari import sites (``envs/atari.py``, ``envs/factory.py``) call this instead of a bare
    ``import ale_py`` so a fresh clone without the optional package fails with an actionable
    ``RuntimeError``, never a raw ``ImportError`` — even if the UI's Atari gate is bypassed by a
    direct API/trainer call. ADR-101 made ale-py opt-in.
    """
    if not atari_available():
        raise RuntimeError(_ALE_MISSING_MSG)
    try:
        import ale_py  # noqa: F401 — import side effect registers the "ALE/*" namespace
    except ImportError as exc:  # find_spec hit but the install is broken — same actionable message
        raise RuntimeError(_ALE_MISSING_MSG) from exc
