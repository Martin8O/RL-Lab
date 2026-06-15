"""Runtime hardware-capability detection (G4a).

A single cached check for whether a CUDA GPU is present. It gates **GPU-only training**
(image-observation envs like Atari, whose PPO needs a ``CnnPolicy`` on CUDA) in both the API
(``GET /api/system``) and the training manager. ``torch`` is imported lazily on first call —
kept out of app startup so ``/health`` and the REST surface stay fast to boot, like the trainers.

The result is cached for the process lifetime (CUDA availability doesn't change at runtime), so
the torch-import cost is paid at most once. Any import/probe failure is treated as "no GPU".
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """True if a CUDA device is available (cached). False if torch is missing or CUDA is absent."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — any probe failure ⇒ treat as CPU-only
        return False
