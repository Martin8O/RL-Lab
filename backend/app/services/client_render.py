"""Client-side rendered environments.

For envs whose physics state the frontend can draw itself (CartPole today), the
server streams the raw state instead of an ``rgb_array`` render — lighter on CPU
and the wire, and crisper on screen. Keep the env set in sync with the frontend's
``CLIENT_RENDER_ENVS`` (frontend/src/components/EnvPreview.tsx).
"""

from __future__ import annotations

from typing import Any


def cart_state(env: Any) -> list[float] | None:
    """Return ``[x, theta]`` for a CartPole-family env, else ``None``.

    ``x`` is the cart position and ``theta`` the pole angle (radians) — enough for
    the frontend to draw the cart + pole. Any other env returns ``None`` so the
    streamer falls back to a server-side image render.
    """
    spec = getattr(env, "spec", None)
    gym_id = getattr(spec, "id", None) or ""
    if not gym_id.startswith("CartPole"):
        return None
    raw = getattr(getattr(env, "unwrapped", env), "state", None)
    if raw is None:
        return None
    try:
        return [float(raw[0]), float(raw[2])]
    except Exception:  # noqa: BLE001 — a malformed state just falls back to image
        return None
