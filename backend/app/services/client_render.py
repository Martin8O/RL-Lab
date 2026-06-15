"""Client-side rendered environments.

For envs whose physics state the frontend can draw itself, the server streams the raw state
instead of an ``rgb_array`` render — lighter on CPU and the wire, crisper on screen, and it lets
us draw a nicer scene than gym's default (e.g. a proper MountainCar hill with margins instead of a
low-res image where the car appears to hit the screen edge). Keep the env set + the per-env state
layout in sync with the frontend's ``CLIENT_RENDER_ENVS`` / drawing (EnvPreview.tsx).
"""

from __future__ import annotations

from typing import Any


def client_state(env: Any, obs: Any = None) -> list[float] | None:
    """Raw physics state for a client-rendered env, else ``None`` (→ server image render).

    All current vector envs are drawn client-side. The frontend knows the per-env layout (EnvPreview
    ``clientKind``) and draws the scene from these raw numbers:

    * CartPole → ``[x, theta]``              (cart position, pole angle)
    * MountainCar(+Continuous) → ``[position, velocity]``
    * Pendulum → ``[theta, theta_dot]``      (rod angle, 0 = upright)
    * Acrobot → ``[theta1, theta2]``         (link angles; theta2 relative to link 1)
    * LunarLander → the full 8-number observation ``[x, y, vx, vy, angle, ang_vel, leg1, leg2]``
      (it has no ``unwrapped.state``, so we forward the observation the loop already has)
    * Toy Text grid-worlds (their state is a single int, in ``unwrapped.s``, not ``.state``):
      FrozenLake / CliffWalking → ``[row, col]`` (the agent's cell); Taxi →
      ``[taxi_row, taxi_col, passenger_loc, destination]`` (decoded from the Taxi state integer).
      The static board (holes / goal / cliff / stops) rides separately in :func:`grid_layout`.

    Any other env returns ``None`` so the streamer falls back to a server-side JPEG.
    """
    spec = getattr(env, "spec", None)
    gym_id = getattr(spec, "id", None) or ""
    try:
        if gym_id.startswith("LunarLander") and obs is not None:
            return [float(v) for v in obs]  # client draws the lander + pad + thruster particles
        # Toy Text: state is a single int in unwrapped.s (the one-hot wrapper doesn't touch it).
        u = getattr(env, "unwrapped", env)
        if gym_id.startswith("FrozenLake"):
            s, ncol = int(u.s), int(u.ncol)
            return [float(s // ncol), float(s % ncol)]
        if gym_id.startswith("CliffWalking"):
            s, ncol = int(u.s), int(u.shape[1])
            return [float(s // ncol), float(s % ncol)]
        if gym_id.startswith("Taxi"):
            taxi_row, taxi_col, pass_loc, dest = list(u.decode(int(u.s)))
            return [float(taxi_row), float(taxi_col), float(pass_loc), float(dest)]
        raw = getattr(u, "state", None)
        if raw is None:
            return None
        if gym_id.startswith("CartPole"):
            return [float(raw[0]), float(raw[2])]
        if gym_id.startswith("MountainCar"):  # MountainCar-v0 AND MountainCarContinuous-v0
            return [float(raw[0]), float(raw[1])]
        if gym_id.startswith("Pendulum"):
            return [float(raw[0]), float(raw[1])]  # [theta, theta_dot]
        if gym_id.startswith("Acrobot"):
            return [float(raw[0]), float(raw[1])]  # [theta1, theta2]
    except Exception:  # noqa: BLE001 — a malformed state just falls back to image
        return None
    return None


def terrain(env: Any) -> list[list[float]] | None:
    """LunarLander only: the per-episode moon surface as obs-space ``[x, y]`` points (else ``None``).

    The real terrain is **randomly generated each episode and is NOT in the observation**, so the
    client otherwise has to guess a flat ground — which makes the lander look like it touches down
    above/below the surface anywhere but the pad. The env keeps the surface in ``unwrapped.sky_polys``;
    we project it into the SAME normalized space as the lander's obs (x centred on the viewport, y
    measured from the pad-touchdown height) so the frontend can draw the lander and the ground in one
    coordinate system. Sent once it's available; the client falls back to a flat moon if absent.
    """
    spec = getattr(env, "spec", None)
    gym_id = getattr(spec, "id", None) or ""
    if not gym_id.startswith("LunarLander"):
        return None
    unwrapped = getattr(env, "unwrapped", env)
    polys = getattr(unwrapped, "sky_polys", None)
    if not polys:
        return None
    try:
        from gymnasium.envs.box2d import lunar_lander as ll

        half_w = ll.VIEWPORT_W / ll.SCALE / 2.0  # obs-x is centred on the viewport, ±1 at the edges
        half_h = ll.VIEWPORT_H / ll.SCALE / 2.0
        ref_y = float(unwrapped.helipad_y) + ll.LEG_DOWN / ll.SCALE  # world y where obs-y = 0

        def to_obs(p: Any) -> list[float]:
            return [(float(p[0]) - half_w) / half_w, (float(p[1]) - ref_y) / half_h]

        # Each sky poly's lower edge is one terrain segment: the surface = first point + every p2.
        return [to_obs(polys[0][0])] + [to_obs(p[1]) for p in polys]
    except Exception:  # noqa: BLE001 — any surprise just falls back to the client's flat default
        return None


def grid_layout(env: Any) -> dict[str, Any] | None:
    """Toy Text only: the static grid-world board the client draws under the agent (else ``None``).

    Returns a :class:`~app.schemas.preview.GridLayout`-shaped dict — ``kind`` + ``rows``/``cols`` + a
    row-major ``cells`` list tagging each square. The dynamic positions (agent / passenger /
    destination) come from :func:`client_state`; this is the fixed board, streamed each frame (tiny)
    so a late-joining client always has it.

    * FrozenLake — read from ``unwrapped.desc`` (the map), so the 4×4 / 8×8 / no-slip variants and
      their hole layouts are all correct.
    * CliffWalking — the fixed 4×12 board (no ``desc``): start bottom-left, goal bottom-right, the
      cliff the squares between them on the bottom row.
    * Taxi — the 5×5 board with the four pickup/drop-off stops marked (R, G, Y, B); the internal
      walls are fixed and drawn client-side.
    """
    spec = getattr(env, "spec", None)
    gym_id = getattr(spec, "id", None) or ""
    u = getattr(env, "unwrapped", env)
    try:
        if gym_id.startswith("FrozenLake"):
            nrow, ncol = int(u.nrow), int(u.ncol)
            tag = {"S": "start", "F": "normal", "H": "hole", "G": "goal"}
            cells = [
                tag.get(u.desc[r][c].decode("ascii"), "normal")
                for r in range(nrow)
                for c in range(ncol)
            ]
            return {"kind": "frozenlake", "rows": nrow, "cols": ncol, "cells": cells}
        if gym_id.startswith("CliffWalking"):
            rows, cols = 4, 12
            cells = ["normal"] * (rows * cols)
            bottom = (rows - 1) * cols
            cells[bottom] = "start"
            cells[bottom + cols - 1] = "goal"
            for c in range(1, cols - 1):
                cells[bottom + c] = "cliff"
            return {"kind": "cliffwalking", "rows": rows, "cols": cols, "cells": cells}
        if gym_id.startswith("Taxi"):
            rows, cols = 5, 5
            cells = ["normal"] * (rows * cols)
            for r, c in ((0, 0), (0, 4), (4, 0), (4, 3)):  # the R, G, Y, B stops
                cells[r * cols + c] = "stop"
            return {"kind": "taxi", "rows": rows, "cols": cols, "cells": cells}
    except Exception:  # noqa: BLE001 — any surprise just falls back to a server image
        return None
    return None
