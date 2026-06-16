"""Multi-agent (PettingZoo) environment adapter — the **5th extensibility seam**.

Every single-agent env goes through :func:`app.envs.factory.make_env`: one ``gym.Env`` driven by
one ``step(action)`` loop. A multi-agent env is a different shape — **N agents act in one shared
world**, each with its own observation / action / reward (PettingZoo's *parallel* API) — so it
cannot ride the single-agent factory. This module is the single place that builds those envs,
mirroring the factory's "one source of truth, every path goes through it" decoupling:

* :func:`make_parallel_env` — the raw PettingZoo **parallel** env. Used by the preview streamer to
  step all agents with the shared policy and read their positions for the swarm render (and any
  future multi-agent play).
* :func:`make_vec_env` — the **SuperSuit bridge** to an SB3-compatible ``VecEnv`` with **parameter
  sharing**: ``pettingzoo_env_to_vec_env_v1`` stacks the N *homogeneous* agents as N sub-envs and
  ``concat_vec_envs_v1`` exposes them as one SB3 ``VecEnv``, so a *single* ``MlpPolicy`` is trained
  over all agents at once — one shared brain (ADR-038). ``VecMonitor`` populates SB3's
  ``ep_info_buffer`` so ``ep_rew_mean`` (the per-agent episode return) drives the existing chart.
* :func:`agent_sprites` / :func:`world_entities` — per-agent + landmark render state (world-space
  ``[x, y]`` positions) the client draws as a "swarm" canvas (ADR-018 client render, ADR-038).

**Homogeneous agents only** — parameter sharing requires identical obs/action spaces across the
agents, which is exactly the cooperative ``simple_spread`` case. Heterogeneous species (e.g.
``simple_tag`` predators vs. prey) need per-species policies and land in **G7b**.

MPE lives in the **``mpe2``** package (the split-out successor to ``pettingzoo.mpe``); both the
PettingZoo/SuperSuit imports and the scenario module are imported **lazily** so app startup and the
REST surface stay ML-free and fast to boot, exactly like the trainers and the single-agent factory.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.envs.registry import EnvSpec


def is_multi_agent(spec: EnvSpec | None) -> bool:
    """True for a PettingZoo multi-agent env (the 5th seam), False for every single-agent env."""
    return spec is not None and spec.family == "petting_zoo"


def _load_scenario(name: str) -> Any:
    """Import an MPE scenario module by name (e.g. ``"simple_spread_v3"``).

    Prefers the modern split-out ``mpe2`` package; falls back to the legacy ``pettingzoo.mpe``
    namespace for older installs (the version split noted in the registry's MPE section).
    """
    for pkg in ("mpe2", "pettingzoo.mpe"):
        try:
            return importlib.import_module(f"{pkg}.{name}")
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        f"MPE scenario '{name}' not found in mpe2 or pettingzoo.mpe — install 'mpe2'"
    )


def make_parallel_env(env_id: str, *, render_mode: str | None = None) -> Any:
    """Build the raw PettingZoo **parallel** env for ``env_id`` from its registry row.

    The scenario module name is the spec's ``gym_id`` (e.g. ``"simple_spread_v3"``); the per-env
    construction kwargs (``N``, ``max_cycles``, ``continuous_actions`` …) ride in ``make_kwargs``.
    Used by the preview streamer (positions for the swarm render) — the trainer uses the vec-env
    bridge below instead.
    """
    from app.envs.registry import get_env

    spec = get_env(env_id)
    if spec is None:
        raise ValueError(f"Unknown multi-agent environment '{env_id}'")
    module = _load_scenario(spec.gym_id)
    kwargs: dict[str, Any] = dict(spec.make_kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return module.parallel_env(**kwargs)


def make_vec_env(env_id: str) -> Any:
    """Build the SB3-compatible, parameter-sharing ``VecEnv`` for ``env_id`` (the training path).

    Bridges the PettingZoo parallel env through SuperSuit so SB3's PPO trains **one shared policy**
    over all N homogeneous agents (parameter sharing), then wraps it in ``VecMonitor`` so episode
    returns populate ``ep_rew_mean`` for the live chart. The env is NOT seeded here — SuperSuit's
    ``ConcatVecEnv`` exposes no ``seed()`` (SB3 2.8 would call it and crash); the trainer seeds the
    policy globally instead (numpy/torch/python), see ``trainer_ppo`` + the reproducibility note.
    """
    import supersuit as ss
    from stable_baselines3.common.vec_env import VecMonitor

    parallel = make_parallel_env(env_id)
    vec = ss.pettingzoo_env_to_vec_env_v1(parallel)
    vec = ss.concat_vec_envs_v1(vec, 1, num_cpus=1, base_class="stable_baselines3")
    return VecMonitor(vec)


# --- render-state extraction (the swarm canvas) --------------------------------------------------
#
# MPE keeps the live scene on ``env.unwrapped.world``: ``world.agents`` (the moving entities) and
# ``world.landmarks`` (cooperative coverage *targets*, or collidable *obstacles*). Each entity's
# position is ``entity.state.p_pos`` — a 2-vector in world space (roughly centred on the origin).
# The client autoscales these to the canvas, so we forward them raw.


def agent_sprites(parallel_env: Any) -> list[dict[str, Any]]:
    """Per-agent render state for the swarm canvas: ``[{x, y, role, size}]`` (world-space).

    ``role`` is ``"adversary"`` for a predator (drives a distinct colour) else ``"agent"`` — so the
    same renderer serves cooperative and (future) predator–prey scenarios.
    """
    world = getattr(getattr(parallel_env, "unwrapped", parallel_env), "world", None)
    if world is None:
        return []
    sprites: list[dict[str, Any]] = []
    for ag in world.agents:
        pos = ag.state.p_pos
        sprites.append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "role": "adversary" if getattr(ag, "adversary", False) else "agent",
                "size": float(getattr(ag, "size", 0.05)),
            }
        )
    return sprites


def world_entities(parallel_env: Any) -> list[dict[str, Any]]:
    """Landmark render state for the swarm canvas: ``[{x, y, kind, size}]`` (world-space).

    ``kind`` is ``"obstacle"`` for a collidable landmark (simple_tag obstacles) else ``"target"``
    (simple_spread coverage points) — the client draws targets as open markers, obstacles as solid.
    """
    world = getattr(getattr(parallel_env, "unwrapped", parallel_env), "world", None)
    if world is None:
        return []
    entities: list[dict[str, Any]] = []
    for lm in world.landmarks:
        pos = lm.state.p_pos
        entities.append(
            {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "kind": "obstacle" if getattr(lm, "collide", False) else "target",
                "size": float(getattr(lm, "size", 0.05)),
            }
        )
    return entities
