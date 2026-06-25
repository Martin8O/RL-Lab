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
import threading
from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.envs.registry import EnvSpec

logger = get_logger(__name__)

# MPE envs (mpe2) call the **process-global, non-thread-safe** ``pygame.init()`` on construction and
# ``pygame.quit()`` on close — even headless (mpe2 builds an off-screen Surface unconditionally). With
# the preview thread and the (per-species) trainer thread building / closing several envs concurrently,
# two of those global calls can overlap and segfault on Windows (a hard access violation, not a Python
# error). Serialize every mpe2 env **construction and close** behind one process lock so the global
# pygame init/quit calls can never race. (The self-play trainer also keeps its envs persistent for the
# whole run — see ``trainer_tag`` — so this lock is only contended a handful of times per run.)
# Reentrant: closing a vec env re-enters via the wrapper's own guarded close on the same thread.
_PYGAME_LOCK = threading.RLock()

# Stiffer contact collisions for the competitive predator–prey world (simple_tag), so obstacles and
# other agents are less easily passed through (the default 1e2 lets fast agents glide straight through).
# MPE caps every entity at its own max_speed, so a stronger contact force just separates overlaps faster
# without flinging anything off-arena. 2.5e2 is a gentler bump than the first 4e2 try — less bounce-back,
# a bit more pass-through, still noticeably firmer than default. Cooperative simple_spread keeps the
# default (its agents don't fight over obstacles). ADR-049.
_TAG_CONTACT_FORCE = 250.0

# Per-episode obstacle randomisation for simple_tag: the count varies per training round / watch session
# (2…6, kept observable at a FIXED obs size by num_landmark_neighbors=2 — the policy sees the 2 nearest),
# and each episode re-rolls every obstacle's size (−30 %…+50 % of the 0.2 base) and re-places them so they
# don't overlap each other (the default reset scatters them and they sometimes touch). ADR-049.
_BASE_OBSTACLE_SIZE = 0.2
_OBSTACLE_SIZE_LO = 0.7
_OBSTACLE_SIZE_HI = 1.5
_OBSTACLE_MIN = 2
_OBSTACLE_MAX = 6


def _stiffen_collisions(parallel_env: Any, force: float) -> None:
    """Raise the MPE world's contact_force on ``parallel_env`` (harder collisions). The world object
    persists across resets, so this sticks for the env's life; EzPickle resets it on a SuperSuit clone,
    so the species wrapper re-applies it at ``reset`` (the clone's runtime entry point)."""
    world = getattr(getattr(parallel_env, "unwrapped", parallel_env), "world", None)
    if world is not None:
        world.contact_force = force


def _randomize_obstacles(world: Any, rng: Any) -> None:
    """Re-roll every obstacle's size + re-place them non-overlapping, in place (call after reset).

    Sizes are ``0.2 × U(0.7, 1.5)`` (−30 %…+50 %); positions are rejection-sampled so no two obstacles
    overlap (with a small margin). Run AFTER ``env.reset`` (which scatters them at the base size), so the
    very first observation of the episode still reflects the reset layout — negligible for a 25-step
    episode, and every subsequent step sees the re-rolled world. Obstacles are the collidable, non-boundary
    landmarks (targets in simple_spread aren't collidable, so this no-ops there)."""
    import numpy as np

    obstacles = [
        lm for lm in getattr(world, "landmarks", [])
        if getattr(lm, "collide", False) and not getattr(lm, "boundary", False)
    ]
    placed: list[Any] = []
    for lm in obstacles:
        lm.size = float(_BASE_OBSTACLE_SIZE * rng.uniform(_OBSTACLE_SIZE_LO, _OBSTACLE_SIZE_HI))
        pos = rng.uniform(-0.9, 0.9, size=2)
        for _ in range(60):  # reject positions that overlap an already-placed obstacle
            if all(
                float(np.linalg.norm(pos - p.state.p_pos)) > (lm.size + p.size) * 1.05
                for p in placed
            ):
                break
            pos = rng.uniform(-0.9, 0.9, size=2)
        lm.state.p_pos = pos
        placed.append(lm)


def random_obstacle_count(rng: Any) -> int:
    """A random obstacle count in ``[2, 6]`` for one training round / watch session (simple_tag)."""
    return int(rng.integers(_OBSTACLE_MIN, _OBSTACLE_MAX + 1))


# Pursuit's render draws each pursuer's field-of-view as a translucent orange wash (alpha 128) which
# dominates the frame and buries the agents. We redraw it far fainter so it's a barely-there hint. Patched
# on the PREVIEW env instance only (render_mode set) — never the training env — so SuperSuit's cloudpickle
# clone (which can't carry a closure/method ref) is untouched.
_VISION_OVERLAY_ALPHA = 16  # was 128 — a faint tint instead of a dominating orange wash


def _dim_vision_overlay(parallel_env: Any) -> None:
    """Make pursuit's pursuer field-of-view overlay near-invisible (preview render only). No-op for any
    env without that overlay, so it's safe to call for the whole SISL-image family."""
    base = getattr(getattr(parallel_env, "unwrapped", parallel_env), "env", None)
    if base is None or not hasattr(base, "draw_pursuers_observations"):
        return
    import pygame

    def _faint() -> None:
        for i in range(base.pursuer_layer.n_agents()):
            x, y = base.pursuer_layer.get_position(i)
            patch = pygame.Surface(
                (base.pixel_scale * base.obs_range, base.pixel_scale * base.obs_range)
            )
            patch.set_alpha(_VISION_OVERLAY_ALPHA)
            patch.fill((255, 152, 72))
            ofst = base.obs_range / 2.0
            base.screen.blit(
                patch, (base.pixel_scale * (x - ofst + 0.5), base.pixel_scale * (y - ofst + 0.5))
            )

    base.draw_pursuers_observations = _faint


def is_multi_agent(spec: EnvSpec | None) -> bool:
    """True for a PettingZoo multi-agent env (the 5th seam), False for every single-agent env."""
    return spec is not None and spec.family == "petting_zoo"


def is_competitive_ma(spec: EnvSpec | None) -> bool:
    """True for a **heterogeneous, competitive** multi-agent env (simple_tag, G7b-2) — the
    per-species self-play case (ADR-048), as opposed to the homogeneous, cooperative parameter-sharing
    case (simple_spread, G7a). These two route to different trainers in the manager."""
    return is_multi_agent(spec) and bool(spec and spec.competitive)


def agent_role(agent: str) -> str:
    """The species an MPE agent belongs to, from its PettingZoo name: ``"adversary"`` for a predator
    (``adversary_0`` …) else ``"agent"`` (the prey ``agent_0`` …). Mirrors ``world.agents[i].adversary``
    used by :func:`agent_sprites`, so the swarm render and the per-species policy routing agree."""
    return "adversary" if agent.startswith("adversary") else "agent"


def species_present(env_id: str) -> list[str]:
    """The distinct species roles in ``env_id``, in a stable order (predators first): e.g.
    ``["adversary", "agent"]`` for simple_tag. Drives the self-play trainer's per-species loop."""
    raw = make_parallel_env(env_id)
    try:
        roles: list[str] = []
        for a in raw.possible_agents:
            role = agent_role(a)
            if role not in roles:
                roles.append(role)
        roles.sort(key=lambda r: 0 if r == "adversary" else 1)  # predators first (the headline)
        return roles
    finally:
        raw.close()


def _load_scenario(name: str) -> Any:
    """Import a PettingZoo scenario module by name (e.g. ``"simple_spread_v3"``, ``"pursuit_v4"``).

    Three families share this loader: **MPE** (the particle worlds — ``simple_spread`` / ``simple_tag``,
    G7a/G7b), **SISL** (the Stanford cooperative-swarm worlds — ``pursuit`` / ``multiwalker``, G7-SISL),
    and **vendored** worlds we ship in-tree (``app.envs.vendored.<id>``) because upstream dropped them
    (``waterworld_v4`` — PettingZoo removed it in 1.25.0; see ``app.envs.vendored``). MPE prefers the
    modern split-out ``mpe2`` package and falls back to the legacy ``pettingzoo.mpe`` namespace (the
    version split noted in the registry's MPE section); SISL lives under ``pettingzoo.sisl``; the
    vendored package is probed **last** so it only catches ids the real libraries no longer carry. We
    probe the packages in order and return the first that has the scenario, so an MPE id resolves from
    ``mpe2``, a stock SISL id from ``pettingzoo.sisl`` and a vendored id from ``app.envs.vendored`` with
    one code path.
    """
    for pkg in ("mpe2", "pettingzoo.mpe", "pettingzoo.sisl", "app.envs.vendored"):
        try:
            return importlib.import_module(f"{pkg}.{name}")
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(
        f"PettingZoo scenario '{name}' not found in mpe2 / pettingzoo.mpe / pettingzoo.sisl / "
        "app.envs.vendored — install 'mpe2' (MPE) or 'pettingzoo[sisl]' (SISL), or add a vendored copy"
    )


def preload_scenario(env_id: str) -> None:
    """Import ``env_id``'s PettingZoo scenario module (and its transitive **pygame** import) on the
    CALLING thread, before the trainer + preview threads spawn for a run.

    The SISL render worlds (``pursuit``, ``multiwalker``) — and the MPE worlds — import ``pygame`` at
    module load. When a run starts, the trainer thread (``make_vec_env``) and the visual preview thread
    (``make_parallel_env``) both cold-import that scenario on the process's first multi-agent run; if
    the two first-time imports race, Python's per-module import locks can **deadlock** (observed as a
    ``_DeadlockError`` on ``_ModuleLock('pygame.mixer')``). This is the exact class of bug the image-env
    ``import stable_baselines3`` preload guards against (ADR-065/076) — a single-threaded preload here,
    before either worker thread spawns, lets both then hit a fully-initialised module cache.

    Idempotent + cheap after the first import. Best-effort: a failed preload must not block the launch
    — the real import error (a missing extra, etc.) still surfaces on the worker thread exactly as
    before."""
    from app.envs.registry import get_env

    spec = get_env(env_id)
    if spec is None:
        return
    try:
        _load_scenario(spec.gym_id)
    except Exception:  # noqa: BLE001 — preload is an optimisation; let the worker thread surface errors
        logger.debug("MA scenario preload failed for %s", env_id, exc_info=True)


def make_parallel_env(
    env_id: str, *, render_mode: str | None = None, obstacle_count: int | None = None
) -> Any:
    """Build the raw PettingZoo **parallel** env for ``env_id`` from its registry row.

    The scenario module name is the spec's ``gym_id`` (e.g. ``"simple_spread_v3"``); the per-env
    construction kwargs (``N``, ``max_cycles``, ``continuous_actions`` …) ride in ``make_kwargs``.
    Used by the preview streamer (positions for the swarm render) — the trainer uses the vec-env
    bridge below instead. ``obstacle_count`` overrides ``num_obstacles`` for the variable-obstacle
    simple_tag world (the obs size stays fixed via the row's ``num_landmark_neighbors``).
    """
    from app.envs.registry import get_env

    spec = get_env(env_id)
    if spec is None:
        raise ValueError(f"Unknown multi-agent environment '{env_id}'")
    module = _load_scenario(spec.gym_id)
    kwargs: dict[str, Any] = dict(spec.make_kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if obstacle_count is not None:
        kwargs["num_obstacles"] = obstacle_count
    with _PYGAME_LOCK:  # construction calls the global pygame.init() — serialize it (see lock note)
        env = module.parallel_env(**kwargs)
    if spec.competitive:  # harder collisions for the predator–prey world (preview path; no clone here)
        _stiffen_collisions(env, _TAG_CONTACT_FORCE)
    if render_mode is not None and spec.ma_render == "image":
        _dim_vision_overlay(env)  # faint pursuit's dominating FOV wash — preview only, training clone safe
    return env


def close_env(env: Any) -> None:
    """Close an MPE env (or its vec-env wrapper) under the shared lock so its global ``pygame.quit()``
    can't race another env's construction/close on a different thread. Best-effort — a teardown hiccup
    must never crash the preview / trainer thread."""
    with _PYGAME_LOCK:
        try:
            env.close()
        except Exception:  # noqa: BLE001 — never let env teardown crash the caller
            logger.debug("MPE env close failed", exc_info=True)


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


# --- competitive self-play (simple_tag, G7b-2) ---------------------------------------------------
#
# Two species with *different* obs sizes + opposite rewards (predator obs 16 vs prey 14) break the
# homogeneous parameter-sharing bridge above. The trainer learns one species at a time against the
# *frozen* snapshot of the other (frozen self-play, ADR-048). The bridge to SB3 is the same SuperSuit
# stack — but applied to a parallel env that exposes **only the learner species** (so its agents are
# homogeneous again) while stepping the opponent species from the frozen policy inside ``step``.


# The wrapper subclasses ``pettingzoo.utils.env.ParallelEnv`` (SuperSuit's bridge isinstance-checks
# it), built lazily + cached so importing this module at app startup stays pettingzoo-free (the manager
# imports ``is_competitive_ma`` at boot). One class, reused for every species env.
_SPECIES_ENV_CLASS: Any = None


def _species_env_class() -> Any:
    """Lazily define + cache the species-filtering ParallelEnv wrapper class (frozen self-play, G7b-2).

    Exposes only the **learner** species (so its agents are homogeneous → SuperSuit can parameter-share
    them), and steps the opponent species from a *frozen* numpy snapshot (ADR-019) injected in ``step``.
    Constant-agent episodes (simple_tag runs the full ``max_cycles`` — no agent dies), so the learner
    set is fixed, which is what SuperSuit's Markov vector wrapper needs. Defined inside the function so
    the ``pettingzoo`` import happens only when a self-play env is actually built."""
    global _SPECIES_ENV_CLASS
    if _SPECIES_ENV_CLASS is not None:
        return _SPECIES_ENV_CLASS

    from pettingzoo.utils.env import ParallelEnv

    class _SpeciesParallelEnv(ParallelEnv):  # type: ignore[misc]  # ParallelEnv is untyped-generic
        def __init__(
            self,
            raw: Any,
            learner_role: str,
            opponent_predict: Any | None = None,
            contact_force: float | None = None,
            randomize_obstacles: bool = False,
        ) -> None:
            self.env = raw
            self._learner_role = learner_role
            self._opponent_predict = opponent_predict
            self._contact_force = contact_force  # re-applied each reset (survives the SuperSuit clone)
            self._randomize_obstacles = randomize_obstacles  # re-roll obstacle size/pos each episode
            self._learner = [a for a in raw.possible_agents if agent_role(a) == learner_role]
            self._opp = [a for a in raw.possible_agents if agent_role(a) != learner_role]
            self.possible_agents = list(self._learner)
            self.agents = list(self._learner)
            self._opp_obs: dict[str, Any] = {}
            self._closed = False
            # SuperSuit's MarkovVectorEnv reads par_env.unwrapped.render_mode; expose it (None is fine —
            # the training env is headless, the preview renders the raw parallel env separately).
            self.render_mode = getattr(raw, "render_mode", None)
            md = dict(getattr(raw, "metadata", {}) or {})
            md["is_parallelizable"] = True
            self.metadata = md

        def observation_space(self, agent: str) -> Any:
            return self.env.observation_space(agent)

        def action_space(self, agent: str) -> Any:
            return self.env.action_space(agent)

        def reset(self, seed: int | None = None, options: Any | None = None) -> tuple[dict, dict]:
            obs, info = self.env.reset(seed=seed, options=options)
            unwrapped = getattr(self.env, "unwrapped", self.env)
            world = getattr(unwrapped, "world", None)
            if self._contact_force is not None and world is not None:  # re-apply after a clone reset
                world.contact_force = self._contact_force
            if self._randomize_obstacles and world is not None:  # re-roll obstacle size + non-overlap
                rng = getattr(unwrapped, "np_random", None)
                if rng is not None:
                    _randomize_obstacles(world, rng)
            self._opp_obs = {a: obs[a] for a in self._opp if a in obs}
            self.agents = [a for a in self.env.agents if a in self._learner]
            return (
                {a: obs[a] for a in self._learner if a in obs},
                {a: info.get(a, {}) for a in self._learner},
            )

        def step(self, actions: dict) -> tuple[dict, dict, dict, dict, dict]:
            merged = dict(actions)
            for a in self._opp:
                merged[a] = self._opponent_action(a)
            obs, rew, term, trunc, info = self.env.step(merged)
            self._opp_obs = {a: obs[a] for a in self._opp if a in obs}
            self.agents = [a for a in self.env.agents if a in self._learner]

            def sel(d: dict) -> dict:
                return {a: d[a] for a in self._learner if a in d}

            return sel(obs), sel(rew), sel(term), sel(trunc), sel(info)

        def _opponent_action(self, agent: str) -> Any:
            if self._opponent_predict is None or agent not in self._opp_obs:
                return self.env.action_space(agent).sample()
            try:
                return int(self._opponent_predict(self._opp_obs[agent]))
            except Exception:  # noqa: BLE001 — a flaky opponent forward must not crash a round
                return self.env.action_space(agent).sample()

        def render(self) -> Any:
            return self.env.render()

        def close(self) -> None:
            # Idempotent: the SB3/gymnasium vec env closes us explicitly AND again via ``__del__`` on
            # GC, and mpe2's close() calls the global pygame.quit() — so guard against a double quit.
            # The pygame-global serialization is done by the CALLER (``close_env`` / ``make_species_vec_env``
            # hold ``_PYGAME_LOCK``); this method must reference no module globals so SuperSuit can still
            # cloudpickle-clone the wrapper (an RLock/logger reference here is unpicklable).
            if self._closed:
                return
            self._closed = True
            try:  # noqa: SIM105 — contextlib.suppress would add a module global ref (unpicklable clone)
                self.env.close()
            except Exception:  # noqa: BLE001 — never let env teardown crash the trainer
                pass

    _SPECIES_ENV_CLASS = _SpeciesParallelEnv
    return _SPECIES_ENV_CLASS


def make_species_vec_env(
    env_id: str,
    learner_role: str,
    opponent_predict: Any | None = None,
    obstacle_count: int | None = None,
) -> Any:
    """SB3-compatible vec env that trains **one species** of ``env_id`` against a frozen opponent.

    Wraps the raw parallel env (learner-only, the opponent species auto-acted by ``opponent_predict``,
    a numpy snapshot baked in here) and bridges it through SuperSuit + ``VecMonitor`` exactly like
    :func:`make_vec_env`. The learner agents are homogeneous, so the result is one parameter-shared
    ``MlpPolicy`` over that species (G7b-2). The trainer rebuilds this per round with the opponent's
    latest snapshot (SuperSuit cloudpickle-clones the env, so the opponent must be baked in at build
    time, not mutated later).

    The whole construction runs under ``_PYGAME_LOCK``: the raw mpe2 env's pygame.init() **and**
    SuperSuit's cloudpickle clone (which re-inits pygame on the copy) must not race the preview thread's
    own mpe2 construction/close (the Windows segfault source)."""
    import supersuit as ss
    from stable_baselines3.common.vec_env import VecMonitor

    from app.envs.registry import get_env

    spec = get_env(env_id)
    competitive = spec is not None and spec.competitive
    force = _TAG_CONTACT_FORCE if competitive else None
    with _PYGAME_LOCK:
        raw = make_parallel_env(env_id, obstacle_count=obstacle_count)  # reentrant lock (also locks)
        wrapped = _species_env_class()(
            raw, learner_role, opponent_predict,
            contact_force=force, randomize_obstacles=competitive,
        )
        vec = ss.pettingzoo_env_to_vec_env_v1(wrapped)
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
