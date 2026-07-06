"""Shared environment factory — the one place every path builds a Gymnasium env.

The trainers (PPO, neuroevolution), the play session and the preview streamer all create
their env through :func:`make_env`, so env construction has a single source of truth. This is
where the **discrete-observation seam (G2)** lives: a Toy Text env's observation is a single
integer (which grid cell / which Taxi configuration), not a vector, so a plain ``MlpPolicy`` or
numpy genome — built for vector obs — can't consume it directly. :class:`OneHotObservation`
turns the int into a length-``n`` one-hot float vector, and the factory applies it automatically
to any ``Discrete``-observation env. So Toy Text trains on the *exact* CartPole MlpPolicy / numpy
path with no engine change; tabular Q-learning (G2b), the native consumer of discrete obs,
decodes the one-hot back to the int with an arg-max.

The factory also resolves per-variant ``make_kwargs`` (e.g. FrozenLake's ``map_name`` / ``is_slippery``,
which share one ``gym_id``) and applies an explicit episode step limit where the env has none
natively (CliffWalking) or a longer one for human play (``play_scale``).

``gymnasium`` is imported at module top, so this module must only ever be imported **lazily** (inside
a worker function), never at startup — keeping /health and the REST surface fast to boot, like the
trainers it serves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.envs.registry import get_env

if TYPE_CHECKING:
    from app.envs.registry import EnvSpec


class OneHotObservation(gym.ObservationWrapper):
    """Map a ``Discrete(n)`` observation (a single int state) to a length-``n`` one-hot vector.

    The discrete-obs seam: with this wrapper a Toy Text env presents a ``Box`` vector observation,
    so the same ``MlpPolicy`` (PPO) and numpy genome (neuroevolution) used for CartPole apply with
    no change. Tabular Q-learning consumes the same wrapped obs by ``argmax``-decoding it back to
    the int — so all three algorithms share one env path.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._n = int(env.observation_space.n)  # type: ignore[attr-defined]
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self._n,), dtype=np.float32)

    def observation(self, obs: Any) -> np.ndarray:
        vec = np.zeros(self._n, dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec


def _episode_limit(spec: EnvSpec, play_scale: int) -> int | None:
    """The explicit ``max_episode_steps`` to pass to ``gym.make``, or ``None`` for the env default.

    Returns ``None`` (use the env's native ``TimeLimit``) for a standard training run of an env that
    already has one — so existing envs are created exactly as before. An explicit limit is returned
    when the env declares ``episode_step_limit`` (e.g. CliffWalking, which has no native limit) or
    when ``play_scale`` > 1 (human play extends short episodes so a person has time to play).
    """
    native = gym.spec(spec.gym_id).max_episode_steps
    base = spec.episode_step_limit if spec.episode_step_limit is not None else native
    if base is None:
        return None
    if spec.episode_step_limit is None and play_scale <= 1:
        return None  # standard run of an env with its own TimeLimit — leave it untouched
    return base * max(1, play_scale)


def make_env(
    env_id: str,
    gym_id: str | None = None,
    *,
    render_mode: str | None = None,
    play_scale: int = 1,
) -> gym.Env:
    """Build the Gymnasium env for ``env_id``, applying the registry's variant kwargs, episode
    limit and (for discrete-obs envs) the one-hot wrapper.

    ``gym_id`` is a fallback used only when ``env_id`` is not in the registry (a defensive path for
    a direct trainer call with an off-registry env); registered envs ignore it and use the spec.
    """
    spec = get_env(env_id)
    gid = spec.gym_id if spec is not None else (gym_id or env_id)

    # Atari (ALE) envs are not in gymnasium's default registry — importing ale_py registers the
    # "ALE/*" namespace as a side effect (no explicit register_envs needed). Done lazily here, before
    # any gym.spec()/gym.make() touches the id, so the ALE import cost is paid only when an Atari env
    # is actually built (G4a; image obs → CnnPolicy/GPU).
    if gid.startswith("ALE/"):
        import ale_py  # noqa: F401 — import side effect registers the ALE namespace
    # MiniGrid envs are registered the same way — by import side effect — so import lazily here, before
    # gym.spec()/gym.make() touches the id (the family has no native gym TimeLimit; it self-truncates). G2c.
    if gid.startswith("MiniGrid"):
        import minigrid  # noqa: F401 — import side effect registers the MiniGrid-* envs
    # VizDoom scenarios (Vizdoom*-v1) register the same way — importing the Gymnasium wrapper is the
    # side effect that adds the ids. Done lazily here so human play (this raw make_env + JPEG path)
    # can build a Doom env; training/AI-play go through image_vec.make_vizdoom, which registers there. G8b.
    if gid.startswith("Vizdoom"):
        from vizdoom import gymnasium_wrapper  # noqa: F401 — side effect registers the Vizdoom* ids

    kwargs: dict[str, Any] = {}
    if spec is not None:
        kwargs.update(spec.make_kwargs)
        limit = _episode_limit(spec, play_scale)
        if limit is not None:
            kwargs["max_episode_steps"] = limit
        # MuJoCo locomotion robots render a finite checker plane and a trained runner outruns it,
        # then appears to sprint over a grey void. Swap in a floor-enlarged copy of the model XML
        # (cosmetic only — the plane collides as infinite regardless of size, so physics/obs/repro are
        # unchanged). Returns None for non-locomotion envs / on failure → stock model. Ant fix.
        if spec.family == "mujoco":
            from app.envs.mujoco_floor import floored_xml_path

            patched = floored_xml_path(gid)
            if patched is not None:
                kwargs.setdefault("xml_file", patched)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(gid, **kwargs)
    # MiniGrid: a Dict obs (7×7×3 image + direction + mission) → FlatObsWrapper flattens it to a 1-D Box
    # vector, so the same MlpPolicy (PPO) / numpy genome (neuroevolution) used for CartPole apply with no
    # engine change — the same idea as the one-hot seam below, a different wrapper. Applied for the whole
    # family on EVERY path (train/play/preview) so the obs shape never drifts between training and AI-play.
    # render() is unaffected (an ObservationWrapper passes it through), so the colourful grid still renders
    # server-side as a JPEG (the family is not in client_render → image path). G2c.
    if spec is not None and spec.family == "minigrid":
        from minigrid.wrappers import FlatObsWrapper

        env = FlatObsWrapper(env)
    # Discrete (single-int) observation → one-hot vector, so vector-obs policies/genomes apply.
    if isinstance(env.observation_space, spaces.Discrete):
        env = OneHotObservation(env)
    return env
