"""Image-observation vec-env builders + the family dispatcher (the GPU CnnPolicy path).

Every image-obs env feeds a ``CnnPolicy`` that must see the **exact same** observation shape on
every path (trainer, live preview, AI play), or the obs drifts between training and inference. So
all three callers build their env through one dispatcher here — :func:`make_image_vec` — which
routes by family to the right builder:

* **Atari** (``family=="atari"``) → :func:`app.envs.atari.make_atari` (AtariWrapper: grayscale
  84×84, 4-frame max-skip, reward-clip, episodic-life; ``Discrete(18)``; obs ``Box(84,84,4)``).
* **CarRacing** (``family=="box2d"`` + image obs) → :func:`make_carracing` below — a *different*
  pipeline: the raw 96×96×3 RGB the env emits (no AtariWrapper) plus a small frame stack, and a
  continuous ``Box(3)`` action. This is the env the seam roadmap flagged as the last ``int``→``box``
  case: image obs **and** a box action (G3c-train).

Like ``atari.py``, the SB3 imports are lazy inside the functions (paid only when an image env
actually runs), and ``render_mode="rgb_array"`` is set so the preview/play loops can grab the raw
colour frame for the JPEG even though the CnnPolicy consumes the stacked tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

    from app.envs.registry import EnvSpec

# Frames stacked into one CarRacing observation. Two is enough to perceive the car's velocity +
# heading (a single frame is positionless); the SB3 zoo uses 2 for CarRacing. The one knob to
# revisit if learning stalls — more stack = more temporal context at a wider obs/buffer cost.
_CARRACING_N_STACK = 2


def make_carracing(
    gym_id: str,
    n_envs: int,
    *,
    make_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> VecEnv:
    """Build the shared CarRacing vec env: ``make_vec_env`` + ``VecFrameStack(2)`` (no AtariWrapper).

    CarRacing is image-obs but **not** an ALE game, so it does NOT take the Atari preprocessing
    (grayscale/84×84/reward-clip/episodic-life would all be wrong here). The obs is the raw 96×96×3
    RGB the env emits; a 2-frame stack (→ ``Box(96,96,6)``, which SB3's ``VecTransposeImage`` makes
    channels-first for the CnnPolicy) adds the velocity/heading a single frame can't show. The
    action is the continuous ``Box(3)`` steer/gas/brake (``continuous=True`` rides in ``make_kwargs``
    from the registry row). Used with ``n_envs=8`` by the trainer and ``n_envs=1`` by the preview /
    AI-play loops so all three see byte-identical obs/action shapes.

    **Multi-env trainer uses ``SubprocVecEnv`` (n_envs>1), not the SB3 default ``DummyVecEnv``.**
    CarRacing's per-step cost is a heavy **pygame** render of the observation (``_render('state_pixels')``
    every step), so DummyVecEnv — which steps all N envs *sequentially on one core* — pins a single core
    and starves the GPU (measured: 8 envs → ~176 steps/s, 1 core busy, GPU ~3 %). SubprocVecEnv runs the
    envs across cores in parallel (measured **4.1×**: ~719 steps/s, 8 cores busy) AND gives each env its
    **own process-local pygame**, which fixes the torn-frame corruption seen when the live preview
    rendered concurrently in the main process (pygame's font/surfarray state is not thread-safe). The
    single-env preview / AI-play case stays **in-process** (``DummyVecEnv``) so its loop can call
    ``venv.render()`` directly and so it adds no spawn cost.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

    env_kwargs: dict[str, Any] = {"render_mode": "rgb_array"}
    if make_kwargs:
        env_kwargs.update(make_kwargs)  # continuous=True (from the registry row)

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = make_vec_env(gym_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs, vec_env_cls=vec_env_cls)
    return VecFrameStack(venv, n_stack=_CARRACING_N_STACK)


def make_image_vec(spec: EnvSpec, n_envs: int, *, seed: int | None = None) -> VecEnv:
    """Dispatch an image-obs env to its vec builder by family — the single seam every image-obs
    caller (trainer / preview / AI play) uses so the CnnPolicy obs shape matches on all three.

    ``family=="atari"`` → the AtariWrapper + frame-stack pipeline (``make_atari``); anything else
    image-obs is CarRacing → the raw-RGB + frame-stack pipeline (``make_carracing``). Both read the
    registry row's ``make_kwargs`` (Atari: ``full_action_space``; CarRacing: ``continuous``) so the
    builder choice stays data-driven.
    """
    if spec.family == "atari":
        from app.envs.atari import make_atari

        return make_atari(spec.gym_id, n_envs, make_kwargs=spec.make_kwargs, seed=seed)
    return make_carracing(spec.gym_id, n_envs, make_kwargs=spec.make_kwargs, seed=seed)
