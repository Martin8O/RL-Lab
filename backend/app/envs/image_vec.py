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
* **VizDoom** (``family=="vizdoom"``) → :func:`make_vizdoom` below (G8b) — a 3rd image pipeline for
  the ZDoom FPS scenarios. Unlike Atari (already a ``Box`` image) the Gymnasium VizDoom wrapper emits
  a **``Dict`` obs** (``{'screen': Box(240,320,3), 'gamevariables': …}``), so a screen-extraction
  wrapper (Dict→``screen`` ``Box``) runs first, then a WarpFrame grayscale/84×84 + frame stack. NOT
  ``AtariWrapper`` (no ALE, no episodic-life / fire-reset / reward-clip) — closer to CarRacing's
  raw-env→WarpFrame→VecFrameStack shape. ``Discrete`` actions.

Like ``atari.py``, the SB3 imports are lazy inside the functions (paid only when an image env
actually runs), and ``render_mode="rgb_array"`` is set so the preview/play loops can grab the raw
colour frame for the JPEG even though the CnnPolicy consumes the stacked tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# gymnasium is imported at module level (the VizDoom screen wrapper subclasses gym.ObservationWrapper,
# which is evaluated at class-definition time). This module is itself imported ONLY lazily inside worker
# functions, so gymnasium's import cost is still paid only when an image env actually runs — never at startup.
import gymnasium as gym

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


# VizDoom (G8b) — a NEW image family (its own vec builder, distinct from Atari + CarRacing).
_VIZDOOM_N_STACK = 4  # frames stacked into one observation (Atari's standard temporal window)
_VIZDOOM_FRAME = 84  # WarpFrame target: the grayscale H×W the CnnPolicy consumes
# Tics advanced per policy step (Atari-style 4-skip, via the wrapper's own frame_skip). The engine
# SUMS reward across the skipped tics, so an episode's *return* is skip-invariant — the registry's
# min_score/solved_score calibration (measured at skip=1) holds. Baked in here (not the registry
# make_kwargs) so it applies ONLY to the CnnPolicy path (trainer / preview / AI-play); human play
# goes through factory.make_env at the smooth default skip=1 (one key press ≠ 4 tics of motion).
_VIZDOOM_FRAME_SKIP = 4


class _VizdoomScreen(gym.ObservationWrapper):
    """Dict obs → just the ``screen`` ``Box`` (the extraction G8a pinned).

    The Gymnasium VizDoom wrapper hands back ``{'screen': Box(240,320,3,uint8), 'gamevariables':
    Box((n,),f32)}``; a ``CnnPolicy`` needs a plain image ``Box``, so this drops everything but the
    screen buffer (matching what Atari already emits) *before* the shared WarpFrame downscale.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = env.observation_space["screen"]  # type: ignore[index]

    def observation(self, observation: Any) -> Any:
        return observation["screen"]


def _make_vizdoom_env(gym_id: str, render_mode: str = "rgb_array") -> gym.Env:
    """One fully-wrapped VizDoom env — module-level so it stays picklable for ``SubprocVecEnv``.

    Registers the ``Vizdoom*`` ids (import side effect), builds the scenario at the 4-tic frame skip,
    extracts the ``screen`` Box, then WarpFrame-downscales to 84×84 grayscale. ``render_mode`` stays
    ``rgb_array`` so the preview/AI-play loops grab the raw colour frame for the JPEG (WarpFrame only
    rewrites the *observation*, so ``render()`` is still the full-colour Doom view).
    """
    from stable_baselines3.common.atari_wrappers import WarpFrame
    from vizdoom import (
        gymnasium_wrapper,  # noqa: F401 — import side effect registers the Vizdoom* ids
    )

    env = gym.make(gym_id, render_mode=render_mode, frame_skip=_VIZDOOM_FRAME_SKIP)
    env = _VizdoomScreen(env)
    return WarpFrame(env, width=_VIZDOOM_FRAME, height=_VIZDOOM_FRAME)


def make_vizdoom(
    gym_id: str,
    n_envs: int,
    *,
    make_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> VecEnv:
    """Build the shared VizDoom vec env: ``make_vec_env`` (screen-extract + WarpFrame) + ``VecFrameStack(4)``.

    A 3rd image pipeline alongside ``make_atari`` / ``make_carracing`` (G8b). The Gymnasium VizDoom
    env emits a ``Dict`` obs, so each env is built through :func:`_make_vizdoom_env` (screen-extraction
    → grayscale 84×84) rather than an ALE ``AtariWrapper``; a 4-frame stack (→ ``Box(84,84,4)``, which
    SB3's ``VecTransposeImage`` makes channels-first for the CnnPolicy) adds the motion a single frame
    can't show. ``gym_id`` is the registered scenario id (``VizdoomBasic-v1`` …); ``make_kwargs`` is
    accepted for dispatcher symmetry (VizDoom carries no extra kwargs — the scenario is the id).

    **``SubprocVecEnv`` for the trainer (n_envs>1), ``DummyVecEnv`` for the single-env preview/AI-play**
    — same split as CarRacing. Each ZDoom instance is a heavy 3D render, so ``DummyVecEnv`` (all envs
    stepped sequentially on one core) starves the GPU; ``SubprocVecEnv`` runs the 8 rollout envs across
    cores in parallel AND gives each its own process-local engine. The n_envs=1 preview / AI-play case
    stays in-process (``DummyVecEnv``) so its loop can call ``venv.render()`` directly.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    venv = make_vec_env(
        _make_vizdoom_env, n_envs=n_envs, seed=seed,
        env_kwargs={"gym_id": gym_id}, vec_env_cls=vec_env_cls,
    )
    return VecFrameStack(venv, n_stack=_VIZDOOM_N_STACK)


def make_image_vec(
    spec: EnvSpec, n_envs: int, *, seed: int | None = None, clip_reward: bool = True
) -> VecEnv:
    """Dispatch an image-obs env to its vec builder by family — the single seam every image-obs
    caller (trainer / preview / AI play) uses so the CnnPolicy obs shape matches on all three.

    ``family=="atari"`` → the AtariWrapper + frame-stack pipeline (``make_atari``); ``family==
    "vizdoom"`` → the screen-extract + WarpFrame + frame-stack pipeline (``make_vizdoom``, G8b);
    anything else image-obs is CarRacing → the raw-RGB + frame-stack pipeline (``make_carracing``).
    Each reads the registry row's ``make_kwargs`` (Atari: ``full_action_space``; CarRacing:
    ``continuous``; VizDoom: none) so the builder choice stays data-driven.

    ``clip_reward`` reaches the AtariWrapper's reward sign-clip (default ``True`` = the training/
    preview recipe). AI play passes ``False`` so its summed reward is the raw game score, on the same
    scale as ``solved_score``; CarRacing + VizDoom never clip, so the flag is a no-op there.
    """
    if spec.family == "atari":
        from app.envs.atari import make_atari

        return make_atari(
            spec.gym_id, n_envs, make_kwargs=spec.make_kwargs, seed=seed, clip_reward=clip_reward
        )
    if spec.family == "vizdoom":
        return make_vizdoom(spec.gym_id, n_envs, make_kwargs=spec.make_kwargs, seed=seed)
    return make_carracing(spec.gym_id, n_envs, make_kwargs=spec.make_kwargs, seed=seed)
