"""Shared Atari (image-obs) vec-env builder — the GPU training/preview path (G4b).

Atari is the project's first **image-observation** family: a ``CnnPolicy`` consumes a stack of
preprocessed 84×84 grayscale frames, not the raw 210×160×3 RGB the env emits. The exact wrapper
stack must be *identical* on every path that feeds the policy, or the obs shape drifts between
training and the live preview. So both the trainer (``n_envs=8``) and the decoupled preview
streamer (``n_envs=1``) build their env through this one function — the image-obs analogue of
``factory.make_env`` (which stays gym-only; this needs SB3's vec wrappers).

Pipeline = SB3 ``make_atari_env`` (applies ``AtariWrapper``: noop-reset, 4-frame max-skip,
episodic-life, fire-reset, grayscale 84×84, reward-clip) → ``VecFrameStack(n_stack=4)`` →
(SB3 auto-adds ``VecTransposeImage`` for the CnnPolicy). The obs space is ``Box(84, 84, 4)``.

Two gotchas are baked in here so callers can't get them wrong:
  * **``frameskip=1``** in ``env_kwargs`` — ``ALE/*-v5`` defaults to an internal frameskip of 4,
    and ``AtariWrapper``'s own ``MaxAndSkip(4)`` would otherwise **double-skip** (16 game frames
    per agent step). Setting the env's own frameskip to 1 leaves the single 4-skip in the wrapper.
  * **``full_action_space=True``** (passed in via the registry row's ``make_kwargs``) — every ALE
    game then exposes all 18 actions at fixed indices, so the trained policy's action space matches
    the shared G4a human keymap (and the future G4c AI-play env). Data-driven, not hardcoded here.

``render_mode="rgb_array"`` is set so the preview can grab the **raw colour frame** for the JPEG:
``WarpFrame`` only rewrites the *observation*, so ``venv.render("rgb_array")`` is still full colour.

Imported lazily inside worker threads (like the trainers/streamers it serves), so SB3/ale_py
import cost is paid only when an image env actually runs — never at app startup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import VecEnv

_N_STACK = 4  # frames stacked into one observation (the standard Atari temporal window)


def make_atari(
    gym_id: str,
    n_envs: int,
    *,
    make_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
    clip_reward: bool = True,
) -> VecEnv:
    """Build the shared Atari vec env: ``make_atari_env`` + ``VecFrameStack(4)``.

    ``gym_id`` is the ``ALE/<Game>-v5`` id; ``make_kwargs`` is the registry row's kwargs (carries
    ``full_action_space=True``). Used with ``n_envs=8`` by the trainer and ``n_envs=1`` by the
    preview streamer so both see byte-identical obs/action shapes.

    ``clip_reward`` is the AtariWrapper's reward sign-clip (``True`` for training/preview — the DQN
    recipe). **AI play passes ``False``** so the per-step reward it sums is the *raw game score*, on
    the same scale as the registry's ``solved_score`` and the training chart's ``ep_rew_mean`` (whose
    Monitor sits inside the clip, so it already reads raw). Clipping touches only the reward, never
    the observation, so the obs stays byte-identical to the clipped paths — the policy doesn't see
    reward at inference, so an unclipped AI-play env can't drift the CnnPolicy.
    """
    from app.services.system_info import require_ale_py

    require_ale_py()  # import ale_py (registers "ALE/*") or raise a clean typed error (R1/ADR-101)
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack

    env_kwargs: dict[str, Any] = {"frameskip": 1, "render_mode": "rgb_array"}
    if make_kwargs:
        env_kwargs.update(make_kwargs)  # full_action_space=True (from the registry row)

    venv = make_atari_env(
        gym_id, n_envs=n_envs, seed=seed, env_kwargs=env_kwargs,
        wrapper_kwargs={"clip_reward": clip_reward},
    )
    return VecFrameStack(venv, n_stack=_N_STACK)
