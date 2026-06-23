"""VecNormalize obs/reward normalization for the MuJoCo PPO path (G5c).

PPO on MuJoCo **stalls without observation normalization** — the raw obs span wildly different scales
(joint angles ~±1 vs contact forces in the hundreds), worst on the high-dim robots (Humanoid 348-D,
Ant 105-D). SB3 rl-zoo3 uses ``normalize: true`` for *every* MuJoCo PPO config: it wraps the training
vec env in :class:`~stable_baselines3.common.vec_env.VecNormalize` (running obs mean/std + reward
scaling), the single biggest lever to make MuJoCo climb toward ``solved``.

The real work is **not** the one-line training wrapper — it is the inference paths. ``VecNormalize``
keeps running obs statistics, and those stats must travel with the policy to **every** inference path
(the decoupled preview numpy forward — ADR-019; AI-play; and a resumed run) or a trained agent gets
*unnormalized* obs at inference and plays like it never trained. This module is the one home for that:

* :func:`wrap_train_env` — ``VecNormalize(DummyVecEnv([Monitor(env)]))``: the Monitor sits **inside** the
  wrapper, so SB3's ``ep_info_buffer`` (→ ``ep_rew_mean`` → the skill meter) records the **raw** episode
  return while the policy still trains on normalized obs + scaled reward (the honesty requirement).
* :func:`embed_stats` / :func:`restore_into` — the stats ride **inside** the ``model.zip`` blob as an
  extra ``vecnormalize.pkl`` member, so the single checkpoint artifact carries everything (no second
  file, no checkpoint-store contract change; ``PPO.load`` ignores the extra member). One blob travels to
  the save slot, the resume path and AI-play alike.
* :func:`obs_normalizer_from_env` / :func:`load_obs_normalizer` — a pure-numpy obs transform
  ``clip((obs - mean) / sqrt(var + eps), -clip, clip)`` for the inference paths, baked either from the
  live wrapper (preview) or from a checkpoint blob (AI-play). **Reward** normalization is training-only
  and is never applied at inference, so reward / skill stay on the raw scale.

Kept SB3-free at import time (SB3 is imported lazily inside the functions, or by ``pickle`` while
reading stats), like the trainers and the play session this module serves.
"""

import io
import pickle
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np

from app.envs.registry import get_env

# The member name the stats ride under inside model.zip (SB3's own convention for a saved VecNormalize).
_VECNORM_MEMBER = "vecnormalize.pkl"
# rl-zoo3's standard MuJoCo clipping (also the SB3 VecNormalize defaults we pass explicitly).
_CLIP_OBS = 10.0
_CLIP_REWARD = 10.0

# A raw obs → normalized obs transform (numpy). None where an env opts out of normalization.
ObsNormalizer = Callable[[Any], np.ndarray]


def should_normalize(env_id: str) -> bool:
    """True if this env opts into obs/reward normalization — the MuJoCo family only (data-driven)."""
    spec = get_env(env_id)
    return spec is not None and spec.normalize_obs


def wrap_train_env(make_base: Callable[[], Any], gamma: float) -> Any:
    """Wrap a freshly-built single env as ``VecNormalize(DummyVecEnv([Monitor(env)]))``.

    The Monitor sits **inside** VecNormalize, so SB3's episode-info buffer records the *raw* episode
    return — ``ep_rew_mean`` stays in real units and the ``[min_score, solved_score]`` skill meter is
    unchanged — while the policy trains on normalized obs + scaled reward (the rl-zoo3 ``normalize:
    true`` recipe). The inner ``DummyVecEnv`` exposes ``seed()``, so SB3 seeds the run normally.
    """
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: Monitor(make_base())])
    return VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=True,
        clip_obs=_CLIP_OBS,
        clip_reward=_CLIP_REWARD,
        gamma=gamma,
    )


def embed_stats(model_zip: bytes, vec_normalize: Any) -> bytes:
    """Return ``model_zip`` with the VecNormalize stats added as a ``vecnormalize.pkl`` member.

    SB3's ``model.zip`` is a plain zip and ``VecNormalize.__getstate__`` strips the venv before
    pickling, so this embeds a self-contained obs/reward-stats payload that ``PPO.load`` ignores.
    Keeping the checkpoint a **single blob** means the stats travel with the model everywhere (the save
    slot, the resume blob, the AI-play load) without touching the ML-free checkpoint store contract.
    """
    payload = pickle.dumps(vec_normalize)
    buf = io.BytesIO(model_zip)
    with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_VECNORM_MEMBER, payload)
    return buf.getvalue()


def _read_stats(model_zip: bytes) -> Any | None:
    """Unpickle the embedded VecNormalize, or ``None`` if the blob carries no stats / can't be read.

    Best-effort by design: an un-normalized env, a pre-G5c checkpoint, or a corrupt/incompatible
    payload all read as ``None`` so the caller falls back to passing obs through un-normalized.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(model_zip)) as zf:
            if _VECNORM_MEMBER not in zf.namelist():
                return None
            payload = zf.read(_VECNORM_MEMBER)
    except (zipfile.BadZipFile, OSError):
        return None
    try:
        return pickle.loads(payload)  # imports SB3's VecNormalize lazily
    except Exception:  # noqa: BLE001 — a corrupt/incompatible payload is treated as "no stats"
        return None


def _normalizer(mean: Any, var: Any, eps: float, clip: float) -> ObsNormalizer:
    """Bake the VecNormalize obs transform into a closure over copied stats (pure numpy)."""
    mean_a = np.asarray(mean, dtype=np.float64)
    var_a = np.asarray(var, dtype=np.float64)

    def normalize(obs: Any) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float64)
        return np.clip((x - mean_a) / np.sqrt(var_a + eps), -clip, clip)

    return normalize


def obs_normalizer_from_env(vec_normalize: Any) -> ObsNormalizer:
    """Bake a pure-numpy obs normalizer from a **live** VecNormalize wrapper (the preview path)."""
    rms = vec_normalize.obs_rms
    return _normalizer(
        rms.mean, rms.var, float(vec_normalize.epsilon), float(vec_normalize.clip_obs)
    )


def load_obs_normalizer(model_zip: bytes) -> ObsNormalizer | None:
    """Bake a pure-numpy obs normalizer from a checkpoint blob's embedded stats (``None`` if absent)."""
    vn = _read_stats(model_zip)
    if vn is None:
        return None
    try:
        return obs_normalizer_from_env(vn)
    except Exception:  # noqa: BLE001 — a missing/odd obs_rms → behave as un-normalized
        return None


def restore_into(vec_normalize_env: Any, model_zip: bytes) -> bool:
    """Copy the saved running stats from ``model_zip`` into a fresh VecNormalize (resume). True if done.

    A resumed run rebuilds a *fresh* VecNormalize (identity stats); without this the policy — trained on
    normalized obs — would see near-raw obs and degrade until the stats re-converged from scratch.
    Copies both ``obs_rms`` (obs normalization) and ``ret_rms`` (reward scaling) so training continues
    seamlessly. No-op (returns ``False``) for an un-normalized env or a pre-G5c checkpoint.
    """
    saved = _read_stats(model_zip)
    if saved is None:
        return False
    try:
        vec_normalize_env.obs_rms = saved.obs_rms
        vec_normalize_env.ret_rms = saved.ret_rms
        return True
    except Exception:  # noqa: BLE001 — a shape/attr mismatch → keep the fresh (identity) stats
        return False
