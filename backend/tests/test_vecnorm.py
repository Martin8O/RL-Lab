"""G5c — VecNormalize for the MuJoCo PPO path: the gate flag, the stats-in-model.zip seam, and the
four places the running obs stats must travel (training wrapper, decoupled preview forward, AI-play,
resume) while ``ep_rew_mean`` stays on the RAW scale (the skill-meter honesty requirement).

These run on **Pendulum** (a fast continuous-box vector env), not a real MuJoCo robot: the seam is
env-agnostic, so the integration test flips Pendulum's ``normalize_obs`` flag on rather than paying for
a multi-million-step MuJoCo run. The registry/gate assertions below cover the real MuJoCo rows.
"""

import io
import zipfile

import gymnasium as gym
import numpy as np
import pytest
from app.envs.registry import get_env
from app.schemas.training import PPOHyperparams, TrainConfig
from app.services import vecnorm
from app.services.checkpoints import CheckpointArtifact, LoadedCheckpoint
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_ppo import train_ppo

# -- the gate flag (data-driven, MuJoCo-only) -------------------------------


def test_normalize_obs_gated_to_mujoco() -> None:
    """All seven MuJoCo rows opt into VecNormalize; the simple vector + image envs stay off."""
    for env_id in ("hopper", "walker2d", "halfcheetah", "ant", "reacher", "swimmer", "humanoid"):
        spec = get_env(env_id)
        assert spec is not None and spec.normalize_obs is True, env_id
    # Off where it must be: the simple vector envs train fine without it, and the image path (CnnPolicy)
    # already scales pixels /255, so normalizing there would be wrong (scope fence).
    for env_id in ("cartpole", "pendulum", "lunarlander", "bipedalwalker", "carracing"):
        spec = get_env(env_id)
        assert spec is not None and spec.normalize_obs is False, env_id

    assert vecnorm.should_normalize("hopper") is True
    assert vecnorm.should_normalize("cartpole") is False
    assert vecnorm.should_normalize("does-not-exist") is False


# -- the helper module (embed / read / normalize / restore) -----------------


def _populated_vecnormalize() -> object:
    """A real VecNormalize over Pendulum, stepped enough to give non-identity running obs stats."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: gym.make("Pendulum-v1")])
    vn = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    vn.reset()
    for _ in range(80):
        vn.step(np.asarray([venv.action_space.sample()]))
    return vn


def test_obs_normalizer_matches_sb3() -> None:
    """The pure-numpy obs transform reproduces SB3's own ``VecNormalize.normalize_obs`` bit-for-bit."""
    vn = _populated_vecnormalize()
    normalize = vecnorm.obs_normalizer_from_env(vn)
    raw = np.array([0.3, -0.7, 1.2], dtype=np.float64)  # a Pendulum-shaped obs
    assert np.allclose(normalize(raw), vn.normalize_obs(raw), atol=1e-5)  # type: ignore[attr-defined]
    # The transform actually moves the obs (non-identity stats) — i.e. normalization is real, not a no-op.
    assert not np.allclose(normalize(raw), raw)


def test_embed_read_roundtrip_inside_model_zip() -> None:
    """The stats ride inside model.zip as an extra member; PPO.load ignores it, the normalizer reads it."""
    from stable_baselines3 import PPO

    model = PPO("MlpPolicy", gym.make("Pendulum-v1"), n_steps=64, batch_size=64, device="cpu")
    buf = io.BytesIO()
    model.save(buf)
    plain = buf.getvalue()
    assert vecnorm.load_obs_normalizer(plain) is None  # no stats yet → un-normalized

    vn = _populated_vecnormalize()
    embedded = vecnorm.embed_stats(plain, vn)

    names = zipfile.ZipFile(io.BytesIO(embedded)).namelist()
    assert "vecnormalize.pkl" in names  # the stats member is present...
    assert "policy.pth" in names  # ...alongside SB3's own members (the zip is still a valid model.zip)
    PPO.load(io.BytesIO(embedded), device="cpu")  # ignores the extra member, loads cleanly

    normalize = vecnorm.load_obs_normalizer(embedded)
    assert normalize is not None
    raw = np.array([0.3, -0.7, 1.2], dtype=np.float64)
    assert np.allclose(normalize(raw), vn.normalize_obs(raw), atol=1e-5)  # type: ignore[attr-defined]


def test_restore_into_copies_running_stats() -> None:
    """Resume copies the saved obs/reward running stats into a fresh (identity) VecNormalize."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO("MlpPolicy", gym.make("Pendulum-v1"), n_steps=64, batch_size=64, device="cpu")
    buf = io.BytesIO()
    model.save(buf)
    saved_vn = _populated_vecnormalize()
    embedded = vecnorm.embed_stats(buf.getvalue(), saved_vn)

    fresh = VecNormalize(DummyVecEnv([lambda: gym.make("Pendulum-v1")]))  # identity stats
    assert np.allclose(fresh.obs_rms.mean, 0.0)
    assert vecnorm.restore_into(fresh, embedded) is True
    assert np.allclose(fresh.obs_rms.mean, saved_vn.obs_rms.mean)  # type: ignore[attr-defined]
    assert fresh.obs_rms.count > 1.0  # the running count travelled too

    # A blob with no embedded stats (pre-G5c / un-normalized) is a clean no-op.
    assert vecnorm.restore_into(fresh, buf.getvalue()) is False


# -- end-to-end through the trainer (Pendulum flagged as normalized) --------


@pytest.fixture
def _pendulum_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flip Pendulum's ``normalize_obs`` on so the trainer exercises the real MuJoCo code path cheaply."""
    spec = get_env("pendulum")
    assert spec is not None
    monkeypatch.setattr(spec, "normalize_obs", True)


def _tiny_normalized_config() -> TrainConfig:
    return TrainConfig(
        env_id="pendulum", algo="ppo", seed=1, total_timesteps=256,
        hyperparams=PPOHyperparams(n_steps=64, batch_size=64),
    )


def test_trained_snapshot_carries_stats_and_reward_stays_raw(_pendulum_normalized: None) -> None:
    """A normalized run: the snapshot embeds the stats, the preview forward is normalized, and
    ``ep_rew_mean`` stays on the RAW Pendulum scale (so the skill meter reads identically to today)."""
    metrics: list = []
    snaps: list[CheckpointArtifact] = []
    captured: dict = {}
    train_ppo(
        _tiny_normalized_config(), "Pendulum-v1", TrainControl(),
        metrics.append, lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=snaps.append,
    )

    # Honesty: ep_rew_mean is the raw return (Pendulum runs deep negative ~ -1000s), NOT the scaled
    # reward VecNormalize feeds the optimizer (which would sit near 0). Monitor-inside-VecNormalize.
    rewarded = [m.ep_rew_mean for m in metrics if m.ep_rew_mean is not None]
    assert rewarded, "no episode reward recorded"
    assert all(r < -100.0 for r in rewarded), rewarded  # clearly raw, not normalized

    # The snapshot blob is a model.zip carrying the embedded VecNormalize stats.
    assert snaps, "no snapshot captured"
    blob = snaps[-1].blob
    assert blob[:2] == b"PK"
    assert "vecnormalize.pkl" in zipfile.ZipFile(io.BytesIO(blob)).namelist()

    # The decoupled preview predict fn normalizes internally and returns a clipped action vector.
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert np.shape(action) == (1,) and -2.0 <= float(action[0]) <= 2.0
    env.step(action)
    env.close()


def test_ai_play_applies_embedded_stats(_pendulum_normalized: None) -> None:
    """AI-play of a normalized checkpoint loads + applies the obs stats (the key 'stats travel' test).

    Without applying them the policy — trained on normalized obs — would get raw obs and act like an
    untrained net; we verify the play predict fn pre-normalizes (so its action differs from what the
    same net produces on un-normalized obs) and still emits a valid clipped action vector.
    """
    snaps: list[CheckpointArtifact] = []
    train_ppo(
        _tiny_normalized_config(), "Pendulum-v1", TrainControl(),
        lambda _m: None, lambda _p: None, None, snaps.append,
    )
    blob = snaps[-1].blob

    config = TrainConfig(env_id="pendulum", algo="ppo", seed=1)
    meta = type("M", (), {"artifact": "model.zip"})()  # minimal stand-in; predict only reads blob+config
    predict = predict_from_checkpoint(LoadedCheckpoint(meta=meta, config=config, blob=blob))  # type: ignore[arg-type]

    obs = np.array([0.5, -0.5, 0.8], dtype=np.float32)
    action = predict(obs)
    assert np.shape(action) == (1,) and -2.0 <= float(action[0]) <= 2.0  # a valid clipped action

    # Prove normalization is actually applied: the same checkpoint WITHOUT the stats (a plain model.zip)
    # produces a different action on the same raw obs.
    plain = _strip_member(blob, "vecnormalize.pkl")
    plain_predict = predict_from_checkpoint(LoadedCheckpoint(meta=meta, config=config, blob=plain))  # type: ignore[arg-type]
    assert not np.allclose(action, plain_predict(obs)), "stats had no effect — they didn't travel"


def test_normalized_resume_continues(_pendulum_normalized: None) -> None:
    """Resuming a normalized run continues the step counter and restores the running stats."""
    snaps: list[CheckpointArtifact] = []
    train_ppo(
        _tiny_normalized_config(), "Pendulum-v1", TrainControl(),
        lambda _m: None, lambda _p: None, None, snaps.append,
    )
    final = snaps[-1]

    metrics: list = []
    resumed_snaps: list[CheckpointArtifact] = []
    train_ppo(
        TrainConfig(
            env_id="pendulum", algo="ppo", seed=1, total_timesteps=512,
            hyperparams=PPOHyperparams(n_steps=64, batch_size=64),
        ),
        "Pendulum-v1", TrainControl(),
        metrics.append, lambda _p: None, None, resumed_snaps.append,
        resume_blob=final.blob,
    )
    assert metrics and metrics[0].timesteps > final.timesteps  # continued, not restarted
    assert "vecnormalize.pkl" in zipfile.ZipFile(io.BytesIO(resumed_snaps[-1].blob)).namelist()


def _strip_member(model_zip: bytes, member: str) -> bytes:
    """Return a copy of the zip without ``member`` (a pre-G5c / un-normalized checkpoint)."""
    src = zipfile.ZipFile(io.BytesIO(model_zip))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in src.namelist():
            if name != member:
                zf.writestr(name, src.read(name))
    return out.getvalue()
