"""S5c — Deep Q-Network (the 7th algorithm): registry gating, the off-policy value-based trainer
reaching a terminal state on a discrete-action env, the decoupled int preview policy, stop, and AI-play
loading.

DQN is the discrete-action mirror of SAC/TD3 and PPO's value-based counterpart — a peer trainer behind
the same manager (ADR-004/028) reusing the whole PPO lane (metrics + progress frames, discrete predict,
skill meter), gated to discrete-action envs. These tests run a *tiny* real DQN on CartPole (low
learning_starts + small buffer/batch) so they exercise the genuine SB3 path while staying fast.
"""

import numpy as np
import pytest
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import DQNHyperparams, TrainConfig
from app.services.checkpoints import CheckpointArtifact, LoadedCheckpoint
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_dqn import _dqn_kwargs, train_dqn
from fastapi.testclient import TestClient

client = TestClient(app)

# The discrete-action envs DQN is offered on (supported_algos) — curated (S5c), not every discrete env.
_DQN_ENVS = ["cartpole", "mountaincar", "acrobot", "lunarlander", "pong"]


def _tiny_dqn_config(total: int = 1500, seed: int = 1) -> TrainConfig:
    """A short CartPole DQN run that does a few hundred real gradient updates in ~seconds on CPU."""
    return TrainConfig(
        env_id="cartpole",
        algo="dqn",
        seed=seed,
        total_timesteps=total,
        dqn=DQNHyperparams(
            buffer_size=2000, batch_size=64, learning_starts=200, train_freq=4,
            target_update_interval=10,
        ),
    )


# -- registry gating --------------------------------------------------------


@pytest.mark.parametrize("env_id", _DQN_ENVS)
def test_dqn_offered_on_discrete_envs(env_id: str) -> None:
    spec = get_env(env_id)
    assert spec is not None, f"{env_id} not registered"
    assert spec.action_space == "discrete"  # DQN is value-based, discrete-action only
    assert "dqn" in spec.supported_algos
    # The dqn hyperparam block must be exposed for every env that lists it (the sidebar reads it).
    assert "dqn" in spec.hyperparams
    block = spec.hyperparams["dqn"]
    assert {
        "learning_rate", "gamma", "buffer_size", "train_freq",
        "target_update_interval", "exploration_fraction", "exploration_final_eps",
    } <= set(block)
    # DQN explores via ε-greedy, not entropy (SAC) or injected action noise (TD3): neither must appear.
    assert "ent_coef" not in block
    assert "train_noise" not in block


def test_dqn_includes_the_atari_birthplace() -> None:
    """DQN must be offered on Atari (its literal birthplace) — an image-obs discrete env on the GPU path."""
    pong = get_env("pong")
    assert pong is not None
    assert pong.obs_type == "image" and "dqn" in pong.supported_algos


def test_dqn_not_offered_on_continuous_board_or_ma_envs() -> None:
    """DQN must stay off every continuous / board / multi-agent env (the complement of SAC/TD3's gate)."""
    for env_id in ("pendulum", "mountaincarcontinuous", "bipedalwalker", "tictactoe", "chess"):
        spec = get_env(env_id)
        assert spec is not None and "dqn" not in spec.supported_algos, env_id


def test_only_discrete_envs_list_dqn() -> None:
    """Defensive: any env that lists DQN must be a discrete action space (no continuous leakage)."""
    for spec in list_envs():
        if "dqn" in spec.supported_algos:
            assert spec.action_space == "discrete", f"{spec.id} lists dqn but isn't discrete"


def test_dqn_offpolicy_budget_set_on_classic_discretes() -> None:
    """The classic discrete DQN envs carry an off-policy ★ budget; Atari reuses the PPO image budget."""
    for env_id in ("cartpole", "acrobot", "lunarlander", "mountaincar"):
        spec = get_env(env_id)
        assert spec is not None and spec.offpolicy_total_timesteps is not None, env_id
    # Atari intentionally has none → DQN reuses default_total_timesteps (the 10M image budget).
    assert get_env("pong").offpolicy_total_timesteps is None


def test_dqn_rejected_on_unsupported_env() -> None:
    """The manager rejects DQN on an env that doesn't list it (Pendulum is continuous) with a clear 400."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "pendulum", "algo": "dqn", "seed": 1, "total_timesteps": 1000,
            "dqn": {"learning_rate": 1e-3, "buffer_size": 1000, "learning_starts": 100},
        },
    )
    assert resp.status_code == 400
    assert "does not support" in resp.json()["detail"]


# -- memory: the Atari replay buffer (the resume bug) -----------------------


def test_dqn_image_path_optimizes_replay_buffer_memory() -> None:
    """Atari (image) DQN must enable optimize_memory_usage so the 84×84×4 replay buffer stores each frame
    ONCE (~4× less RAM). Without it, DQN.load on resume re-allocates the full ~2.6 GB buffer and can
    MemoryError or thrash to ~10 steps/s (the reported S5c resume bug). Vector envs must NOT use it —
    they keep handle_timeout_termination so a CartPole TimeLimit truncation isn't treated as terminal."""
    cfg = _tiny_dqn_config()
    image = _dqn_kwargs(cfg, is_image=True)
    assert image["optimize_memory_usage"] is True
    assert image["replay_buffer_kwargs"] == {"handle_timeout_termination": False}
    assert image["gradient_steps"] == 1  # Nature recipe: one update per train_freq collected steps
    assert image["batch_size"] == 32

    vector = _dqn_kwargs(cfg, is_image=False)
    assert "optimize_memory_usage" not in vector  # tiny vector buffer; keep default timeout handling
    assert "replay_buffer_kwargs" not in vector


# -- the off-policy trainer -------------------------------------------------


def test_train_dqn_reaches_terminal_and_snapshots() -> None:
    """A short DQN run finishes, hands up a checkpoint snapshot (algo="dqn"), and publishes a decoupled
    preview policy whose action is a plain int in CartPole's action set (the ADR-021 discrete arm)."""
    import gymnasium as gym

    snapshots: list[CheckpointArtifact] = []
    captured: dict = {}
    terminal = train_dqn(
        _tiny_dqn_config(),
        "CartPole-v1",
        TrainControl(),
        lambda _m: None,
        lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=lambda art: snapshots.append(art),
    )
    assert terminal == "finished"
    assert snapshots, "no checkpoint snapshot captured"
    assert snapshots[-1].algo == "dqn" and len(snapshots[-1].blob) > 0

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert isinstance(action, int) and action in (0, 1)  # a discrete action, not a float vector
    env.step(action)  # the env accepts the discrete action without error
    env.close()


def test_train_dqn_stop_aborts() -> None:
    """A stop requested before learning ends the run promptly (off-policy: the per-step callback sees it)."""
    control = TrainControl()
    control.request_stop()
    terminal = train_dqn(
        _tiny_dqn_config(total=4000), "CartPole-v1", control, lambda _m: None, lambda _p: None
    )
    assert terminal == "stopped"


def test_dqn_resume_does_not_emit_every_step() -> None:
    """Resuming must NOT fire the metrics/snapshot callback on every step (the reported "load a DQN save
    and it crawls at ~12/s for a minute, then jumps" bug). The callback seeds its emit threshold relative
    to the model's CURRENT step count: a fixed 2000 would already be exceeded by a resumed counter, so
    _emit (a full model.save snapshot) fired on every step until the threshold caught up. Force a tiny
    interval so the catch-up would be obvious, then resume <1 interval of new steps and count snapshots."""
    import app.services.trainer_dqn as trainer_dqn

    orig_interval = trainer_dqn._METRICS_INTERVAL_STEPS
    trainer_dqn._METRICS_INTERVAL_STEPS = 100  # small so the bug's per-step catch-up is unmistakable
    try:
        snaps1: list[CheckpointArtifact] = []
        train_dqn(
            _tiny_dqn_config(total=1000), "CartPole-v1", TrainControl(),
            lambda _m: None, lambda _p: None, on_snapshot=snaps1.append,
        )
        start = snaps1[-1].timesteps  # ~1000, far above the 100-step interval
        snaps2: list[CheckpointArtifact] = []
        train_dqn(
            _tiny_dqn_config(total=start + 350), "CartPole-v1", TrainControl(),
            lambda _m: None, lambda _p: None, on_snapshot=snaps2.append, resume_blob=snaps1[-1].blob,
        )
    finally:
        trainer_dqn._METRICS_INTERVAL_STEPS = orig_interval
    # The FIRST snapshot on resume must land at least one interval *past* the resume point. With the bug
    # the fixed threshold (100) was already far below the resumed counter (~1000), so _emit fired on the
    # very first step → a snapshot stamped at ≈start. With the fix the threshold is seeded to start+100,
    # so the earliest snapshot is ≥ start+100.
    assert snaps2, "no snapshot captured on resume"
    first = min(s.timesteps for s in snaps2)
    assert first >= start + 90, (
        f"first resumed snapshot at {first} (resumed from {start}) — _emit fired immediately (regression)"
    )


def test_dqn_checkpoint_loads_for_ai_play() -> None:
    """A saved DQN model.zip turns into a deterministic AI-play predict fn (a plain int action)."""
    snapshots: list[CheckpointArtifact] = []
    train_dqn(
        _tiny_dqn_config(), "CartPole-v1", TrainControl(),
        lambda _m: None, lambda _p: None, on_snapshot=lambda art: snapshots.append(art),
    )
    art = snapshots[-1]
    loaded = LoadedCheckpoint(
        meta=CheckpointMeta(
            id="t", label="t", env_id="cartpole", algo="dqn", seed=1, created_at="now",
            reward=art.reward, timesteps=art.timesteps, total_timesteps=art.total_timesteps,
            iteration=art.iteration, generation=None, total_generations=None,
            artifact=art.artifact_name,
        ),
        config=_tiny_dqn_config(),
        blob=art.blob,
    )
    predict = predict_from_checkpoint(loaded)
    action = predict(np.zeros(4, dtype=np.float32))  # CartPole obs is 4-D
    assert isinstance(action, int) and action in (0, 1)
