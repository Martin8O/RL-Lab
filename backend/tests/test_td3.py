"""S5b — Twin Delayed DDPG (the 6th algorithm): registry gating, the off-policy trainer reaching a
terminal state on a continuous-Box env, the decoupled box preview policy, stop, and AI-play loading.

TD3 is SAC's off-policy sibling — a peer trainer behind the same manager (ADR-004/028) reusing the whole
PPO lane (metrics + progress frames, box predict, skill meter), gated to the same continuous-action envs.
These tests run a *tiny* real TD3 on Pendulum (low learning_starts + small buffer/batch) so they exercise
the genuine SB3 path while staying fast.
"""

import numpy as np
import pytest
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import TD3Hyperparams, TrainConfig
from app.services.checkpoints import CheckpointArtifact, LoadedCheckpoint
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_td3 import train_td3
from fastapi.testclient import TestClient

client = TestClient(app)

# The continuous-Box envs TD3 is offered on (supported_algos) — the same set as SAC.
_TD3_ENVS = [
    "pendulum", "mountaincarcontinuous", "bipedalwalker", "bipedalwalkerhardcore",
    "hopper", "walker2d", "halfcheetah", "ant", "reacher", "swimmer", "humanoid",
]


def _tiny_td3_config(total: int = 1500, seed: int = 1) -> TrainConfig:
    """A short Pendulum TD3 run that does a few hundred real gradient updates in ~seconds on CPU."""
    return TrainConfig(
        env_id="pendulum",
        algo="td3",
        seed=seed,
        total_timesteps=total,
        td3=TD3Hyperparams(buffer_size=2000, batch_size=64, learning_starts=200, train_freq=1),
    )


# -- registry gating --------------------------------------------------------


@pytest.mark.parametrize("env_id", _TD3_ENVS)
def test_td3_offered_on_continuous_box_envs(env_id: str) -> None:
    spec = get_env(env_id)
    assert spec is not None, f"{env_id} not registered"
    assert spec.action_space == "box"  # TD3 is continuous-action only
    assert "td3" in spec.supported_algos
    # The td3 hyperparam block must be exposed for every env that lists it (the sidebar reads it).
    assert "td3" in spec.hyperparams
    block = spec.hyperparams["td3"]
    assert {"learning_rate", "gamma", "tau", "buffer_size", "train_freq", "train_noise"} <= set(block)
    # TD3 has no entropy temperature (deterministic policy): the SAC-only categorical must NOT appear.
    assert "ent_coef" not in block


def test_td3_offered_on_exactly_the_sac_envs() -> None:
    """TD3 shares SAC's continuous-Box gate exactly — neither set should drift from the other."""
    td3_envs = {s.id for s in list_envs() if "td3" in s.supported_algos}
    sac_envs = {s.id for s in list_envs() if "sac" in s.supported_algos}
    assert td3_envs == sac_envs


def test_td3_not_offered_on_discrete_board_or_ma_envs() -> None:
    """TD3 must stay off every non-continuous env: discrete control, Toy Text, board, multi-agent."""
    for env_id in ("cartpole", "mountaincar", "acrobot", "frozenlake", "tictactoe", "chess"):
        spec = get_env(env_id)
        assert spec is not None and "td3" not in spec.supported_algos, env_id
    # And never on an image env (CarRacing is continuous-Box but image-obs → off-policy excluded initially).
    assert "td3" not in get_env("carracing").supported_algos


def test_only_box_envs_list_td3() -> None:
    """Defensive: any env that lists TD3 must be a continuous-Box action space (no discrete leakage)."""
    for spec in list_envs():
        if "td3" in spec.supported_algos:
            assert spec.action_space == "box", f"{spec.id} lists td3 but isn't box"


def test_td3_rejected_on_unsupported_env() -> None:
    """The manager rejects TD3 on an env that doesn't list it (CartPole is discrete) with a clear 400."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "cartpole", "algo": "td3", "seed": 1, "total_timesteps": 1000,
            "td3": {"learning_rate": 1e-3, "buffer_size": 1000, "learning_starts": 100},
        },
    )
    assert resp.status_code == 400
    assert "does not support" in resp.json()["detail"]


# -- the off-policy trainer -------------------------------------------------


def test_train_td3_reaches_terminal_and_snapshots() -> None:
    """A short TD3 run finishes, hands up a checkpoint snapshot (algo="td3"), and publishes a decoupled
    box preview policy whose action is a clipped float *vector* the env can step (the ADR-021 box arm)."""
    import gymnasium as gym

    snapshots: list[CheckpointArtifact] = []
    captured: dict = {}
    terminal = train_td3(
        _tiny_td3_config(),
        "Pendulum-v1",
        TrainControl(),
        lambda _m: None,
        lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=lambda art: snapshots.append(art),
    )
    assert terminal == "finished"
    assert snapshots, "no checkpoint snapshot captured"
    assert snapshots[-1].algo == "td3" and len(snapshots[-1].blob) > 0

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert np.shape(action) == (1,)  # a 1-D action vector, not a scalar/int
    assert -2.0 <= float(action[0]) <= 2.0  # clipped into Pendulum's [-2, 2] torque bounds
    env.step(action)  # the env accepts the continuous action without error
    env.close()


def test_train_td3_no_exploration_noise_ok() -> None:
    """train_noise=0 (no injected exploration noise) is a valid config: action_noise is None, run finishes."""
    cfg = TrainConfig(
        env_id="pendulum", algo="td3", seed=2, total_timesteps=1200,
        td3=TD3Hyperparams(buffer_size=2000, batch_size=64, learning_starts=200, train_noise=0.0),
    )
    terminal = train_td3(cfg, "Pendulum-v1", TrainControl(), lambda _m: None, lambda _p: None)
    assert terminal == "finished"


def test_train_td3_stop_aborts() -> None:
    """A stop requested before learning ends the run promptly (off-policy: the per-step callback sees it)."""
    control = TrainControl()
    control.request_stop()
    terminal = train_td3(
        _tiny_td3_config(total=4000), "Pendulum-v1", control, lambda _m: None, lambda _p: None
    )
    assert terminal == "stopped"


def test_td3_checkpoint_loads_for_ai_play() -> None:
    """A saved TD3 model.zip turns into a deterministic AI-play predict fn (box vector, no VecNormalize)."""
    snapshots: list[CheckpointArtifact] = []
    train_td3(
        _tiny_td3_config(), "Pendulum-v1", TrainControl(),
        lambda _m: None, lambda _p: None, on_snapshot=lambda art: snapshots.append(art),
    )
    art = snapshots[-1]
    loaded = LoadedCheckpoint(
        meta=CheckpointMeta(
            id="t", label="t", env_id="pendulum", algo="td3", seed=1, created_at="now",
            reward=art.reward, timesteps=art.timesteps, total_timesteps=art.total_timesteps,
            iteration=art.iteration, generation=None, total_generations=None,
            artifact=art.artifact_name,
        ),
        config=_tiny_td3_config(),
        blob=art.blob,
    )
    predict = predict_from_checkpoint(loaded)
    action = predict(np.zeros(3, dtype=np.float32))  # Pendulum obs is 3-D
    assert np.shape(action) == (1,) and -2.0 <= float(action[0]) <= 2.0
