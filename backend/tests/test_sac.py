"""S5a — Soft Actor-Critic (the 5th algorithm): registry gating, the off-policy trainer reaching a
terminal state on a continuous-Box env, the decoupled box preview policy, stop, and AI-play loading.

SAC is a peer trainer behind the same manager (ADR-004/028) and reuses the whole PPO lane (metrics +
progress frames, box predict, skill meter) — only off-policy and gated to continuous-action envs. These
tests run a *tiny* real SAC on Pendulum (low learning_starts + small buffer/batch) so they exercise the
genuine SB3 path while staying fast.
"""

import numpy as np
import pytest
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import SACHyperparams, TrainConfig
from app.services.checkpoints import CheckpointArtifact, LoadedCheckpoint
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_sac import train_sac
from fastapi.testclient import TestClient

client = TestClient(app)

# The continuous-Box envs SAC is offered on (supported_algos), and the kinds it must stay OFF.
_SAC_ENVS = [
    "pendulum", "mountaincarcontinuous", "bipedalwalker", "bipedalwalkerhardcore",
    "hopper", "walker2d", "halfcheetah", "ant", "reacher", "swimmer", "humanoid",
]


def _tiny_sac_config(total: int = 1500, seed: int = 1) -> TrainConfig:
    """A short Pendulum SAC run that does a few hundred real gradient updates in ~seconds on CPU/GPU."""
    return TrainConfig(
        env_id="pendulum",
        algo="sac",
        seed=seed,
        total_timesteps=total,
        sac=SACHyperparams(buffer_size=2000, batch_size=64, learning_starts=200, train_freq=1),
    )


# -- registry gating --------------------------------------------------------


@pytest.mark.parametrize("env_id", _SAC_ENVS)
def test_sac_offered_on_continuous_box_envs(env_id: str) -> None:
    spec = get_env(env_id)
    assert spec is not None, f"{env_id} not registered"
    assert spec.action_space == "box"  # SAC is continuous-action only
    assert "sac" in spec.supported_algos
    # The sac hyperparam block must be exposed for every env that lists it (the sidebar reads it).
    assert "sac" in spec.hyperparams
    block = spec.hyperparams["sac"]
    assert {"learning_rate", "gamma", "tau", "buffer_size", "train_freq", "ent_coef"} <= set(block)
    assert block["ent_coef"].type == "categorical" and "auto" in (block["ent_coef"].choices or [])


def test_sac_not_offered_on_discrete_board_or_ma_envs() -> None:
    """SAC must stay off every non-continuous env: discrete control, Toy Text, board, multi-agent."""
    for env_id in ("cartpole", "mountaincar", "acrobot", "frozenlake", "tictactoe", "chess"):
        spec = get_env(env_id)
        assert spec is not None and "sac" not in spec.supported_algos, env_id
    # And never on an image env (CarRacing is continuous-Box but image-obs → SAC excluded initially).
    assert "sac" not in get_env("carracing").supported_algos


def test_only_box_envs_list_sac() -> None:
    """Defensive: any env that lists SAC must be a continuous-Box action space (no discrete leakage)."""
    for spec in list_envs():
        if "sac" in spec.supported_algos:
            assert spec.action_space == "box", f"{spec.id} lists sac but isn't box"


def test_sac_rejected_on_unsupported_env() -> None:
    """The manager rejects SAC on an env that doesn't list it (CartPole is discrete) with a clear 400."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "cartpole", "algo": "sac", "seed": 1, "total_timesteps": 1000,
            "sac": {"learning_rate": 3e-4, "buffer_size": 1000, "learning_starts": 100},
        },
    )
    assert resp.status_code == 400
    assert "does not support" in resp.json()["detail"]


# -- the off-policy trainer -------------------------------------------------


def test_train_sac_reaches_terminal_and_snapshots() -> None:
    """A short SAC run finishes, hands up a checkpoint snapshot (algo="sac"), and publishes a decoupled
    box preview policy whose action is a clipped float *vector* the env can step (the ADR-021 box arm)."""
    import gymnasium as gym

    snapshots: list[CheckpointArtifact] = []
    captured: dict = {}
    terminal = train_sac(
        _tiny_sac_config(),
        "Pendulum-v1",
        TrainControl(),
        lambda _m: None,
        lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=lambda art: snapshots.append(art),
    )
    assert terminal == "finished"
    assert snapshots, "no checkpoint snapshot captured"
    assert snapshots[-1].algo == "sac" and len(snapshots[-1].blob) > 0

    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert np.shape(action) == (1,)  # a 1-D action vector, not a scalar/int
    assert -2.0 <= float(action[0]) <= 2.0  # clipped into Pendulum's [-2, 2] torque bounds
    env.step(action)  # the env accepts the continuous action without error
    env.close()


def test_train_sac_stop_aborts() -> None:
    """A stop requested before learning ends the run promptly (off-policy: the per-step callback sees it)."""
    control = TrainControl()
    control.request_stop()
    terminal = train_sac(
        _tiny_sac_config(total=4000), "Pendulum-v1", control, lambda _m: None, lambda _p: None
    )
    assert terminal == "stopped"


def test_sac_checkpoint_loads_for_ai_play() -> None:
    """A saved SAC model.zip turns into a deterministic AI-play predict fn (box vector, no VecNormalize)."""
    snapshots: list[CheckpointArtifact] = []
    train_sac(
        _tiny_sac_config(), "Pendulum-v1", TrainControl(),
        lambda _m: None, lambda _p: None, on_snapshot=lambda art: snapshots.append(art),
    )
    art = snapshots[-1]
    loaded = LoadedCheckpoint(
        meta=CheckpointMeta(
            id="t", label="t", env_id="pendulum", algo="sac", seed=1, created_at="now",
            reward=art.reward, timesteps=art.timesteps, total_timesteps=art.total_timesteps,
            iteration=art.iteration, generation=None, total_generations=None,
            artifact=art.artifact_name,
        ),
        config=_tiny_sac_config(),
        blob=art.blob,
    )
    predict = predict_from_checkpoint(loaded)
    action = predict(np.zeros(3, dtype=np.float32))  # Pendulum obs is 3-D
    assert np.shape(action) == (1,) and -2.0 <= float(action[0]) <= 2.0
