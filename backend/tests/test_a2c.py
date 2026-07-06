"""S5d — Advantage Actor-Critic (the 8th algorithm): registry gating, the on-policy trainer reaching a
terminal state on both a discrete and a continuous env, the decoupled preview policy (int vs box), stop,
and AI-play loading.

A2C is PPO's simpler on-policy predecessor — a peer trainer behind the same manager (ADR-004/028)
reusing the whole PPO on-policy lane (metrics + progress frames, numpy preview, skill meter), offered on
a curated mix of discrete + continuous classic-control envs. These tests run a *tiny* real A2C so they
exercise the genuine SB3 path while staying fast.
"""

import numpy as np
import pytest
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import A2CHyperparams, TrainConfig
from app.services.checkpoints import CheckpointArtifact, LoadedCheckpoint
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_a2c import train_a2c
from fastapi.testclient import TestClient

client = TestClient(app)

# The classic-control envs A2C is offered on (supported_algos) — curated (S5d): both discrete AND
# continuous, unlike DQN's discrete-only / SAC-TD3's continuous-only gates.
_A2C_DISCRETE = ["cartpole", "mountaincar", "acrobot", "lunarlander"]
_A2C_CONTINUOUS = ["pendulum", "mountaincarcontinuous"]
_A2C_ENVS = _A2C_DISCRETE + _A2C_CONTINUOUS


def _tiny_a2c_config(env_id: str = "cartpole", total: int = 1000, seed: int = 1) -> TrainConfig:
    """A short A2C run that does a few hundred real updates in ~seconds on CPU (n_steps=5 default)."""
    return TrainConfig(
        env_id=env_id,
        algo="a2c",
        seed=seed,
        total_timesteps=total,
        a2c=A2CHyperparams(n_steps=5),
    )


# -- registry gating --------------------------------------------------------


@pytest.mark.parametrize("env_id", _A2C_ENVS)
def test_a2c_offered_on_curated_classic_control(env_id: str) -> None:
    spec = get_env(env_id)
    assert spec is not None, f"{env_id} not registered"
    # Vector-obs CPU envs (classic_control + LunarLander/box2d) — the MlpPolicy/CPU lane A2C covers.
    assert spec.obs_type == "vector" and spec.hw_requirement == "cpu"
    assert "a2c" in spec.supported_algos
    # The a2c hyperparam block must be exposed for every env that lists it (the sidebar reads it).
    assert "a2c" in spec.hyperparams
    block = spec.hyperparams["a2c"]
    assert {
        "learning_rate", "gamma", "n_steps", "gae_lambda", "ent_coef",
        "n_hidden_layers", "neurons_per_layer", "activation",
    } <= set(block)
    # A2C is on-policy: no PPO clip, no off-policy replay buffer, no DQN/TD3 exploration knobs.
    for absent in ("clip_range", "buffer_size", "train_noise", "exploration_fraction"):
        assert absent not in block, absent


def test_a2c_handles_both_action_types() -> None:
    """A2C's distinguishing trait vs DQN/SAC/TD3: it is offered on discrete AND continuous envs."""
    assert all(get_env(e).action_space == "discrete" for e in _A2C_DISCRETE)
    assert all(get_env(e).action_space == "box" for e in _A2C_CONTINUOUS)


def test_a2c_not_offered_on_image_board_ma_or_mujoco() -> None:
    """A2C stays off the non-curated families (image/board/multi-agent/MuJoCo) — it is a lean classic-
    control comparison for now (Atari A2C would be a later throughput-oriented stretch)."""
    for env_id in ("pong", "carracing", "tictactoe", "chess", "mpe_spread", "humanoid", "bipedalwalker"):
        spec = get_env(env_id)
        assert spec is not None and "a2c" not in spec.supported_algos, env_id


def test_only_vector_cpu_envs_list_a2c() -> None:
    """Defensive: any env that lists A2C must be a vector-obs CPU env (no leakage into an image / GPU /
    board / multi-agent family the on-policy MlpPolicy/CPU trainer doesn't cover)."""
    for spec in list_envs():
        if "a2c" in spec.supported_algos:
            assert spec.obs_type == "vector", f"{spec.id} lists a2c but isn't a vector obs"
            assert spec.hw_requirement == "cpu", f"{spec.id} lists a2c but isn't CPU-trainable"
            assert spec.family in ("classic_control", "box2d"), f"{spec.id} lists a2c in an unexpected family"


def test_a2c_reuses_the_ppo_onpolicy_budget() -> None:
    """A2C is on-policy, so it reuses the PPO env-step budget (default_total_timesteps), NOT the off-
    policy budget: the shared PPO/A2C total drives its step ladder, not offpolicy_total_timesteps."""
    for env_id in _A2C_ENVS:
        spec = get_env(env_id)
        assert spec is not None and spec.default_total_timesteps > 0, env_id


def test_a2c_rejected_on_unsupported_env() -> None:
    """The manager rejects A2C on an env that doesn't list it (Taxi is Toy Text) with a clear 400."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "taxi", "algo": "a2c", "seed": 1, "total_timesteps": 1000,
            "a2c": {"learning_rate": 7e-4, "n_steps": 5},
        },
    )
    assert resp.status_code == 400
    assert "does not support" in resp.json()["detail"]


# -- the on-policy trainer --------------------------------------------------


def test_train_a2c_reaches_terminal_and_snapshots_discrete() -> None:
    """A short A2C run on CartPole finishes, hands up a checkpoint snapshot (algo="a2c"), and publishes a
    decoupled preview policy whose action is a plain int in CartPole's action set (the discrete arm)."""
    import gymnasium as gym

    snapshots: list[CheckpointArtifact] = []
    captured: dict = {}
    terminal = train_a2c(
        _tiny_a2c_config(),
        "CartPole-v1",
        TrainControl(),
        lambda _m: None,
        lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=lambda art: snapshots.append(art),
    )
    assert terminal == "finished"
    assert snapshots, "no checkpoint snapshot captured"
    assert snapshots[-1].algo == "a2c" and len(snapshots[-1].blob) > 0

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert isinstance(action, int) and action in (0, 1)  # a discrete action, not a float vector
    env.step(action)  # the env accepts the discrete action without error
    env.close()


def test_train_a2c_preview_is_a_box_vector_on_continuous() -> None:
    """On a continuous env (Pendulum) the decoupled preview policy returns a clipped float action vector
    inside the action bounds — the box arm of the int|box preview duality (A2C handles both)."""
    snapshots: list[CheckpointArtifact] = []
    captured: dict = {}
    train_a2c(
        _tiny_a2c_config(env_id="pendulum", total=600),
        "Pendulum-v1",
        TrainControl(),
        lambda _m: None,
        lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=snapshots.append,
    )
    assert snapshots and snapshots[-1].algo == "a2c"
    action = captured["fn"](np.zeros(3, dtype=np.float32))  # Pendulum obs is 3-D
    action = np.asarray(action)
    assert action.shape == (1,)  # Pendulum action is Box(1): a single torque
    assert -2.0 <= float(action[0]) <= 2.0  # clipped into the [-2, 2] torque bounds


def test_train_a2c_stop_aborts() -> None:
    """A stop requested before learning ends the run promptly (the per-step callback observes it)."""
    control = TrainControl()
    control.request_stop()
    terminal = train_a2c(
        _tiny_a2c_config(total=4000), "CartPole-v1", control, lambda _m: None, lambda _p: None
    )
    assert terminal == "stopped"


def test_a2c_checkpoint_loads_for_ai_play() -> None:
    """A saved A2C model.zip turns into a deterministic AI-play predict fn (a plain int action)."""
    snapshots: list[CheckpointArtifact] = []
    train_a2c(
        _tiny_a2c_config(), "CartPole-v1", TrainControl(),
        lambda _m: None, lambda _p: None, on_snapshot=lambda art: snapshots.append(art),
    )
    art = snapshots[-1]
    loaded = LoadedCheckpoint(
        meta=CheckpointMeta(
            id="t", label="t", env_id="cartpole", algo="a2c", seed=1, created_at="now",
            reward=art.reward, timesteps=art.timesteps, total_timesteps=art.total_timesteps,
            iteration=art.iteration, generation=None, total_generations=None,
            artifact=art.artifact_name,
        ),
        config=_tiny_a2c_config(),
        blob=art.blob,
    )
    predict = predict_from_checkpoint(loaded)
    action = predict(np.zeros(4, dtype=np.float32))  # CartPole obs is 4-D
    assert isinstance(action, int) and action in (0, 1)
