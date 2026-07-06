"""S5e — Quantile-Regression DQN (the 9th algorithm): registry gating, the distributional off-policy
trainer reaching a terminal state on a discrete-action env, the decoupled int preview policy, the
n_quantiles knob, stop, and AI-play loading.

QR-DQN is the distributional DQN — a peer trainer behind the same manager (ADR-004/028) reusing the whole
off-policy lane (metrics + progress frames, discrete predict, skill meter, empty-buffer resume gate),
gated to the same discrete-action envs as DQN. These tests run a *tiny* real QR-DQN on CartPole (low
learning_starts + small buffer/batch + few quantiles) so they exercise the genuine SB3-contrib path while
staying fast.
"""

import numpy as np
import pytest
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import QRDQNHyperparams, TrainConfig
from app.services.checkpoints import CheckpointArtifact, LoadedCheckpoint
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_qrdqn import _policy_kwargs, _qrdqn_kwargs, train_qrdqn
from fastapi.testclient import TestClient

client = TestClient(app)

# The discrete-action envs QR-DQN is offered on (supported_algos) — the SAME set as DQN (S5e).
_QRDQN_ENVS = ["cartpole", "mountaincar", "acrobot", "lunarlander", "pong"]


def _tiny_qrdqn_config(total: int = 1500, seed: int = 1) -> TrainConfig:
    """A short CartPole QR-DQN run that does a few hundred real gradient updates in ~seconds on CPU."""
    return TrainConfig(
        env_id="cartpole",
        algo="qrdqn",
        seed=seed,
        total_timesteps=total,
        qrdqn=QRDQNHyperparams(
            n_quantiles=10, buffer_size=2000, batch_size=64, learning_starts=200, train_freq=4,
            target_update_interval=10,
        ),
    )


# -- registry gating --------------------------------------------------------


@pytest.mark.parametrize("env_id", _QRDQN_ENVS)
def test_qrdqn_offered_on_discrete_envs(env_id: str) -> None:
    spec = get_env(env_id)
    assert spec is not None, f"{env_id} not registered"
    assert spec.action_space == "discrete"  # QR-DQN is value-based, discrete-action only
    assert "qrdqn" in spec.supported_algos
    # The qrdqn hyperparam block must be exposed for every env that lists it (the sidebar reads it).
    assert "qrdqn" in spec.hyperparams
    block = spec.hyperparams["qrdqn"]
    assert {
        "learning_rate", "gamma", "n_quantiles", "buffer_size", "train_freq",
        "target_update_interval", "exploration_fraction", "exploration_final_eps",
    } <= set(block)
    # QR-DQN explores via ε-greedy, not entropy (SAC) or injected action noise (TD3): neither must appear.
    assert "ent_coef" not in block
    assert "train_noise" not in block


def test_qrdqn_is_offered_wherever_dqn_is() -> None:
    """QR-DQN mirrors DQN's gating exactly — it is offered on the SAME discrete envs (its whole point is
    the DQN-vs-QR-DQN comparison on the same game), and never where DQN isn't."""
    for spec in list_envs():
        assert ("qrdqn" in spec.supported_algos) == ("dqn" in spec.supported_algos), spec.id


def test_qrdqn_includes_the_atari_birthplace() -> None:
    """QR-DQN (distributional RL's historical home) must be offered on Atari — an image-obs discrete env."""
    pong = get_env("pong")
    assert pong is not None
    assert pong.obs_type == "image" and "qrdqn" in pong.supported_algos


def test_qrdqn_not_offered_on_continuous_board_or_ma_envs() -> None:
    """QR-DQN must stay off every continuous / board / multi-agent env (the complement of SAC/TD3's gate)."""
    for env_id in ("pendulum", "mountaincarcontinuous", "bipedalwalker", "tictactoe", "chess"):
        spec = get_env(env_id)
        assert spec is not None and "qrdqn" not in spec.supported_algos, env_id


def test_only_discrete_envs_list_qrdqn() -> None:
    """Defensive: any env that lists QR-DQN must be a discrete action space (no continuous leakage)."""
    for spec in list_envs():
        if "qrdqn" in spec.supported_algos:
            assert spec.action_space == "discrete", f"{spec.id} lists qrdqn but isn't discrete"


def test_qrdqn_cartpole_recipe_uses_ten_quantiles() -> None:
    """The per-env ★ n_quantiles is set from _QRDQN_TUNED (CartPole = the rl-zoo3 recipe's 10), while the
    generic classic-control block default is 25 — so the block is genuinely per-env tuned."""
    cartpole = get_env("cartpole")
    assert cartpole is not None
    nq = cartpole.hyperparams["qrdqn"]["n_quantiles"]
    assert nq.recommended == 10 and nq.default == 10
    acrobot = get_env("acrobot")
    assert acrobot is not None
    assert acrobot.hyperparams["qrdqn"]["n_quantiles"].recommended == 25  # block default (untuned)


def test_qrdqn_rejected_on_unsupported_env() -> None:
    """The manager rejects QR-DQN on an env that doesn't list it (Pendulum is continuous) with a 400."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "pendulum", "algo": "qrdqn", "seed": 1, "total_timesteps": 1000,
            "qrdqn": {"learning_rate": 1e-3, "buffer_size": 1000, "learning_starts": 100},
        },
    )
    assert resp.status_code == 400
    assert "does not support" in resp.json()["detail"]


# -- n_quantiles is a policy kwarg + Atari memory ---------------------------


def test_qrdqn_n_quantiles_is_a_policy_kwarg_not_a_model_kwarg() -> None:
    """n_quantiles is QR-DQN's one knob beyond DQN and lives in policy_kwargs (SB3-contrib), NOT the model
    kwargs — the vector path also carries net_arch there; the Atari path carries only n_quantiles."""
    cfg = _tiny_qrdqn_config()
    assert "n_quantiles" not in _qrdqn_kwargs(cfg, is_image=False)  # not a model kwarg
    vec_pk = _policy_kwargs(cfg, is_image=False)
    assert vec_pk["n_quantiles"] == 10 and "net_arch" in vec_pk
    img_pk = _policy_kwargs(cfg, is_image=True)
    assert img_pk["n_quantiles"] == 10 and "net_arch" not in img_pk  # CnnPolicy keeps the NatureCNN default


def test_qrdqn_image_path_optimizes_replay_buffer_memory() -> None:
    """Atari (image) QR-DQN must enable optimize_memory_usage so the 84×84×4 replay buffer stores each
    frame once (~4× less RAM) — the same handling as DQN (ADR-069). Vector envs must NOT use it (they keep
    handle_timeout_termination so a CartPole TimeLimit truncation isn't treated as terminal)."""
    cfg = _tiny_qrdqn_config()
    image = _qrdqn_kwargs(cfg, is_image=True)
    assert image["optimize_memory_usage"] is True
    assert image["replay_buffer_kwargs"] == {"handle_timeout_termination": False}
    assert image["gradient_steps"] == 1  # Nature recipe: one update per train_freq collected steps
    assert image["batch_size"] == 32

    vector = _qrdqn_kwargs(cfg, is_image=False)
    assert "optimize_memory_usage" not in vector  # tiny vector buffer; keep default timeout handling
    assert "replay_buffer_kwargs" not in vector


# -- the distributional off-policy trainer ----------------------------------


def test_train_qrdqn_reaches_terminal_and_snapshots() -> None:
    """A short QR-DQN run finishes, hands up a checkpoint snapshot (algo="qrdqn"), and publishes a
    decoupled preview policy whose action is a plain int in CartPole's action set (the ADR-021 arm)."""
    import gymnasium as gym

    snapshots: list[CheckpointArtifact] = []
    captured: dict = {}
    terminal = train_qrdqn(
        _tiny_qrdqn_config(),
        "CartPole-v1",
        TrainControl(),
        lambda _m: None,
        lambda _p: None,
        on_policy=lambda fn: captured.update(fn=fn),
        on_snapshot=lambda art: snapshots.append(art),
    )
    assert terminal == "finished"
    assert snapshots, "no checkpoint snapshot captured"
    assert snapshots[-1].algo == "qrdqn" and len(snapshots[-1].blob) > 0

    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert isinstance(action, int) and action in (0, 1)  # a discrete action, not a float vector
    env.step(action)  # the env accepts the discrete action without error
    env.close()


def test_train_qrdqn_stop_aborts() -> None:
    """A stop requested before learning ends the run promptly (off-policy: the per-step callback sees it)."""
    control = TrainControl()
    control.request_stop()
    terminal = train_qrdqn(
        _tiny_qrdqn_config(total=4000), "CartPole-v1", control, lambda _m: None, lambda _p: None
    )
    assert terminal == "stopped"


def test_qrdqn_resume_does_not_emit_every_step() -> None:
    """Resuming must NOT fire the metrics/snapshot callback on every step (the off-policy resume bug,
    ADR-069, shared with DQN/SAC/TD3). The callback seeds its emit threshold relative to the model's
    CURRENT step count, so a resumed counter doesn't trip a fixed threshold on every step."""
    import app.services.trainer_qrdqn as trainer_qrdqn

    orig_interval = trainer_qrdqn._METRICS_INTERVAL_STEPS
    trainer_qrdqn._METRICS_INTERVAL_STEPS = 100  # small so the bug's per-step catch-up is unmistakable
    try:
        snaps1: list[CheckpointArtifact] = []
        train_qrdqn(
            _tiny_qrdqn_config(total=1000), "CartPole-v1", TrainControl(),
            lambda _m: None, lambda _p: None, on_snapshot=snaps1.append,
        )
        start = snaps1[-1].timesteps  # ~1000, far above the 100-step interval
        snaps2: list[CheckpointArtifact] = []
        train_qrdqn(
            _tiny_qrdqn_config(total=start + 350), "CartPole-v1", TrainControl(),
            lambda _m: None, lambda _p: None, on_snapshot=snaps2.append, resume_blob=snaps1[-1].blob,
        )
    finally:
        trainer_qrdqn._METRICS_INTERVAL_STEPS = orig_interval
    assert snaps2, "no snapshot captured on resume"
    first = min(s.timesteps for s in snaps2)
    assert first >= start + 90, (
        f"first resumed snapshot at {first} (resumed from {start}) — _emit fired immediately (regression)"
    )


def test_qrdqn_checkpoint_loads_for_ai_play() -> None:
    """A saved QR-DQN model.zip turns into a deterministic AI-play predict fn (a plain int action)."""
    snapshots: list[CheckpointArtifact] = []
    train_qrdqn(
        _tiny_qrdqn_config(), "CartPole-v1", TrainControl(),
        lambda _m: None, lambda _p: None, on_snapshot=lambda art: snapshots.append(art),
    )
    art = snapshots[-1]
    loaded = LoadedCheckpoint(
        meta=CheckpointMeta(
            id="t", label="t", env_id="cartpole", algo="qrdqn", seed=1, created_at="now",
            reward=art.reward, timesteps=art.timesteps, total_timesteps=art.total_timesteps,
            iteration=art.iteration, generation=None, total_generations=None,
            artifact=art.artifact_name,
        ),
        config=_tiny_qrdqn_config(),
        blob=art.blob,
    )
    predict = predict_from_checkpoint(loaded)
    action = predict(np.zeros(4, dtype=np.float32))  # CartPole obs is 4-D
    assert isinstance(action, int) and action in (0, 1)
