"""B2 — PPO training service: trainer reproducibility/stop, manager lifecycle, WS stream."""

import pytest
from app.main import app
from app.schemas.training import PPOHyperparams, TrainConfig
from app.services.connection_manager import manager
from app.services.train_control import TrainControl
from app.services.trainer_ppo import train_ppo
from app.services.training_manager import (
    AlreadyRunningError,
    InvalidConfigError,
    TrainingManager,
    training_manager,
)
from fastapi.testclient import TestClient

client = TestClient(app)


def _tiny_config(total: int = 512, n_steps: int = 64, seed: int = 42) -> TrainConfig:
    """A few-rollout CartPole run that finishes in well under a second on CPU."""
    return TrainConfig(
        env_id="cartpole",
        algo="ppo",
        seed=seed,
        total_timesteps=total,
        hyperparams=PPOHyperparams(n_steps=n_steps, batch_size=64),
    )


# -- trainer ----------------------------------------------------------------


def test_train_ppo_reproducible_with_same_seed() -> None:
    def run() -> list[tuple[int, float | None, float | None]]:
        seen: list[tuple[int, float | None, float | None]] = []
        train_ppo(
            _tiny_config(),
            "CartPole-v1",
            TrainControl(),
            lambda m: seen.append((m.timesteps, m.ep_rew_mean, m.ep_len_mean)),
            lambda _p: None,  # progress sink unused (tiny run finishes under the 1s tick)
        )
        return seen

    first, second = run(), run()
    assert len(first) == 8  # 512 timesteps / 64 n_steps
    assert first == second


def test_train_ppo_continuous_box_action() -> None:
    """G1b: PPO trains on a continuous (box) env and the decoupled numpy preview predict returns a
    clipped action *vector* the env can step — not an int (the int→box seam in _build_numpy_predict)."""
    import gymnasium as gym
    import numpy as np

    captured: dict = {}
    train_ppo(
        TrainConfig(
            env_id="pendulum", algo="ppo", seed=1, total_timesteps=256,
            hyperparams=PPOHyperparams(n_steps=64, batch_size=64),
        ),
        "Pendulum-v1", TrainControl(),
        lambda _m: None, lambda _p: None, on_policy=lambda fn: captured.update(fn=fn),
    )
    env = gym.make("Pendulum-v1")
    obs, _ = env.reset(seed=0)
    action = captured["fn"](obs)
    assert np.shape(action) == (1,)  # a 1-D action vector, not a scalar/int
    assert -2.0 <= float(action[0]) <= 2.0  # clipped into Pendulum's [-2, 2] torque bounds
    env.step(action)  # the env accepts the continuous action without error
    env.close()


def test_train_ppo_stop_aborts_early() -> None:
    control = TrainControl()
    seen: list[object] = []

    def sink(metrics: object) -> None:
        seen.append(metrics)
        control.request_stop()  # stop right after the first rollout

    terminal = train_ppo(_tiny_config(total=4096), "CartPole-v1", control, sink, lambda _p: None)
    assert terminal == "stopped"
    assert len(seen) == 1


def test_interruptible_ppo_matches_stock_ppo() -> None:
    """The epoch-sliced _InterruptiblePPO must train *bit-identically* to stock SB3 PPO at the same
    seed — the between-epochs stop hook must not perturb the trajectory (ADR-038 follow-up: the fix
    for the multi-agent 'Stopping' hang slices the hookless update into single epochs)."""
    import numpy as np
    from app.envs.factory import make_env
    from app.services.trainer_ppo import _InterruptiblePPO
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    def run(cls: type[PPO]) -> list[np.ndarray]:
        set_random_seed(0)
        model = cls(
            "MlpPolicy", make_env("cartpole", "CartPole-v1"),
            seed=0, n_steps=128, batch_size=64, device="cpu", verbose=0,
        )
        model.learn(total_timesteps=128 * 2)
        return [p.detach().cpu().numpy().copy() for p in model.policy.parameters()]

    stock, interruptible = run(PPO), run(_InterruptiblePPO)
    assert all(np.array_equal(a, b) for a, b in zip(stock, interruptible, strict=True))


def test_interruptible_ppo_bails_between_epochs() -> None:
    """A Stop requested during PPO's (callback-less) update phase ends it after the current epoch,
    not the whole n_epochs pass — keeping Stop responsive on a heavy multi-agent update."""
    from app.envs.factory import make_env
    from app.services.trainer_ppo import _InterruptiblePPO

    model = _InterruptiblePPO(
        "MlpPolicy", make_env("cartpole", "CartPole-v1"),
        seed=0, n_steps=128, batch_size=64, n_epochs=10, device="cpu", verbose=0,
    )
    model.stop_check = lambda: True  # a Stop is already pending when the update begins
    model.learn(total_timesteps=128)  # one rollout → one (interrupted) update
    assert model._n_updates == 1  # bailed after epoch 1 (would be 10 without the hook)
    assert model.n_epochs == 10  # n_epochs restored, so a resumed run still does full updates


# -- manager ----------------------------------------------------------------


def test_manager_rejects_unknown_env() -> None:
    mgr = TrainingManager(manager)
    with pytest.raises(InvalidConfigError):
        mgr.start(TrainConfig(env_id="does-not-exist"))


def test_manager_single_active_run_and_clean_stop() -> None:
    mgr = TrainingManager(manager)  # no loop bound → broadcasts are skipped
    status = mgr.start(_tiny_config(total=200_000, n_steps=256))
    try:
        assert status.state == "running"
        with pytest.raises(AlreadyRunningError):
            mgr.start(_tiny_config())
    finally:
        mgr.stop()
        mgr.join(timeout=30)
    assert mgr.status().state == "stopped"


# -- REST + WS --------------------------------------------------------------


def test_status_endpoint_shape() -> None:
    body = client.get("/api/train/status").json()
    for key in (
        "type", "state", "env_id", "algo", "seed",
        "timesteps", "total_timesteps", "config", "last_metrics", "last_evolution", "error",
    ):
        assert key in body
    assert body["type"] == "status"


def test_start_streams_metrics_over_ws() -> None:
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        resp = c.post(
            "/api/train/start",
            json={
                "env_id": "cartpole", "algo": "ppo", "seed": 7,
                "total_timesteps": 256,
                "hyperparams": {"n_steps": 64, "batch_size": 64},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["state"] == "running"

        metrics = None
        for _ in range(50):
            frame = ws.receive_json()
            if frame.get("type") == "metrics":
                metrics = frame
                break

        assert metrics is not None, "no metrics frame arrived over WS"
        assert metrics["total_timesteps"] == 256
        assert metrics["iteration"] >= 1
        assert "ep_rew_mean" in metrics

        c.post("/api/train/stop")
        training_manager.join(timeout=30)
