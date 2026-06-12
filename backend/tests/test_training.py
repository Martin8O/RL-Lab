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


def test_train_ppo_stop_aborts_early() -> None:
    control = TrainControl()
    seen: list[object] = []

    def sink(metrics: object) -> None:
        seen.append(metrics)
        control.request_stop()  # stop right after the first rollout

    terminal = train_ppo(_tiny_config(total=4096), "CartPole-v1", control, sink, lambda _p: None)
    assert terminal == "stopped"
    assert len(seen) == 1


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
