"""X3 — Seed-sweep launcher: queue one config across N seeds, sharing one experiment_id.

The orchestration tests stub the PPO trainer with a fast in-process fake (no torch / SB3) so the
manager's queue mechanics — sequential drain, experiment tagging, cancel-drops-the-rest — are
exercised deterministically in milliseconds instead of via real training.
"""

import time
from pathlib import Path

import pytest
from app.main import app
from app.schemas.training import SweepRequest, TrainConfig, TrainingMetrics
from app.services.runs import RunStore
from app.services.training_manager import InvalidConfigError, training_manager
from fastapi.testclient import TestClient

# -- seed resolution (pure) -------------------------------------------------


def test_resolve_seeds_from_count() -> None:
    req = SweepRequest(config=TrainConfig(env_id="cartpole", seed=42), seed_count=3)
    assert training_manager._resolve_seeds(req) == [42, 43, 44]


def test_resolve_seeds_explicit_list_is_deduped() -> None:
    req = SweepRequest(config=TrainConfig(env_id="cartpole"), seeds=[7, 9, 7, 11])
    assert training_manager._resolve_seeds(req) == [7, 9, 11]  # order preserved, dupes dropped


def test_resolve_seeds_explicit_wins_over_count() -> None:
    req = SweepRequest(config=TrainConfig(env_id="cartpole", seed=42), seeds=[1, 2], seed_count=9)
    assert training_manager._resolve_seeds(req) == [1, 2]


def test_resolve_seeds_rejects_empty_and_oversized() -> None:
    with pytest.raises(InvalidConfigError):
        training_manager._resolve_seeds(SweepRequest(config=TrainConfig()))  # neither given
    with pytest.raises(InvalidConfigError):
        training_manager._resolve_seeds(SweepRequest(config=TrainConfig(), seed_count=999))


# -- experiment_id persistence ----------------------------------------------

_PPO_FRAMES = [
    {"type": "metrics", "iteration": 1, "timesteps": 4096, "ep_rew_mean": 500.0, "loss": 0.2},
]


def test_experiment_id_round_trips_through_run_store(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    cfg = TrainConfig(
        env_id="cartpole", algo="ppo", seed=7,
        experiment_id="exp-test-001", experiment_label="my sweep",
    )
    meta = store.save(
        cfg, _PPO_FRAMES, state="finished",
        started_at="2026-07-03T10:00:00+00:00", solved_score=500.0,
    )
    assert meta.experiment_id == "exp-test-001" and meta.experiment_label == "my sweep"
    detail = store.get(meta.id)
    assert detail is not None
    assert detail.meta.experiment_id == "exp-test-001"
    assert detail.config.experiment_id == "exp-test-001"  # config.json carries it too (reproducibility)


def test_single_run_has_no_experiment_id(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    meta = store.save(
        TrainConfig(env_id="cartpole", algo="ppo"), _PPO_FRAMES,
        state="finished", started_at="2026-07-03T10:00:00+00:00", solved_score=500.0,
    )
    assert meta.experiment_id is None and meta.experiment_label is None


# -- queue orchestration (fast fake trainer) --------------------------------


def _fast_ppo(config, gym_id, control, emit_metrics, emit_progress, publish, snapshot, resume):
    """A drop-in for train_ppo: emit one archivable frame (reward ≥ 10% of solved) and finish."""
    emit_metrics(
        TrainingMetrics(
            iteration=1, timesteps=config.total_timesteps, total_timesteps=config.total_timesteps,
            ep_rew_mean=500.0, ep_len_mean=200.0, loss=0.1, learning_rate=3e-4, elapsed=0.1,
        )
    )
    return "finished"


def _blocking_ppo(config, gym_id, control, emit_metrics, emit_progress, publish, snapshot, resume):
    """Like _fast_ppo but blocks until a stop is requested, so a cancel can land mid-run."""
    emit_metrics(
        TrainingMetrics(
            iteration=1, timesteps=config.total_timesteps, total_timesteps=config.total_timesteps,
            ep_rew_mean=500.0, ep_len_mean=200.0, loss=0.1, learning_rate=3e-4, elapsed=0.1,
        )
    )
    for _ in range(500):  # ≤5 s guard
        if control.stop_requested:
            return "stopped"
        time.sleep(0.01)
    return "finished"


def _wait_until(pred, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met within timeout")


def test_sweep_queues_and_runs_all_seeds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.services.trainer_ppo.train_ppo", _fast_ppo)
    monkeypatch.setattr(training_manager, "_runs", RunStore(tmp_path / "runs"))
    with TestClient(app) as c:
        resp = c.post(
            "/api/train/sweep",
            json={
                "config": {"env_id": "cartpole", "algo": "ppo", "seed": 100, "total_timesteps": 4096},
                "seed_count": 3,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # The response is the first seed's status, already tagged with the sweep.
        assert body["sweep"]["total"] == 3
        assert body["sweep"]["index"] == 1
        assert body["sweep"]["running_seed"] == 100
        experiment_id = body["sweep"]["experiment_id"]

        # Drain: all three seeds run back-to-back, then the sweep clears.
        _wait_until(lambda: training_manager.status().sweep is None and training_manager.status().state == "finished")

        runs = training_manager._runs.list()
        mine = [r for r in runs if r.experiment_id == experiment_id]
        assert len(mine) == 3, "all three queued seeds should have been archived"
        assert sorted(r.seed for r in mine) == [100, 101, 102]
        assert all(r.experiment_label is None for r in mine)


def test_cancel_sweep_drops_the_queue(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("app.services.trainer_ppo.train_ppo", _blocking_ppo)
    monkeypatch.setattr(training_manager, "_runs", RunStore(tmp_path / "runs"))
    with TestClient(app) as c:
        resp = c.post(
            "/api/train/sweep",
            json={
                "config": {"env_id": "cartpole", "algo": "ppo", "seed": 200, "total_timesteps": 4096},
                "seed_count": 3,
            },
        )
        assert resp.status_code == 200
        experiment_id = resp.json()["sweep"]["experiment_id"]
        # Seed 1 is now blocked inside the fake; cancel the whole sweep.
        _wait_until(lambda: training_manager.status().state == "running")
        c.post("/api/train/stop")

        _wait_until(lambda: training_manager.status().sweep is None)
        runs = training_manager._runs.list()
        mine = [r for r in runs if r.experiment_id == experiment_id]
        assert len(mine) == 1, "only the running seed should archive; the rest are dropped"
        assert mine[0].seed == 200


def test_sweep_rejects_unknown_env() -> None:
    with TestClient(app) as c:
        resp = c.post(
            "/api/train/sweep",
            json={"config": {"env_id": "does-not-exist", "algo": "ppo"}, "seed_count": 2},
        )
        assert resp.status_code == 400
