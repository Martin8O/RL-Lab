"""D1 — Save / Load / Export checkpoints: store round-trip, snapshot+resume, REST surface."""

from pathlib import Path

from app.main import app
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import EvolutionHyperparams, EvolutionMetrics, PPOHyperparams, TrainConfig
from app.services.checkpoints import CheckpointArtifact, CheckpointStore
from app.services.train_control import TrainControl
from app.services.trainer_evolution import train_evolution
from app.services.trainer_ppo import train_ppo
from app.services.training_manager import training_manager
from fastapi.testclient import TestClient

client = TestClient(app)


def _store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "checkpoints")


def _artifact() -> CheckpointArtifact:
    return CheckpointArtifact(
        algo="ppo", blob=b"fake-model-bytes", artifact_name="model.zip",
        reward=123.4, timesteps=40_000, total_timesteps=50_000, iteration=12,
    )


# -- store round-trip -------------------------------------------------------


def test_save_list_load_export_delete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.list() == []

    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=7)
    meta = store.save(cfg, _artifact(), [{"type": "metrics", "iteration": 1}], label="run A")
    assert meta.label == "run A" and meta.env_id == "cartpole" and meta.artifact == "model.zip"
    assert meta.reward == 123.4 and meta.timesteps == 40_000

    listed = store.list()
    assert len(listed) == 1 and listed[0].id == meta.id

    loaded = store.load(meta.id)
    assert loaded is not None
    assert loaded.blob == b"fake-model-bytes"
    assert loaded.config.seed == 7 and loaded.config.env_id == "cartpole"

    exported = store.export_zip(meta.id)
    assert exported is not None
    data, filename = exported
    assert filename == f"{meta.id}.zip" and data[:2] == b"PK"  # zip magic

    assert store.delete(meta.id) is True
    assert store.list() == []
    assert store.load(meta.id) is None
    assert store.delete(meta.id) is False


def test_list_newest_first_and_traversal_guard(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo")
    a = store.save(cfg, _artifact(), [])
    b = store.save(cfg, _artifact(), [])
    # created_at carries microsecond precision, so the second save sorts first.
    assert [m.id for m in store.list()] == [b.id, a.id]
    # Path-traversal ids are rejected, never resolved against the filesystem.
    assert store.load("../secret") is None
    assert store.export_zip("..") is None
    assert store.delete("a/b") is False


# -- PPO snapshot + resume --------------------------------------------------


def _tiny_ppo(total: int = 256, seed: int = 1) -> TrainConfig:
    return TrainConfig(
        env_id="cartpole", algo="ppo", seed=seed, total_timesteps=total,
        hyperparams=PPOHyperparams(n_steps=64, batch_size=64),
    )


def test_ppo_snapshot_then_resume_continues_timesteps() -> None:
    snaps: list[CheckpointArtifact] = []
    train_ppo(
        _tiny_ppo(total=256), "CartPole-v1", TrainControl(),
        lambda _m: None, lambda _p: None, None, snaps.append,
    )
    assert snaps, "no snapshot captured"
    final = snaps[-1]
    assert final.algo == "ppo" and final.artifact_name == "model.zip"
    assert final.blob[:2] == b"PK" and final.timesteps >= 256

    # Resume from the snapshot toward a higher absolute target; the model must continue from
    # the saved step count rather than restart at 0. The first metrics frame after resuming lands
    # past the checkpoint's timesteps (≈ final.timesteps + n_steps), not back near zero.
    metrics: list = []
    train_ppo(
        _tiny_ppo(total=512), "CartPole-v1", TrainControl(),
        metrics.append, lambda _p: None, None,
        resume_blob=final.blob,
    )
    assert metrics and metrics[0].timesteps > final.timesteps


# -- evolution snapshot + resume --------------------------------------------


def _tiny_evo(generations: int = 2, seed: int = 3) -> TrainConfig:
    return TrainConfig(
        env_id="cartpole", algo="neuroevolution", seed=seed,
        evolution=EvolutionHyperparams(
            population_size=6, top_k_parents=3, mutation_rate=0.1,
            crossover_rate=0.5, generations=generations, episodes=1,
        ),
    )


def test_evolution_snapshot_then_resume_continues_generations() -> None:
    snaps: list[CheckpointArtifact] = []
    train_evolution(
        _tiny_evo(generations=2), "CartPole-v1", TrainControl(),
        lambda _m: None, lambda _p: None, snaps.append,
    )
    assert snaps and snaps[-1].generation == 2
    assert snaps[-1].artifact_name == "population.npz" and snaps[-1].total_generations == 2

    frames: list[EvolutionMetrics] = []
    train_evolution(
        _tiny_evo(generations=2), "CartPole-v1", TrainControl(),
        frames.append, lambda _p: None, None, snaps[-1].blob,
    )
    # Resumed run picks generation numbering up where the checkpoint left off.
    assert [f.generation for f in frames] == [3, 4]
    assert all(f.total_generations == 4 for f in frames)


# -- manager + REST ---------------------------------------------------------


def test_save_rejected_with_no_snapshot() -> None:
    # Fresh manager state (no run yet) → nothing to save.
    resp = client.post("/api/checkpoints")
    assert resp.status_code == 400


def test_rest_save_load_export_delete_round_trip() -> None:
    # Context-managed client runs the app lifespan, so the manager's WS broadcasts land on a
    # live loop instead of leaving un-awaited coroutines during the real training run.
    with TestClient(app) as c:
        # Train a tiny PPO run to completion so the manager holds a terminal snapshot.
        started = c.post(
            "/api/train/start",
            json={
                "env_id": "cartpole", "algo": "ppo", "seed": 5,
                "total_timesteps": 256,
                "hyperparams": {"n_steps": 64, "batch_size": 64},
            },
        )
        assert started.status_code == 200
        training_manager.join(timeout=30)

        saved = c.post("/api/checkpoints", json={"label": "tiny"})
        assert saved.status_code == 200
        meta = saved.json()
        CheckpointMeta.model_validate(meta)
        cid = meta["id"]
        assert meta["label"] == "tiny" and meta["algo"] == "ppo"

        assert any(m["id"] == cid for m in c.get("/api/checkpoints").json())

        export = c.get(f"/api/checkpoints/{cid}/export")
        assert export.status_code == 200
        assert export.headers["content-type"] == "application/zip"
        assert export.content[:2] == b"PK"

        # Load resumes training; stop it cleanly so the suite isn't left with a live run.
        loaded = c.post(f"/api/checkpoints/{cid}/load")
        assert loaded.status_code == 200 and loaded.json()["state"] == "running"
        c.post("/api/train/stop")
        training_manager.join(timeout=30)

        assert c.delete(f"/api/checkpoints/{cid}").status_code == 204
        assert all(m["id"] != cid for m in c.get("/api/checkpoints").json())


def test_load_unknown_checkpoint_is_404() -> None:
    assert client.post("/api/checkpoints/does-not-exist/load").status_code == 404
    assert client.get("/api/checkpoints/does-not-exist/export").status_code == 404
    assert client.delete("/api/checkpoints/does-not-exist").status_code == 404


def test_load_unknown_env_is_rejected(tmp_path: Path) -> None:
    # A checkpoint whose env is not in the registry must be rejected with a clear 400.
    from app.services.checkpoints import checkpoint_store

    cfg = TrainConfig(env_id="ghost-env", algo="ppo", seed=1)
    meta = checkpoint_store.save(cfg, _artifact(), [])
    resp = client.post(f"/api/checkpoints/{meta['id'] if isinstance(meta, dict) else meta.id}/load")
    assert resp.status_code == 400
    assert "ghost-env" in resp.json()["detail"]
