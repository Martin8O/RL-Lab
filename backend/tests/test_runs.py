"""D2 — Run history & version compare: store round-trip, derivation, archive gate, REST."""

from pathlib import Path

from app.main import app
from app.schemas.runs import RunDetail
from app.schemas.training import TrainConfig
from app.services.runs import RunStore, final_score, should_archive
from app.services.training_manager import training_manager
from fastapi.testclient import TestClient

client = TestClient(app)

_SOLVED = 500.0  # CartPole solved score (10% = 50)


def _store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


# A PPO run that crosses the solved score (500) at 6144 steps.
_PPO_FRAMES = [
    {"type": "metrics", "iteration": 1, "timesteps": 2048, "ep_rew_mean": 30.0, "loss": 0.5},
    {"type": "metrics", "iteration": 2, "timesteps": 4096, "ep_rew_mean": 500.0, "loss": 0.3},
    {"type": "metrics", "iteration": 3, "timesteps": 6144, "ep_rew_mean": 500.0, "loss": 0.2},
]
# An evolution run that crosses the solved score at generation 2.
_EVO_FRAMES = [
    {"type": "evolution", "generation": 1, "total_generations": 3, "best_fitness": 120.0,
     "timesteps": 6000},
    {"type": "evolution", "generation": 2, "total_generations": 3, "best_fitness": 500.0,
     "timesteps": 12000},
]


# -- archive gate + derivation helpers --------------------------------------


def test_should_archive_gate() -> None:
    # Every terminal-success run is kept regardless of score — low-skill filtering is a Data Lab UI
    # choice now, not a save-time gate (a 0%-skill run that showed real learning must still archive).
    assert should_archive("finished") is True
    assert should_archive("stopped") is True
    assert should_archive("error") is False     # a crashed run isn't a result
    assert should_archive("running") is False   # not terminal
    assert should_archive("idle") is False


def test_final_score_per_algo() -> None:
    ppo = TrainConfig(env_id="cartpole", algo="ppo")
    evo = TrainConfig(env_id="cartpole", algo="neuroevolution")
    assert final_score(ppo, _PPO_FRAMES) == 500.0
    assert final_score(evo, _EVO_FRAMES) == 500.0
    assert final_score(ppo, []) is None


# -- store round-trip + solved_at -------------------------------------------


def test_save_ppo_solved_at_in_timesteps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=7, total_timesteps=50_000)
    meta = store.save(
        cfg, _PPO_FRAMES, state="finished",
        started_at="2026-06-12T10:00:00+00:00", solved_score=_SOLVED,
    )
    assert meta.final_reward == 500.0 and meta.iteration == 3
    assert meta.solved_at == 4096.0  # first frame to reach 500 was at 4096 steps
    assert meta.frames == 3

    detail = store.get(meta.id)
    assert detail is not None and isinstance(detail, RunDetail)
    assert detail.config.seed == 7 and len(detail.metrics) == 3

    assert store.delete(meta.id) is True
    assert store.get(meta.id) is None


def test_save_evolution_solved_at_in_generations(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="neuroevolution", seed=3)
    meta = store.save(
        cfg, _EVO_FRAMES, state="stopped",
        started_at="2026-06-12T11:00:00+00:00", solved_score=_SOLVED,
    )
    assert meta.final_reward == 500.0 and meta.generation == 2 and meta.total_generations == 3
    assert meta.solved_at == 2.0  # solved at generation 2
    assert meta.iteration is None and "gen 2" in meta.label


def test_save_never_solved_has_none_solved_at(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=1)
    frames = [{"type": "metrics", "iteration": 1, "timesteps": 2048, "ep_rew_mean": 120.0}]
    meta = store.save(
        cfg, frames, state="finished",
        started_at="2026-06-12T10:00:00+00:00", solved_score=_SOLVED,
    )
    assert meta.solved_at is None and meta.final_reward == 120.0


def test_list_newest_first_and_traversal_guard(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo")
    a = store.save(cfg, _PPO_FRAMES, state="finished", started_at="2026-06-12T10:00:00+00:00")
    b = store.save(cfg, _PPO_FRAMES, state="finished", started_at="2026-06-12T10:05:00+00:00")
    # id carries microsecond-precise timestamp, so the second save sorts first.
    assert [m.id for m in store.list()] == [b.id, a.id]
    # Path-traversal ids are rejected, never resolved against the filesystem.
    assert store.get("../secret") is None
    assert store.delete("a/b") is False


# -- manager + REST ---------------------------------------------------------


def test_low_skill_run_is_still_archived() -> None:
    # Every finished run is archived now, regardless of skill: a tiny 256-step PPO run stays far below
    # CartPole's solved 500 (0% skill) yet must still reach history — low-skill filtering is a Data Lab
    # UI choice, not a save-time gate (so a real-but-sub-solved learning curve is never silently lost).
    # Context-managed client runs the app lifespan for live WS broadcasts.
    with TestClient(app) as c:
        seed = 4242  # unique so we can find exactly our run
        before = {r["id"] for r in c.get("/api/runs").json()}
        started = c.post(
            "/api/train/start",
            json={
                "env_id": "cartpole", "algo": "ppo", "seed": seed,
                "total_timesteps": 256,
                "hyperparams": {"n_steps": 64, "batch_size": 64},
            },
        )
        assert started.status_code == 200
        training_manager.join(timeout=60)

        runs = c.get("/api/runs").json()
        mine = [r for r in runs if r["seed"] == seed and r["id"] not in before]
        assert len(mine) == 1, "a finished run should be archived even below the old skill threshold"
        c.delete(f"/api/runs/{mine[0]['id']}")  # keep the shared data/ dir clean


def test_get_and_delete_unknown_run_is_404() -> None:
    assert client.get("/api/runs/does-not-exist").status_code == 404
    assert client.delete("/api/runs/does-not-exist").status_code == 404
