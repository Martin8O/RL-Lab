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


# -- X7 curation: label / note / experiment tag / exclude / bulk delete ------


def test_update_meta_is_partial_and_sidecar_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=7)
    meta = store.save(cfg, _PPO_FRAMES, state="finished", started_at="2026-06-12T10:00:00+00:00")
    original_label = meta.label

    # A partial patch touches only the given field; the rest of the meta is preserved.
    updated = store.update_meta(meta.id, {"note": "my best run"})
    assert updated is not None and updated.note == "my best run"
    assert updated.label == original_label and updated.excluded is False

    # Immutable artifacts are never rewritten — config + metrics still read back intact.
    detail = store.get(meta.id)
    assert detail is not None and detail.config.seed == 7 and len(detail.metrics) == 3

    # Setting the label + exclude persists across a fresh read from disk.
    store.update_meta(meta.id, {"label": "Baseline", "excluded": True})
    reread = next(m for m in store.list() if m.id == meta.id)
    assert reread.label == "Baseline" and reread.excluded is True and reread.note == "my best run"

    # An unknown id is a no-op returning None (never raises / creates a dir).
    assert store.update_meta("does-not-exist", {"note": "x"}) is None
    assert store.update_meta("../escape", {"note": "x"}) is None


def test_delete_many_counts_only_existing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo")
    a = store.save(cfg, _PPO_FRAMES, state="finished", started_at="2026-06-12T10:00:00+00:00")
    b = store.save(cfg, _PPO_FRAMES, state="finished", started_at="2026-06-12T10:05:00+00:00")
    assert store.delete_many([a.id, b.id, "ghost"]) == 2  # 'ghost' didn't exist
    assert store.get(a.id) is None and store.get(b.id) is None


def test_patch_route_edits_curation_fields(tmp_path: Path, monkeypatch) -> None:
    from app.api import runs as runs_api

    store = _store(tmp_path)
    monkeypatch.setattr(runs_api, "run_store", store)
    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=5)
    rid = store.save(cfg, _PPO_FRAMES, state="finished", started_at="2026-06-12T10:00:00+00:00").id

    resp = client.patch(f"/api/runs/{rid}", json={"note": "annotated", "excluded": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["note"] == "annotated" and body["excluded"] is True and body["seed"] == 5
    assert client.patch("/api/runs/nope", json={"note": "x"}).status_code == 404


def test_group_and_bulk_delete_routes(tmp_path: Path, monkeypatch) -> None:
    from app.api import runs as runs_api

    store = _store(tmp_path)
    monkeypatch.setattr(runs_api, "run_store", store)
    cfg = TrainConfig(env_id="cartpole", algo="ppo")
    ids = [
        store.save(cfg, _PPO_FRAMES, state="finished", started_at=f"2026-06-12T10:0{i}:00+00:00").id
        for i in range(3)
    ]

    # Group two of them under one named experiment.
    resp = client.post(
        "/api/runs/group",
        json={"run_ids": ids[:2], "experiment_id": "manual:my-study", "experiment_label": "My study"},
    )
    assert resp.status_code == 200
    grouped = resp.json()
    assert {r["experiment_id"] for r in grouped} == {"manual:my-study"}
    assert all(r["experiment_label"] == "My study" for r in grouped)

    # Ungroup (clear the tag) is the same route with a null id.
    cleared = client.post("/api/runs/group", json={"run_ids": [ids[0]], "experiment_id": None}).json()
    assert cleared[0]["experiment_id"] is None

    # Bulk delete reports how many existed.
    resp = client.post("/api/runs/delete", json={"run_ids": [*ids, "ghost"]})
    assert resp.status_code == 200 and resp.json()["deleted"] == 3
    assert store.list() == []
