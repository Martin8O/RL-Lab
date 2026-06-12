"""C2 — Persistent high scores: record-only-on-beat, atomic persistence, REST surface."""

from pathlib import Path

from app.main import app
from app.schemas.highscores import HighScore
from app.services.highscores import HighScoreStore, make_meta
from fastapi.testclient import TestClient

client = TestClient(app)


def _store(tmp_path: Path) -> HighScoreStore:
    return HighScoreStore(tmp_path / "highscores.json")


# -- store ------------------------------------------------------------------


def test_record_only_when_strictly_better(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.get("cartpole") is None

    first = store.record("cartpole", 100.0, make_meta("ppo", 42, iteration=3))
    assert first is not None and first.score == 100.0

    # Equal or worse → no new record, stored best unchanged.
    assert store.record("cartpole", 100.0, make_meta("ppo", 42, iteration=4)) is None
    assert store.record("cartpole", 80.0, make_meta("ppo", 42, iteration=5)) is None
    assert store.get("cartpole").score == 100.0

    # Strictly better → recorded.
    better = store.record("cartpole", 250.0, make_meta("neuroevolution", 7, generation=9))
    assert better is not None and better.score == 250.0
    assert better.meta.algo == "neuroevolution"
    assert better.meta.generation == 9 and better.meta.iteration is None


def test_persists_across_store_instances(tmp_path: Path) -> None:
    _store(tmp_path).record("cartpole", 321.0, make_meta("ppo", 1, iteration=1))
    # A brand-new store over the same file (simulating a server restart) sees the best.
    reloaded = _store(tmp_path)
    best = reloaded.get("cartpole")
    assert best is not None and best.score == 321.0


def test_corrupt_file_starts_fresh(tmp_path: Path) -> None:
    path = tmp_path / "highscores.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = HighScoreStore(path)
    assert store.all() == []  # tolerated, not crashed
    assert store.record("cartpole", 10.0, make_meta("ppo", 0, iteration=0)) is not None


# -- REST -------------------------------------------------------------------


def test_highscores_endpoints_empty_then_populated(tmp_path: Path) -> None:
    # Autouse fixture redirects the singleton at a tmp file → starts empty.
    assert client.get("/api/highscores").json() == []
    assert client.get("/api/highscores/cartpole").json() is None

    from app.services.highscores import highscores

    highscores.record("cartpole", 500.0, make_meta("ppo", 42, iteration=12))

    listed = client.get("/api/highscores").json()
    assert len(listed) == 1 and listed[0]["env_id"] == "cartpole"
    detail = client.get("/api/highscores/cartpole").json()
    assert detail["type"] == "highscore" and detail["score"] == 500.0
    # Validate the response shape against the contract.
    HighScore.model_validate(detail)
