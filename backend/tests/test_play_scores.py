"""E2 — play leaderboards: top-N trimming, AI de-dup by model, persistence, REST surface."""

from pathlib import Path

from app.main import app
from app.schemas.play_scores import TOP_N, PlayScores
from app.services.play_scores import PlayScoreStore
from fastapi.testclient import TestClient

client = TestClient(app)


def _store(tmp_path: Path) -> PlayScoreStore:
    return PlayScoreStore(tmp_path / "play_scores.json")


# -- store ------------------------------------------------------------------


def test_human_board_keeps_top_n_sorted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    # Submit more than the cap, ascending; the board keeps the best TOP_N, best first.
    for s in range(TOP_N + 3):
        store.submit("cartpole", "human", f"P{s}", float(s * 10), steps=s)
    board = store.get("cartpole").human
    assert len(board) == TOP_N
    scores = [e.score for e in board]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == (TOP_N + 2) * 10  # the highest submitted survived

    # A score that can't beat the lowest kept entry does not qualify.
    res = store.submit("cartpole", "human", "weak", 1.0)
    assert res.qualified is False and res.rank is None


def test_ai_board_dedupes_by_model_keeping_best(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.submit("cartpole", "ai", "ppo·2k", 120.0, model_id="ckpt-1", algo="ppo")
    assert first.qualified and first.rank == 1

    # A weaker run of the same model is ignored (no duplicate row).
    weaker = store.submit("cartpole", "ai", "ppo·2k", 90.0, model_id="ckpt-1", algo="ppo")
    assert weaker.qualified is False
    assert len(store.get("cartpole").ai) == 1

    # A better run of the same model replaces it (still one row, higher score).
    better = store.submit("cartpole", "ai", "ppo·2k", 300.0, model_id="ckpt-1", algo="ppo")
    assert better.qualified
    ai = store.get("cartpole").ai
    assert len(ai) == 1 and ai[0].score == 300.0


def test_boards_are_per_env_and_per_category(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.submit("cartpole", "human", "A", 50.0)
    store.submit("cartpole", "ai", "M", 50.0, model_id="m1")
    assert len(store.get("cartpole").human) == 1
    assert len(store.get("cartpole").ai) == 1
    assert store.get("other-env").human == []


def test_persists_across_store_instances(tmp_path: Path) -> None:
    _store(tmp_path).submit("cartpole", "human", "Martin", 432.0, steps=432)
    reloaded = _store(tmp_path)  # simulates a restart over the same file
    board = reloaded.get("cartpole").human
    assert len(board) == 1 and board[0].name == "Martin" and board[0].score == 432.0


# -- REST -------------------------------------------------------------------


def test_playscores_endpoints_empty_then_populated() -> None:
    # Autouse fixture redirects the singleton at a tmp file → starts empty.
    empty = client.get("/api/playscores/cartpole").json()
    assert empty["env_id"] == "cartpole" and empty["human"] == [] and empty["ai"] == []
    PlayScores.model_validate(empty)

    posted = client.post(
        "/api/playscores/cartpole",
        json={"category": "human", "name": "Ada", "score": 275.0, "steps": 275},
    ).json()
    assert posted["qualified"] is True and posted["rank"] == 1
    assert posted["scores"]["human"][0]["name"] == "Ada"

    listed = client.get("/api/playscores/cartpole").json()
    assert listed["human"][0]["score"] == 275.0
