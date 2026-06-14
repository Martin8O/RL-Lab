"""E1 — play sessions: human + AI episodes, action handling, config guards, REST + WS routing."""

import json
import types

import pytest
from app.main import app
from app.schemas.play import PlayConfig
from app.schemas.training import PPOHyperparams, TrainConfig
from app.services.checkpoints import CheckpointStore
from app.services.connection_manager import manager
from app.services.play_session import (
    AlreadyPlayingError,
    InvalidPlayConfigError,
    PlayCheckpointNotFoundError,
    PlaySession,
    play_session,
)
from app.services.train_control import TrainControl
from app.services.trainer_ppo import train_ppo
from fastapi.testclient import TestClient

client = TestClient(app)


def _ppo_checkpoint(store: CheckpointStore) -> str:
    """Train a tiny PPO run and persist its snapshot into ``store``; return the slot id."""
    snaps: list = []
    train_ppo(
        TrainConfig(
            env_id="cartpole", algo="ppo", seed=1, total_timesteps=256,
            hyperparams=PPOHyperparams(n_steps=64, batch_size=64),
        ),
        "CartPole-v1", TrainControl(),
        lambda _m: None, lambda _p: None, None, snaps.append,
    )
    assert snaps, "no PPO snapshot captured"
    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=1)
    return store.save(cfg, snaps[-1], []).id


# -- human episode (no loop bound → broadcasts are skipped, like the manager tests) --------


def test_human_session_runs_accepts_actions_and_rates() -> None:
    sess = PlaySession(manager)
    status = sess.start(PlayConfig(env_id="cartpole", mode="human", seed=0, speed=20.0))
    assert status.state == "playing"
    sess.submit_action(1)  # a keyboard action is accepted while playing
    sess.join(timeout=30)

    final = sess.status()
    assert final.state == "finished"
    assert final.result is not None
    assert final.result.mode == "human"
    assert final.step >= 1 and final.result.steps == final.step
    assert final.result.rating.band in {
        "child", "below_average", "average", "above_average", "superhuman"
    }


# -- AI episode from a checkpoint -------------------------------------------


def test_ai_session_plays_from_checkpoint(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    cid = _ppo_checkpoint(store)

    sess = PlaySession(manager, checkpoints=store)
    status = sess.start(
        PlayConfig(env_id="cartpole", mode="ai", checkpoint_id=cid, seed=0, speed=20.0)
    )
    assert status.state == "playing"
    sess.join(timeout=60)

    final = sess.status()
    assert final.state == "finished"
    assert final.result is not None and final.result.mode == "ai"
    assert final.score >= 1
    assert final.result.rating.score == final.score


# -- config guards ----------------------------------------------------------


def test_ai_session_requires_valid_checkpoint(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    sess = PlaySession(manager, checkpoints=store)

    with pytest.raises(InvalidPlayConfigError):
        sess.start(PlayConfig(env_id="cartpole", mode="ai", checkpoint_id=None))
    with pytest.raises(PlayCheckpointNotFoundError):
        sess.start(PlayConfig(env_id="cartpole", mode="ai", checkpoint_id="nope"))


def test_unknown_env_is_rejected() -> None:
    sess = PlaySession(manager)
    with pytest.raises(InvalidPlayConfigError):
        sess.start(PlayConfig(env_id="ghost-env", mode="human"))


def test_single_active_session() -> None:
    sess = PlaySession(manager)
    sess.start(PlayConfig(env_id="cartpole", mode="human", seed=0, speed=1.0))
    try:
        with pytest.raises(AlreadyPlayingError):
            sess.start(PlayConfig(env_id="cartpole", mode="human"))
    finally:
        sess.stop()
        sess.join(timeout=30)


# -- action handling (white-box: avoids spinning a full render loop) --------


def test_human_action_clamped_to_action_space() -> None:
    sess = PlaySession(manager)
    sess._mode = "human"
    sess._n_actions = 2
    fake_env = types.SimpleNamespace(
        action_space=types.SimpleNamespace(n=2, sample=lambda: 0)
    )
    sess._latest_action = 5
    assert sess._choose_action(fake_env, None) == 1  # clamped to n-1
    sess._latest_action = -3
    assert sess._choose_action(fake_env, None) == 0  # clamped to 0


def test_submit_action_ignored_when_not_playing() -> None:
    sess = PlaySession(manager)
    sess.submit_action(1)  # no active session → no-op
    assert sess._latest_action == 0


def test_idle_action_initialises_held_action() -> None:
    # CartPole has no idle (idle_action None) → the held action defaults to 0, as before.
    sess = PlaySession(manager)
    sess.start(PlayConfig(env_id="cartpole", mode="human", seed=0, speed=20.0))
    try:
        assert sess._latest_action == 0
    finally:
        sess.stop()
        sess.join(timeout=30)

    # MountainCar's idle is 1 (no acceleration). The session must hold that until a key is
    # pressed — not the default 0, which means "push left" and shoves the car before any input.
    sess2 = PlaySession(manager)
    sess2.start(
        PlayConfig(env_id="mountaincar", mode="human", seed=0, speed=20.0, idle_action=1)
    )
    try:
        assert sess2._latest_action == 1
    finally:
        sess2.stop()
        sess2.join(timeout=30)


def test_set_speed_updates_and_clamps() -> None:
    sess = PlaySession(manager)
    sess.start(PlayConfig(env_id="cartpole", mode="human", seed=0, speed=1.0))
    try:
        assert sess.set_speed(8.0).speed == 8.0
        assert sess._current_speed() == 8.0
        assert sess.set_speed(999.0).speed == 20.0  # clamped to the play max
        assert sess.set_speed(0.0).speed == 0.1  # clamped to the play min
    finally:
        sess.stop()
        sess.join(timeout=30)


# -- REST + WS --------------------------------------------------------------


def test_play_status_endpoint_shape() -> None:
    body = client.get("/api/play/status").json()
    for key in (
        "type", "state", "env_id", "mode", "checkpoint_id",
        "seed", "speed", "step", "score", "result", "error",
    ):
        assert key in body
    assert body["type"] == "play_status"


def test_play_start_validation_errors() -> None:
    assert client.post(
        "/api/play/start", json={"env_id": "ghost", "mode": "human", "speed": 1.0}
    ).status_code == 400
    assert client.post(
        "/api/play/start",
        json={"env_id": "cartpole", "mode": "ai", "speed": 1.0},
    ).status_code == 400  # AI without a checkpoint
    assert client.post(
        "/api/play/start",
        json={"env_id": "cartpole", "mode": "ai", "checkpoint_id": "nope", "speed": 1.0},
    ).status_code == 404


def test_rest_human_start_stop() -> None:
    # Context-managed client runs the lifespan so the session's WS broadcasts land on a live loop.
    with TestClient(app) as c:
        started = c.post(
            "/api/play/start",
            json={"env_id": "cartpole", "mode": "human", "seed": 0, "speed": 1.0},
        )
        assert started.status_code == 200 and started.json()["state"] == "playing"
        stopped = c.post("/api/play/stop")
        assert stopped.status_code == 200
        play_session.join(timeout=30)
        assert c.get("/api/play/status").json()["state"] in {"stopped", "finished"}


def test_play_speed_endpoint_updates_live_session() -> None:
    with TestClient(app) as c:
        c.post(
            "/api/play/start",
            json={"env_id": "cartpole", "mode": "human", "seed": 0, "speed": 1.0},
        )
        try:
            r = c.post("/api/play/speed", json={"speed": 4.0})
            assert r.status_code == 200 and r.json()["speed"] == 4.0
        finally:
            c.post("/api/play/stop")
            play_session.join(timeout=30)


def test_ws_action_routes_and_text_still_echoes() -> None:
    with TestClient(app) as c, c.websocket_connect("/ws") as ws:
        ws.send_text("hello")
        assert ws.receive_json() == {"echo": "hello"}
        # An action frame is consumed (routed to the play session), not echoed; a following
        # plain-text message still round-trips, proving only action frames are intercepted.
        ws.send_text(json.dumps({"type": "action", "action": 1}))
        ws.send_text("again")
        assert ws.receive_json() == {"echo": "again"}
