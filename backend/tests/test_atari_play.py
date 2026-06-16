"""G4c — Atari Play-vs-AI: the image-obs CnnPolicy inference branch in the play session.

A saved Atari ``model.zip`` is a CnnPolicy that consumes the 84×84×4 frame stack, not the raw
210×160×3 RGB the env emits — so ``model.predict`` on the raw obs shape-errors. G4c gives AI play
a dedicated loop on the shared ``make_atari`` vec env (G4b) so the obs shape matches the checkpoint.

These tests run on a **CPU machine too**: constructing a CnnPolicy needs no GPU (only our app's
*training* is CUDA-gated), and AI-play inference runs on CPU by design (ADR-019/044). A full Pong
game is too long to play to completion in a unit test (CPU CNN forward per step), so we prove the
seam — the predict accepts the stacked obs, and the loop steps + stops cleanly — not a finished game.
"""

import time
from io import BytesIO

import pytest
from app.envs.atari import make_atari
from app.schemas.play import PlayConfig
from app.schemas.training import TrainConfig
from app.services.checkpoints import CheckpointArtifact, CheckpointStore
from app.services.connection_manager import manager
from app.services.play_session import PlaySession
from app.services.policy import predict_from_checkpoint

_PONG = "ALE/Pong-v5"
_KW = {"full_action_space": True}


@pytest.fixture(scope="module")
def cnn_blob() -> bytes:
    """An (untrained) CnnPolicy serialized to ``model.zip`` bytes, built on CPU.

    Constructing the NatureCNN policy needs no GPU — only our training path is CUDA-gated — so this
    gives a loadable Atari checkpoint that exercises the AI-play branch on any machine. Module-scoped
    so the (slowish) policy build is paid once for the file.
    """
    from stable_baselines3 import PPO

    venv = make_atari(_PONG, 1, make_kwargs=_KW)
    try:
        model = PPO("CnnPolicy", venv, device="cpu", n_steps=64, batch_size=64)
        buf = BytesIO()
        model.save(buf)
    finally:
        venv.close()
    return buf.getvalue()


def _save_blob(store: CheckpointStore, blob: bytes) -> str:
    """Persist a CnnPolicy blob as a Pong PPO checkpoint; return the slot id."""
    cfg = TrainConfig(env_id="pong", algo="ppo", seed=0)
    artifact = CheckpointArtifact(algo="ppo", blob=blob, artifact_name="model.zip")
    return store.save(cfg, artifact, []).id


# -- the crux: the checkpoint policy accepts the stacked obs (was a shape error) ------------


def test_checkpoint_predict_accepts_stacked_obs(tmp_path, cnn_blob) -> None:
    """``predict_from_checkpoint`` over an Atari CnnPolicy accepts the make_atari 84×84×4 stack and
    returns a valid Atari action index — the exact case that shape-errored on the raw obs pre-G4c."""
    store = CheckpointStore(tmp_path / "checkpoints")
    loaded = store.load(_save_blob(store, cnn_blob))
    assert loaded is not None
    predict = predict_from_checkpoint(loaded)

    venv = make_atari(_PONG, 1, make_kwargs=_KW)
    try:
        obs = venv.reset()
        assert obs.shape == (1, 84, 84, 4)  # the stacked obs the policy trained on
        action = predict(obs[0])  # SB3 predict transposes (84,84,4) → (4,84,84) for the CnnPolicy
    finally:
        venv.close()
    assert isinstance(action, int) and 0 <= action < 18


# -- the loop: AI image play steps the make_atari vec env and stops cleanly ------------------


def test_ai_image_play_runs_steps_and_stops(tmp_path, cnn_blob) -> None:
    """A Pong AI session dispatches to the image loop, steps the CnnPolicy over the vec env and tears
    down on Stop. We stop after a couple of steps rather than finishing the game (CPU CNN inference
    makes a full first-to-21 match far too long for a unit test)."""
    store = CheckpointStore(tmp_path / "checkpoints")
    cid = _save_blob(store, cnn_blob)

    sess = PlaySession(manager, checkpoints=store)
    status = sess.start(
        PlayConfig(env_id="pong", mode="ai", checkpoint_id=cid, seed=0, speed=4.0)
    )
    assert status.state == "playing"

    deadline = time.monotonic() + 30
    while sess.status().step < 1 and time.monotonic() < deadline:
        time.sleep(0.1)
    assert sess.status().step >= 1, "the image-AI loop did not step the vec env"

    sess.stop()
    sess.join(timeout=15)
    assert sess.status().state == "stopped"  # clean teardown, no crash


def test_ai_image_play_rates_a_symmetric_score(tmp_path, cnn_blob, monkeypatch) -> None:
    """End-to-end finalize: a finished image-AI episode rates its score on Pong's symmetric −21…21
    meter. We don't wait for a real game — we drive ``_run_image_ai`` with a vec env stubbed to end
    immediately at a chosen score, proving the score → skill-band wiring (the audit's skill check)."""
    import numpy as np

    store = CheckpointStore(tmp_path / "checkpoints")
    cid = _save_blob(store, cnn_blob)
    sess = PlaySession(manager, checkpoints=store)

    class _OneStepVenv:
        """A make_atari stand-in: one step returns done with reward +21 (a Pong shutout win)."""

        action_space = type("A", (), {"n": 18, "sample": staticmethod(lambda: 0)})()

        def reset(self):
            return np.zeros((1, 84, 84, 4), dtype=np.uint8)

        def step(self, _action):
            return (
                np.zeros((1, 84, 84, 4), dtype=np.uint8),
                np.array([21.0], dtype=np.float32),
                np.array([True]),
                [{}],
            )

        def render(self, mode="rgb_array"):
            return np.zeros((210, 160, 3), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr("app.envs.atari.make_atari", lambda *a, **k: _OneStepVenv())
    sess.start(PlayConfig(env_id="pong", mode="ai", checkpoint_id=cid, seed=0, speed=20.0))
    sess.join(timeout=15)

    final = sess.status()
    assert final.state == "finished"
    assert final.result is not None and final.result.mode == "ai"
    assert final.result.score == 21.0
    # +21 is the top of Pong's [-21, 21] range → the strongest band, ratio 1.0.
    assert final.result.rating.band == "superhuman"
    assert final.result.rating.ratio == pytest.approx(1.0)
