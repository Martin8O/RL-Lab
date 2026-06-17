"""G4a — Atari (ALE) family: registry rows, the image-obs factory path, GPU-training gating.

These envs are the first **image-observation** envs and are **human-playable now / GPU-trained
later**: training is gated (hw_requirement="gpu") while human play uses the existing JPEG render
path (client_render returns None for an image obs). Covers the registry data, the shared factory
building a real ALE env, and the /api/system + training-gate backstop.
"""

import numpy as np
import pytest
from app.envs.factory import make_env
from app.envs.registry import get_env, list_envs
from app.main import app
from app.services.client_render import client_state
from app.services.preview_streamer import encode_frame
from app.services.system_info import gpu_available
from fastapi.testclient import TestClient

client = TestClient(app)

# A representative slice — the prompt's named batch plus a symmetric-score game (Pong) and a
# negative-floor sport (Boxing). The whole family is data-driven from one table, so spot-checks suffice.
SAMPLE = ["pong", "breakout", "spaceinvaders", "mspacman", "qbert", "seaquest", "boxing"]


# -- registry ---------------------------------------------------------------


def test_atari_family_registered() -> None:
    atari = [e for e in list_envs() if e.family == "atari"]
    assert len(atari) >= 50, "G4a registers a large Atari batch (50+ games)"
    for spec in atari:
        assert spec.gym_id.startswith("ALE/") and spec.gym_id.endswith("-v5")
        assert spec.obs_type == "image"
        assert spec.action_space == "discrete"
        assert spec.supported_algos == ["ppo"]  # image obs → CnnPolicy/GPU; evo + Q-learning opt out
        assert spec.hw_requirement == "gpu"  # training gated to a CUDA machine
        assert spec.human_playable is True  # but playable by hand now
        assert spec.make_kwargs == {"full_action_space": True}  # uniform 18-action keyboard map


def test_atari_sample_specs() -> None:
    for eid in SAMPLE:
        spec = get_env(eid)
        assert spec is not None, f"{eid} not registered"
    pong = get_env("pong")
    assert pong is not None
    assert pong.gym_id == "ALE/Pong-v5"
    # Symmetric game → a negative floor so the meter fills through the red (like LunarLander).
    assert pong.min_score == -21.0 and pong.solved_score == 21.0
    boxing = get_env("boxing")
    assert boxing is not None and boxing.min_score == -100.0 and boxing.solved_score == 100.0
    # A one-directional arcade score fills 0 → a "really good" target.
    breakout = get_env("breakout")
    assert breakout is not None and breakout.min_score == 0.0 and breakout.solved_score > 0


# -- the image-obs factory path ---------------------------------------------


def test_make_env_builds_an_atari_env() -> None:
    """The shared factory imports ale_py, applies full_action_space, and leaves the image obs alone
    (no one-hot wrapper). reset/step/render is the exact human-play path."""
    env = make_env("pong", render_mode="rgb_array", play_scale=1)
    try:
        assert int(env.action_space.n) == 18  # full action space → uniform keymap
        assert len(env.observation_space.shape) == 3  # image obs (H, W, C), not a vector/discrete
        obs, _ = env.reset(seed=0)
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        # Image obs → no client-side render; the streamer falls back to a server JPEG.
        assert client_state(env, obs) is None
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.shape == (210, 160, 3)
        image, w, h = encode_frame(rgb)
        assert isinstance(image, str) and len(image) > 0 and (w, h) == (160, 210)
    finally:
        env.close()


# -- /api/system + the GPU-training gate ------------------------------------


def test_system_reports_gpu_flag() -> None:
    body = client.get("/api/system").json()
    assert isinstance(body["gpu_available"], bool)
    assert body["gpu_available"] == gpu_available()


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train Atari")
def test_gpu_env_training_is_gated_without_a_gpu() -> None:
    """On a CPU-only machine, starting training for a GPU env is rejected with a clear 400 — the
    UI also disables Run, this is the backstop (G4a)."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "pong",
            "algo": "ppo",
            "seed": 1,
            "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "GPU" in resp.json()["detail"]


def test_atari_skill_bands_span_symmetric_range() -> None:
    """Per-game bands come straight from [min_score, solved_score] — no bespoke band table (Pong ±21)."""
    skill = client.get("/api/skill/pong").json()
    assert skill["min_score"] == -21.0 and skill["max_score"] == 21.0
    assert skill["bands"][0]["min_score"] == -21.0  # weakest band starts at the floor
    assert skill["bands"][-1]["id"] == "superhuman"


# -- training implemented vs GPU-gated (the capability-aware gate, pre-migration hardening) -----


def test_train_implemented_split_after_g4b() -> None:
    """After G4b the image-obs family splits: **Atari trains** (the CnnPolicy + AtariWrapper/frame-stack
    + CUDA seam is built), while **CarRacing stays gated** (its non-Atari image trainer is G3c-train).
    Every vector/discrete env — including the GPU-gated *vector* heavies (BipedalWalker/MuJoCo, MlpPolicy)
    and the competitive multi-agent ``simple_tag`` envs (per-species self-play, G7b-2) — trains too.
    ``train_implemented`` is False for exactly the last not-yet-built trainer: the image CarRacing
    (G3c-train); a GPU box still keeps it gated via the backstop."""
    not_yet = {"carracing"}  # the only trainer not built yet (G3c-train — image CnnPolicy)
    for spec in list_envs():
        expected = spec.id not in not_yet
        assert spec.train_implemented is expected, (
            f"{spec.id}: train_implemented={spec.train_implemented} (obs_type={spec.obs_type})"
        )
    # Spot-check both sides of the split.
    assert get_env("pong").train_implemented is True  # type: ignore[union-attr]  # image, Atari trainer built (G4b)
    assert get_env("carracing").train_implemented is False  # type: ignore[union-attr]  # image, G3c-train pending
    assert get_env("bipedalwalker").train_implemented is True  # type: ignore[union-attr]
    assert get_env("mpe_tag").train_implemented is True  # type: ignore[union-attr]  # per-species self-play (G7b-2)


def test_image_env_training_gated_even_with_a_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key backstop: an image-obs env's training is rejected **even on a CUDA machine**, because
    its CnnPolicy trainer isn't built. Without this, a GPU desktop (or someone building from source on
    a GPU) would un-gate Atari/CarRacing via the gpu check and crash inside the MlpPolicy trainer."""
    monkeypatch.setattr("app.services.training_manager.gpu_available", lambda: True)
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "carracing",
            "algo": "ppo",
            "seed": 1,
            "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "later version" in resp.json()["detail"]  # the "trainer not built yet" message, not "no GPU"
