"""G3c-play — CarRacing (image obs + continuous Box(3) action): registry row, the image+box factory
path, GPU-training gating, the steer/gas/brake play action, and the skill bands.

CarRacing is the env that finally combines BOTH previously-separate seams for *human play*: an image
observation (96×96×3 → the existing server-JPEG render path, like Atari/MiniGrid) AND a continuous
Box(3) action (steer/gas/brake → the G1b/G3b continuous-box play path). So it needs NO new engine
code — only data + content. Training needs the CnnPolicy seam and is gated to a GPU machine (G3c-train).
"""

import types

import numpy as np
import pytest
from app.envs.factory import make_env
from app.envs.registry import get_env
from app.main import app
from app.services.client_render import client_state
from app.services.connection_manager import manager
from app.services.play_session import PlaySession
from app.services.preview_streamer import encode_frame
from app.services.system_info import gpu_available
from fastapi.testclient import TestClient

client = TestClient(app)


# -- registry ---------------------------------------------------------------


def test_carracing_registered() -> None:
    spec = get_env("carracing")
    assert spec is not None, "carracing not registered"
    assert spec.gym_id == "CarRacing-v3"
    assert spec.family == "box2d"
    assert spec.obs_type == "image"  # 96×96×3 pixels — the first image-obs human-playable env
    assert spec.action_space == "box"  # continuous Box(3): steer / gas / brake — the G1b/G3b seam
    assert spec.supported_algos == ["ppo"]  # evolution opted out as data (a flat-vector genome can't take pixels)
    assert spec.hw_requirement == "gpu"  # image obs needs the CnnPolicy seam (G3c-train); play available now
    assert spec.human_playable is True
    assert spec.competitive is False
    assert spec.make_kwargs == {"continuous": True}  # the continuous variant (vs Discrete(5))
    # do-nothing/off-track floor (ADR-026); the −100 off-field penalty is terminal, not per-step.
    assert spec.solved_score == 900.0 and spec.min_score == -100.0
    assert spec.floor_scales_with_steps is False
    assert spec.play_step_scale == 1


# -- the image + continuous-box factory path --------------------------------


def test_make_env_builds_carracing_image_box() -> None:
    """The shared factory builds a real CarRacing env: image obs (96×96×3), continuous Box(3) action,
    and an image render → server JPEG (client_state is None). This is the exact human-play path."""
    env = make_env("carracing", render_mode="rgb_array", play_scale=1)
    try:
        assert env.observation_space.shape == (96, 96, 3)
        assert env.action_space.shape == (3,)  # steer, gas, brake
        assert getattr(env.action_space, "n", None) is None  # box, not discrete
        # steer spans [-1, 1]; gas/brake span [0, 1] — confirms the continuous variant.
        assert np.allclose(env.action_space.low, [-1.0, 0.0, 0.0])
        assert np.allclose(env.action_space.high, [1.0, 1.0, 1.0])
        obs, _ = env.reset(seed=0)
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert client_state(env, obs) is None  # image obs → server JPEG, not a client-render state
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        image, w, h = encode_frame(rgb)
        assert isinstance(image, str) and len(image) > 0 and w > 0 and h > 0
    finally:
        env.close()


# -- steer/gas/brake play action (content side of the continuous-box seam) ----


def test_carracing_play_action_reshaped_and_clipped() -> None:
    """A human's steer/gas/brake vector command is reshaped into the Box(3) action and clipped to
    [low, high] (steer [-1,1], gas/brake [0,1]); the keymap's scalar idle 0 fills a zero vector."""
    sess = PlaySession(manager)
    fake_box = types.SimpleNamespace(
        action_space=types.SimpleNamespace(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            sample=lambda: np.zeros(3, dtype=np.float32),
        )
    )
    sess._capture_action_space(fake_box)
    assert sess._box_shape == (3,)
    sess._mode = "human"
    # full left + gas, with out-of-range entries clipped (brake -1 → 0 since brake's low is 0).
    sess._latest_action = [-2.0, 5.0, -1.0]
    assert np.allclose(sess._choose_action(fake_box, None), [-1.0, 1.0, 0.0])
    # the scalar idle 0 fills the whole action with zero (coast — no steer/gas/brake).
    sess._latest_action = 0
    assert np.allclose(sess._choose_action(fake_box, None), [0.0, 0.0, 0.0])


# -- /api/system + the GPU-training gate ------------------------------------


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train CarRacing")
def test_carracing_training_is_gated_without_a_gpu() -> None:
    """On a CPU-only machine, starting CarRacing training is rejected with a clear 400 (the image obs
    needs CnnPolicy/CUDA); the UI also disables Run. Mirrors Atari (G4a) / BipedalWalker (G3b)."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "carracing", "algo": "ppo", "seed": 1, "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "GPU" in resp.json()["detail"]


def test_skill_bands_span_offtrack_to_solved() -> None:
    """Bands come straight from [min_score, solved_score] = [-100, 900] — no bespoke band table."""
    skill = client.get("/api/skill/carracing").json()
    assert skill["min_score"] == -100.0 and skill["max_score"] == 900.0
    assert skill["bands"][0]["min_score"] == -100.0  # weakest band starts at the off-track floor
    assert skill["bands"][-1]["id"] == "superhuman"
