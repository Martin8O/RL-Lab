"""G3b — Box2D heavies (BipedalWalker + Hardcore): registry rows, the continuous-box factory path,
GPU-training gating, and the multi-joint vector play action.

Like Atari (G4a) these are **human-playable now / GPU-trained later**: training is gated
(hw_requirement="gpu") because learning to walk takes millions of steps, while human play uses the
existing server-JPEG render and the G1b continuous-box seam. The action is Box(4) — four leg-joint
torques — so play sends a per-joint VECTOR (summed client-side); the play session reshapes + clips it.
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

VARIANTS = ["bipedalwalker", "bipedalwalkerhardcore"]


# -- registry ---------------------------------------------------------------


@pytest.mark.parametrize("eid", VARIANTS)
def test_bipedalwalker_registered(eid: str) -> None:
    spec = get_env(eid)
    assert spec is not None, f"{eid} not registered"
    assert spec.gym_id == "BipedalWalker-v3"
    assert spec.family == "box2d"
    assert spec.obs_type == "vector"
    assert spec.action_space == "box"  # continuous Box(4) — the G1b seam
    assert spec.supported_algos == ["ppo", "sac", "td3"]  # PPO + SAC + TD3 (S5a/S5b); evolution opted out (hard 4-DoF)
    assert spec.hw_requirement == "gpu"  # training gated to a powerful machine
    assert spec.human_playable is True   # but playable by hand now
    assert spec.competitive is False
    # fall/timeout floor (ADR-026), LunarLander-shaped; the floor does not scale with steps.
    assert spec.solved_score == 300.0 and spec.min_score == -100.0
    assert spec.floor_scales_with_steps is False
    assert spec.play_step_scale == 1


def test_hardcore_variant_uses_make_kwargs() -> None:
    base = get_env("bipedalwalker")
    hard = get_env("bipedalwalkerhardcore")
    assert base is not None and hard is not None
    assert base.make_kwargs == {}
    assert hard.make_kwargs == {"hardcore": True}  # same gym id, harder terrain via kwargs
    assert hard.default_total_timesteps > base.default_total_timesteps  # needs a bigger budget


# -- the continuous-box factory path ----------------------------------------


@pytest.mark.parametrize("eid", VARIANTS)
def test_make_env_builds_a_bipedalwalker(eid: str) -> None:
    """The shared factory builds a real Box2D env: vector obs (24), continuous Box(4) action, and an
    image render → server JPEG (no client-side state). This is the exact human-play path; the
    Hardcore variant exercises the make_kwargs={"hardcore": True} path."""
    env = make_env(eid, render_mode="rgb_array", play_scale=1)
    try:
        assert env.observation_space.shape == (24,)
        assert env.action_space.shape == (4,)  # four leg-joint torques
        assert getattr(env.action_space, "n", None) is None  # box, not discrete
        obs, _ = env.reset(seed=0)
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert client_state(env, obs) is None  # vector-but-not-client-rendered → server JPEG
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3
        image, w, h = encode_frame(rgb)
        assert isinstance(image, str) and len(image) > 0 and w > 0 and h > 0
    finally:
        env.close()


# -- multi-joint vector play action (the G3b content side of the G1b seam) ----


def test_vector_play_action_reshaped_and_clipped() -> None:
    """A human's per-joint vector command is reshaped into the Box(4) action and clipped; the
    keymap's scalar idle 0 fills the whole action with zero torque."""
    sess = PlaySession(manager)
    fake_box = types.SimpleNamespace(
        action_space=types.SimpleNamespace(
            low=np.full(4, -1.0, dtype=np.float32),
            high=np.full(4, 1.0, dtype=np.float32),
            shape=(4,),
            sample=lambda: np.zeros(4, dtype=np.float32),
        )
    )
    sess._capture_action_space(fake_box)
    assert sess._box_shape == (4,)
    sess._mode = "human"
    # a 4-element vector passes through, out-of-range entries clipped to [-1, 1]
    sess._latest_action = [1.0, -1.0, 5.0, -9.0]
    assert np.allclose(sess._choose_action(fake_box, None), [1.0, -1.0, 1.0, -1.0])
    # the scalar idle 0 fills the whole action with zero torque (all legs limp)
    sess._latest_action = 0
    assert np.allclose(sess._choose_action(fake_box, None), [0.0, 0.0, 0.0, 0.0])


# -- /api/system + the GPU-training gate ------------------------------------


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train BipedalWalker")
@pytest.mark.parametrize("eid", VARIANTS)
def test_gpu_env_training_is_gated_without_a_gpu(eid: str) -> None:
    """On a CPU-only machine, starting training for a GPU-gated env is rejected with a clear 400 —
    the UI also disables Run; this is the backstop (mirrors Atari G4a)."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": eid, "algo": "ppo", "seed": 1, "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "GPU" in resp.json()["detail"]


def test_skill_bands_span_fall_to_solved() -> None:
    """Bands come straight from [min_score, solved_score] = [-100, 300] (LunarLander-shaped) — no
    bespoke band table, exactly like every other env."""
    skill = client.get("/api/skill/bipedalwalker").json()
    assert skill["min_score"] == -100.0 and skill["max_score"] == 300.0
    assert skill["bands"][0]["min_score"] == -100.0  # weakest band starts at the fall floor
    assert skill["bands"][-1]["id"] == "superhuman"
