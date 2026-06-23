"""G5a — MuJoCo family (vector obs + continuous Box action): registry rows, the vector+box factory
path (server-JPEG render), the continuous-box play action, GPU-training gating, and the skill bands.

MuJoCo robots are image-free continuous-control envs, so — like BipedalWalker — they reuse two
existing seams with no engine code: the G1b/G3b continuous-box action path (box-aware play/predict,
a per-joint vector keymap) and the server-JPEG render path (client_state returns None → env.render()).
The risk the prompt flagged (offscreen rgb_array on Windows) was checked first and works. Training
takes millions of steps and is gated to a GPU machine; human play needs no model and works now.
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

# The six G5a MuJoCo envs and their continuous action dimensions. These all share native [-1, 1]
# torque bounds, so the parametrized build test below asserts that range. Humanoid (the 7th robot,
# G5b) is NOT in this dict: its native bounds are Box(-0.4, 0.4), so it gets its own test below.
_MUJOCO_ENVS = {
    "hopper": ("Hopper-v5", 3),
    "walker2d": ("Walker2d-v5", 6),
    "halfcheetah": ("HalfCheetah-v5", 6),
    "ant": ("Ant-v5", 8),
    "reacher": ("Reacher-v5", 2),
    "swimmer": ("Swimmer-v5", 2),
}


# -- registry ---------------------------------------------------------------


@pytest.mark.parametrize("env_id", list(_MUJOCO_ENVS))
def test_mujoco_registered(env_id: str) -> None:
    spec = get_env(env_id)
    assert spec is not None, f"{env_id} not registered"
    assert spec.gym_id == _MUJOCO_ENVS[env_id][0]
    assert spec.family == "mujoco"
    assert spec.obs_type == "vector"  # a float state → MlpPolicy (no CnnPolicy); server-JPEG render
    assert spec.action_space == "box"  # continuous per-joint torques — the G1b/G3b seam
    assert spec.supported_algos == ["ppo"]  # evolution opted out as data (hard multi-joint control)
    assert spec.hw_requirement == "gpu"  # millions of steps → desktop; play available now
    assert spec.human_playable is True
    assert spec.competitive is False
    # The skill floor is the venv-measured idle baseline (ADR-026), so it sits below the solved score.
    assert spec.min_score < spec.solved_score


def test_mujoco_skill_floor_calibration() -> None:
    """Spot-check the venv-measured idle floors: a do-nothing agent should read ~0% (ADR-026)."""
    # Locomotion envs floor at their measured idle (standing) return; Reacher is a step-penalty env.
    assert get_env("hopper").min_score == 120.0 and get_env("hopper").solved_score == 3800.0
    assert get_env("ant").min_score == 980.0 and get_env("ant").solved_score == 6000.0
    assert get_env("halfcheetah").min_score == 0.0  # idle ≈ 0 (never terminates); random flailing < 0
    # Reacher is the one step-penalty env: its floor widens with the longer play episode.
    assert get_env("reacher").floor_scales_with_steps is True
    assert get_env("reacher").play_step_scale == 6
    # The locomotion envs do not widen their floor (a fall is terminal) and play at native length.
    for env_id in ("hopper", "walker2d", "halfcheetah", "ant", "swimmer"):
        assert get_env(env_id).floor_scales_with_steps is False
        assert get_env(env_id).play_step_scale == 1


def test_humanoid_registered_and_built() -> None:
    """Humanoid-v5 is the 7th MuJoCo robot (G5b) — the same data-only family seam, just the hardest
    member. It carries the family flags (vector obs, box action, PPO-only, GPU-gated, trainable now)
    but differs from the other six in two ways: a much larger 348-number state and native action
    bounds of Box(-0.4, 0.4) instead of [-1, 1] — which is exactly why it has its own test."""
    spec = get_env("humanoid")
    assert spec is not None, "humanoid not registered"
    assert spec.gym_id == "Humanoid-v5"
    assert spec.family == "mujoco"
    assert spec.obs_type == "vector"  # a flat 348-float state → MlpPolicy; server-JPEG render
    assert spec.action_space == "box"  # 17 continuous per-joint torques
    assert spec.supported_algos == ["ppo"]
    assert spec.hw_requirement == "gpu"  # one of the hardest tasks → millions of steps on the desktop
    assert spec.train_implemented is True  # MlpPolicy trainer exists; only gated by step count
    assert spec.human_playable is True and spec.competitive is False
    assert spec.difficulty == "advanced"
    # A 17-joint humanoid topples even faster than Hopper/Walker2d → the same modest play slow-down.
    assert spec.human_play_slowdown == 2.5
    # The skill floor is the venv-measured zero-torque idle return (≈198 → 200), well below "solved".
    assert spec.min_score == 200.0 and spec.solved_score == 5000.0
    assert spec.min_score < spec.solved_score
    assert spec.play_step_scale == 1 and spec.floor_scales_with_steps is False

    # Build the real env: a 348-vector obs and a continuous Box(17) action in the native [-0.4, 0.4].
    env = make_env("humanoid", render_mode="rgb_array", play_scale=1)
    try:
        assert env.observation_space.shape == (348,)
        assert env.action_space.shape == (17,)
        assert getattr(env.action_space, "n", None) is None  # box, not discrete
        assert np.allclose(env.action_space.low, -0.4) and np.allclose(env.action_space.high, 0.4)
        env.reset(seed=0)
        obs, _reward, _term, _trunc, _ = env.step(env.action_space.sample())
        assert client_state(env, obs) is None  # not client-rendered → server JPEG
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # offscreen rgb_array works on Windows
    finally:
        env.close()


def test_hopper_walker_human_play_slowdown() -> None:
    """Hopper/Walker2d render at 125 fps and fall in ~1 s, so human play gets a MODEST slow-down
    (human_play_slowdown 2.5 → ~10–15 fps, ~15 s) so a beginner can see the robot move and fall — the
    earlier 8× overshot into an unplayably choppy slideshow. The other envs run slow enough already
    and keep the default 1.0 (no slow-down)."""
    assert get_env("hopper").human_play_slowdown == 2.5
    assert get_env("walker2d").human_play_slowdown == 2.5
    for env_id in ("halfcheetah", "ant", "reacher", "swimmer"):
        assert get_env(env_id).human_play_slowdown == 1.0


# -- the vector + continuous-box factory path -------------------------------


@pytest.mark.parametrize("env_id", list(_MUJOCO_ENVS))
def test_make_env_builds_mujoco_vector_box(env_id: str) -> None:
    """The shared factory builds a real MuJoCo env: a 1-D vector obs, a continuous Box action, and an
    rgb_array render → server JPEG (client_state is None). This is the exact human-play path, and it
    confirms offscreen rendering works on this machine (the risk the prompt flagged)."""
    _gym_id, n_act = _MUJOCO_ENVS[env_id]
    env = make_env(env_id, render_mode="rgb_array", play_scale=1)
    try:
        assert len(env.observation_space.shape) == 1  # a flat vector → MlpPolicy
        assert env.action_space.shape == (n_act,)  # continuous per-joint torques
        assert getattr(env.action_space, "n", None) is None  # box, not discrete
        assert np.allclose(env.action_space.low, -1.0) and np.allclose(env.action_space.high, 1.0)
        obs, _ = env.reset(seed=0)
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert client_state(env, obs) is None  # not client-rendered → server JPEG
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # offscreen rgb_array works on Windows
        image, w, h = encode_frame(rgb)
        assert isinstance(image, str) and len(image) > 0 and w > 0 and h > 0
    finally:
        env.close()


def test_reacher_play_scale_extends_short_episode() -> None:
    """Reacher's native 50-step episode is too short to play, so play_step_scale=6 lengthens it (the
    factory multiplies the cap); the locomotion envs keep their native length (play_scale=1 → None)."""
    env = make_env("reacher", play_scale=6)
    try:
        assert env.spec.max_episode_steps == 300  # 50 × 6 — a few real seconds at 50 fps
    finally:
        env.close()


# -- continuous-box play action (content side of the seam) ------------------


def test_mujoco_play_action_reshaped_and_clipped() -> None:
    """A human's per-joint torque vector is reshaped into the Box action and clipped to [-1, 1]; the
    keymap's scalar idle 0 fills a zero-torque vector. Uses Hopper's Box(3) (thigh/knee/ankle)."""
    sess = PlaySession(manager)
    fake_box = types.SimpleNamespace(
        action_space=types.SimpleNamespace(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            shape=(3,),
            sample=lambda: np.zeros(3, dtype=np.float32),
        )
    )
    sess._capture_action_space(fake_box)
    assert sess._box_shape == (3,)
    sess._mode = "human"
    # held thigh + knee torque with an out-of-range entry clipped to the [-1, 1] bound.
    sess._latest_action = [-2.0, 1.0, 0.0]
    assert np.allclose(sess._choose_action(fake_box, None), [-1.0, 1.0, 0.0])
    # the scalar idle 0 fills the whole action with zero torque (joints go limp).
    sess._latest_action = 0
    assert np.allclose(sess._choose_action(fake_box, None), [0.0, 0.0, 0.0])


# -- /api/system + the GPU-training gate ------------------------------------


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train MuJoCo")
def test_mujoco_training_is_gated_without_a_gpu() -> None:
    """On a CPU-only machine, starting MuJoCo training is rejected with a clear 400 (a good gait needs
    millions of steps — impractical on the laptop CPU); the UI also disables Run. Mirrors BipedalWalker."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "hopper", "algo": "ppo", "seed": 1, "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "GPU" in resp.json()["detail"]


def test_skill_bands_span_idle_to_solved() -> None:
    """Bands come straight from [min_score, solved_score] = [120, 3800] for Hopper — no bespoke table."""
    skill = client.get("/api/skill/hopper").json()
    assert skill["min_score"] == 120.0 and skill["max_score"] == 3800.0
    assert skill["bands"][0]["min_score"] == 120.0  # weakest band starts at the idle floor
    assert skill["bands"][-1]["id"] == "superhuman"
