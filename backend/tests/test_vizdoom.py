"""G8b — VizDoom family (image-obs discrete FPS on the G4 image seam).

VizDoom is a NEW image family: it rides the existing CnnPolicy/CUDA image path (server-JPEG render,
AI-play _run_image_ai, raw-JPEG human play) but through its OWN vec builder
(``envs/image_vec.py::make_vizdoom``) — the Gymnasium wrapper emits a **Dict** obs, so a
screen-extraction wrapper (Dict→``screen`` Box) runs before the WarpFrame 84×84 + 4-frame stack (NOT
the ALE AtariWrapper). Training is GPU-gated (hw_requirement="gpu"); human play uses the raw JPEG
render path now. Covers the registry data, the two build paths (make_image_vec + the raw human-play
factory env), the image-obs render, and the GPU-training gate. Mirrors test_atari.py / test_carracing.py.
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

SCENARIOS = [
    "doom_basic",
    "doom_defend_center",
    "doom_health_gathering",
    "doom_defend_line",  # G8d-1
    "doom_health_gathering_supreme",  # G8d-1
]


# -- registry ---------------------------------------------------------------


def test_vizdoom_family_registered() -> None:
    fam = [e for e in list_envs() if e.family == "vizdoom"]
    assert len(fam) == 5, "G8b registers 3 VizDoom scenarios; G8d-1 adds 2 more"
    for spec in fam:
        assert spec.gym_id.startswith("Vizdoom") and spec.gym_id.endswith("-v1")  # v0 deprecated in 1.3.0
        assert spec.obs_type == "image"  # a Dict screen buffer → screen-extract + WarpFrame → Box(84,84,4)
        assert spec.action_space == "discrete"
        assert spec.supported_algos == ["ppo", "dqn", "qrdqn"]  # image obs → CnnPolicy/GPU; evo + Q can't take pixels
        assert spec.hw_requirement == "gpu"  # CnnPolicy training gated to a CUDA machine; human play stays
        assert spec.human_playable is True
        assert spec.competitive is False
        assert spec.train_implemented is True  # rides the existing image trainer via make_vizdoom
        assert spec.make_kwargs == {}  # the scenario is the gym_id; frame_skip is baked into make_vizdoom
        assert spec.min_score < spec.solved_score  # a real [floor, solved] band (ADR-026)
        assert spec.recommended_algo == "ppo"


def test_vizdoom_sample_specs() -> None:
    for eid in SCENARIOS:
        assert get_env(eid) is not None, f"{eid} not registered"
    basic = get_env("doom_basic")
    assert basic is not None and basic.gym_id == "VizdoomBasic-v1"
    # Idle times out at -1/tic over 300 tics → a deep negative floor; a fast kill (≈+100) is "solved".
    assert basic.min_score == -300.0 and basic.solved_score == 90.0

    # G8d-1 — Defend the Line: monsters INFIGHT, so an idle agent banks ~1-2 free kills (probed idle
    # mean 1.9, mode 1) → a positive floor of 1 (NOT -1 like Defend the Center); competent turret ~20.
    line = get_env("doom_defend_line")
    assert line is not None and line.gym_id == "VizdoomDefendLine-v1"
    assert line.min_score == 1.0 and line.solved_score == 20.0

    # G8d-1 — Health Gathering Supreme: identical reward config to Health Gathering, so the idle probe
    # lands on the SAME +284 floor (survives on start health) → the positive band carries over verbatim.
    supreme = get_env("doom_health_gathering_supreme")
    assert supreme is not None and supreme.gym_id == "VizdoomHealthGatheringSupreme-v1"
    assert supreme.min_score == 280.0 and supreme.solved_score == 2000.0


def test_vizdoom_not_in_offpolicy_budgets() -> None:
    """DQN/QR-DQN on VizDoom reuse the PPO image budget (default_total_timesteps) — like Atari, VizDoom
    is intentionally left out of the off-policy budget map, so offpolicy_total_timesteps stays None."""
    for eid in SCENARIOS:
        spec = get_env(eid)
        assert spec is not None and spec.offpolicy_total_timesteps is None


# -- the image vec builder (make_vizdoom via the shared dispatcher) ----------


def test_make_image_vec_builds_vizdoom_screen_stack() -> None:
    """make_image_vec dispatches the vizdoom family to make_vizdoom: the Dict obs is screen-extracted
    and WarpFrame-downscaled to grayscale 84×84, then 4-stacked → Box(84,84,4) the CnnPolicy consumes.
    n_envs=1 (DummyVecEnv, in-process) is the preview / AI-play shape."""
    from app.envs.image_vec import make_image_vec

    spec = get_env("doom_basic")
    assert spec is not None
    venv = make_image_vec(spec, 1, seed=0)
    try:
        assert venv.observation_space.shape == (84, 84, 4)  # grayscale 84×84 × 4-frame stack
        assert int(venv.action_space.n) == 4  # Discrete(4): NO-OP + strafe L/R + attack
        obs = venv.reset()
        assert obs.shape == (1, 84, 84, 4)
        obs, reward, dones, _ = venv.step(np.array([venv.action_space.sample()]))
        assert obs.shape == (1, 84, 84, 4)
        # The raw colour Doom frame is still available for the JPEG (WarpFrame only rewrites the obs).
        rgb = np.asarray(venv.render(mode="rgb_array"), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3
    finally:
        venv.close()


# -- the raw human-play factory path (Dict obs, ignored; render → JPEG) -------


def test_make_env_builds_raw_vizdoom_for_human_play() -> None:
    """Human play goes through the shared factory (NOT make_vizdoom): it registers the Vizdoom ids and
    builds the raw env. The obs is a Dict (the player supplies the action, so it is never fed to a
    policy), client_state returns None → the streamer falls back to a server JPEG of the 3D view."""
    env = make_env("doom_basic", render_mode="rgb_array", play_scale=1)
    try:
        assert int(env.action_space.n) == 4  # Discrete(4)
        obs, _ = env.reset(seed=0)
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        assert client_state(env, obs) is None  # Dict/image obs → server JPEG, not a client-render state
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # the raw 3D colour frame
        image, w, h = encode_frame(rgb)
        assert isinstance(image, str) and len(image) > 0 and w > 0 and h > 0
    finally:
        env.close()


# -- /api/system + the GPU-training gate ------------------------------------


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train VizDoom")
def test_vizdoom_training_is_gated_without_a_gpu() -> None:
    """On a CPU-only machine, starting VizDoom training is rejected with a clear 400 (image obs needs
    CnnPolicy/CUDA); the UI also disables Run. Mirrors Atari (G4b) / CarRacing (G3c-train)."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "doom_basic", "algo": "ppo", "seed": 1, "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "GPU" in resp.json()["detail"]


def test_vizdoom_skill_bands_span_floor_to_solved() -> None:
    """Bands come straight from [min_score, solved_score] — no bespoke band table (Basic [-300, 90])."""
    skill = client.get("/api/skill/doom_basic").json()
    assert skill["min_score"] == -300.0 and skill["max_score"] == 90.0
    assert skill["bands"][0]["min_score"] == -300.0  # weakest band starts at the idle/timeout floor
    assert skill["bands"][-1]["id"] == "superhuman"
