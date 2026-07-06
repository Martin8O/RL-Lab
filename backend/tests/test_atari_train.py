"""G4b — Atari GPU training seam: the shared CnnPolicy + AtariWrapper/frame-stack + CUDA path.

Covers the data (the flipped ``train_implemented`` gate), the shared ``app/envs/atari.make_atari``
vec builder (obs/action shapes + the raw-colour render), the trainer's image-obs policy/device
branch, and the CNN preview snapshot that replaces the MLP-only numpy forward for image envs. The
CUDA-dependent pieces are skipped on a CPU-only machine; the shape/gating pieces run anywhere.
"""

import numpy as np
import pytest
from app.envs.atari import make_atari
from app.envs.registry import get_env
from app.main import app
from app.schemas.training import TrainConfig
from app.services import trainer_ppo
from app.services.system_info import gpu_available
from fastapi.testclient import TestClient

client = TestClient(app)


# -- registry: the gate is flipped on for Atari (only) ----------------------


def test_pong_registered_and_trainable() -> None:
    spec = get_env("pong")
    assert spec is not None
    assert spec.gym_id == "ALE/Pong-v5"
    assert spec.family == "atari"
    assert spec.obs_type == "image"
    assert spec.supported_algos == ["ppo", "dqn", "qrdqn"]  # pixels → no evolution/Q-learning; dqn = DQN's birthplace (S5c), qrdqn = distributional DQN (S5e)
    assert spec.hw_requirement == "gpu"  # CnnPolicy needs CUDA; CPU box still gates Run
    assert spec.train_implemented is True  # G4b built the trainer — un-gated on a GPU box
    assert spec.make_kwargs == {"full_action_space": True}  # 18 actions → shared keymap + policy parity


def test_carracing_now_trainable() -> None:
    """G4b was Atari-only (CarRacing stayed gated then); G3c-train later built CarRacing's own
    (non-Atari) CnnPolicy + box pipeline and flipped its flag on. See test_carracing_train.py."""
    spec = get_env("carracing")
    assert spec is not None and spec.train_implemented is True


# -- the shared Atari vec builder (CPU-safe: building envs needs no GPU) -----


def test_make_atari_obs_and_action_shapes() -> None:
    """make_atari yields the 84×84×4 frame stack + the full 18-action space, and renders the RAW
    colour frame (WarpFrame only rewrites the observation) so the preview JPEG stays full-colour."""
    venv = make_atari("ALE/Pong-v5", 2, make_kwargs={"full_action_space": True})
    try:
        assert venv.observation_space.shape == (84, 84, 4)  # 4 stacked grayscale frames
        assert venv.action_space.n == 18  # full_action_space → shared keymap / policy parity
        obs = venv.reset()
        assert obs.shape == (2, 84, 84, 4) and obs.dtype == np.uint8
        rgb = np.asarray(venv.render(mode="rgb_array"), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # raw colour, NOT the grayscale obs
    finally:
        venv.close()


# -- the trainer's image-obs CnnPolicy/CUDA branch + the CNN preview snapshot --


@pytest.mark.skipif(not gpu_available(), reason="CnnPolicy training needs a CUDA device")
def test_atari_builds_cnn_on_cuda_and_snapshot_predicts() -> None:
    """_build_model picks CnnPolicy + device=cuda for an image env; a few steps train; the decoupled
    preview snapshot dispatches to the CPU torch forward (not the MLP numpy one) and returns an int."""
    cfg = TrainConfig(env_id="pong", algo="ppo", seed=7, total_timesteps=256)
    cfg.hyperparams.n_steps = 64
    cfg.hyperparams.batch_size = 64
    model = trainer_ppo._build_model(cfg, "ALE/Pong-v5")
    try:
        assert type(model.policy).__name__ == "ActorCriticCnnPolicy"
        assert next(model.policy.parameters()).device.type == "cuda"
        assert model.observation_space.shape == (4, 84, 84)  # SB3 VecTransposeImage → channels-first
        model.learn(total_timesteps=128)
        predict = trainer_ppo._build_preview_predict(model)  # image → CNN snapshot path
        action = predict(model.observation_space.sample())
        assert isinstance(action, int) and 0 <= action < 18
    finally:
        if model.env is not None:
            model.env.close()


# -- the training gate (mirrors CarRacing/BipedalWalker) --------------------


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train Atari now")
def test_atari_training_rejected_without_a_gpu() -> None:
    """On a CPU-only box, starting Pong training is rejected with a clear 400 (needs a CUDA GPU)."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "pong", "algo": "ppo", "seed": 1, "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
    assert "GPU" in resp.json()["detail"]


def test_atari_gating_tracks_gpu_presence() -> None:
    """Documents the un-gate: with the trainer built (``train_implemented``), Atari is gated *iff*
    no CUDA device is present — so a GPU box admits the run and a CPU box still rejects it."""
    spec = get_env("pong")
    assert spec is not None
    not_implemented = spec.train_implemented is False
    needs_absent_gpu = spec.hw_requirement == "gpu" and not gpu_available()
    gated = not_implemented or needs_absent_gpu
    assert gated is (not gpu_available())
