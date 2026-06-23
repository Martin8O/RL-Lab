"""G3c-train — CarRacing GPU training: the CNN + continuous-box image seam (the last gated env).

CarRacing is image-obs like Atari but a *different* pipeline: raw 96×96×3 RGB (no AtariWrapper) +
a 2-frame stack, and a continuous ``Box(3)`` action (steer/gas/brake). This covers the data (the
flipped ``train_implemented`` gate), the shared ``image_vec.make_carracing`` builder + the
``make_image_vec`` family dispatcher (obs/action shapes + the raw-colour render), and the
box-aware CNN preview snapshot. The CUDA build is skipped on a CPU-only machine; the shape /
gating / snapshot pieces run anywhere (building a CnnPolicy on CPU needs no GPU — only *training*
is gated).
"""

import numpy as np
import pytest
from app.envs.image_vec import make_carracing, make_image_vec
from app.envs.registry import get_env
from app.main import app
from app.schemas.training import TrainConfig
from app.services import trainer_ppo
from app.services.system_info import gpu_available
from fastapi.testclient import TestClient

client = TestClient(app)


# -- registry: the gate is flipped on (CarRacing trains now) ----------------


def test_carracing_registered_and_trainable() -> None:
    spec = get_env("carracing")
    assert spec is not None
    assert spec.gym_id == "CarRacing-v3"
    assert spec.family == "box2d"
    assert spec.obs_type == "image"
    assert spec.action_space == "box"  # continuous steer/gas/brake — the int→box image seam
    assert spec.supported_algos == ["ppo"]  # pixels → no evolution/Q-learning
    assert spec.hw_requirement == "gpu"  # CnnPolicy needs CUDA; a CPU box still gates Run
    assert spec.train_implemented is True  # G3c-train built the trainer — un-gated on a GPU box
    assert spec.make_kwargs == {"continuous": True}  # the Box(3) variant (vs Discrete(5))
    # image-obs CnnPolicy hyperparam shape, shared with Atari (small rollout + fuller batch)
    assert spec.hyperparams["ppo"]["n_steps"].default == 256
    assert spec.hyperparams["ppo"]["batch_size"].default == 256
    assert spec.hyperparams["ppo"]["n_epochs"].default == 4


# -- the shared CarRacing vec builder + the family dispatcher (CPU-safe) -----


def test_make_carracing_obs_and_action_shapes() -> None:
    """make_carracing yields a 96×96×6 (2-frame-stacked RGB) obs + the continuous Box(3) action,
    and renders the RAW colour frame (the obs preprocessing only rewrites the observation).

    Uses n_envs=1 — the in-process DummyVecEnv path the preview / AI-play loops use (fast + can call
    .render()). The n_envs>1 trainer path uses SubprocVecEnv (parallel cores + isolated pygame; ~4.1×
    throughput) — exercised by the GPU build test below, not here, to keep this case spawn-free."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = make_carracing("CarRacing-v3", 1, make_kwargs={"continuous": True})
    try:
        assert type(venv.venv).__name__ == DummyVecEnv.__name__  # single env stays in-process
        assert venv.observation_space.shape == (96, 96, 6)  # 2 stacked RGB frames (3×2 channels)
        assert venv.action_space.shape == (3,)  # steer, gas, brake
        assert getattr(venv.action_space, "n", None) is None  # continuous (Box), not Discrete
        obs = venv.reset()
        assert obs.shape == (1, 96, 96, 6) and obs.dtype == np.uint8
        rgb = np.asarray(venv.render(mode="rgb_array"), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # raw colour, NOT the stacked obs
    finally:
        venv.close()


def test_make_carracing_trainer_uses_subprocs() -> None:
    """The n_envs>1 trainer path selects SubprocVecEnv (parallel env stepping across cores + a
    process-local pygame per env). Asserts the class without stepping/rendering, then closes fast."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = make_carracing("CarRacing-v3", 2, make_kwargs={"continuous": True})
    try:
        assert type(venv.venv).__name__ == SubprocVecEnv.__name__
    finally:
        venv.close()


def test_make_image_vec_dispatches_by_family() -> None:
    """make_image_vec routes CarRacing → the raw-RGB+stack builder and Atari → the AtariWrapper one,
    so every image-obs caller (trainer/preview/play) gets the right CnnPolicy obs shape."""
    car = make_image_vec(get_env("carracing"), 1)
    try:
        assert car.observation_space.shape == (96, 96, 6)
        assert getattr(car.action_space, "n", None) is None  # box
    finally:
        car.close()
    atari = make_image_vec(get_env("pong"), 1)
    try:
        assert atari.observation_space.shape == (84, 84, 4)
        assert atari.action_space.n == 18  # Atari pipeline unchanged (full_action_space)
    finally:
        atari.close()


# -- the box-aware CNN preview snapshot (CPU-safe: building a CnnPolicy needs no GPU) --


def test_box_cnn_snapshot_returns_a_clipped_vector() -> None:
    """The decoupled preview snapshot for an image+box env (obs rank 3 → the CNN path) returns a
    clipped float action vector, not an int — the continuous-box seam in _build_cnn_predict.

    Built on CPU on purpose: instantiating a CnnPolicy needs no CUDA (only *training* is gated), so
    this runs everywhere and locks the box behaviour without a GPU."""
    from stable_baselines3 import PPO

    venv = make_carracing("CarRacing-v3", 1, make_kwargs={"continuous": True})
    model = PPO("CnnPolicy", venv, device="cpu", n_steps=16, batch_size=16, n_epochs=1)
    try:
        predict = trainer_ppo._build_preview_predict(model)  # image obs → CNN snapshot path
        action = predict(model.observation_space.sample())
        arr = np.asarray(action)
        assert arr.shape == (3,) and arr.dtype == np.float32  # a Box(3) vector, not an int
        low = np.asarray(model.action_space.low, dtype=np.float32)
        high = np.asarray(model.action_space.high, dtype=np.float32)
        assert np.all(arr >= low) and np.all(arr <= high)  # clipped into [low, high]
    finally:
        if model.env is not None:
            model.env.close()


# -- the trainer's image-obs CnnPolicy/CUDA branch (GPU only) ----------------


@pytest.mark.skipif(not gpu_available(), reason="CnnPolicy training needs a CUDA device")
def test_carracing_builds_cnn_on_cuda_and_snapshot_predicts() -> None:
    """_build_model picks CnnPolicy + device=cuda for CarRacing; a few steps train; the decoupled
    preview snapshot returns a clipped Box(3) action over the channels-first stacked obs."""
    cfg = TrainConfig(env_id="carracing", algo="ppo", seed=7, total_timesteps=64)
    cfg.hyperparams.n_steps = 32
    cfg.hyperparams.batch_size = 32
    model = trainer_ppo._build_model(cfg, "CarRacing-v3")
    try:
        assert type(model.policy).__name__ == "ActorCriticCnnPolicy"
        assert next(model.policy.parameters()).device.type == "cuda"
        assert model.observation_space.shape == (6, 96, 96)  # VecTransposeImage → channels-first
        model.learn(total_timesteps=64)
        predict = trainer_ppo._build_preview_predict(model)
        action = np.asarray(predict(model.observation_space.sample()))
        assert action.shape == (3,)  # a continuous steer/gas/brake vector
    finally:
        if model.env is not None:
            model.env.close()


# -- the training gate (mirrors Atari) --------------------------------------


@pytest.mark.skipif(gpu_available(), reason="a CUDA machine can actually train CarRacing now")
def test_carracing_training_rejected_without_a_gpu() -> None:
    """On a CPU-only box, starting CarRacing training is rejected with a clear 400 (needs a CUDA GPU)."""
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
