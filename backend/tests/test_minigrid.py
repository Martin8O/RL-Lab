"""G2c — MiniGrid family + the FlatObs (image-Dict → vector) seam.

MiniGrid's native observation is a Dict (a 7×7×3 partial-view image + direction + a mission
string). The shared factory wraps the family in ``minigrid.wrappers.FlatObsWrapper``, flattening
it to a length-2835 Box vector, so the SAME MlpPolicy (PPO) / numpy genome (neuroevolution) used
for CartPole apply with no engine change — the same idea as the Toy Text one-hot seam. Unlike
Atari (image obs → CnnPolicy/GPU) these train on CPU now, so the family is NOT GPU-gated. Rendering
stays server-side (the family is not in client_render → JPEG), and play is turn-based.

Covers the registry rows, the FlatObs factory path (+ server-JPEG render), self-truncation (no
native TimeLimit), the sparse-reward skill bands, and a smoke train through both trainers.
"""

import numpy as np
from app.envs.factory import make_env
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.training import EvolutionHyperparams, PPOHyperparams, TrainConfig
from app.services.client_render import client_state, grid_layout
from app.services.preview_streamer import encode_frame
from app.services.train_control import TrainControl
from app.services.trainer_evolution import train_evolution
from app.services.trainer_ppo import train_ppo
from fastapi.testclient import TestClient
from minigrid.wrappers import FlatObsWrapper

client = TestClient(app)

MINIGRID = ["minigrid_empty", "minigrid_fourrooms", "minigrid_doorkey", "minigrid_keycorridor"]


# -- registry ---------------------------------------------------------------


def test_minigrid_family_registered() -> None:
    families = {e.id: e.family for e in list_envs()}
    for eid in MINIGRID:
        spec = get_env(eid)
        assert spec is not None, f"{eid} not registered"
        assert families[eid] == "minigrid"
        assert spec.gym_id.startswith("MiniGrid-")
        assert spec.obs_type == "vector"  # after FlatObsWrapper (applied in the factory)
        assert spec.action_space == "discrete"
        assert spec.supported_algos == ["ppo", "neuroevolution"]  # no q_learning (obs is a vector)
        assert spec.hw_requirement == "cpu"  # FlatObs + MlpPolicy trains on CPU — NOT GPU-gated
        assert spec.turn_based is True
        assert spec.sparse_reward is True  # 0 until the goal → play meter "measures" until episode end
        assert spec.human_playable is True
        assert spec.competitive is False
        # No gym reward_threshold; success pays 1 − 0.9·steps/max ≈ 0.9–0.97 → solved 0.95, floor 0.
        assert spec.solved_score == 0.95 and spec.min_score == 0.0


def test_minigrid_exact_gym_ids() -> None:
    # The KeyCorridor id has NO dashes inside the S3R1 suffix (a common mis-spelling).
    assert get_env("minigrid_keycorridor").gym_id == "MiniGrid-KeyCorridorS3R1-v0"  # type: ignore[union-attr]
    assert get_env("minigrid_empty").gym_id == "MiniGrid-Empty-5x5-v0"  # type: ignore[union-attr]
    assert get_env("minigrid_doorkey").gym_id == "MiniGrid-DoorKey-5x5-v0"  # type: ignore[union-attr]
    assert get_env("minigrid_fourrooms").gym_id == "MiniGrid-FourRooms-v0"  # type: ignore[union-attr]


# -- the FlatObs seam (factory) + server-JPEG render ------------------------


def test_make_env_flattens_minigrid_obs() -> None:
    """The factory applies FlatObsWrapper, so a Dict obs presents as a (2835,) vector + Discrete(7)."""
    for eid in MINIGRID:
        env = make_env(eid)
        try:
            assert isinstance(env, FlatObsWrapper)
            assert env.observation_space.shape == (2835,)  # 7×7×3 image + mission one-hot
            assert int(env.action_space.n) == 7  # turn-left/right, forward, pickup, drop, toggle, done
            obs, _ = env.reset(seed=0)
            assert obs.shape == (2835,)
        finally:
            env.close()


def test_minigrid_renders_server_side_jpeg() -> None:
    """The family is not client-rendered (client_state → None), so env.render() rgb → a server JPEG —
    exactly like Atari, but without a retro skin. render() works despite the obs wrapper."""
    env = make_env("minigrid_fourrooms", render_mode="rgb_array", play_scale=1)
    try:
        obs, _ = env.reset(seed=0)
        assert client_state(env, obs) is None  # → server image render
        assert grid_layout(env) is None  # not a Toy Text client-render grid
        rgb = np.asarray(env.render(), dtype=np.uint8)
        assert rgb.ndim == 3 and rgb.shape[2] == 3  # FourRooms renders 608×608×3
        image, w, h = encode_frame(rgb)
        assert isinstance(image, str) and len(image) > 0 and w > 0 and h > 0
    finally:
        env.close()


def test_minigrid_self_truncates_without_a_native_timelimit() -> None:
    """MiniGrid has no gym TimeLimit (spec.max_episode_steps is None) but truncates itself at its
    internal max_steps — so no episode_step_limit is needed (unlike CliffWalking) and a run can't hang."""
    env = make_env("minigrid_empty")
    try:
        assert env.spec is not None and env.spec.max_episode_steps is None
        env.reset(seed=0)
        done = False
        for _ in range(500):  # well past Empty-5x5's internal cap
            _, _, term, trunc, _ = env.step(0)  # spin in place (turn left) — never reaches the goal
            if term or trunc:
                done = True
                break
        assert done, "MiniGrid must self-truncate even when the goal is never reached"
    finally:
        env.close()


# -- sparse-reward skill bands ----------------------------------------------


def test_minigrid_skill_bands_span_zero_to_solved() -> None:
    """Positive sparse reward → bands climb [0, 0.95] like FrozenLake (no negative floor)."""
    skill = client.get("/api/skill/minigrid_empty").json()
    assert skill["min_score"] == 0.0 and skill["max_score"] == 0.95
    assert skill["bands"][0]["min_score"] == 0.0  # weakest band starts at the floor
    assert skill["bands"][-1]["id"] == "superhuman"


# -- the seam through both trainers -----------------------------------------


def test_ppo_trains_on_minigrid_via_flatobs() -> None:
    """PPO's MlpPolicy consumes the flattened vector — MiniGrid trains with no engine change, on CPU."""
    metrics: list = []
    train_ppo(
        TrainConfig(
            env_id="minigrid_empty", algo="ppo", seed=0, total_timesteps=512,
            hyperparams=PPOHyperparams(n_steps=128, batch_size=64),
        ),
        "MiniGrid-Empty-5x5-v0", TrainControl(),
        metrics.append, lambda _p: None,
    )
    assert len(metrics) == 4  # 512 / 128 rollouts
    assert metrics[-1].timesteps == 512


def test_evolution_runs_on_minigrid() -> None:
    """Neuroevolution's numpy genome consumes the flattened vector too (weak on the big obs, but it
    runs end-to-end without crashing — the user-selected second algorithm for the family)."""
    frames: list = []
    train_evolution(
        TrainConfig(
            env_id="minigrid_empty", algo="neuroevolution", seed=0,
            evolution=EvolutionHyperparams(
                population_size=4, top_k_parents=2, mutation_rate=0.1,
                crossover_rate=0.5, generations=2, episodes=1,
            ),
        ),
        "MiniGrid-Empty-5x5-v0", TrainControl(),
        frames.append, lambda _f: None,
    )
    assert [f.generation for f in frames] == [1, 2]
