import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_list_envs() -> None:
    response = client.get("/api/envs")
    assert response.status_code == 200
    envs = response.json()
    assert isinstance(envs, list)
    assert len(envs) >= 1
    assert any(e["id"] == "cartpole" for e in envs)


def test_cartpole_detail() -> None:
    response = client.get("/api/envs/cartpole")
    assert response.status_code == 200
    env = response.json()
    assert env["id"] == "cartpole"
    assert env["gym_id"] == "CartPole-v1"
    ppo = env["hyperparams"]["ppo"]
    assert ppo["learning_rate"]["recommended"] == pytest.approx(3e-4)
    assert ppo["gamma"]["recommended"] == pytest.approx(0.99)
    assert ppo["n_steps"]["recommended"] == 2048


def test_cartpole_structure() -> None:
    env = client.get("/api/envs/cartpole").json()
    assert env["obs_type"] == "vector"
    assert env["action_space"] == "discrete"
    assert "ppo" in env["supported_algos"]
    assert "neuroevolution" in env["supported_algos"]
    assert env["hw_requirement"] == "cpu"
    assert env["difficulty"] == "beginner"
    assert env["human_playable"] is True
    assert env["competitive"] is False


def test_unknown_env_returns_404() -> None:
    response = client.get("/api/envs/nonexistent")
    assert response.status_code == 404


# -- G1a: classic-control completion (MountainCar + Acrobot) -----------------


@pytest.mark.parametrize(
    "env_id,gym_id,solved,floor",
    [
        ("mountaincar", "MountainCar-v0", -110.0, -200.0),
        ("acrobot", "Acrobot-v1", -100.0, -500.0),
    ],
)
def test_classic_control_g1a_envs(
    env_id: str, gym_id: str, solved: float, floor: float
) -> None:
    """The two new envs register as discrete/vector CPU envs (data-only, like CartPole)."""
    env = client.get(f"/api/envs/{env_id}").json()
    assert env["gym_id"] == gym_id
    assert env["family"] == "classic_control"
    assert env["obs_type"] == "vector"
    assert env["action_space"] == "discrete"
    assert env["hw_requirement"] == "cpu"
    assert env["supported_algos"] == ["ppo", "neuroevolution", "dqn"]  # + DQN on the discrete envs (S5c)
    assert env["solved_score"] == solved
    # negative-reward envs: the skill floor sits below the solved score (meter fills the red)
    assert env["min_score"] == floor < env["solved_score"]


def test_play_step_scale_extends_short_envs_only() -> None:
    """Short/by-hand-slow envs run longer play episodes (play_step_scale=3); CartPole/MCC keep 1×."""
    scale = {e["id"]: e["play_step_scale"] for e in client.get("/api/envs").json()}
    assert scale["mountaincar"] == 3 and scale["acrobot"] == 3 and scale["pendulum"] == 3
    assert scale["lunarlander"] == 3  # Box2D landing takes a while by hand — give a human 3× the steps
    assert scale["cartpole"] == 1  # balancing has no fixed goal — already runs to the step cap
    assert scale["mountaincarcontinuous"] == 1  # already 999 steps — long enough


def test_standard_hyperparams_shared_across_vector_envs() -> None:
    """Every vector/discrete env exposes the identical PPO + neuroevolution param surface.

    The DQN block (S5c) is intentionally **per-env tuned** (rl-zoo3 recipes — CartPole's fast target
    sync vs the others), so it is excluded from this shared-surface check.
    """
    ids = ["cartpole", "lunarlander", "mountaincar", "acrobot"]
    surfaces = [
        {k: v for k, v in client.get(f"/api/envs/{i}").json()["hyperparams"].items() if k != "dqn"}
        for i in ids
    ]
    assert all(s == surfaces[0] for s in surfaces[1:])


# -- G1b: classic-control continuous-action members (Pendulum + MountainCarContinuous) ------


@pytest.mark.parametrize(
    "env_id,gym_id,solved,floor,difficulty",
    [
        ("pendulum", "Pendulum-v1", -150.0, -1600.0, "intermediate"),
        ("mountaincarcontinuous", "MountainCarContinuous-v0", 90.0, 0.0, "advanced"),
    ],
)
def test_classic_control_g1b_continuous_envs(
    env_id: str, gym_id: str, solved: float, floor: float, difficulty: str
) -> None:
    """The continuous (box) members register as vector/CPU envs with a `box` action space — the
    int→box seam is in the trainers/play loop, not the registry, so these stay data rows."""
    env = client.get(f"/api/envs/{env_id}").json()
    assert env["gym_id"] == gym_id
    assert env["family"] == "classic_control"
    assert env["obs_type"] == "vector"
    assert env["action_space"] == "box"  # the distinguishing trait of G1b
    assert env["hw_requirement"] == "cpu"
    assert env["supported_algos"] == ["ppo", "neuroevolution", "sac", "td3"]  # + SAC/TD3 on the box envs (S5a/S5b)
    assert env["solved_score"] == solved
    assert env["min_score"] == floor
    assert env["difficulty"] == difficulty
    # Same shared PPO + neuroevolution param surface as the discrete envs (only the env differs). The
    # DQN block is excluded: these box envs don't offer DQN, and CartPole's DQN block is per-env tuned.
    surface = {k: v for k, v in env["hyperparams"].items() if k != "dqn"}
    cartpole_surface = {
        k: v for k, v in client.get("/api/envs/cartpole").json()["hyperparams"].items() if k != "dqn"
    }
    assert surface == cartpole_surface


# -- per-env ★ recommended algorithm (the picker marker) ---------------------


def test_recommended_algo_is_always_a_supported_algo() -> None:
    """Every env exposes a single ★ recommended algorithm, and it is always one the env supports
    (a curated value that wasn't in supported_algos would mark a non-existent picker option)."""
    for env in client.get("/api/envs").json():
        rec = env["recommended_algo"]
        assert rec is not None, env["id"]
        assert rec in env["supported_algos"], (env["id"], rec, env["supported_algos"])


def test_recommended_algo_defaults_to_first_supported() -> None:
    """When an env doesn't curate one, the recommendation is supported_algos[0] (the PPO baseline)."""
    by_id = {e["id"]: e for e in client.get("/api/envs").json()}
    # Not curated → falls back to the first supported algo.
    assert by_id["cartpole"]["recommended_algo"] == "ppo"
    assert by_id["lunarlander"]["recommended_algo"] == "ppo"
    # chess is AlphaZero-only, so its sole supported algo is also the recommendation.
    assert by_id["chess"]["recommended_algo"] == "alphazero"


@pytest.mark.parametrize(
    "env_id,expected",
    [
        # off-policy continuous control wins on the swing-up + the MuJoCo robots
        ("pendulum", "sac"),
        ("halfcheetah", "sac"),
        ("humanoid", "sac"),
        # the sparse exploration trap — population search reaches the flag where PPO stalls
        ("mountaincarcontinuous", "neuroevolution"),
        # tabular Q-learning is the natural fit for the Toy-Text grid-worlds
        ("frozenlake", "q_learning"),
        ("taxi", "q_learning"),
        ("cliffwalking", "q_learning"),
        # AlphaZero is the board-game algorithm
        ("tictactoe", "alphazero"),
        ("connect_four", "alphazero"),
    ],
)
def test_recommended_algo_curated_overrides(env_id: str, expected: str) -> None:
    """Where a non-PPO algo measurably wins (recorded findings), the env curates its ★ recommendation."""
    env = client.get(f"/api/envs/{env_id}").json()
    assert env["recommended_algo"] == expected
