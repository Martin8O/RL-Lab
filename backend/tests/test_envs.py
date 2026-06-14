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
    assert env["supported_algos"] == ["ppo", "neuroevolution"]
    assert env["solved_score"] == solved
    # negative-reward envs: the skill floor sits below the solved score (meter fills the red)
    assert env["min_score"] == floor < env["solved_score"]


def test_standard_hyperparams_shared_across_vector_envs() -> None:
    """Every vector/discrete env exposes the identical PPO + neuroevolution param surface."""
    ids = ["cartpole", "lunarlander", "mountaincar", "acrobot"]
    surfaces = [client.get(f"/api/envs/{i}").json()["hyperparams"] for i in ids]
    assert all(s == surfaces[0] for s in surfaces[1:])
