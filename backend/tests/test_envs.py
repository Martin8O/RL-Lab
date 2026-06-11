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
