"""E1 — skill evaluation: registry-derived bands + the score→band evaluator + REST surface."""

from app.main import app
from app.schemas.skill import EnvSkill
from app.services.skill import env_skill, rate
from fastapi.testclient import TestClient

client = TestClient(app)


# -- bands derived from the registry ----------------------------------------


def test_cartpole_band_thresholds() -> None:
    skill = env_skill("cartpole")
    assert skill is not None
    assert skill.max_score == 500.0
    # 0 / 10% / 30% / 60% / 95% of 500
    assert [(b.id, b.min_score) for b in skill.bands] == [
        ("child", 0.0),
        ("below_average", 50.0),
        ("average", 150.0),
        ("above_average", 300.0),
        ("superhuman", 475.0),
    ]


def test_env_skill_unknown_env_is_none() -> None:
    assert env_skill("does-not-exist") is None
    assert rate("does-not-exist", 100.0) is None


# -- the evaluator ----------------------------------------------------------


def test_rate_maps_scores_to_bands() -> None:
    cases = {
        30.0: "child",
        49.99: "child",
        50.0: "below_average",
        149.0: "below_average",
        150.0: "average",
        299.0: "average",
        300.0: "above_average",
        474.0: "above_average",
        475.0: "superhuman",
        500.0: "superhuman",
    }
    for score, band in cases.items():
        rating = rate("cartpole", score)
        assert rating is not None and rating.band == band, score


def test_rate_ratio_clamped_to_unit_interval() -> None:
    # below 0 → child, ratio floored at 0; above max → superhuman, ratio capped at 1
    low = rate("cartpole", -10.0)
    high = rate("cartpole", 600.0)
    assert low is not None and low.band == "child" and low.ratio == 0.0
    assert high is not None and high.band == "superhuman" and high.ratio == 1.0
    mid = rate("cartpole", 250.0)
    assert mid is not None and mid.ratio == 0.5


# -- REST -------------------------------------------------------------------


def test_skill_endpoint_shape_and_404() -> None:
    body = client.get("/api/skill/cartpole").json()
    EnvSkill.model_validate(body)
    assert body["env_id"] == "cartpole" and body["max_score"] == 500.0
    assert len(body["bands"]) == 5
    assert client.get("/api/skill/does-not-exist").status_code == 404
