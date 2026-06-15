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


def test_negative_reward_env_bands_span_the_red() -> None:
    """G1a: MountainCar's bands climb through the negatives, from min_score up to solved_score."""
    skill = env_skill("mountaincar")
    assert skill is not None
    assert skill.min_score == -200.0 and skill.max_score == -110.0
    assert skill.bands[0].min_score == -200.0  # 'child' starts at the floor, not 0
    assert skill.bands[-1].id == "superhuman"
    # a "solved" score (-110) fills the meter; a flat -200 run rates child at 0%
    solved = rate("mountaincar", -110.0)
    floor = rate("mountaincar", -200.0)
    assert solved is not None and solved.band == "superhuman" and solved.ratio == 1.0
    assert floor is not None and floor.band == "child" and floor.ratio == 0.0


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


def test_min_scale_widens_the_floor_for_longer_play_episodes() -> None:
    """A play episode at play_step_scale=3 runs 3× the steps, so its failure floor is ~3× deeper.
    `min_scale` scales min_score (not solved_score), so a mid run rates higher over the longer
    episode and the bands span the wider range — used by play_session for these envs."""
    base = env_skill("mountaincar")
    scaled = env_skill("mountaincar", min_scale=3.0)
    assert base is not None and scaled is not None
    assert base.min_score == -200.0 and scaled.min_score == -600.0  # 3× deeper floor
    assert scaled.max_score == base.max_score == -110.0             # success score unchanged
    assert scaled.bands[0].min_score == -600.0
    # A −400 run is below the standard floor (child, 0%) but mid-range over the longer episode.
    assert rate("mountaincar", -400.0).band == "child"
    longer = rate("mountaincar", -400.0, min_scale=3.0)
    assert longer is not None and longer.band in {"below_average", "average"} and longer.ratio > 0.0


def test_floor_widening_gated_to_step_penalty_envs() -> None:
    """LunarLander is shaped/terminal (a crash ends the episode early ≈ −100), so floor_scales_with_steps
    is False — its play floor must NOT widen with play_step_scale, or a crash rates as a near-success."""
    base = env_skill("lunarlander")
    played = env_skill("lunarlander", min_scale=3.0)
    assert base is not None and played is not None
    assert base.min_score == played.min_score == -200.0  # floor unchanged despite play_step_scale=3
    # a typical crash (≈ −100) stays in a low band, not above_average (which the un-gated −600 floor gave)
    crash = rate("lunarlander", -100.0, min_scale=3.0)
    assert crash is not None and crash.band in {"child", "below_average"}


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
