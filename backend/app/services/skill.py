"""Skill evaluation — turn a play-session score into a beginner-friendly band.

Thresholds are **data-driven from the env registry**: each band's lower bound is a fixed
fraction of the env's ``solved_score`` (100% of the goal). So CartPole (solved=500) yields
child<50, below-avg<150, avg<300, above-avg<475, superhuman≥475, and any future game with a
``solved_score`` gets sensibly scaled bands for free — no per-env skill config to maintain.

The fraction scheme assumes a non-negative score that climbs toward ``solved_score`` (true of
CartPole and the Atari/Box2D envs on the desktop roadmap). A symmetric-score game like Pong
(−21…21) would want a custom band table; when that lands, add an optional ``skill_bands``
override to :class:`~app.envs.registry.EnvSpec` and prefer it here. Until then this stays
deliberately small and registry-sourced.
"""

from app.envs.registry import get_env
from app.schemas.skill import EnvSkill, SkillBand, SkillBandId, SkillRating

# Band lower bounds as a fraction of the env's solved_score, weakest → strongest. Documented
# here (and surfaced verbatim via GET /api/skill/{env}) so the thresholds are explicit.
_BAND_FRACTIONS: list[tuple[SkillBandId, float]] = [
    ("child", 0.00),
    ("below_average", 0.10),
    ("average", 0.30),
    ("above_average", 0.60),
    ("superhuman", 0.95),
]


def env_skill(env_id: str, min_scale: float = 1.0) -> EnvSkill | None:
    """The concrete band thresholds for ``env_id``, or ``None`` if the env is unknown.

    Bands span the env's ``[min_score, solved_score]`` range, so a shaped env that starts in
    the red (LunarLander, ``min_score=-100``) gets bands that climb through the negatives
    instead of all bunching at 0.

    ``min_scale`` widens the floor for longer **play** episodes: an env played at
    ``play_step_scale=3`` runs 3× the steps, so its failure floor (≈ −1 × max_steps) is ~3×
    deeper while a *success* score is unchanged — pass the play scale so the meter/rating span
    matches the longer episode (default 1.0 = the standard training span).
    """
    spec = get_env(env_id)
    if spec is None:
        return None
    max_score = spec.solved_score
    min_score = spec.min_score * min_scale
    span = max_score - min_score
    bands = [
        SkillBand(id=band_id, min_score=round(min_score + frac * span, 4))
        for band_id, frac in _BAND_FRACTIONS
    ]
    return EnvSkill(env_id=env_id, max_score=max_score, min_score=min_score, bands=bands)


def rate(env_id: str, score: float, min_scale: float = 1.0) -> SkillRating | None:
    """Rate a finished session's ``score`` against ``env_id``'s bands.

    Returns the highest band whose ``min_score`` the score reaches (the lowest band starts at
    the env's ``min_score``, so a score is always rated), plus a clamped fill ratio for the
    meter measured across ``[min_score, solved_score]``. ``None`` only if the env is unknown.
    ``min_scale`` widens the floor for longer play episodes (see :func:`env_skill`).
    """
    skill = env_skill(env_id, min_scale)
    if skill is None:
        return None
    band: SkillBandId = skill.bands[0].id
    for candidate in skill.bands:
        if score >= candidate.min_score:
            band = candidate.id
    span = skill.max_score - skill.min_score
    ratio = (score - skill.min_score) / span if span > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    return SkillRating(band=band, score=score, max_score=skill.max_score, ratio=ratio)
