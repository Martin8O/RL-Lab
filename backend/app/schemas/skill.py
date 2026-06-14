"""Skill-band contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

A *skill band* maps a finished play session's score to a beginner-friendly rating
(child → below-average → average → above-average → superhuman). The concrete numeric
thresholds are derived per-env from the registry's ``solved_score`` (see services/skill.py),
so adding a game needs no extra skill config — the bands scale with its goal automatically.

``EnvSkill`` is the documented threshold table returned by ``GET /api/skill/{env_id}``;
``SkillRating`` is the evaluation of one finished session, embedded in a play result.
"""

from typing import Literal

from pydantic import BaseModel

# The five fixed rating bands, weakest → strongest. The frontend (E2) maps each id to a
# localized label + meter colour; the backend deals only in ids + numeric thresholds.
SkillBandId = Literal[
    "child", "below_average", "average", "above_average", "superhuman"
]


class SkillBand(BaseModel):
    """One band: its id and the inclusive lower score bound that qualifies for it.

    A score belongs to the highest band whose ``min_score`` it reaches. The upper bound is
    implicitly the next band's ``min_score`` (or the env's max score for the top band).
    """

    id: SkillBandId
    min_score: float


class EnvSkill(BaseModel):
    """The skill-band thresholds for one env — returned by ``GET /api/skill/{env_id}``."""

    env_id: str
    max_score: float  # the env's solved_score (100% of the goal)
    min_score: float  # the score that reads as 0% (0 for CartPole, negative for LunarLander)
    bands: list[SkillBand]  # ascending by min_score; lowest band starts at min_score


class SkillRating(BaseModel):
    """How a finished session scored — the rated band plus the raw figures behind it.

    ``ratio`` is ``(score - min_score) / (max_score - min_score)`` clamped to [0, 1], a
    ready-made fill fraction for the E2 skill meter (it does not have to recompute the band
    boundaries to draw the bar). For an env with ``min_score == 0`` this is just the old
    ``score / max_score``.
    """

    band: SkillBandId
    score: float
    max_score: float
    ratio: float
