"""Play-leaderboard contracts — defined once here (pydantic), mirrored in frontend types.ts.

Separate from the training all-time best (schemas/highscores.py): this is a *named* hall of
fame for interactive play sessions, kept per environment and split into two boards —
``human`` (you, at the keyboard) and ``ai`` (a checkpoint playing itself). Persisted to a
gitignored ``data/play_scores.json`` (see services/play_scores.py) so it is per-device and
survives restarts. Each board keeps the top :data:`TOP_N` entries, best first.

For the AI board the "name" is a model identifier (the checkpoint label); ``model_id`` is the
checkpoint id, so the board can keep one row per distinct model (best score wins).
"""

from typing import Literal

from pydantic import BaseModel, Field

# How many entries each board keeps *and* shows (the frontend renders exactly this many). Kept
# equal to the displayed count so "did I make the board?" matches what the user sees — a score
# only qualifies if a slot is free or it beats the lowest visible entry.
TOP_N = 5

PlayCategory = Literal["human", "ai"]


class PlayScoreEntry(BaseModel):
    """One leaderboard row: who, what they scored, and when."""

    name: str
    score: float
    steps: int
    achieved_at: str  # ISO-8601 UTC timestamp
    # AI rows only: which checkpoint played (id for de-duping, algo for the badge).
    model_id: str | None = None
    algo: str | None = None


class PlayScores(BaseModel):
    """Both boards for one env — returned by GET /api/playscores/{env_id}."""

    env_id: str
    human: list[PlayScoreEntry] = Field(default_factory=list)
    ai: list[PlayScoreEntry] = Field(default_factory=list)


class PlayScoreSubmit(BaseModel):
    """Submit a finished session's score to a board (POST /api/playscores/{env_id})."""

    category: PlayCategory
    name: str
    score: float
    steps: int = 0
    model_id: str | None = None
    algo: str | None = None


class PlayScoreResult(BaseModel):
    """Outcome of a submit: the updated boards + where (if at all) the entry landed."""

    scores: PlayScores
    qualified: bool          # did the score make the top-N board?
    rank: int | None = None  # 1-based position on its board when qualified, else None
