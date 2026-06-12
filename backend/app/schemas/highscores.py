"""High-score contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

A high score is the *all-time best* score an env has ever reached on this machine. It is
persisted to a gitignored ``data/highscores.json`` (see services/highscores.py) so it is
per-device and survives server restarts. ``HighScore`` doubles as the WS push frame
(``{type:"highscore"}``) and the REST response, mirroring how PreviewState/TrainStatus carry
a ``type`` discriminator on both channels.
"""

from typing import Literal

from pydantic import BaseModel

from app.schemas.training import Algo


class HighScoreMeta(BaseModel):
    """How a high score was achieved — enough to understand and reproduce the run.

    ``generation`` is set for neuroevolution, ``iteration`` for PPO (the other stays None),
    matching the prompt's "generation/iteration" provenance.
    """

    algo: Algo
    seed: int
    generation: int | None = None
    iteration: int | None = None
    achieved_at: str  # ISO-8601 UTC timestamp


class HighScore(BaseModel):
    """The persisted all-time best for one environment.

    Pushed over WS as {type:"highscore"} whenever a run beats the stored best, and returned
    by GET /api/highscores[/{env_id}].
    """

    type: Literal["highscore"] = "highscore"
    env_id: str
    score: float
    meta: HighScoreMeta
