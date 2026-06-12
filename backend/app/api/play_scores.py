"""Play-leaderboard endpoints — read the Human/AI boards for an env and submit a score.

Read-only ``GET`` for the bottom-panel boards + the skill-meter record markers; ``POST`` to
place a finished session (name supplied by the UI for human runs, the model label for AI).
"""

from fastapi import APIRouter

from app.schemas.play_scores import PlayScoreResult, PlayScores, PlayScoreSubmit
from app.services.play_scores import play_scores

router = APIRouter(prefix="/api/playscores", tags=["playscores"])


@router.get("/{env_id}", response_model=PlayScores)
async def get_play_scores(env_id: str) -> PlayScores:
    return play_scores.get(env_id)


@router.post("/{env_id}", response_model=PlayScoreResult)
async def submit_play_score(env_id: str, body: PlayScoreSubmit) -> PlayScoreResult:
    return play_scores.submit(
        env_id,
        body.category,
        body.name,
        body.score,
        steps=body.steps,
        model_id=body.model_id,
        algo=body.algo,
    )
