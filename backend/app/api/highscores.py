"""High-score endpoints. The all-time best per env is also pushed over WS as
``{type:"highscore"}`` when a run beats it; these REST routes let a freshly-loaded client
read the persisted values (e.g. for the TopBar "Best" chip) before any new frame arrives.
"""

from fastapi import APIRouter

from app.schemas.highscores import HighScore
from app.services.highscores import highscores

router = APIRouter(prefix="/api/highscores", tags=["highscores"])


@router.get("", response_model=list[HighScore])
async def list_highscores() -> list[HighScore]:
    return highscores.all()


@router.get("/{env_id}", response_model=HighScore | None)
async def get_highscore(env_id: str) -> HighScore | None:
    # 200 with a null body when the env has no recorded score yet (simpler for the client
    # than a 404 it would have to special-case).
    return highscores.get(env_id)
