"""Skill-band endpoint — the documented score thresholds for an env's skill meter.

Read-only: thresholds are derived from the registry's ``solved_score`` (see services/skill.py).
The play session does the *rating* of a finished episode internally and pushes it in the
``{type:"play_result"}`` frame; this endpoint lets the UI draw the band scale up front.
"""

from fastapi import APIRouter, HTTPException

from app.schemas.skill import EnvSkill
from app.services.skill import env_skill

router = APIRouter(prefix="/api/skill", tags=["skill"])


@router.get("/{env_id}", response_model=EnvSkill)
async def get_env_skill(env_id: str) -> EnvSkill:
    skill = env_skill(env_id)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Unknown environment '{env_id}'")
    return skill
