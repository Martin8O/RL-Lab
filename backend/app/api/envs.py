from fastapi import APIRouter, HTTPException

from app.envs.registry import EnvSpec, get_env, list_envs

router = APIRouter(prefix="/api/envs", tags=["envs"])


@router.get("", response_model=list[EnvSpec])
def get_envs() -> list[EnvSpec]:
    return list_envs()


@router.get("/{env_id}", response_model=EnvSpec)
def get_env_by_id(env_id: str) -> EnvSpec:
    spec = get_env(env_id)
    if spec is None:
        raise HTTPException(status_code=404, detail=f"Environment '{env_id}' not found")
    return spec
