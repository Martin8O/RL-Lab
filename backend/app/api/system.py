"""System capability endpoint (G4a).

Exposes runtime hardware facts the UI uses to gate features — currently just whether a CUDA GPU
is available, which decides if GPU-only training (Atari and other image-obs envs) can run here.
Defined as a sync route so FastAPI runs the (lazy, one-off) torch probe in a threadpool rather
than blocking the event loop.
"""

from fastapi import APIRouter

from app.schemas.system import SystemInfo
from app.services.system_info import gpu_available

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("", response_model=SystemInfo)
def get_system() -> SystemInfo:
    return SystemInfo(gpu_available=gpu_available())
