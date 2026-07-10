"""System capability endpoint (G4a, R1).

Exposes runtime facts the UI uses to gate features — whether a CUDA GPU is available (decides if
GPU-only training of Atari and other image-obs envs can run here) and whether the optional ale-py
package is installed (decides if the Atari family is usable at all — ADR-101 made it opt-in).
Defined as a sync route so FastAPI runs the (lazy, one-off) probes in a threadpool rather than
blocking the event loop.
"""

from fastapi import APIRouter

from app.schemas.system import SystemInfo
from app.services.system_info import atari_available, gpu_available

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("", response_model=SystemInfo)
def get_system() -> SystemInfo:
    return SystemInfo(gpu_available=gpu_available(), atari_available=atari_available())
