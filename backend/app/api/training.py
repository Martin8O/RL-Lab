"""Training control endpoints. The WS metric/status frames are pushed via the
connection manager from the trainer thread; these REST routes drive the lifecycle.
"""

import asyncio

from fastapi import APIRouter, HTTPException

from app.schemas.training import SweepRequest, TrainConfig, TrainStatus
from app.services.training_manager import (
    AlreadyRunningError,
    InvalidConfigError,
    training_manager,
)

router = APIRouter(prefix="/api/train", tags=["training"])


@router.post("/start", response_model=TrainStatus)
async def start_training(config: TrainConfig | None = None) -> TrainStatus:
    # /start runs on the event loop, so capture it here for the worker thread's broadcasts.
    training_manager.bind_loop(asyncio.get_running_loop())
    try:
        return training_manager.start(config or TrainConfig())
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/sweep", response_model=TrainStatus)
async def start_sweep(request: SweepRequest) -> TrainStatus:
    """Launch a seed-sweep (X3): one config queued across N seeds, run back-to-back, each archived with
    a shared experiment_id + its own seed. Returns the status of the first (now running) seed."""
    training_manager.bind_loop(asyncio.get_running_loop())
    try:
        return training_manager.start_sweep(request)
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/pause", response_model=TrainStatus)
async def pause_training() -> TrainStatus:
    return training_manager.pause()


@router.post("/resume", response_model=TrainStatus)
async def resume_training() -> TrainStatus:
    return training_manager.resume()


@router.post("/stop", response_model=TrainStatus)
async def stop_training() -> TrainStatus:
    return training_manager.stop()


@router.get("/status", response_model=TrainStatus)
async def get_training_status() -> TrainStatus:
    return training_manager.status()
