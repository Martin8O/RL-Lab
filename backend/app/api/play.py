"""Play-session control endpoints — start/stop one interactive episode + read its status.

Human actions arrive over the WebSocket as ``{type:"action"}`` frames (routed to the session
in main.py); these REST routes drive the lifecycle, mirroring /api/train/*. Rendered frames and
the final skill result are pushed over WS by the session's worker thread.
"""

import asyncio

from fastapi import APIRouter, HTTPException

from app.schemas.play import PlayConfig, PlayStatus
from app.services.play_session import (
    AlreadyPlayingError,
    InvalidPlayConfigError,
    PlayCheckpointNotFoundError,
    play_session,
)

router = APIRouter(prefix="/api/play", tags=["play"])


@router.post("/start", response_model=PlayStatus)
async def start_play(config: PlayConfig | None = None) -> PlayStatus:
    # /start spins up a worker thread, so capture the loop here for its WS broadcasts.
    play_session.bind_loop(asyncio.get_running_loop())
    try:
        return play_session.start(config or PlayConfig())
    except PlayCheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AlreadyPlayingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidPlayConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop", response_model=PlayStatus)
async def stop_play() -> PlayStatus:
    return play_session.stop()


@router.get("/status", response_model=PlayStatus)
async def get_play_status() -> PlayStatus:
    return play_session.status()
