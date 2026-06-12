"""Preview control endpoints. Frames themselves are pushed over WS by the streamer;
these REST routes read and update the visual on/off toggle and the speed multiplier.
"""

import asyncio

from fastapi import APIRouter

from app.schemas.preview import PreviewConfig, PreviewState
from app.services.preview_streamer import preview_streamer

router = APIRouter(prefix="/api/preview", tags=["preview"])


@router.get("", response_model=PreviewState)
async def get_preview() -> PreviewState:
    return preview_streamer.state()


@router.post("", response_model=PreviewState)
async def update_preview(config: PreviewConfig) -> PreviewState:
    # Capture the loop so the streamer's worker-thread broadcasts can reach it even if a
    # toggle arrives before any training run has started.
    preview_streamer.bind_loop(asyncio.get_running_loop())
    if config.visual is not None:
        preview_streamer.set_visual(config.visual)
    if config.speed is not None:
        preview_streamer.set_speed(config.speed)
    return preview_streamer.state()
