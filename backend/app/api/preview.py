"""Preview control endpoints. Frames themselves are pushed over WS by the streamer;
these REST routes read and update the visual on/off toggle and the speed multiplier.
"""

import asyncio

from fastapi import APIRouter

from app.schemas.preview import PreviewConfig, PreviewState, PreviewWatch
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


@router.post("/watch", response_model=PreviewState)
async def watch_preview(req: PreviewWatch) -> PreviewState:
    # "Watch the ecosystem" (G7b): preview an env with no training attached. With a ``checkpoint_id`` it
    # is **Watch AI** — the saved multi-agent model plays itself (both species' brains); without one it's
    # a random rollout. Bind the loop here too — a watch can be the first thing that touches the streamer.
    preview_streamer.bind_loop(asyncio.get_running_loop())
    if req.on:
        # Load the saved per-species policies off the event loop (SB3 load is blocking) BEFORE starting
        # the watch, so the swarm renders the trained ecosystem from the first frames.
        predicts = None
        if req.checkpoint_id:
            from app.services.checkpoints import checkpoint_store
            from app.services.trainer_tag import load_species_predicts

            loaded = await asyncio.to_thread(checkpoint_store.load, req.checkpoint_id)
            if loaded is not None:
                predicts = await asyncio.to_thread(load_species_predicts, loaded.blob)
        preview_streamer.start_watch(req.env_id)
        if predicts:
            preview_streamer.set_policies(predicts)
    else:
        preview_streamer.stop_watch()
    return preview_streamer.state()
