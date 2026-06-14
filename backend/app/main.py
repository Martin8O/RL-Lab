import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.checkpoints import router as checkpoints_router
from app.api.envs import router as envs_router
from app.api.highscores import router as highscores_router
from app.api.play import router as play_router
from app.api.play_scores import router as play_scores_router
from app.api.preview import router as preview_router
from app.api.runs import router as runs_router
from app.api.skill import router as skill_router
from app.api.training import router as training_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services.connection_manager import manager
from app.services.play_session import play_session
from app.services.preview_streamer import preview_streamer
from app.services.training_manager import training_manager

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Backend starting (version %s)", settings.app_version)
    # Let the trainer + preview threads marshal their broadcasts onto this loop.
    loop = asyncio.get_running_loop()
    training_manager.bind_loop(loop)
    preview_streamer.bind_loop(loop)
    play_session.bind_loop(loop)
    yield
    logger.info("Backend shutting down")


app = FastAPI(title="RL Dashboard", version=settings.app_version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(envs_router)
app.include_router(training_router)
app.include_router(preview_router)
app.include_router(highscores_router)
app.include_router(checkpoints_router)
app.include_router(runs_router)
app.include_router(play_router)
app.include_router(play_scores_router)
app.include_router(skill_router)


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": settings.app_version}


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Human play input arrives as a JSON {type:"action", action:<int>} frame and is
            # routed to the live play session; anything else keeps the A3 echo behaviour.
            message = _parse_json(data)
            if isinstance(message, dict) and message.get("type") == "action":
                _route_action(message)
                continue
            await manager.send(ws, {"echo": data})
    except WebSocketDisconnect:
        manager.disconnect(ws)


def _parse_json(data: str) -> object | None:
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        return None


def _route_action(message: dict) -> None:
    """Forward a human ``{type:"action"}`` frame to the play session (ignoring malformed ones).

    The action is passed through as received — an int/float for a discrete/continuous scalar
    action, or a list of floats for a continuous vector — and the play session interprets it
    against the live env's action space (no int() cast, which would break continuous actions).
    """
    action = message.get("action")
    if isinstance(action, (int, float, list)):
        play_session.submit_action(action)
    else:
        logger.debug("Ignoring malformed action frame: %r", message)
