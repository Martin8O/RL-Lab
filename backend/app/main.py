import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.api.envs import router as envs_router
from app.api.preview import router as preview_router
from app.api.training import router as training_router
from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.services.connection_manager import manager
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
            await manager.send(ws, {"echo": data})
    except WebSocketDisconnect:
        manager.disconnect(ws)
