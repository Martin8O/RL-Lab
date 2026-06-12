"""Checkpoint endpoints — save the current run, list/load/export/delete saved slots.

Save and load go through the training manager (it holds the live model snapshot and owns the
run lifecycle); list/export/delete are pure store reads served straight from the checkpoint
store. Artifacts live only under the gitignored ``data/checkpoints/`` and are never committed.
"""

import asyncio

from fastapi import APIRouter, HTTPException, Response

from app.schemas.checkpoints import CheckpointMeta, CheckpointSaveRequest
from app.schemas.training import TrainStatus
from app.services.checkpoints import checkpoint_store
from app.services.training_manager import (
    AlreadyRunningError,
    CheckpointNotFoundError,
    InvalidConfigError,
    NothingToSaveError,
    training_manager,
)

router = APIRouter(prefix="/api/checkpoints", tags=["checkpoints"])


@router.get("", response_model=list[CheckpointMeta])
async def list_checkpoints() -> list[CheckpointMeta]:
    return checkpoint_store.list()


@router.post("", response_model=CheckpointMeta)
async def save_checkpoint(body: CheckpointSaveRequest | None = None) -> CheckpointMeta:
    try:
        return training_manager.save_checkpoint(body.label if body else None)
    except NothingToSaveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/{checkpoint_id}/load", response_model=TrainStatus)
async def load_checkpoint(checkpoint_id: str) -> TrainStatus:
    # /load spins up a worker thread, so capture the loop here for its broadcasts.
    training_manager.bind_loop(asyncio.get_running_loop())
    try:
        return training_manager.load_checkpoint(checkpoint_id)
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{checkpoint_id}/export")
async def export_checkpoint(checkpoint_id: str) -> Response:
    result = checkpoint_store.export_zip(checkpoint_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    data, filename = result
    return Response(
        content=data,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/{checkpoint_id}", status_code=204)
async def delete_checkpoint(checkpoint_id: str) -> Response:
    if not checkpoint_store.delete(checkpoint_id):
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return Response(status_code=204)
