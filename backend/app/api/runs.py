"""Run-history endpoints — list finished runs, fetch one for overlay, delete one.

These are pure store reads/writes served straight from the run store; runs are recorded
automatically by the training manager when a run reaches a terminal state. Artifacts live
only under the gitignored ``data/runs/`` and are never committed.
"""

from fastapi import APIRouter, HTTPException, Response

from app.schemas.runs import RunDetail, RunMeta
from app.services.runs import run_store

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("", response_model=list[RunMeta])
async def list_runs() -> list[RunMeta]:
    return run_store.list()


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> RunDetail:
    detail = run_store.get(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str) -> Response:
    if not run_store.delete(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return Response(status_code=204)
