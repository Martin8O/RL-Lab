"""Run-history endpoints — list finished runs, fetch one for overlay, delete one.

These are pure store reads/writes served straight from the run store; runs are recorded
automatically by the training manager when a run reaches a terminal state. Artifacts live
only under the gitignored ``data/runs/`` and are never committed.
"""

from fastapi import APIRouter, HTTPException, Response

from app.schemas.runs import (
    BulkDeleteRequest,
    BulkDeleteResult,
    GroupRequest,
    RunDetail,
    RunMeta,
    RunMetaPatch,
)
from app.services.runs import run_store

router = APIRouter(prefix="/api/runs", tags=["runs"])


@router.get("", response_model=list[RunMeta])
async def list_runs() -> list[RunMeta]:
    return run_store.list()


# Bulk curation ops (X7) are POST on literal sub-paths so they don't collide with the ``/{run_id}``
# item routes; they're defined before ``/{run_id}`` for clarity (FastAPI matches by method + literal
# path first regardless of order).


@router.post("/group", response_model=list[RunMeta])
async def group_runs(req: GroupRequest) -> list[RunMeta]:
    """Tag the given runs into one named experiment (or ungroup them when ``experiment_id`` is null) —
    sidecar-only edits. Returns the updated metas (missing ids are skipped)."""
    fields = {"experiment_id": req.experiment_id, "experiment_label": req.experiment_label}
    return [m for rid in req.run_ids if (m := run_store.update_meta(rid, fields)) is not None]


@router.post("/delete", response_model=BulkDeleteResult)
async def bulk_delete_runs(req: BulkDeleteRequest) -> BulkDeleteResult:
    """Delete several runs at once (bulk / whole-experiment), returning how many existed."""
    return BulkDeleteResult(deleted=run_store.delete_many(req.run_ids))


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: str) -> RunDetail:
    detail = run_store.get(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return detail


@router.patch("/{run_id}", response_model=RunMeta)
async def patch_run(run_id: str, patch: RunMetaPatch) -> RunMeta:
    """Edit a run's curation fields (label / note / experiment tag / excluded) — meta.json sidecar only.
    A partial update: only the fields present in the body are applied."""
    updated = run_store.update_meta(run_id, patch.model_dump(exclude_unset=True))
    if updated is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return updated


@router.delete("/{run_id}", status_code=204)
async def delete_run(run_id: str) -> Response:
    if not run_store.delete(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    return Response(status_code=204)
