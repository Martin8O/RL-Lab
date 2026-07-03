"""Analysis endpoints (Phase X / DataLab) — server-computed summary statistics for finished runs.

The backend holds the *full* raw metric history on disk (``data/runs/<id>/metrics.json``); the frontend
store is only a capped live ring buffer. So the summary numbers a comparison table needs are computed
here, from disk, as the source of truth. This route wires the pure engine (:mod:`app.services.analysis.stats`)
to the run store: load each requested run, look up its env's skill range, reduce to a
:class:`~app.schemas.analysis.RunSummary`. Unknown run ids are skipped (one object per *found* run).
"""

from typing import Annotated

from fastapi import APIRouter, Query

from app.envs.registry import get_env
from app.schemas.analysis import RunSummary
from app.services.analysis.stats import summarize
from app.services.runs import run_store

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.get("/summary", response_model=list[RunSummary])
async def run_summaries(
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> list[RunSummary]:
    """Summary statistics for each requested run (``?run_ids=a&run_ids=b``).

    Reads full history from disk (the store backfills the X1 canonical axes on load) and normalizes each
    curve against its env's ``[min_score, solved_score]``. Missing / unreadable runs are omitted rather
    than failing the whole request, so a partially-stale id list still returns what it can.
    """
    summaries: list[RunSummary] = []
    for rid in run_ids or []:
        detail = run_store.get(rid)
        if detail is None:
            continue
        spec = get_env(detail.config.env_id)
        min_score = spec.min_score if spec else 0.0
        solved_score = spec.solved_score if spec else 0.0
        summaries.append(
            summarize(
                run_id=detail.meta.id,
                env_id=detail.config.env_id,
                algo=detail.config.algo,
                seed=detail.config.seed,
                frames=detail.metrics,
                min_score=min_score,
                solved_score=solved_score,
            )
        )
    return summaries
