"""Analysis endpoints (Phase X / DataLab) — server-computed summary statistics for finished runs.

The backend holds the *full* raw metric history on disk (``data/runs/<id>/metrics.json``); the frontend
store is only a capped live ring buffer. So the summary numbers a comparison table needs are computed
here, from disk, as the source of truth. This route wires the pure engine (:mod:`app.services.analysis.stats`)
to the run store: load each requested run, look up its env's skill range, reduce to a
:class:`~app.schemas.analysis.RunSummary`. Unknown run ids are skipped (one object per *found* run).
"""

from typing import Annotated, Literal

from fastapi import APIRouter, Query, Response

from app.envs.registry import get_env
from app.schemas.analysis import RunSummary
from app.services.analysis import export as export_engine
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


# The Wave-1 export formats (X5) are thin plugins over the shared load → normalize pipeline in
# app.services.analysis.export; one route per format below dispatches through the registry there.


def _export_response(
    fmt: str, run_ids: list[str] | None, pivot: Literal["game", "algo"]
) -> Response:
    """Load the selected runs server-side (full history on disk) and stream the built format as a file
    download. An empty / all-unknown selection still returns a valid (header-only) artifact, not an error,
    so the client can hand it straight to the user."""
    content, media_type, filename = export_engine.export(fmt, run_ids or [], pivot)
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export.csv")
async def export_csv(
    run_ids: Annotated[list[str] | None, Query()] = None,
    pivot: Literal["game", "algo"] = "game",
) -> Response:
    """Tidy long CSV at full resolution — one row per ``(run, frame, metric)``, both pivots derivable."""
    return _export_response("csv", run_ids, pivot)


@router.get("/export.xlsx")
async def export_xlsx(
    run_ids: Annotated[list[str] | None, Query()] = None,
    pivot: Literal["game", "algo"] = "game",
) -> Response:
    """Publication XLSX — ``Summary`` + a per-game (``pivot=game``, raw reward) or per-algorithm
    (``pivot=algo``, normalized skill-%) sheet each with a native chart + ``Config`` + ``Methods``."""
    return _export_response("xlsx", run_ids, pivot)


@router.get("/export.repro")
async def export_repro(
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """The reproducibility card(s): a citable sha256 config-hash + BibTeX + reproduce command per run."""
    return _export_response("repro", run_ids, "game")


@router.get("/export.tex")
async def export_latex(
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """A paste-ready booktabs results table of the X2 summary statistics."""
    return _export_response("latex", run_ids, "game")
