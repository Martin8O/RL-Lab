"""Analysis endpoints (Phase X / DataLab) — server-computed summary statistics for finished runs.

The backend holds the *full* raw metric history on disk (``data/runs/<id>/metrics.json``); the frontend
store is only a capped live ring buffer. So the summary numbers a comparison table needs are computed
here, from disk, as the source of truth. This route wires the pure engine (:mod:`app.services.analysis.stats`)
to the run store: load each requested run, look up its env's skill range, reduce to a
:class:`~app.schemas.analysis.RunSummary`. Unknown run ids are skipped (one object per *found* run).
"""

from typing import Annotated, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from app.envs.registry import get_env
from app.schemas.analysis import (
    AggregateResponse,
    ExperimentInfo,
    MethodRliable,
    PerformanceProfile,
    ProbabilityOfImprovement,
    RliableEstimate,
    RliableResult,
    RunSummary,
)
from app.schemas.training import Algo
from app.services.analysis import aggregate as aggregate_engine
from app.services.analysis import export as export_engine
from app.services.analysis import rliable_metrics as rliable
from app.services.analysis.aggregate import SeedCurve
from app.services.analysis.stats import score_of_frame, skill_pct, summarize
from app.services.runs import run_store

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Axis / metric a caller may aggregate on — the canonical X1 axes and the two curve scales.
Axis = Literal["env_steps", "wall_clock"]
Metric = Literal["reward", "skill_pct"]


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
    fmt: str,
    run_ids: list[str] | None,
    pivot: Literal["game", "algo"],
    lang: Literal["en", "cz"] = "en",
) -> Response:
    """Load the selected runs server-side (full history on disk) and stream the built format as a file
    download. ``lang`` localizes descriptive text (XLSX headings, the SVG figure) to the app's language.
    An empty / all-unknown selection still returns a valid (header-only) artifact, not an error, so the
    client can hand it straight to the user."""
    content, media_type, filename = export_engine.export(fmt, run_ids or [], pivot, lang)
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
    lang: Literal["en", "cz"] = "en",
) -> Response:
    """Publication XLSX — ``Summary`` + a per-game (``pivot=game``, raw reward) or per-algorithm
    (``pivot=algo``, normalized skill-%) sheet each with a native chart + ``Config`` + ``Methods``.
    Sheet tabs + headings are in ``lang`` (the app's language); data columns stay English."""
    return _export_response("xlsx", run_ids, pivot, lang)


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


@router.get("/export.svg")
async def export_figure(
    run_ids: Annotated[list[str] | None, Query()] = None,
    pivot: Literal["game", "algo"] = "game",
    lang: Literal["en", "cz"] = "en",
) -> Response:
    """A standalone vector figure (SVG) of the selected runs — raw reward (``pivot=game``) or normalized
    skill-% (``pivot=algo``) vs env_steps, ready to drop into a paper or slides. Title/axes in ``lang``."""
    return _export_response("figure", run_ids, pivot, lang)


@router.get("/export.zip")
async def export_tensorboard(
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """A ZIP of TensorBoard event files (one log dir per run) — unzip and point ``tensorboard --logdir``
    at it to browse the curves interactively."""
    return _export_response("tensorboard", run_ids, "game")


@router.get("/export.npz")
async def export_scorematrix(
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> Response:
    """The normalized ``runs × tasks`` score matrix per algorithm as an ``.npz`` (the exact rliable input,
    so a researcher can re-run their own analysis). Round-trips with a plain ``numpy.load``."""
    return _export_response("scorematrix", run_ids, "game")


# ---------------------------------------------------------------------------
# X4 — experiment aggregation across seeds (ADR-086)
# ---------------------------------------------------------------------------


def _all_experiments() -> list[ExperimentInfo]:
    """Group the whole run history into experiments (light path — reads meta + config, not the curves)."""
    pairs: list[tuple] = []
    for meta in run_store.list():
        config = run_store.get_config(meta.id)
        if config is not None:
            pairs.append((meta, config))
    return aggregate_engine.group_experiments(pairs)


def _experiment_run_ids(experiment_id: str) -> list[str]:
    """The member run ids of an experiment. An explicit X3 sweep id resolves straight from ``meta.json``
    (no config loads); an ``auto:`` group needs the config-hash grouping over the full history."""
    if not experiment_id.startswith("auto:"):
        return [m.id for m in run_store.list() if m.experiment_id == experiment_id]
    match = next((e for e in _all_experiments() if e.experiment_id == experiment_id), None)
    return match.run_ids if match else []


@router.get("/experiments", response_model=list[ExperimentInfo])
async def experiments() -> list[ExperimentInfo]:
    """Every experiment auto-grouped from the run history (a seed sweep = one experiment; single runs are
    1-seed experiments). The membership + seeds each aggregation (``/aggregate``) will fold together."""
    return _all_experiments()


def _seed_curve(run: export_engine.LoadedRun, axis: str, metric: str) -> SeedCurve:
    """Extract one run's ``(x, y)`` points on the chosen axis (``env_steps`` / ``wall_clock``) and metric
    (raw ``reward`` / normalized ``skill_pct``) — the input the pure aggregator rebins."""
    xs: list[float | None] = []
    ys: list[float | None] = []
    for frame in run.frames:
        xs.append(frame.get(axis))
        reward = score_of_frame(run.config.algo, frame)
        ys.append(reward if metric == "reward" else skill_pct(reward, run.min_score, run.solved_score))
    return SeedCurve(seed=run.config.seed, xs=xs, ys=ys)


@router.get("/aggregate", response_model=AggregateResponse)
async def aggregate(
    experiment_id: str | None = None,
    run_ids: Annotated[list[str] | None, Query()] = None,
    axis: Axis = "env_steps",
    metric: Metric = "reward",
    seeds: Annotated[list[int] | None, Query()] = None,
    points: int = 100,
) -> AggregateResponse:
    """The across-seed aggregate for one experiment: a band-ready mean ± std / CI curve (rebinned onto a
    common ``axis`` grid) + the mean ± std of every X2 summary scalar.

    Select the runs by ``experiment_id`` (its sweep members) **or** an explicit ``run_ids`` list; an
    optional ``seeds`` subset includes/excludes individual seeds so the band recomputes live. ``metric``
    picks the raw reward or the normalized skill %; ``points`` sets the grid resolution.
    """
    ids = list(run_ids) if run_ids else (_experiment_run_ids(experiment_id) if experiment_id else [])
    loaded = export_engine.load_runs(ids)
    if seeds is not None:
        keep = set(seeds)
        loaded = [r for r in loaded if r.config.seed in keep]
    if not loaded:
        raise HTTPException(status_code=404, detail="No runs matched the aggregation request")

    curves = [_seed_curve(r, axis, metric) for r in loaded]
    band = aggregate_engine.aggregate_curves(curves, axis=axis, metric=metric, points=points)
    summary = aggregate_engine.aggregate_summaries([r.summary for r in loaded])
    return AggregateResponse(
        experiment_id=experiment_id,
        env_id=loaded[0].config.env_id,
        algo=loaded[0].config.algo,
        band=band,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# X4 — rliable estimators (IQM / CIs / performance profiles / prob-of-improvement)
# ---------------------------------------------------------------------------


def _estimate(matrix: np.ndarray, name: str, reps: int, rng_seed: int) -> RliableEstimate:
    value, lo, hi = rliable.aggregate_estimate(matrix, name, reps=reps, rng_seed=rng_seed)
    return RliableEstimate(value=value, ci_low=lo, ci_high=hi)


@router.get("/rliable", response_model=RliableResult)
async def rliable_analysis(
    run_ids: Annotated[list[str] | None, Query()] = None,
    reps: int = 2000,
    profile_points: int = 51,
) -> RliableResult:
    """The rliable analysis (Agarwal et al.) over a selection of runs: per-algorithm IQM / mean / median /
    optimality-gap with stratified-bootstrap 95 % CIs, a performance profile, and — when ≥2 algorithms
    share tasks — the probability that the first improves on the second. Scores are each run's final
    skill % ÷ 100 (the normalized ``runs × tasks`` matrix); runs group into methods by algorithm."""
    loaded = export_engine.load_runs(run_ids or [])
    by_algo: dict[Algo, list[export_engine.LoadedRun]] = {}
    for run in loaded:
        by_algo.setdefault(run.config.algo, []).append(run)

    methods: list[MethodRliable] = []
    score_matrices: dict[Algo, rliable.ScoreMatrix] = {}
    for algo, group in by_algo.items():
        entries = [(r.config.env_id, r.config.seed, export_engine.normalized_score(r)) for r in group]
        sm = rliable.build_score_matrix(entries)
        if sm.matrix.size == 0:
            continue  # no seed common to every task → no honest matrix for this method
        score_matrices[algo] = sm
        taus = rliable.default_taus(sm.matrix, profile_points)
        methods.append(
            MethodRliable(
                algo=algo, n_runs=sm.matrix.shape[0], tasks=sm.tasks, seeds=sm.seeds,
                matrix=sm.matrix.tolist(),
                iqm=_estimate(sm.matrix, "iqm", reps, 0),
                mean=_estimate(sm.matrix, "mean", reps, 1),
                median=_estimate(sm.matrix, "median", reps, 2),
                optimality_gap=_estimate(sm.matrix, "optimality_gap", reps, 3),
                profile=PerformanceProfile(taus=taus, fractions=rliable.performance_profile(sm.matrix, taus)),
            )
        )

    poi: ProbabilityOfImprovement | None = None
    if len(methods) >= 2:
        a, b = methods[0].algo, methods[1].algo
        ma, mb, shared = rliable.shared_task_columns(score_matrices[a], score_matrices[b])
        if shared:
            value, lo, hi = rliable.probability_of_improvement(ma, mb, reps=reps, rng_seed=4)
            poi = ProbabilityOfImprovement(algo_x=a, algo_y=b, value=value, ci_low=lo, ci_high=hi)

    return RliableResult(
        normalization="final skill % / 100 (per-task normalized to [0, 1])",
        methods=methods,
        prob_of_improvement=poi,
    )
