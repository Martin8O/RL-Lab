"""Analysis contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

The DataLab (Phase X) distils a finished run's *curve* into the scalars a paper reports. A learning
curve answers "how", but a comparison **table** needs one number per run per metric — the content of
X5's ``Summary`` sheet and X6's ranking table. :class:`RunSummary` is that per-run row; the pure,
unit-tested engine that computes it is :mod:`app.services.analysis.stats`, and it is served by
``GET /api/analysis/summary?run_ids=...`` (one object per requested run).

All fields are ``| None`` where a short / sparse / never-solved run can't produce the number — the
engine never crashes on a 3-point or all-``None`` curve, it returns ``None`` for what it can't compute.
"""

from pydantic import BaseModel

from app.schemas.training import Algo


class RunSummary(BaseModel):
    """The standard RL summary statistics for one finished run (X2, ADR-083).

    Computed over the run's recorded metric frames on the **canonical comparison axes** (X1):
    ``env_steps`` (cumulative environment interactions) and ``wall_clock`` (elapsed seconds), against
    the env's ``[min_score, solved_score]`` skill range. The six metric groups the DataLab ranks on:

    1. **Final performance** — ``final_reward`` (mean over the last ~10 % of the curve, robust to the
       last-point noise) + ``final_skill_pct`` (that reward normalized to the 0–100 % skill scale, the
       same ``(r−min)/(solved−min)`` clamp the live chart uses).
    2. **Sample efficiency** — ``solved_env_steps`` / ``solved_wall_clock``: where the run first reached
       ``solved_score`` (``None`` if it never did).
    3. **AUC** — ``auc_normalized``: the trapezoidal area under the normalized-skill curve over
       ``env_steps``, divided by the step range → the *mean* skill (0–1) across the run ("how fast **and**
       how high" in one number; the natural ranking key).
    4. **Stability** — ``late_reward_std``: roughness (population std) of the reward over the late curve.
       ``across_seed_std`` is a **hook for X4** (mean ± std across a seed sweep); always ``None`` here.
    5. **Throughput** — ``mean_steps_per_sec`` (``env_steps`` per ``wall_clock`` over the whole run) +
       the run totals ``final_env_steps`` / ``final_wall_clock`` for downstream efficiency ratios.
    6. **Peak vs final** — ``peak_reward`` / ``peak_env_steps`` / ``peak_skill_pct`` and ``collapse_pct``
       (how many skill points the run gave back from its peak — common for value-based collapse).
    """

    run_id: str
    env_id: str
    algo: Algo
    seed: int
    n_frames: int
    # The skill range the normalization used (echoed so the client needn't re-fetch the env registry).
    min_score: float
    solved_score: float

    # 1. Final performance
    final_reward: float | None = None
    final_skill_pct: float | None = None  # 0–100, clamped — matches the live chart's solvedPct formula

    # 2. Sample efficiency / time-to-threshold
    solved_env_steps: int | None = None
    solved_wall_clock: float | None = None

    # 3. Area under the (normalized) learning curve
    auc_normalized: float | None = None  # 0–1 mean skill across the run (trapezoid over env_steps)

    # 4. Stability / variance
    late_reward_std: float | None = None
    across_seed_std: float | None = None  # hook: filled by the X4 seed-aggregation pass, None for a single run

    # 5. Throughput / wall-clock efficiency
    final_env_steps: int = 0
    final_wall_clock: float = 0.0
    mean_steps_per_sec: float | None = None

    # 6. Peak vs final
    peak_reward: float | None = None
    peak_env_steps: int | None = None
    peak_skill_pct: float | None = None
    collapse_pct: float | None = None  # max(0, peak_skill_pct − final_skill_pct); 0 = no post-peak collapse


# ---------------------------------------------------------------------------
# X4 — experiment aggregation across seeds (ADR-086). Computed by the pure
# app.services.analysis.aggregate + rliable_metrics modules, served by /api/analysis/*.
# ---------------------------------------------------------------------------


class ExperimentInfo(BaseModel):
    """One experiment = the set of runs sharing ``(env, algo, hyperparameters)`` — i.e. a seed sweep (X4).

    Runs launched by an X3 seed-sweep carry an explicit ``experiment_id`` and group by it; every other run
    is auto-grouped by a seed-independent config hash (``group_hash``), under a derived
    ``auto:<env>:<algo>:<hash>`` id. ``GET /api/analysis/experiments`` returns one of these per group.
    """

    experiment_id: str  # explicit sweep id, or a derived "auto:<env>:<algo>:<hash>" key
    label: str | None = None
    env_id: str
    algo: Algo
    group_hash: str  # seed-independent config hash — what "the same experiment" means
    run_ids: list[str]
    seeds: list[int]
    n_seeds: int


class SeedStat(BaseModel):
    """One X2 scalar aggregated across an experiment's seeds: mean ± sample std (ddof=1) + 95 % t-CI.

    ``std`` / ``ci_low`` / ``ci_high`` are ``None`` when only one seed contributed a value (no spread to
    report). ``n`` is the number of seeds that had a non-``None`` value for this metric (a seed that never
    solved contributes no ``solved_env_steps``, so a metric can have fewer seeds than the experiment)."""

    n: int
    mean: float
    std: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class AggregatedSummary(BaseModel):
    """Across-seed aggregation of the X2 summary statistics for one experiment.

    ``metrics`` maps each :class:`RunSummary` scalar field name → its :class:`SeedStat` across the seeds.
    This is the mean ± std the X5 ``Summary`` sheet's ± columns and the DataLab ranking table read (it
    realizes the ``RunSummary.across_seed_std`` hook at the *group* level, where a std across seeds
    actually lives — a single run can't have one)."""

    n_seeds: int
    seeds: list[int]
    metrics: dict[str, SeedStat]


class AggregateBand(BaseModel):
    """A band-ready mean learning curve across seeds, rebinned onto a common axis grid (X4).

    Seeds log at *different* ``env_steps`` densities, so raw rows can't be averaged: each seed's curve is
    linearly interpolated onto a shared grid over the seeds' **overlapping** x-range first, then reduced
    per grid point. ``std`` / ``ci_*`` are ``None`` when only one seed contributes (the band collapses to
    a single line); ``lo`` / ``hi`` are the per-point min/max envelope across seeds."""

    axis: str  # "env_steps" | "wall_clock"
    metric: str  # "reward" | "skill_pct"
    n_seeds: int
    seeds: list[int]
    x: list[float]
    mean: list[float]
    std: list[float] | None = None
    ci_low: list[float] | None = None  # 95 % t-based CI band
    ci_high: list[float] | None = None
    lo: list[float]  # per-point min envelope across seeds
    hi: list[float]  # per-point max envelope across seeds


class AggregateResponse(BaseModel):
    """The full aggregate for one experiment: the band-ready curve + the across-seed summary stats."""

    experiment_id: str | None = None
    env_id: str
    algo: Algo
    band: AggregateBand | None = None  # None when no seed had ≥2 plottable points
    summary: AggregatedSummary


class RliableEstimate(BaseModel):
    """A point estimate with its stratified-bootstrap 95 % CI (Agarwal et al., 2021)."""

    value: float
    ci_low: float
    ci_high: float


class PerformanceProfile(BaseModel):
    """A run-score performance profile: the fraction of ``(run, task)`` scores above each threshold τ.

    A monotonically non-increasing curve; the area under it is a robust aggregate, and non-crossing
    profiles imply stochastic dominance (Agarwal et al.)."""

    taus: list[float]
    fractions: list[float]


class MethodRliable(BaseModel):
    """The rliable aggregate metrics for one method (algorithm) over a ``runs × tasks`` score matrix.

    ``matrix`` (with ``seeds`` as its row labels and ``tasks`` as its columns) is the exact normalized
    input the estimates were computed from — echoed so a researcher can re-run their own analysis (the
    JSON companion to the NPZ score-matrix export)."""

    algo: Algo
    n_runs: int  # rows (seeds) per task in the rectangular matrix
    tasks: list[str]  # the env_ids (columns)
    seeds: list[int]  # the seed per row
    matrix: list[list[float]]  # the n_runs × n_tasks normalized-score (0–1) matrix
    iqm: RliableEstimate
    mean: RliableEstimate
    median: RliableEstimate
    optimality_gap: RliableEstimate  # mean shortfall from a perfect (1.0) score — lower is better
    profile: PerformanceProfile


class ProbabilityOfImprovement(BaseModel):
    """P(algo_x > algo_y) averaged over the shared tasks (per-task Mann–Whitney), + a bootstrap CI.

    0.5 = indistinguishable; >0.5 favours ``algo_x``. Computed only when the selection has ≥2 algorithms
    sharing at least one task."""

    algo_x: Algo
    algo_y: Algo
    value: float
    ci_low: float
    ci_high: float


class RliableResult(BaseModel):
    """The rliable analysis over a selection of runs: per-method aggregates + pairwise prob-of-improvement.

    Scores are the per-run **final skill %** (X2) divided by 100 → the normalized ``runs × tasks`` matrix
    Agarwal et al. take as input; runs are grouped into methods by algorithm, tasks are the games."""

    normalization: str
    methods: list[MethodRliable]
    prob_of_improvement: ProbabilityOfImprovement | None = None
