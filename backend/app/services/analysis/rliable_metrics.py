"""rliable estimators (X4, ADR-086) — reimplemented in pure numpy/scipy, no ``rliable`` pip dep.

The robust aggregate metrics from Agarwal, Schwarzer, Castro, Courville & Bellemare (2021), *Deep
Reinforcement Learning at the Edge of the Statistical Precipice* (NeurIPS). Point-estimate-with-error-bar
reporting over a handful of seeds is unreliable; the paper's fix — **IQM** (interquartile mean) with
**stratified-bootstrap CIs**, **performance profiles**, and **probability of improvement** — is the modern
standard. We reimplement the estimators (they're small, well-specified maths) rather than take the
dependency, matching the X2/aggregate "pure, tested numpy" style.

Everything operates on a **normalized ``runs × tasks`` score matrix**: rows are independent runs (seeds),
columns are tasks (games), entries in ~[0, 1] (a per-task-normalized skill fraction). All functions are
pure; the bootstrap takes an explicit ``rng_seed`` so its CIs are reproducible (the tests pin them).

Reference: https://arxiv.org/abs/2108.13264
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

Matrix = np.ndarray

_DEFAULT_REPS = 2000
_MAX_REPS = 10_000
_CI_PERCENTILES = (2.5, 97.5)  # 95 % interval


# ---------------------------------------------------------------------------
# Score matrix construction
# ---------------------------------------------------------------------------


@dataclass
class ScoreMatrix:
    """A rectangular ``runs × tasks`` normalized-score matrix + its row (seed) and column (task) labels."""

    tasks: list[str]  # column labels (env_ids)
    seeds: list[int]  # row labels (the seed per row)
    matrix: Matrix  # shape (len(seeds), len(tasks)); dtype float


def build_score_matrix(entries: Sequence[tuple[str, int, float | None]]) -> ScoreMatrix:
    """Assemble a rectangular ``runs × tasks`` matrix from ``(task, seed, score)`` rows — the rliable input.

    rliable needs a hole-free matrix (equal runs per task), so we keep only the **seeds present in every
    task** (the intersection) and drop any task/seed that would leave a gap; a ``None`` score simply means
    that seed is absent for that task. Returns an empty matrix (``matrix.size == 0``) when no seed is common
    to all tasks. Columns are sorted by task id, rows by seed, so the matrix is deterministic.
    """
    tasks = sorted({task for task, _, _ in entries})
    per_task: dict[str, dict[int, float]] = {t: {} for t in tasks}
    for task, seed, score in entries:
        if score is not None:
            per_task[task][seed] = float(score)

    if not tasks:
        return ScoreMatrix(tasks=[], seeds=[], matrix=np.empty((0, 0), dtype=float))

    common: set[int] | None = None
    for t in tasks:
        seeds_here = set(per_task[t].keys())
        common = seeds_here if common is None else (common & seeds_here)
    seeds = sorted(common or set())
    if not seeds:
        return ScoreMatrix(tasks=tasks, seeds=[], matrix=np.empty((0, len(tasks)), dtype=float))

    matrix = np.array([[per_task[t][s] for t in tasks] for s in seeds], dtype=float)
    return ScoreMatrix(tasks=tasks, seeds=seeds, matrix=matrix)


# ---------------------------------------------------------------------------
# Aggregate point estimators (each: matrix -> scalar)
# ---------------------------------------------------------------------------


def _flat(matrix: Matrix) -> Matrix:
    """The finite scores of a matrix, flattened (drops NaN so a partial matrix still reduces)."""
    a = np.asarray(matrix, dtype=float).ravel()
    return a[~np.isnan(a)]


def interquartile_mean(matrix: Matrix) -> float:
    """IQM — the mean of the middle 50 % of scores (25 % trimmed each tail). The paper's headline estimator:
    robust to the outlier runs that make the naive mean noisy, yet far more efficient than the median."""
    flat = _flat(matrix)
    return float(scipy_stats.trim_mean(flat, 0.25)) if flat.size else float("nan")


def aggregate_mean(matrix: Matrix) -> float:
    """The plain mean score (reported for contrast — the estimator IQM improves on)."""
    flat = _flat(matrix)
    return float(flat.mean()) if flat.size else float("nan")


def aggregate_median(matrix: Matrix) -> float:
    """The median score (robust but statistically inefficient — the paper's other contrast estimator)."""
    flat = _flat(matrix)
    return float(np.median(flat)) if flat.size else float("nan")


def optimality_gap(matrix: Matrix, gamma: float = 1.0) -> float:
    """Mean shortfall from a perfect score: ``mean(max(0, gamma − score))``. 0 = every run reached the
    target (``gamma``, default 1.0 = solved); higher = further from mastery. Lower is better."""
    flat = _flat(matrix)
    return float(np.mean(np.maximum(0.0, gamma - flat))) if flat.size else float("nan")


ESTIMATORS: dict[str, Callable[[Matrix], float]] = {
    "iqm": interquartile_mean,
    "mean": aggregate_mean,
    "median": aggregate_median,
    "optimality_gap": optimality_gap,
}


# ---------------------------------------------------------------------------
# Stratified bootstrap CIs
# ---------------------------------------------------------------------------


def _stratified_resample(matrix: Matrix, rng: np.random.Generator) -> Matrix:
    """One stratified bootstrap replicate: resample the runs (rows) **within each task (column)**
    independently, with replacement. Stratifying by task preserves the per-task run count, which the
    across-task aggregate depends on (Agarwal et al.)."""
    n_runs, n_tasks = matrix.shape
    out = np.empty_like(matrix)
    for j in range(n_tasks):
        idx = rng.integers(0, n_runs, size=n_runs)
        out[:, j] = matrix[idx, j]
    return out


def bootstrap_ci(
    matrix: Matrix,
    estimator: Callable[[Matrix], float],
    *,
    reps: int = _DEFAULT_REPS,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """A stratified-bootstrap 95 % CI for ``estimator`` over ``matrix``. Deterministic given ``rng_seed``.

    Returns ``(nan, nan)`` for an empty matrix. With few seeds the interval is wide *by construction* —
    that honesty (a 2-seed result can't claim a tight bound) is the whole point of the method."""
    m = np.asarray(matrix, dtype=float)
    if m.size == 0 or m.shape[0] == 0:
        return float("nan"), float("nan")
    reps = max(1, min(int(reps), _MAX_REPS))
    rng = np.random.default_rng(rng_seed)
    samples = np.array([estimator(_stratified_resample(m, rng)) for _ in range(reps)])
    lo, hi = np.nanpercentile(samples, _CI_PERCENTILES)
    return float(lo), float(hi)


def aggregate_estimate(
    matrix: Matrix, name: str, *, reps: int = _DEFAULT_REPS, rng_seed: int = 0
) -> tuple[float, float, float]:
    """``(value, ci_low, ci_high)`` for a named estimator (``iqm`` / ``mean`` / ``median`` /
    ``optimality_gap``) — the point estimate plus its stratified-bootstrap 95 % CI."""
    estimator = ESTIMATORS[name]
    value = estimator(np.asarray(matrix, dtype=float))
    lo, hi = bootstrap_ci(matrix, estimator, reps=reps, rng_seed=rng_seed)
    return value, lo, hi


# ---------------------------------------------------------------------------
# Performance profiles
# ---------------------------------------------------------------------------


def performance_profile(matrix: Matrix, taus: Sequence[float]) -> list[float]:
    """The score distribution's run-score performance profile: for each τ, the fraction of ``(run, task)``
    scores **strictly above** τ. Non-increasing in τ; the area under it is a robust aggregate and
    non-crossing profiles imply one method stochastically dominates another (Agarwal et al.)."""
    flat = _flat(matrix)
    if flat.size == 0:
        return [0.0 for _ in taus]
    return [float(np.mean(flat > t)) for t in taus]


def default_taus(matrix: Matrix, points: int = 51) -> list[float]:
    """An evenly-spaced τ grid from 0 to the matrix max (≥1.0), for :func:`performance_profile`."""
    flat = _flat(matrix)
    top = max(1.0, float(flat.max())) if flat.size else 1.0
    points = max(2, min(int(points), 501))
    return np.linspace(0.0, top, points).tolist()


# ---------------------------------------------------------------------------
# Probability of improvement (per-task Mann–Whitney, averaged)
# ---------------------------------------------------------------------------


def _prob_improvement(x: Matrix, y: Matrix) -> float:
    """P(X > Y) averaged over tasks: per task, the fraction of run pairs with ``x_run > y_run`` (ties count
    half) — the normalized Mann–Whitney U statistic — then the mean across tasks. ``x`` and ``y`` must share
    columns (tasks); their row counts (seeds) may differ."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    per_task: list[float] = []
    for j in range(x.shape[1]):
        xs = x[:, j][~np.isnan(x[:, j])]
        ys = y[:, j][~np.isnan(y[:, j])]
        if xs.size == 0 or ys.size == 0:
            continue
        pairwise = xs[:, None] - ys[None, :]
        wins = float(np.mean(pairwise > 0) + 0.5 * np.mean(pairwise == 0))
        per_task.append(wins)
    return float(np.mean(per_task)) if per_task else float("nan")


def probability_of_improvement(
    x: Matrix, y: Matrix, *, reps: int = _DEFAULT_REPS, rng_seed: int = 0
) -> tuple[float, float, float]:
    """``(value, ci_low, ci_high)`` for P(X > Y) over shared tasks, with a stratified-bootstrap 95 % CI.

    0.5 = the two methods are indistinguishable; >0.5 favours ``x``. The CI resamples both methods' runs
    within each task; deterministic given ``rng_seed``."""
    xm = np.asarray(x, dtype=float)
    ym = np.asarray(y, dtype=float)
    value = _prob_improvement(xm, ym)
    if xm.size == 0 or ym.size == 0:
        return value, float("nan"), float("nan")
    reps = max(1, min(int(reps), _MAX_REPS))
    rng = np.random.default_rng(rng_seed)
    samples = np.array(
        [_prob_improvement(_stratified_resample(xm, rng), _stratified_resample(ym, rng)) for _ in range(reps)]
    )
    lo, hi = np.nanpercentile(samples, _CI_PERCENTILES)
    return value, float(lo), float(hi)


def shared_task_columns(a: ScoreMatrix, b: ScoreMatrix) -> tuple[Matrix, Matrix, list[str]]:
    """Restrict two score matrices to their **common tasks** (columns), in a shared order — the aligned
    input :func:`probability_of_improvement` needs. Rows (seeds) are left independent per method."""
    common = [t for t in a.tasks if t in set(b.tasks)]
    ai = [a.tasks.index(t) for t in common]
    bi = [b.tasks.index(t) for t in common]
    ma = a.matrix[:, ai] if common else np.empty((a.matrix.shape[0], 0), dtype=float)
    mb = b.matrix[:, bi] if common else np.empty((b.matrix.shape[0], 0), dtype=float)
    return ma, mb, common
