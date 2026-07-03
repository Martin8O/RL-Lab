"""Experiment aggregation (X4, ADR-086) — turn N single-seed runs into one mean±band + across-seed stats.

Pure functions only: no I/O, no global state (the caller loads the runs from the store and hands the
curves + summaries here). Two jobs:

* :func:`aggregate_curves` — the **rebin-then-average** method. Seeds log at *different* ``env_steps``
  densities, so their rows can't be averaged directly; each seed's curve is linearly interpolated onto a
  shared grid over the seeds' **overlapping** x-range, then reduced to a per-grid-point mean, sample std
  (``ddof=1``) and a 95 % t-based CI. Degrades to a clean single line when only one seed is usable.
* :func:`aggregate_summaries` — the same idea for the scalars: mean ± std / CI across seeds for every X2
  summary field (fills the X5 ``Summary`` sheet's ± columns and the DataLab ranking table).

:func:`group_experiments` folds a set of runs into experiments (a seed sweep = one experiment): runs with
an explicit X3 ``experiment_id`` group by it; the rest auto-group by a seed-independent config hash.

The 95 % CI is **t-based** (``t_{0.975, n-1}``), which is honest for the small seed counts a student runs:
with 2 seeds the interval is deliberately wide — that width *is* the message.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy import stats as scipy_stats

from app.schemas.analysis import (
    AggregateBand,
    AggregatedSummary,
    ExperimentInfo,
    RunSummary,
    SeedStat,
)
from app.schemas.runs import RunMeta
from app.schemas.training import TrainConfig
from app.services.analysis import provenance

_DEFAULT_POINTS = 100
_MAX_POINTS = 2000

# The RunSummary scalar fields aggregated across seeds (ids / labels excluded — only the numbers a paper
# reports as mean ± std). ``across_seed_std`` itself is excluded: it's the per-run hook this pass realizes.
_SUMMARY_FIELDS = (
    "final_reward", "final_skill_pct", "solved_env_steps", "solved_wall_clock",
    "auc_normalized", "late_reward_std", "mean_steps_per_sec", "final_env_steps",
    "final_wall_clock", "peak_reward", "peak_skill_pct", "collapse_pct",
)


@dataclass
class SeedCurve:
    """One seed's curve for aggregation: its seed + the raw ``(x, y)`` points on the chosen axis/metric.

    ``xs`` / ``ys`` may contain ``None`` and duplicate / unsorted x — :func:`aggregate_curves` cleans them
    (drops ``None``, dedups on x keeping the later value, sorts) before interpolating."""

    seed: int
    xs: Sequence[float | None]
    ys: Sequence[float | None]


def _clean_curve(xs: Sequence[float | None], ys: Sequence[float | None]) -> tuple[list[float], list[float]]:
    """Drop ``None`` points, dedup on x (keep the later value), sort strictly increasing in x — the shape
    ``numpy.interp`` needs (a monotonic ``xp``)."""
    by_x: dict[float, float] = {}
    for x, y in zip(xs, ys, strict=True):
        if x is None or y is None:
            continue
        by_x[float(x)] = float(y)
    items = sorted(by_x.items())
    return [x for x, _ in items], [y for _, y in items]


def _mean_ci(values: list[float]) -> SeedStat:
    """Mean ± sample std (``ddof=1``) + 95 % t-CI over a list of per-seed values (``n`` = its length)."""
    n = len(values)
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if n < 2:
        return SeedStat(n=n, mean=mean)
    std = float(arr.std(ddof=1))
    half = float(scipy_stats.t.ppf(0.975, df=n - 1)) * std / sqrt(n)
    return SeedStat(n=n, mean=mean, std=std, ci_low=mean - half, ci_high=mean + half)


def _single_line(axis: str, metric: str, seed: int, xs: list[float], ys: list[float]) -> AggregateBand:
    """A degenerate band: one seed → the raw curve as the mean line, no std / CI, envelope = the line."""
    return AggregateBand(
        axis=axis, metric=metric, n_seeds=1, seeds=[seed],
        x=xs, mean=ys, std=None, ci_low=None, ci_high=None, lo=ys, hi=ys,
    )


def aggregate_curves(
    curves: Sequence[SeedCurve], *, axis: str, metric: str, points: int = _DEFAULT_POINTS
) -> AggregateBand | None:
    """Rebin the seed curves onto a common grid and reduce to a mean ± std / CI band.

    Interpolates each seed onto ``points`` evenly-spaced x over the seeds' **overlapping** x-range (so no
    seed is extrapolated), then computes the per-point mean, sample std (``ddof=1``), 95 % t-CI and the
    min/max envelope. Returns ``None`` when no seed has a plottable curve; degrades to a single line when
    only one seed is usable or the seeds share no common x-range (no honest band is possible there).
    """
    points = max(2, min(int(points), _MAX_POINTS))
    cleaned: list[tuple[int, list[float], list[float]]] = []
    for c in curves:
        xs, ys = _clean_curve(c.xs, c.ys)
        if len(xs) >= 2:
            cleaned.append((c.seed, xs, ys))

    if not cleaned:
        # No seed has ≥2 points to interpolate — fall back to any single-point seed as a flat line.
        for c in curves:
            xs, ys = _clean_curve(c.xs, c.ys)
            if xs:
                return _single_line(axis, metric, c.seed, xs, ys)
        return None

    if len(cleaned) == 1:
        seed, xs, ys = cleaned[0]
        return _single_line(axis, metric, seed, xs, ys)

    lo_x = max(xs[0] for _, xs, _ in cleaned)
    hi_x = min(xs[-1] for _, xs, _ in cleaned)
    if hi_x <= lo_x:
        # The seeds never overlap in x → no common grid; show the longest curve alone (honest, no band).
        seed, xs, ys = max(cleaned, key=lambda c: len(c[1]))
        return _single_line(axis, metric, seed, xs, ys)

    grid = np.linspace(lo_x, hi_x, points)
    stacked = np.vstack([np.interp(grid, xs, ys) for _, xs, ys in cleaned])  # (n_seeds, points)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0, ddof=1)
    n = len(cleaned)
    half = float(scipy_stats.t.ppf(0.975, df=n - 1)) * std / sqrt(n)
    return AggregateBand(
        axis=axis, metric=metric, n_seeds=n, seeds=[s for s, _, _ in cleaned],
        x=grid.tolist(), mean=mean.tolist(), std=std.tolist(),
        ci_low=(mean - half).tolist(), ci_high=(mean + half).tolist(),
        lo=stacked.min(axis=0).tolist(), hi=stacked.max(axis=0).tolist(),
    )


def aggregate_summaries(summaries: Sequence[RunSummary]) -> AggregatedSummary:
    """Mean ± std / CI across seeds for every X2 summary scalar (skipping ``None`` values per field)."""
    seeds = [s.seed for s in summaries]
    metrics: dict[str, SeedStat] = {}
    for field in _SUMMARY_FIELDS:
        values = [float(v) for s in summaries if (v := getattr(s, field)) is not None]
        if values:
            metrics[field] = _mean_ci(values)
    return AggregatedSummary(n_seeds=len(summaries), seeds=seeds, metrics=metrics)


def group_experiments(runs: Sequence[tuple[RunMeta, TrainConfig]]) -> list[ExperimentInfo]:
    """Fold runs into experiments: explicit X3 ``experiment_id`` wins; the rest auto-group by config.

    Runs from one seed-sweep share an ``experiment_id`` (and, being one config, one ``group_hash``); a run
    with no sweep id is grouped by ``auto:<env>:<algo>:<group_hash8>`` so two runs differing only by seed
    still land together. Returned sorted by ``(env_id, algo, experiment_id)`` for a stable listing.
    """
    groups: dict[str, dict] = {}
    for meta, config in runs:
        ghash = provenance.config_group_hash(config)
        key = meta.experiment_id or f"auto:{meta.env_id}:{meta.algo}:{ghash[:12]}"
        group = groups.setdefault(
            key,
            {"label": meta.experiment_label, "env_id": meta.env_id, "algo": meta.algo,
             "group_hash": ghash, "run_ids": [], "seeds": []},
        )
        group["run_ids"].append(meta.id)
        group["seeds"].append(meta.seed)
        if group["label"] is None and meta.experiment_label:
            group["label"] = meta.experiment_label

    experiments = [
        ExperimentInfo(
            experiment_id=key, label=g["label"], env_id=g["env_id"], algo=g["algo"],
            group_hash=g["group_hash"], run_ids=g["run_ids"], seeds=g["seeds"],
            n_seeds=len(g["run_ids"]),
        )
        for key, g in groups.items()
    ]
    experiments.sort(key=lambda e: (e.env_id, e.algo, e.experiment_id))
    return experiments
