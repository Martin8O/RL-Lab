"""X4 — experiment aggregation + the rliable estimators (pure) + their API routes.

Covers the Definition of Done:
  * ``aggregate.py`` rebins seeds onto a common grid (incl. **uneven densities**) → mean / std / CI, and
    degrades to a clean single line at N=1.
  * across-seed summary stats (mean ± std / CI per X2 scalar), skipping ``None`` values.
  * experiment grouping — explicit ``experiment_id`` and auto (seed-independent config hash).
  * ``rliable_metrics.py`` — IQM / optimality-gap / performance-profile / prob-of-improvement with
    **stratified-bootstrap CIs**, incl. the few-seed → wide-CI honesty point; no ``rliable`` pip dep.
  * ``GET /api/analysis/{experiments,aggregate,rliable}`` round-trip; the NPZ score-matrix loads back.
"""

import io
import math
from pathlib import Path

import numpy as np
from app.api import analysis as analysis_api
from app.main import app
from app.schemas.analysis import RunSummary
from app.schemas.runs import RunMeta
from app.schemas.training import TrainConfig
from app.services.analysis import aggregate as agg
from app.services.analysis import export as export_engine
from app.services.analysis import provenance
from app.services.analysis import rliable_metrics as rl
from app.services.runs import RunStore
from fastapi.testclient import TestClient
from scipy import stats as scipy_stats


def _frame(env_steps: int, ep_rew_mean: float | None, elapsed: float) -> dict:
    return {
        "type": "metrics", "env_steps": env_steps, "timesteps": env_steps,
        "ep_rew_mean": ep_rew_mean, "wall_clock": elapsed, "elapsed": elapsed,
    }


def _summary(seed: int, **fields: float | None) -> RunSummary:
    return RunSummary(
        run_id=f"r{seed}", env_id="cartpole", algo="ppo", seed=seed,
        n_frames=3, min_score=0.0, solved_score=500.0, **fields,
    )


# ===========================================================================
# aggregate.py — rebin curves
# ===========================================================================


def test_rebin_uneven_densities_recovers_the_line() -> None:
    # Two seeds tracing the same line y=x but sampled at *different* densities → after rebinning both
    # interpolate to y=grid, so the mean is the line and the std is ~0 everywhere.
    a = agg.SeedCurve(seed=1, xs=[0.0, 50.0, 100.0], ys=[0.0, 50.0, 100.0])
    b = agg.SeedCurve(seed=2, xs=[0.0, 25.0, 50.0, 75.0, 100.0], ys=[0.0, 25.0, 50.0, 75.0, 100.0])
    band = agg.aggregate_curves([a, b], axis="env_steps", metric="reward", points=11)
    assert band is not None and band.n_seeds == 2
    assert band.x[0] == 0.0 and band.x[-1] == 100.0
    assert all(abs(m - x) < 1e-9 for m, x in zip(band.mean, band.x, strict=True))  # mean == the line
    assert band.std is not None and all(s < 1e-9 for s in band.std)  # no spread


def test_rebin_mean_std_ci_at_a_grid_point() -> None:
    # Three seeds diverging linearly to 0 / 10 / 20 at x=100 → at the last grid point mean=10, sample
    # std(ddof=1)=10, and a t-based 95% CI half-width of t(.975,2)*10/sqrt(3).
    curves = [
        agg.SeedCurve(seed=1, xs=[0.0, 100.0], ys=[0.0, 0.0]),
        agg.SeedCurve(seed=2, xs=[0.0, 100.0], ys=[0.0, 10.0]),
        agg.SeedCurve(seed=3, xs=[0.0, 100.0], ys=[0.0, 20.0]),
    ]
    band = agg.aggregate_curves(curves, axis="env_steps", metric="reward", points=2)
    assert band is not None and band.seeds == [1, 2, 3]
    assert math.isclose(band.mean[-1], 10.0)
    assert band.std is not None and math.isclose(band.std[-1], 10.0)
    half = float(scipy_stats.t.ppf(0.975, df=2)) * 10.0 / math.sqrt(3)
    assert band.ci_low is not None and band.ci_high is not None
    assert math.isclose(band.ci_low[-1], 10.0 - half) and math.isclose(band.ci_high[-1], 10.0 + half)
    assert band.lo[-1] == 0.0 and band.hi[-1] == 20.0  # min/max envelope


def test_single_seed_is_a_clean_line_no_band() -> None:
    curve = agg.SeedCurve(seed=7, xs=[0.0, 1.0, 2.0], ys=[1.0, 2.0, 3.0])
    band = agg.aggregate_curves([curve], axis="env_steps", metric="reward", points=50)
    assert band is not None and band.n_seeds == 1
    assert band.x == [0.0, 1.0, 2.0] and band.mean == [1.0, 2.0, 3.0]
    assert band.std is None and band.ci_low is None and band.ci_high is None


def test_no_plottable_points_returns_none() -> None:
    curve = agg.SeedCurve(seed=1, xs=[0.0, 1.0], ys=[None, None])
    assert agg.aggregate_curves([curve], axis="env_steps", metric="reward") is None


def test_curve_cleaning_drops_none_and_dedups_x() -> None:
    # A None y is dropped; a duplicate x keeps the later value; order is restored.
    curve = agg.SeedCurve(seed=1, xs=[0.0, 10.0, 10.0, 5.0], ys=[0.0, 1.0, 99.0, None])
    band = agg.aggregate_curves([curve], axis="env_steps", metric="reward")
    assert band is not None
    assert band.x == [0.0, 10.0] and band.mean == [0.0, 99.0]


# ===========================================================================
# aggregate.py — across-seed summary stats
# ===========================================================================


def test_aggregate_summaries_mean_std_and_skips_none() -> None:
    summaries = [
        _summary(1, final_skill_pct=10.0, solved_env_steps=1000),
        _summary(2, final_skill_pct=20.0, solved_env_steps=None),  # never solved → no solved_env_steps
        _summary(3, final_skill_pct=30.0, solved_env_steps=3000),
    ]
    out = agg.aggregate_summaries(summaries)
    assert out.n_seeds == 3 and out.seeds == [1, 2, 3]
    fs = out.metrics["final_skill_pct"]
    assert fs.n == 3 and math.isclose(fs.mean, 20.0) and fs.std is not None and math.isclose(fs.std, 10.0)
    # solved_env_steps had a None → aggregated over the 2 seeds that solved.
    assert out.metrics["solved_env_steps"].n == 2
    assert math.isclose(out.metrics["solved_env_steps"].mean, 2000.0)


def test_aggregate_summaries_single_seed_has_no_std() -> None:
    out = agg.aggregate_summaries([_summary(1, final_skill_pct=42.0)])
    stat = out.metrics["final_skill_pct"]
    assert stat.n == 1 and stat.mean == 42.0 and stat.std is None and stat.ci_low is None


# ===========================================================================
# aggregate.py — experiment grouping + the seed-independent hash
# ===========================================================================


def _meta(rid: str, seed: int, experiment_id: str | None = None) -> RunMeta:
    return RunMeta(
        id=rid, label=rid, env_id="cartpole", algo="ppo", seed=seed,
        created_at="2026-07-03T10:00:00+00:00", finished_at="2026-07-03T10:05:00+00:00",
        state="finished", experiment_id=experiment_id,
    )


def test_config_group_hash_is_seed_independent() -> None:
    base = TrainConfig(env_id="cartpole", algo="ppo", seed=1)
    other_seed = TrainConfig(env_id="cartpole", algo="ppo", seed=99, experiment_id="x", experiment_label="L")
    diff_budget = TrainConfig(env_id="cartpole", algo="ppo", seed=1, total_timesteps=123456)
    assert provenance.config_group_hash(base) == provenance.config_group_hash(other_seed)  # seed ignored
    assert provenance.config_group_hash(base) != provenance.config_group_hash(diff_budget)  # config matters
    # The plain (run-level) hash still separates the seeds — grouping is the only seed-blind view.
    assert provenance.config_hash(base) != provenance.config_hash(other_seed)


def test_group_experiments_explicit_and_auto() -> None:
    cfg = lambda seed, **kw: TrainConfig(env_id="cartpole", algo="ppo", seed=seed, **kw)  # noqa: E731
    runs = [
        (_meta("s1", 42, "expA"), cfg(42, experiment_id="expA")),
        (_meta("s2", 43, "expA"), cfg(43, experiment_id="expA")),
        (_meta("a1", 1), cfg(1)),  # no experiment_id → auto-grouped with a2 (same config, diff seed)
        (_meta("a2", 2), cfg(2)),
        (_meta("b1", 1), cfg(1, total_timesteps=999)),  # different config → its own auto group
    ]
    exps = agg.group_experiments(runs)
    by_id = {e.experiment_id: e for e in exps}
    assert by_id["expA"].n_seeds == 2 and sorted(by_id["expA"].seeds) == [42, 43]
    auto = [e for e in exps if e.experiment_id.startswith("auto:")]
    assert len(auto) == 2  # the shared-config pair and the odd-budget singleton
    pair = next(e for e in auto if e.n_seeds == 2)
    assert sorted(pair.seeds) == [1, 2] and set(pair.run_ids) == {"a1", "a2"}


# ===========================================================================
# rliable_metrics.py — aggregate estimators
# ===========================================================================


def test_iqm_is_the_trimmed_mean() -> None:
    m = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
    assert math.isclose(rl.interquartile_mean(m), 0.5)
    assert math.isclose(rl.interquartile_mean(m), float(scipy_stats.trim_mean(m.ravel(), 0.25)))


def test_mean_median_optimality_gap() -> None:
    m = np.array([[0.2, 0.6], [0.4, 0.8]])
    assert math.isclose(rl.aggregate_mean(m), 0.5)
    assert math.isclose(rl.aggregate_median(m), 0.5)
    # optimality gap = mean(max(0, 1 - score)) = mean(.8,.4,.6,.2) = .5
    assert math.isclose(rl.optimality_gap(m), 0.5)


def test_estimators_ignore_nan() -> None:
    m = np.array([[0.5, np.nan], [0.5, 0.5]])
    assert math.isclose(rl.aggregate_mean(m), 0.5)  # the NaN cell is dropped, not propagated


# ===========================================================================
# rliable_metrics.py — performance profiles
# ===========================================================================


def test_performance_profile_is_non_increasing() -> None:
    m = np.array([[0.1], [0.4], [0.7], [1.0]])
    taus = [0.0, 0.5, 0.9]
    frac = rl.performance_profile(m, taus)
    assert frac == [1.0, 0.5, 0.25]  # >0: 4/4, >0.5: {.7,1}, >0.9: {1}
    assert all(frac[i] >= frac[i + 1] for i in range(len(frac) - 1))


def test_default_taus_span_zero_to_at_least_one() -> None:
    taus = rl.default_taus(np.array([[0.2], [0.8]]), points=6)
    assert taus[0] == 0.0 and taus[-1] == 1.0 and len(taus) == 6


# ===========================================================================
# rliable_metrics.py — stratified-bootstrap CIs (incl. few-seed → wide)
# ===========================================================================


def test_bootstrap_ci_is_deterministic_and_brackets() -> None:
    m = np.array([[0.2], [0.4], [0.6], [0.8]])
    a = rl.bootstrap_ci(m, rl.aggregate_mean, reps=500, rng_seed=0)
    b = rl.bootstrap_ci(m, rl.aggregate_mean, reps=500, rng_seed=0)
    assert a == b  # same seed → identical CI
    lo, hi = a
    assert lo <= 0.5 <= hi  # the point estimate sits inside its interval


def test_few_seeds_give_wider_cis_than_many() -> None:
    # The honesty point: 2 seeds can't claim a tight bound. Same spread, more seeds → tighter CI.
    two = np.array([[0.2], [0.8]])
    many = np.array([[x] for x in np.linspace(0.2, 0.8, 8)])
    lo2, hi2 = rl.bootstrap_ci(two, rl.aggregate_mean, reps=1000, rng_seed=0)
    lo8, hi8 = rl.bootstrap_ci(many, rl.aggregate_mean, reps=1000, rng_seed=0)
    assert (hi2 - lo2) > (hi8 - lo8)


def test_aggregate_estimate_bundles_value_and_ci() -> None:
    m = np.array([[0.0], [0.5], [1.0]])
    value, lo, hi = rl.aggregate_estimate(m, "iqm", reps=300, rng_seed=0)
    assert lo <= value <= hi


# ===========================================================================
# rliable_metrics.py — probability of improvement
# ===========================================================================


def test_prob_of_improvement_dominant_and_tie() -> None:
    x = np.array([[1.0], [0.9], [0.8]])
    y = np.array([[0.1], [0.2], [0.0]])
    val, lo, hi = rl.probability_of_improvement(x, y, reps=300, rng_seed=0)
    assert math.isclose(val, 1.0)  # every x beats every y
    tie, _, _ = rl.probability_of_improvement(x, x, reps=300, rng_seed=0)
    assert math.isclose(tie, 0.5)  # identical → coin flip (ties count half)


# ===========================================================================
# rliable_metrics.py — score-matrix construction
# ===========================================================================


def test_build_score_matrix_keeps_common_seeds_only() -> None:
    entries = [
        ("cartpole", 1, 0.5), ("cartpole", 2, 0.6), ("cartpole", 3, None),  # None dropped
        ("acrobot", 1, 0.7),  # only seed 1 → intersection is {1}
    ]
    sm = rl.build_score_matrix(entries)
    assert sm.tasks == ["acrobot", "cartpole"] and sm.seeds == [1]
    assert sm.matrix.shape == (1, 2)
    assert list(sm.matrix[0]) == [0.7, 0.5]  # columns sorted by task id


def test_build_score_matrix_empty_when_no_common_seed() -> None:
    sm = rl.build_score_matrix([("cartpole", 1, 0.5), ("acrobot", 2, 0.6)])
    assert sm.matrix.size == 0


def test_shared_task_columns_aligns_two_methods() -> None:
    a = rl.build_score_matrix([("cartpole", 1, 0.5), ("acrobot", 1, 0.6)])
    b = rl.build_score_matrix([("cartpole", 1, 0.2), ("mountaincar", 1, 0.9)])
    ma, mb, shared = rl.shared_task_columns(a, b)
    assert shared == ["cartpole"] and ma.shape == (1, 1) and mb.shape == (1, 1)
    assert ma[0, 0] == 0.5 and mb[0, 0] == 0.2


# ===========================================================================
# API routes
# ===========================================================================


def _store(tmp_path: Path, monkeypatch) -> RunStore:
    store = RunStore(tmp_path / "runs")
    monkeypatch.setattr(export_engine, "run_store", store)
    monkeypatch.setattr(analysis_api, "run_store", store)
    return store


def _rising(env_steps_top: int, top: float) -> list[dict]:
    return [_frame(0, 0.0, 0.0), _frame(env_steps_top // 2, top / 2, 1.0), _frame(env_steps_top, top, 2.0)]


def _save_sweep(store: RunStore, env_id: str, algo: str, seeds: list[int], exp: str, top: float) -> None:
    for s in seeds:
        cfg = TrainConfig(env_id=env_id, algo=algo, seed=s, experiment_id=exp, experiment_label="sweep")
        store.save(cfg, _rising(6000, top), state="finished",
                   started_at="2026-07-03T10:00:00+00:00", solved_score=500.0)


def test_experiments_route_groups_a_sweep(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    _save_sweep(store, "cartpole", "ppo", [42, 43, 44], "expA", 480.0)
    client = TestClient(app)
    exps = client.get("/api/analysis/experiments").json()
    match = next(e for e in exps if e["experiment_id"] == "expA")
    assert match["n_seeds"] == 3 and sorted(match["seeds"]) == [42, 43, 44]


def test_aggregate_route_band_and_seed_subset(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    _save_sweep(store, "cartpole", "ppo", [42, 43, 44], "expA", 480.0)
    client = TestClient(app)

    full = client.get("/api/analysis/aggregate", params={
        "experiment_id": "expA", "axis": "env_steps", "metric": "skill_pct", "points": 20}).json()
    assert full["band"]["n_seeds"] == 3 and len(full["band"]["x"]) == 20
    assert "final_skill_pct" in full["summary"]["metrics"]

    # Excluding a seed recomputes the band from just the two remaining seeds.
    subset = client.get("/api/analysis/aggregate", params={
        "experiment_id": "expA", "seeds": [42, 43]}).json()
    assert subset["band"]["n_seeds"] == 2 and sorted(subset["band"]["seeds"]) == [42, 43]


def test_aggregate_route_404_when_empty(tmp_path: Path, monkeypatch) -> None:
    _store(tmp_path, monkeypatch)
    client = TestClient(app)
    assert client.get("/api/analysis/aggregate", params={"experiment_id": "nope"}).status_code == 404


def test_rliable_route_two_methods_and_prob_improvement(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    # Two algorithms over the same two games, 2 seeds each. ppo clearly stronger than the weak run.
    for algo, top in (("ppo", 480.0), ("sac", 60.0)):
        for env_id in ("cartpole", "acrobot"):
            for s in (1, 2):
                cfg = TrainConfig(env_id=env_id, algo=algo, seed=s)
                store.save(cfg, _rising(6000, top), state="finished",
                           started_at="2026-07-03T10:00:00+00:00", solved_score=500.0)
    run_ids = [m.id for m in store.list()]
    client = TestClient(app)
    result = client.get("/api/analysis/rliable", params={"run_ids": run_ids, "reps": 200}).json()
    algos = {m["algo"] for m in result["methods"]}
    assert {"ppo", "sac"} <= algos
    for method in result["methods"]:
        assert method["n_runs"] == 2 and set(method["tasks"]) == {"cartpole", "acrobot"}
        est = method["iqm"]
        assert est["ci_low"] <= est["value"] <= est["ci_high"]
    assert result["prob_of_improvement"] is not None


def test_npz_score_matrix_round_trips(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path, monkeypatch)
    _save_sweep(store, "cartpole", "ppo", [1, 2, 3], "expA", 480.0)
    run_ids = [m.id for m in store.list()]
    client = TestClient(app)
    resp = client.get("/api/analysis/export.npz", params={"run_ids": run_ids})
    assert resp.status_code == 200
    loaded = np.load(io.BytesIO(resp.content))  # no allow_pickle needed
    assert loaded["ppo__matrix"].shape == (3, 1)  # 3 seeds × 1 task
    assert list(loaded["ppo__tasks"]) == ["cartpole"]
    assert sorted(loaded["ppo__seeds"].tolist()) == [1, 2, 3]
