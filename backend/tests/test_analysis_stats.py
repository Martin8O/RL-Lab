"""X2 — the summary-statistics engine (pure) + its API route.

Two things under test:
  1. :func:`app.services.analysis.stats.summarize` computes all six metric groups correctly on
     hand-built curves, and degrades gracefully (no crash, ``None`` results) on degenerate ones.
  2. ``GET /api/analysis/summary`` round-trips a saved run and skips unknown ids.

The normalized skill % is asserted to match the SAME ``(r−min)/(solved−min)`` clamp the live chart
uses, so the summary and the meter stay one source of truth.
"""

import math
from pathlib import Path

from app.api import analysis as analysis_api
from app.main import app
from app.schemas.training import TrainConfig
from app.services.analysis.stats import skill_pct, summarize
from app.services.runs import RunStore
from fastapi.testclient import TestClient


def _frame(env_steps: int, ep_rew_mean: float | None, elapsed: float, **extra: object) -> dict:
    """A minimal metrics-frame dict with the X1 canonical axes present."""
    return {
        "type": "metrics", "env_steps": env_steps, "timesteps": env_steps,
        "ep_rew_mean": ep_rew_mean, "wall_clock": elapsed, "elapsed": elapsed, **extra,
    }


# -- skill % is the chart's formula ----------------------------------------


def test_skill_pct_matches_chart_formula_and_clamps() -> None:
    # Chart: clamp((r - min) / (solved - min) * 100, 0, 100). LunarLander-shaped range.
    assert skill_pct(-54.0, -200.0, 200.0) == ((-54.0 - -200.0) / (200.0 - -200.0)) * 100.0
    assert skill_pct(500.0, 0.0, 500.0) == 100.0
    assert skill_pct(9999.0, 0.0, 500.0) == 100.0  # clamped high
    assert skill_pct(-999.0, 0.0, 500.0) == 0.0  # clamped low
    assert skill_pct(5.0, 10.0, 10.0) is None  # degenerate range
    assert skill_pct(None, 0.0, 500.0) is None


# -- the six metric groups on a clean rising curve --------------------------


def test_summarize_full_rising_curve() -> None:
    # A CartPole-like climb to solved (500) then flat, 0..500 skill range.
    frames = [
        _frame(2048, 20.0, 1.0),
        _frame(4096, 200.0, 2.0),
        _frame(6144, 500.0, 3.0),
        _frame(8192, 500.0, 4.0),
        _frame(10240, 490.0, 5.0),
    ]
    s = summarize(
        run_id="r1", env_id="cartpole", algo="ppo", seed=7,
        frames=frames, min_score=0.0, solved_score=500.0,
    )
    assert s.n_frames == 5
    # 1. final = mean of last 10% → last frame only (round(5*0.1)=1) → 490.
    assert s.final_reward == 490.0
    assert s.final_skill_pct == 98.0
    # 2. first frame to hit solved (500) is env_steps 6144 / wall_clock 3.0.
    assert s.solved_env_steps == 6144
    assert s.solved_wall_clock == 3.0
    # 3. AUC in (0, 1) and below 1 (it starts low).
    assert s.auc_normalized is not None and 0.0 < s.auc_normalized < 1.0
    # 5. throughput = last env_steps / last wall_clock.
    assert s.mean_steps_per_sec == 10240 / 5.0
    assert s.final_env_steps == 10240 and s.final_wall_clock == 5.0
    # 6. peak = 500 (first reached at 6144); gave back 2 skill points (100 → 98).
    assert s.peak_reward == 500.0 and s.peak_env_steps == 6144
    assert s.peak_skill_pct == 100.0
    assert s.collapse_pct == 2.0


def test_summarize_detects_post_peak_collapse() -> None:
    # Value-based-style collapse: climbs to 400 then falls back to 100.
    frames = [_frame(1000, 100.0, 1.0), _frame(2000, 400.0, 2.0), _frame(3000, 100.0, 3.0)]
    s = summarize(
        run_id="r", env_id="cartpole", algo="ppo", seed=1,
        frames=frames, min_score=0.0, solved_score=500.0,
    )
    assert s.peak_reward == 400.0 and s.peak_env_steps == 2000
    # peak skill 80, final skill 20 → 60 points given back.
    assert s.collapse_pct == 60.0
    assert s.solved_env_steps is None  # never reached 500


def test_summarize_neuroevolution_reads_best_fitness() -> None:
    frames = [
        {"type": "evolution", "env_steps": 6000, "timesteps": 6000, "best_fitness": 120.0, "wall_clock": 4.0, "elapsed": 4.0},
        {"type": "evolution", "env_steps": 12000, "timesteps": 12000, "best_fitness": 480.0, "wall_clock": 8.0, "elapsed": 8.0},
    ]
    s = summarize(
        run_id="e", env_id="cartpole", algo="neuroevolution", seed=3,
        frames=frames, min_score=0.0, solved_score=500.0,
    )
    assert s.final_reward == 480.0  # read from best_fitness, not ep_rew_mean
    assert s.peak_reward == 480.0


# -- degenerate / sparse curves must not crash ------------------------------


def test_summarize_empty_curve() -> None:
    s = summarize(
        run_id="empty", env_id="cartpole", algo="ppo", seed=0,
        frames=[], min_score=0.0, solved_score=500.0,
    )
    assert s.n_frames == 0
    assert s.final_reward is None and s.auc_normalized is None
    assert s.peak_reward is None and s.mean_steps_per_sec is None


def test_summarize_single_point() -> None:
    s = summarize(
        run_id="one", env_id="cartpole", algo="ppo", seed=0,
        frames=[_frame(2048, 250.0, 2.0)], min_score=0.0, solved_score=500.0,
    )
    assert s.final_reward == 250.0
    assert s.auc_normalized == 0.5  # single point → its own skill fraction
    assert s.late_reward_std == 0.0  # one value → no roughness
    assert s.collapse_pct == 0.0


def test_summarize_all_none_scores() -> None:
    # A run whose reward never populated (all None) still returns axes, no reward stats, no crash.
    frames = [_frame(1000, None, 1.0), _frame(2000, None, 2.0)]
    s = summarize(
        run_id="n", env_id="cartpole", algo="ppo", seed=0,
        frames=frames, min_score=0.0, solved_score=500.0,
    )
    assert s.final_reward is None and s.auc_normalized is None and s.peak_reward is None
    assert s.final_env_steps == 2000  # axes still computed
    assert s.mean_steps_per_sec == 1000.0


def test_summarize_unknown_env_range_gives_no_skill() -> None:
    # solved <= min (unknown env fallback 0/0) → no normalization, but raw stats still populate.
    s = summarize(
        run_id="u", env_id="mystery", algo="ppo", seed=0,
        frames=[_frame(1000, 5.0, 1.0), _frame(2000, 9.0, 2.0)], min_score=0.0, solved_score=0.0,
    )
    assert s.final_reward == 9.0
    assert s.final_skill_pct is None and s.auc_normalized is None
    assert s.solved_env_steps is None


def test_summarize_zero_wallclock_no_div_by_zero() -> None:
    s = summarize(
        run_id="z", env_id="cartpole", algo="ppo", seed=0,
        frames=[_frame(1000, 10.0, 0.0)], min_score=0.0, solved_score=500.0,
    )
    assert s.mean_steps_per_sec is None
    assert not math.isinf(s.final_wall_clock)


# -- the API route: store round-trip + unknown-id skip ----------------------


def test_summary_endpoint_roundtrips_saved_run(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runs")
    monkeypatch.setattr(analysis_api, "run_store", store)

    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=7)
    frames = [_frame(2048, 30.0, 1.0), _frame(4096, 500.0, 2.0), _frame(6144, 500.0, 3.0)]
    meta = store.save(cfg, frames, state="finished", started_at="2026-07-02T10:00:00+00:00", solved_score=500.0)

    client = TestClient(app)
    resp = client.get("/api/analysis/summary", params={"run_ids": [meta.id, "does-not-exist"]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1  # the unknown id is skipped, not fatal
    (summary,) = data
    assert summary["run_id"] == meta.id
    assert summary["env_id"] == "cartpole" and summary["algo"] == "ppo"
    assert summary["solved_env_steps"] == 4096
    assert summary["peak_reward"] == 500.0


def test_summary_endpoint_empty_run_ids() -> None:
    resp = TestClient(app).get("/api/analysis/summary")
    assert resp.status_code == 200 and resp.json() == []
