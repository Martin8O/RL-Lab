"""Summary-statistics engine (X2, ADR-083) — the scalars a paper reports, from one run's curve.

Pure functions only: no I/O, no global state. The caller loads a run from the run store and hands the
recorded metric frames + the env's ``[min_score, solved_score]`` here; :func:`summarize` distils them
into a :class:`~app.schemas.analysis.RunSummary`. Everything is computed on the **canonical comparison
axes** (X1) — ``env_steps`` (cumulative environment interactions) and ``wall_clock`` (elapsed seconds) —
so runs from any algorithm are reduced the same way.

Robust to sparse / short / ``None``-laden curves: a 3-point run, an all-``None`` reward series, a
zero-length x-axis all return ``None`` for what can't be computed rather than raising.
"""

from statistics import pstdev
from typing import Any

from app.schemas.analysis import RunSummary
from app.schemas.training import Algo

# Fraction of the curve treated as its "late" tail for the noise-robust final performance + stability.
_LATE_FRACTION = 0.10


def score_of_frame(algo: str, frame: dict[str, Any]) -> float | None:
    """The per-frame headline score on the env's reward scale.

    ``best_fitness`` for neuroevolution (its curve is fitness), ``ep_rew_mean`` for every other trainer
    (PPO / off-policy / Q-learning / board / competitive self-play all report the mean episode return).
    Mirrors ``services.runs._frame_score`` — kept here so the analysis engine stays self-contained.
    """
    return frame.get("best_fitness") if algo == "neuroevolution" else frame.get("ep_rew_mean")


def skill_pct(reward: float | None, min_score: float, solved_score: float) -> float | None:
    """Normalize a reward to the 0–100 % skill scale — the SAME ``(r−min)/(solved−min)`` clamp the live
    reward chart uses (``RewardChart.tsx`` ``solvePct``), so a run's summary reads consistently with the
    meter. ``None`` when the reward is missing or the env has no usable range (``solved ≤ min``)."""
    if reward is None or solved_score <= min_score:
        return None
    frac = (reward - min_score) / (solved_score - min_score)
    return max(0.0, min(100.0, frac * 100.0))


def _frame_x(frame: dict[str, Any]) -> int:
    """A frame's canonical env-steps x (X1); falls back to ``timesteps`` for a not-yet-backfilled dict."""
    return int(frame.get("env_steps", frame.get("timesteps", 0)) or 0)


def _frame_w(frame: dict[str, Any]) -> float:
    """A frame's canonical wall-clock seconds (X1); falls back to ``elapsed``."""
    return float(frame.get("wall_clock", frame.get("elapsed", 0.0)) or 0.0)


def _late_slice(values: list[Any]) -> list[Any]:
    """The last ``_LATE_FRACTION`` of a series (at least one element) — the noise-robust tail."""
    if not values:
        return []
    n = max(1, round(len(values) * _LATE_FRACTION))
    return values[-n:]


def _auc_normalized(xs: list[int], skills: list[float | None]) -> float | None:
    """Trapezoidal area under the normalized-skill (0–1) curve over ``env_steps``, divided by the step
    range → the mean skill across the run (comparable across runs of different length). Needs points with
    a real score; a single point returns its skill; a zero-width x-range returns the plain mean."""
    pts = [(x, s) for x, s in zip(xs, skills, strict=True) if s is not None]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1]
    x0, x1 = pts[0][0], pts[-1][0]
    span = x1 - x0
    if span <= 0:
        return sum(s for _, s in pts) / len(pts)
    area = 0.0
    for (xa, sa), (xb, sb) in zip(pts, pts[1:], strict=False):
        area += 0.5 * (sa + sb) * (xb - xa)
    return area / span


def summarize(
    *,
    run_id: str,
    env_id: str,
    algo: Algo,
    seed: int,
    frames: list[dict[str, Any]],
    min_score: float,
    solved_score: float,
) -> RunSummary:
    """Reduce one run's recorded metric ``frames`` into the standard RL summary statistics.

    ``min_score`` / ``solved_score`` are the env's registry skill range (the same span the chart + skill
    meter use). Pure: the caller loads ``frames`` (via the run store, which backfills the X1 axes) and the
    range (via the env registry) and passes them in. See :class:`~app.schemas.analysis.RunSummary`.
    """
    base = RunSummary(
        run_id=run_id, env_id=env_id, algo=algo, seed=seed,
        n_frames=len(frames), min_score=min_score, solved_score=solved_score,
    )
    if not frames:
        return base

    xs = [_frame_x(f) for f in frames]
    ws = [_frame_w(f) for f in frames]
    scores = [score_of_frame(algo, f) for f in frames]
    valid = [s for s in scores if s is not None]

    base.final_env_steps = xs[-1]
    base.final_wall_clock = ws[-1]
    base.mean_steps_per_sec = xs[-1] / ws[-1] if ws[-1] > 0 else None

    # 1. Final performance — mean reward over the late tail (robust to a noisy last point), + skill %.
    late_scores = [s for s in _late_slice(scores) if s is not None]
    if late_scores:
        base.final_reward = sum(late_scores) / len(late_scores)
        base.final_skill_pct = skill_pct(base.final_reward, min_score, solved_score)

    # 2. Sample efficiency — the first frame to reach the solved score, on both canonical axes.
    if solved_score > min_score:
        for x, w, s in zip(xs, ws, scores, strict=True):
            if s is not None and s >= solved_score:
                base.solved_env_steps = x
                base.solved_wall_clock = w
                break

    # 3. AUC of the normalized (0–1 skill) learning curve over env_steps. A None skill (missing score OR
    #    a degenerate env range) stays None so a run with no usable skill scale yields no AUC (not a bogus 0).
    def _skill_frac(s: float | None) -> float | None:
        p = skill_pct(s, min_score, solved_score)
        return None if p is None else p / 100.0

    base.auc_normalized = _auc_normalized(xs, [_skill_frac(s) for s in scores])

    # 4. Stability — roughness (population std) of the reward over the late tail. (across-seed std: X4.)
    if len(late_scores) >= 1:
        base.late_reward_std = pstdev(late_scores) if len(late_scores) >= 2 else 0.0

    # 6. Peak vs final — did it give back skill after its best point?
    if valid:
        peak = max(valid)
        base.peak_reward = peak
        base.peak_env_steps = next(x for x, s in zip(xs, scores, strict=True) if s == peak)
        base.peak_skill_pct = skill_pct(peak, min_score, solved_score)
        if base.peak_skill_pct is not None and base.final_skill_pct is not None:
            base.collapse_pct = max(0.0, base.peak_skill_pct - base.final_skill_pct)

    return base
