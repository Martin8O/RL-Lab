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
