"""Run-history contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

A *run* is a finished training run's full config + metric history, persisted under the
gitignored ``data/runs/`` (see services/runs.py) so past experiments can be listed and
overlaid on the chart for comparison (the cookbook's v1/v2/v3 idea). ``RunMeta`` is the
listing/provenance row returned by ``GET /api/runs``; ``RunDetail`` adds the full config +
the recorded metric frames the chart overlay needs, returned by ``GET /api/runs/{id}``.
"""

from typing import Any

from pydantic import BaseModel

from app.schemas.training import Algo, TrainConfig, TrainState


class RunMeta(BaseModel):
    """One finished run — enough to list it and understand/reproduce the experiment.

    Progress is reported two ways depending on the algorithm: PPO fills
    ``timesteps``/``total_timesteps``/``iteration``; neuroevolution fills
    ``generation``/``total_generations`` (the other stays at its default).
    """

    id: str
    label: str
    env_id: str
    algo: Algo
    seed: int
    created_at: str  # ISO-8601 UTC — when the run started
    finished_at: str  # ISO-8601 UTC — when it reached its terminal state
    state: TrainState  # terminal state that produced this run: "finished" | "stopped"
    final_reward: float | None = None  # last ep_rew_mean (PPO) / best_fitness (evolution)
    # Where the run first reached the solved score (100% of goal): a timestep (PPO) or a
    # generation (evolution), in the same x-unit as that algorithm's chart. None = never
    # solved. The cleanest "how efficient was this run" metric, surfaced in the compare view.
    solved_at: float | None = None
    timesteps: int = 0
    total_timesteps: int = 0
    iteration: int | None = None
    generation: int | None = None
    total_generations: int | None = None
    frames: int = 0  # number of recorded metric frames (overlay resolution)
    # Seed-sweep grouping (X3): runs from one sweep share this id, so the analysis suite (X4) can
    # aggregate the N seeds of an experiment. None for a plain single run. ``experiment_label`` is an
    # optional human name for the sweep.
    experiment_id: str | None = None
    experiment_label: str | None = None


class RunDetail(BaseModel):
    """A run read back in full: its listing row, the reproducible config, and every
    recorded metric frame (each a :class:`~app.schemas.training.TrainingMetrics` or
    :class:`~app.schemas.training.EvolutionMetrics` dump) for the chart overlay."""

    meta: RunMeta
    config: TrainConfig
    metrics: list[dict[str, Any]]
