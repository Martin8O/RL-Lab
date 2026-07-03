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
    # Curation (X7) — editable sidecar fields the user sets from the Data Lab to organize the archive.
    # ``note`` is a free-text annotation; ``excluded`` curates a run *out of analysis* without deleting it
    # (the Data Lab hides excluded runs from the picker by default, so they can't enter an overlay/export).
    note: str | None = None
    excluded: bool = False


# The mutable subset of RunMeta a user may edit (X7). Every field is optional so a PATCH is a partial
# update: only the fields actually sent (``model_fields_set``) are applied, so sending ``note: null``
# clears the note while omitting it leaves it untouched. Immutable provenance (id, config, metrics) is
# never touched — curation edits only the meta.json sidecar.
class RunMetaPatch(BaseModel):
    label: str | None = None
    note: str | None = None
    experiment_id: str | None = None
    experiment_label: str | None = None
    excluded: bool | None = None


class GroupRequest(BaseModel):
    """Tag a set of runs into one named experiment (X7) — the manual counterpart of an X3 seed sweep's
    auto-shared ``experiment_id``. Passing ``experiment_id=None`` ungroups the runs (clears the tag)."""

    run_ids: list[str]
    experiment_id: str | None = None
    experiment_label: str | None = None


class BulkDeleteRequest(BaseModel):
    run_ids: list[str]


class BulkDeleteResult(BaseModel):
    deleted: int  # how many of the requested runs actually existed and were removed


class RunDetail(BaseModel):
    """A run read back in full: its listing row, the reproducible config, and every
    recorded metric frame (each a :class:`~app.schemas.training.TrainingMetrics` or
    :class:`~app.schemas.training.EvolutionMetrics` dump) for the chart overlay."""

    meta: RunMeta
    config: TrainConfig
    metrics: list[dict[str, Any]]
