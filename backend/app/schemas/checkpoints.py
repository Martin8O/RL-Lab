"""Checkpoint contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

A checkpoint is a saved training slot persisted under the gitignored ``data/checkpoints/``
(see services/checkpoints.py) so it is per-device and survives restarts. ``CheckpointMeta`` is
the listing/provenance row returned by ``GET /api/checkpoints[/{id}]`` and ``POST`` (save); the
model artifact and full config/metrics live alongside it on disk and are streamed by the export
endpoint.
"""

from pydantic import BaseModel

from app.schemas.training import Algo


class CheckpointMeta(BaseModel):
    """One saved checkpoint slot — enough to list it and understand/reproduce the run.

    Progress is reported two ways depending on the algorithm: PPO fills
    ``timesteps``/``total_timesteps``/``iteration``; neuroevolution fills
    ``generation``/``total_generations`` (the other stays at its default).
    """

    id: str
    label: str
    env_id: str
    algo: Algo
    seed: int
    created_at: str  # ISO-8601 UTC timestamp
    reward: float | None = None  # PPO ep_rew_mean / evolution best_fitness at save time
    timesteps: int = 0
    total_timesteps: int = 0
    iteration: int | None = None
    generation: int | None = None
    total_generations: int | None = None
    artifact: str  # on-disk model filename: "model.zip" (PPO) | "population.npz" (evolution)


class CheckpointSaveRequest(BaseModel):
    """Optional body for ``POST /api/checkpoints`` — a human label for the slot."""

    label: str | None = None
