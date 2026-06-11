"""Training contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

These shapes are shared by the REST control endpoints (/api/train/*) and the WebSocket
metric/status frames, so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field

Algo = Literal["ppo"]  # neuroevolution joins in Phase C
TrainState = Literal[
    "idle", "running", "paused", "stopping", "stopped", "finished", "error"
]


class PPOHyperparams(BaseModel):
    """Tunable PPO knobs. Defaults match the cookbook's ★ recommended values."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_range: float = 0.2
    ent_coef: float = 0.0
    n_steps: int = 2048
    batch_size: int = 64
    n_hidden_layers: int = 2
    neurons_per_layer: int = 64
    activation: Literal["tanh", "relu"] = "tanh"


class TrainConfig(BaseModel):
    """Full, reproducible description of a training run (echoed back in status)."""

    env_id: str = "cartpole"
    algo: Algo = "ppo"
    seed: int = 42
    total_timesteps: int = 50_000
    hyperparams: PPOHyperparams = Field(default_factory=PPOHyperparams)


class TrainingMetrics(BaseModel):
    """One per-rollout metrics frame, pushed over WS as {type:"metrics", ...}."""

    type: Literal["metrics"] = "metrics"
    iteration: int
    timesteps: int
    total_timesteps: int
    ep_rew_mean: float | None
    ep_len_mean: float | None
    loss: float | None
    learning_rate: float | None
    elapsed: float


class TrainStatus(BaseModel):
    """Lifecycle snapshot — returned by /api/train/* and pushed as {type:"status", ...}."""

    type: Literal["status"] = "status"
    state: TrainState
    env_id: str | None = None
    algo: Algo | None = None
    seed: int | None = None
    timesteps: int = 0
    total_timesteps: int = 0
    config: TrainConfig | None = None
    last_metrics: TrainingMetrics | None = None
    error: str | None = None
