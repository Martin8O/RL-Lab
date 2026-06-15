"""Training contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

These shapes are shared by the REST control endpoints (/api/train/*) and the WebSocket
metric/status frames, so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field

Algo = Literal["ppo", "neuroevolution", "q_learning"]
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


class EvolutionHyperparams(BaseModel):
    """Tunable neuroevolution knobs. Defaults match the cookbook's ★ recommended values.

    ``episodes`` (eval episodes per genome) is a non-UI knob — fitness is the *mean*
    return over this many episodes, which steadies selection without a slider of its own.
    """

    population_size: int = 50
    top_k_parents: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    generations: int = 30
    episodes: int = 3


class QLearningHyperparams(BaseModel):
    """Tunable tabular Q-learning knobs (the 3rd algorithm, G2b).

    Q-learning learns a ``[n_states × n_actions]`` table of action values directly from a
    *discrete* observation (Toy Text). ``learning_rate`` (α) is the table-update step (far larger
    than PPO's gradient step). ε-greedy exploration anneals from ``epsilon_start`` to
    ``epsilon_end`` over the first ``epsilon_decay`` *fraction* of the episode budget, then holds
    at the end value — a budget-relative schedule so it behaves the same whether a game wants
    3 000 or 20 000 episodes. ``episodes`` is the training budget (this algorithm's "Total Steps").
    """

    learning_rate: float = 0.1  # α — the Bellman update step (much larger than PPO's gradient lr)
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.5  # fraction of the episode budget to anneal ε over (then hold at end)
    episodes: int = 5_000


class TrainConfig(BaseModel):
    """Full, reproducible description of a training run (echoed back in status).

    ``hyperparams`` configures PPO; ``evolution`` configures neuroevolution; ``q_learning``
    configures tabular Q-learning. Exactly one applies, selected by ``algo``; the others stay
    None so the recorded config is clean.
    """

    env_id: str = "cartpole"
    algo: Algo = "ppo"
    seed: int = 42
    total_timesteps: int = 50_000
    hyperparams: PPOHyperparams = Field(default_factory=PPOHyperparams)
    evolution: EvolutionHyperparams | None = None
    q_learning: QLearningHyperparams | None = None


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


class TrainingProgress(BaseModel):
    """Lightweight ~1 Hz progress frame, pushed over WS as {type:"progress", ...}.

    Emitted by a decoupled ticker thread (not SB3's per-step callback, which is dormant
    during the PPO update phase) so the live stats refresh at a steady ~1 Hz regardless of
    training phase. Carries the rolling reward/length means too, so the reward chart can be
    plotted at ~1 Hz instead of only once per rollout.
    """

    type: Literal["progress"] = "progress"
    iteration: int
    timesteps: int
    total_timesteps: int
    steps_per_sec: float
    ep_rew_mean: float | None = None
    ep_len_mean: float | None = None
    elapsed: float


class EvolutionChild(BaseModel):
    """One ranked genome in a generation's Top-K leaderboard.

    ``avg_reward`` is the genome's fitness (mean episode return). ``seed`` is the
    deterministic env seed it was scored with — surfaced (instead of a meaningless γ/α)
    so a child's run can be reproduced exactly.
    """

    id: int  # unique + increasing across the run: (generation-1)*population + rank
    total_reward: float
    avg_reward: float
    steps: int
    seed: int


class MutationDist(BaseModel):
    """Histogram of the weight perturbations applied to breed this generation's offspring."""

    bins: list[float]  # bin edges; len == len(counts) + 1
    counts: list[int]


class EvolutionMetrics(BaseModel):
    """One per-generation frame, pushed over WS as {type:"evolution", ...}."""

    type: Literal["evolution"] = "evolution"
    generation: int
    total_generations: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    children: list[EvolutionChild]  # Top-5
    mutation_dist: MutationDist
    timesteps: int  # cumulative env steps simulated so far (across all generations)
    elapsed: float


class QLearningMetrics(BaseModel):
    """One periodic Q-learning frame, pushed over WS as {type:"q_learning", ...}.

    Q-learning is *episodic* (not rollout/timestep- or generation-based), so its x-axis is the
    episode counter. ``ep_rew_mean`` is the mean return over the most recent batch of episodes
    (the headline learning curve — for FrozenLake this is literally the success rate). The
    table itself rides in the separate, unlogged :class:`QTableFrame` so the per-frame history
    stays light. Emitted every ``report_every`` episodes (plus a final frame).
    """

    type: Literal["q_learning"] = "q_learning"
    iteration: int  # report index (1, 2, 3 …) — the "work done" counter, mirrors PPO's iteration
    episode: int
    total_episodes: int
    epsilon: float
    ep_rew_mean: float | None
    ep_len_mean: float | None
    timesteps: int  # cumulative env steps simulated so far (across all episodes)
    elapsed: float


class QTable(BaseModel):
    """A snapshot of the learned action-value table for the heatmap (the "watch it fill in" view).

    ``values`` is row-major ``[n_states][n_actions]`` Q-values. Kept out of the logged metric
    history (it is large for Taxi's 500 states) and streamed only in :class:`QTableFrame`.
    """

    n_states: int
    n_actions: int
    values: list[list[float]]


class QTableFrame(BaseModel):
    """WS frame: {type:"qtable", ...} — the current Q-table snapshot for the live heatmap.

    Decoupled from :class:`QLearningMetrics` (and never logged into the run's metric history) so
    the table can be streamed at a steady cadence and a late-joining client can repopulate the
    heatmap from :class:`TrainStatus.last_qtable` without bloating checkpoints / run history.
    """

    type: Literal["qtable"] = "qtable"
    episode: int
    total_episodes: int
    table: QTable


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
    # Latest neuroevolution frame, retained so a client that connects mid-run (or after a
    # finished run) can repopulate the leaderboard / Evolution Stats / Fitness chart without
    # waiting for the next generation. None for PPO runs (use ``last_metrics`` there).
    last_evolution: EvolutionMetrics | None = None
    # Latest tabular Q-learning frame + Q-table snapshot, retained so a client connecting mid-run
    # (or after one finishes) repopulates the chart / stats / heatmap without waiting for the next
    # report. None for PPO / neuroevolution runs.
    last_q_learning: QLearningMetrics | None = None
    last_qtable: QTableFrame | None = None
    error: str | None = None
