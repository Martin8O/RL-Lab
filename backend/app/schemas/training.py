"""Training contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

These shapes are shared by the REST control endpoints (/api/train/*) and the WebSocket
metric/status frames, so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field

Algo = Literal["ppo", "neuroevolution", "q_learning", "alphazero"]
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
    n_epochs: int = 10  # passes over each rollout per update; Atari uses fewer (4) for throughput
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


class SelfPlayHyperparams(BaseModel):
    """Tunable knobs for competitive multi-agent self-play (simple_tag, G7b-2).

    The per-species PPO networks reuse :class:`PPOHyperparams` (one shared net per species, frozen
    self-play, ADR-048). The only self-play-specific knob is ``rounds``: how many times the two
    species alternate (each round = both species get one learning turn against the other's frozen
    snapshot). More rounds = a deeper arms race but a longer run; the per-round budget is
    ``total_timesteps / (rounds × n_species)``.
    """

    rounds: int = 8


class AlphaZeroHyperparams(BaseModel):
    """Tunable AlphaZero-lite knobs (the 4th algorithm, G6f — the board branch's algorithm jump).

    AlphaZero learns a board game purely by **self-play**: each move runs ``simulations`` of
    neural-guided MCTS (a CNN policy+value net guiding tree search in place of G6a's random rollout), and
    the search's visit counts train the net — no human data and no MCTS *teacher*, unlike the G6b
    MaskablePPO trainer it competes with on the same board. The budget is ``iterations`` ×
    ``games_per_iter`` self-play games (this algorithm's "Total Steps"); more ``simulations`` = sharper
    move targets and stronger play, but slower self-play. The net size + replay/exploration knobs are
    fixed at sensible defaults rather than surfaced as sliders, to keep the panel focused.

    G6g rebuilt the engine to be **GPU-bound**: ``parallel_games`` self-play games run concurrently and
    every MCTS step batches their leaf evaluations into one wide GPU forward (G6f did batch-1 forwards,
    where a GPU sits idle/slower), feeding a bigger ResNet (``channels`` × ``blocks`` with GroupNorm).

    G6h swaps the self-play **search**: ``use_gumbel`` runs Gumbel AlphaZero (Gumbel-Top-``gumbel_considered``
    + Sequential Halving over ``gumbel_sims`` simulations) instead of PUCT. Gumbel provably improves the
    policy at far fewer simulations, so the self-play search budget drops (``gumbel_sims`` ★16 vs the old
    ``simulations`` ★30–50) for an equal-or-better target and ~2× the games/s. ``simulations`` is kept as
    the PUCT fallback (``use_gumbel=False``); the per-iteration **eval** stays PUCT/argmax (honest yardstick).
    """

    learning_rate: float = 5e-4  # Adam step for the net update (gentler than PPO's, for stability)
    # Gumbel self-play search (G6h, the default) — a low-sim, best-arm-identification root search.
    use_gumbel: bool = True  # self-play uses Gumbel-Top-k + Sequential Halving (False ⇒ PUCT `simulations`)
    gumbel_sims: int = 16  # simulations per move under Gumbel — far fewer than PUCT for the same strength
    gumbel_considered: int = 16  # root moves Sequential Halving picks among (m; capped by the legal count)
    simulations: int = 50  # PUCT fallback: neural-MCTS sims per move when use_gumbel is False
    games_per_iter: int = 24  # self-play games generated per iteration
    iterations: int = 30  # training iterations — this algorithm's budget
    # Non-UI knobs (sensible fixed defaults; not exposed as sliders). Tuned via Local/_probe_g6f_learn.py:
    # gentle training (a few epochs over the buffer, not a fixed large step count) avoids the value-head
    # overfit that poisons the MCTS→target feedback loop — the difference between learning and stalling.
    c_puct: float = 2.0  # PUCT exploration constant for the self-play MCTS
    # The batched-GPU engine (G6g): a bigger ResNet (128 channels × 10 residual blocks) with batch-size-
    # independent GroupNorm, and `parallel_games` self-play games run concurrently so each MCTS step is one
    # wide GPU forward (G6f's batch-1 forwards left the GPU idle/slower — measured). `parallel_games` is the
    # batch cap; the effective batch is min(parallel_games, games_per_iter).
    channels: int = 128  # CNN width (G6f "lite" was 64)
    blocks: int = 10  # CNN residual blocks (G6f "lite" was 4)
    norm: str = "group"  # GroupNorm in the tower (batch-independent → no BatchNorm train/eval pitfalls)
    parallel_games: int = 128  # concurrent self-play games batched into one GPU forward per MCTS step
    # (G6g review "A": raised 64→128 so a games_per_iter=128 run actually batches 128 wide — the profiled
    # ceiling on the RTX 5070, ~12% over a 64-cohort and a fuller GPU; small games stay capped by their own
    # games_per_iter so this only widens the high-throughput chess runs. VRAM-validated at batch 128.)
    batch_size: int = 128
    train_epochs: float = 2.0  # passes over the replay buffer per iteration (gentle — avoids overfit)
    buffer_size: int = 80_000  # replay window (self-play positions) — wider for the bigger net + batches
    temp_moves: int = 6  # opening plies sampled from the visit distribution (exploration), then greedy
    # Inference (with search) — the net's REAL strength: a few MCTS sims guided by the CNN, used for the
    # honest eval-vs-reference curve and the human's Play-vs-net opponent (NOT the bare policy head). The
    # eval count is small so the per-iteration eval stays a few seconds; play can afford more per move.
    eval_simulations: int = 30  # neural-MCTS sims per move when scoring the curve
    play_simulations: int = 60  # neural-MCTS sims per move when a human plays the trained net


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
    configures tabular Q-learning; ``alphazero`` configures the AlphaZero-lite board trainer.
    Exactly one applies, selected by ``algo``; the others stay None so the recorded config is clean.
    """

    env_id: str = "cartpole"
    algo: Algo = "ppo"
    seed: int = 42
    total_timesteps: int = 50_000
    hyperparams: PPOHyperparams = Field(default_factory=PPOHyperparams)
    evolution: EvolutionHyperparams | None = None
    q_learning: QLearningHyperparams | None = None
    # Present only for competitive multi-agent self-play runs (simple_tag); None otherwise. The
    # per-species PPO uses ``hyperparams``; this carries only the self-play round schedule (G7b-2).
    self_play: SelfPlayHyperparams | None = None
    # Present only for AlphaZero-lite board runs (algo=="alphazero", G6f); None otherwise. The budget
    # is iterations × games_per_iter self-play games, so total_timesteps is set to match by the client.
    alphazero: AlphaZeroHyperparams | None = None


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


class HwStats(BaseModel):
    """Live hardware telemetry, sampled onto each 1 Hz progress frame (G4b).

    CPU + RAM are always present (``psutil``); the GPU fields are **optional** — ``None`` when
    NVML/``pynvml`` is unavailable (a non-NVIDIA machine) so the panel can show ``—`` rather than a
    misleading 0. ``cpu_process_pct`` is this process's CPU use normalised to 0–100 % of the whole
    machine (raw ``psutil`` per-process % can exceed 100 on multi-core); memory is reported in MB.
    """

    cpu_process_pct: float
    ram_used_mb: float
    ram_total_mb: float
    gpu_util_pct: float | None = None
    gpu_vram_used_mb: float | None = None
    gpu_vram_total_mb: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None


class HwStatsFrame(BaseModel):
    """WS frame: {type:"hwstats", ...} — one 1 Hz hardware-telemetry sample (G4b).

    Broadcast by the training manager for the lifetime of *any* active run, independent of the
    algorithm: PPO, neuroevolution and tabular Q-learning all get the HW panel (the PPO progress
    ticker is the wrong home — evolution/Q-learning emit no progress frame). Decoupled + unlogged.
    """

    type: Literal["hwstats"] = "hwstats"
    stats: HwStats


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


class SpeciesMetrics(BaseModel):
    """One species' current learning stats inside a competitive self-play frame (simple_tag, G7b-2).

    ``role`` is PettingZoo's species tag — ``"adversary"`` (predators) or ``"agent"`` (prey). The
    return is per-agent (the shared net's mean episode return for that species). ``timesteps`` is the
    species' own cumulative learned steps (it only grows during that species' learning turns).
    """

    role: str
    ep_rew_mean: float | None
    ep_len_mean: float | None
    timesteps: int


class MultiAgentMetrics(BaseModel):
    """One competitive self-play frame, pushed over WS as {type:"ma_metrics", ...} (simple_tag, G7b-2).

    Two species learn by alternating frozen-opponent rounds (ADR-048), so a single reward line can't
    describe the run — this frame carries **both** species at once for the two-line "ecosystem" chart.
    ``learning_role`` is whichever species is optimising right now (the other plays frozen this round).
    ``ep_rew_mean`` mirrors the **predator** headline return so the generic run-history / high-score
    paths (which read ``ep_rew_mean``) keep working without special-casing this frame.
    """

    type: Literal["ma_metrics"] = "ma_metrics"
    round: int
    total_rounds: int
    learning_role: str
    species: list[SpeciesMetrics]
    ep_rew_mean: float | None  # predator (adversary) headline — drives high-score / archive
    timesteps: int  # cumulative env steps across both species (the chart x-axis)
    total_timesteps: int
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
    # Latest neuroevolution frame, retained so a client that connects mid-run (or after a
    # finished run) can repopulate the leaderboard / Evolution Stats / Fitness chart without
    # waiting for the next generation. None for PPO runs (use ``last_metrics`` there).
    last_evolution: EvolutionMetrics | None = None
    # Latest tabular Q-learning frame + Q-table snapshot, retained so a client connecting mid-run
    # (or after one finishes) repopulates the chart / stats / heatmap without waiting for the next
    # report. None for PPO / neuroevolution runs.
    last_q_learning: QLearningMetrics | None = None
    last_qtable: QTableFrame | None = None
    # Latest competitive self-play frame (simple_tag), retained so a client connecting mid-run (or
    # after one finishes) repopulates the two-line ecosystem chart without waiting for the next round.
    # None for every single-policy run (PPO / neuroevolution / Q-learning).
    last_ma_metrics: MultiAgentMetrics | None = None
    error: str | None = None
