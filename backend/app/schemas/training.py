"""Training contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

These shapes are shared by the REST control endpoints (/api/train/*) and the WebSocket
metric/status frames, so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Algo = Literal[
    "ppo", "neuroevolution", "q_learning", "alphazero", "sac", "td3", "dqn", "a2c", "qrdqn"
]
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
    # Parallel self-play across independent GPU actor processes (G6i, ADR-062). 1 ⇒ today's single
    # in-process threaded actor (byte-identical, the default). >1 ⇒ that many separate worker processes,
    # each with its OWN CUDA net, self-playing locally (no inference server, no per-round IPC) into a shared
    # result queue. Risk-gated on the RTX 5070: 2 workers give ~1.6× chess self-play at GPU 49→94 % (the
    # in-process actor shares the GIL with the learner thread and starves during training bursts; separate
    # processes escape that AND add a core). 2 is the Windows sweet spot — 3 ≈ worse, 4 collapses (Windows
    # WDDM has no MPS to share the GPU). Only takes effect on CUDA; on CPU it falls back to the single actor
    # (the 128×10 net is too heavy for parallel CPU workers — measured ~10× slower).
    actor_processes: int = 1
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


class SACHyperparams(BaseModel):
    """Tunable Soft Actor-Critic knobs (the 5th algorithm, S5a — off-policy continuous control).

    SAC is **off-policy**: it fills a replay buffer of past transitions and learns twin soft-Q
    critics + a squashed-Gaussian actor with entropy regularization — far more sample-efficient than
    PPO on the high-DoF MuJoCo robots (it actually *solves* Humanoid). Gated to continuous-action
    (``Box``) envs only (MuJoCo + BipedalWalker + Pendulum + MountainCarContinuous). Trains on raw
    obs/rewards (NOT VecNormalize — that running reward scaling is on-policy-shaped and would drift
    against a replay buffer; the standard SAC recipe needs neither), so ``ep_rew_mean`` stays raw and
    the ``[min_score, solved_score]`` skill meter reads exactly like PPO's.

    The defaults are SB3's MuJoCo recipe. ``ent_coef`` is a string: ``"auto"`` lets SAC tune the
    entropy temperature itself (the recommended default, almost always best), or a numeric string
    (e.g. ``"0.1"``) pins it. ``batch_size`` / ``learning_starts`` / ``gradient_steps`` are fixed
    (advanced knobs, not sliders, like PPO's ``n_steps``/``n_epochs``); ``gradient_steps`` tracks
    ``train_freq`` in the trainer so the update:collection ratio stays 1:1.
    """

    learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # target-network soft-update coefficient (Polyak averaging)
    buffer_size: int = 1_000_000  # replay-buffer capacity (past transitions to learn from)
    batch_size: int = 256  # minibatch sampled from the buffer per gradient step (fixed, not a slider)
    learning_starts: int = 10_000  # random warmup steps before the first gradient update (fixed)
    train_freq: int = 1  # env steps collected between update phases; gradient_steps tracks this
    ent_coef: str = "auto"  # "auto" = self-tuned entropy temperature; a numeric string pins it


class TD3Hyperparams(BaseModel):
    """Tunable TD3 knobs (the 6th algorithm, S5b — Twin Delayed DDPG, off-policy continuous control).

    TD3 is SAC's sibling: same off-policy machinery (a replay buffer + twin critics + slow target
    nets), gated to the same continuous-action (``Box``) envs, on the same raw obs/rewards (NOT
    VecNormalize), so ``ep_rew_mean`` and the ``[min_score, solved_score]`` skill meter read exactly
    like PPO's. The difference is the *policy*: TD3's actor is **deterministic** (one action per
    state, no entropy), so it has no ``ent_coef``. Its three signature tricks — twin clipped critics,
    **delayed** policy updates (``policy_delay``), and **target-policy smoothing**
    (``target_policy_noise`` clipped to ``target_noise_clip``) — are SB3 defaults and kept fixed.

    Because the policy is deterministic it must explore by *injecting* noise into the actions it
    collects: ``train_noise`` is the std of that Gaussian exploration noise (SAC explores via entropy
    instead — the conceptual analogue). The defaults are SB3's / the TD3 paper's MuJoCo recipe
    (``learning_rate`` 1e-3, net [400, 300]). ``batch_size`` / ``learning_starts`` / ``gradient_steps``
    / ``policy_delay`` / ``target_policy_noise`` / ``target_noise_clip`` are fixed (advanced, not
    sliders); ``gradient_steps`` tracks ``train_freq`` so the update:collection ratio stays 1:1.
    """

    learning_rate: float = 1e-3  # TD3's canonical value (SB3 default; paper uses [400,300] + 1e-3)
    gamma: float = 0.99
    tau: float = 0.005  # target-network soft-update coefficient (Polyak averaging)
    buffer_size: int = 1_000_000  # replay-buffer capacity (past transitions to learn from)
    batch_size: int = 256  # minibatch sampled from the buffer per gradient step (fixed, not a slider)
    learning_starts: int = 10_000  # random warmup steps before the first gradient update (fixed)
    train_freq: int = 1  # env steps collected between update phases; gradient_steps tracks this
    train_noise: float = 0.1  # std of Gaussian exploration noise added to actions (TD3 has no entropy)
    policy_delay: int = 2  # update the actor (+ targets) once per this many critic updates (fixed)
    target_policy_noise: float = 0.2  # std of smoothing noise added to the target action (fixed)
    target_noise_clip: float = 0.5  # the target-smoothing noise is clipped to ±this (fixed)


class DQNHyperparams(BaseModel):
    """Tunable DQN knobs (the 7th algorithm, S5c — Deep Q-Network, off-policy value-based control).

    DQN is the **value-based** counterpart to policy-gradient PPO and the discrete-action mirror of
    SAC/TD3: same off-policy machinery (a replay buffer + a slow target network), but it learns an
    **action-value function** (Q) with a neural net and acts by taking the highest-Q action, instead
    of learning a policy directly. Gated to **discrete-action** envs only (the exact complement of
    SAC/TD3's continuous-``Box`` gate: CartPole + the classic-control discretes + LunarLander + Atari).
    Trains on raw obs/rewards (like SAC/TD3, no VecNormalize), so ``ep_rew_mean`` and the
    ``[min_score, solved_score]`` skill meter read exactly like PPO's.

    The one conceptual difference from SAC/TD3 is *how it explores*: not entropy (SAC's ``ent_coef``)
    or injected action noise (TD3's ``train_noise``), but **ε-greedy** — it plays a random action with
    probability ε, which anneals from 1.0 down to ``exploration_final_eps`` over the first
    ``exploration_fraction`` of the budget, then holds. ``target_update_interval`` is how often (in
    steps) the slow target network is hard-copied from the live Q-net (DQN's analogue of SAC/TD3's
    soft τ update). The ★ recommended values are per-env from rl-zoo3's tuned recipes (CartPole likes a
    high ``train_freq`` + a fast target sync; Atari uses the Nature-DQN recipe — set on the registry).

    ``batch_size`` / ``learning_starts`` are fixed (advanced, not sliders, like SAC's); the trainer
    budget-scales ``learning_starts`` so a short run doesn't burn a fifth of itself on random warmup,
    and sets ``gradient_steps`` itself (Atari does the Nature 1-update-per-collect; the vector envs do
    one update per collected step).
    """

    learning_rate: float = 1e-3  # gradient step for the Q-net (rl-zoo3 classic-control range)
    gamma: float = 0.99
    buffer_size: int = 100_000  # replay-buffer capacity (smaller than SAC/TD3's 1M — Atari is RAM-heavy)
    batch_size: int = 128  # minibatch sampled from the buffer per gradient step (fixed, not a slider)
    learning_starts: int = 1_000  # random warmup steps before the first gradient update (budget-scaled)
    train_freq: int = 4  # env steps collected between update phases (gradient_steps set by the trainer)
    target_update_interval: int = 250  # steps between hard copies of the live Q-net into the target net
    exploration_fraction: float = 0.2  # fraction of the budget to anneal ε over (then hold at the final)
    exploration_final_eps: float = 0.05  # the ε value held after annealing (residual random exploration)


class A2CHyperparams(BaseModel):
    """Tunable A2C knobs (the 8th algorithm, S5d — Advantage Actor-Critic, on-policy).

    A2C is PPO's **simpler predecessor** and its on-policy sibling: the same actor-critic shape
    (one network with a policy head + a value head, trained with gradients on freshly-collected
    rollouts, **no replay buffer**), but *without* PPO's clipped-surrogate objective — it does one
    plain policy-gradient update per rollout instead of several clipped epochs. That makes it the
    natural **PPO-vs-A2C teaching comparison**: same family, so any gap is down to PPO's clipping +
    multi-epoch reuse. Handles **both** discrete and continuous (``Box``) actions (unlike DQN's
    discrete-only / SAC-TD3's continuous-only gates), so it is offered on a curated mix of both.

    Its signature difference from PPO is the **short rollout**: ``n_steps`` defaults to 5 (PPO uses
    2048) — A2C updates after only a handful of steps, which is why the original recipe leans on many
    parallel envs (we run one, so the ★ per-env value is nudged up a little to steady the gradient).
    ``gae_lambda`` defaults to 1.0 (A2C classically uses full Monte-Carlo returns, no GAE smoothing);
    lower it toward PPO's 0.95 to trade variance for bias. ``ent_coef`` is the same exploration-
    entropy bonus as PPO. ``vf_coef`` / ``max_grad_norm`` / RMSprop are SB3 defaults, kept fixed
    (advanced, not sliders). Trains on raw obs/rewards (no VecNormalize — it is offered only on the
    classic-control envs, none of which are the MuJoCo family that normalizes), so ``ep_rew_mean`` and
    the ``[min_score, solved_score]`` skill meter read exactly like PPO's.
    """

    learning_rate: float = 7e-4  # A2C's canonical value (SB3 default; RMSprop, larger than PPO's 3e-4)
    gamma: float = 0.99
    n_steps: int = 5  # steps collected per update — A2C's signature short rollout (PPO uses 2048)
    gae_lambda: float = 1.0  # 1.0 = full Monte-Carlo returns (A2C classic); lower → GAE bias/variance trade
    ent_coef: float = 0.0  # entropy bonus (exploration), same knob as PPO
    n_hidden_layers: int = 2
    neurons_per_layer: int = 64
    activation: Literal["tanh", "relu"] = "tanh"


class QRDQNHyperparams(BaseModel):
    """Tunable QR-DQN knobs (the 9th algorithm, S5e — Quantile-Regression DQN, distributional off-policy).

    QR-DQN is **DQN made distributional**. Plain DQN learns a single number per action — the *mean*
    expected return (Q). QR-DQN instead learns the whole **return distribution**, represented as a set
    of ``n_quantiles`` values that carve the distribution into equal-probability slices (the median,
    the quartiles, …); it still *acts* on the mean of those quantiles (``argmax``), so the greedy
    policy is comparable to DQN's, but the richer learning target (a **quantile-Huber** regression
    loss instead of a single-scalar TD error) is often a more stable signal. It is one of the
    ingredients of Rainbow, and the natural **DQN-vs-QR-DQN teaching comparison** — same off-policy
    value-based machinery, so any difference isolates *distributional* learning.

    Everything else is DQN: the same replay buffer + slow target network, the same **ε-greedy**
    exploration (anneal ε from 1.0 to ``exploration_final_eps`` over the first ``exploration_fraction``
    of the budget, then hold), the same per-env rl-zoo3 recipes, gated to the **same discrete-action**
    envs (classic-control discretes + LunarLander + Atari — its distributional edge historically showed
    on Atari). Trains on raw obs/rewards (no VecNormalize), so ``ep_rew_mean`` and the
    ``[min_score, solved_score]`` skill meter read exactly like DQN's / PPO's.

    The one knob DQN doesn't have is **``n_quantiles``** — how many quantiles represent each action's
    return distribution (more = a finer distribution but a heavier net; SB3's default is 200, the
    rl-zoo3 classic-control recipes use far fewer). Like DQN, ``batch_size`` / ``learning_starts`` are
    fixed (advanced, not sliders) and the trainer budget-scales ``learning_starts`` + sets
    ``gradient_steps`` itself.
    """

    learning_rate: float = 1e-3  # gradient step for the quantile net (rl-zoo3 classic-control range)
    gamma: float = 0.99
    n_quantiles: int = 25  # quantiles per action's return distribution (DQN's single Q → a distribution)
    buffer_size: int = 100_000  # replay-buffer capacity (smaller than SAC/TD3's 1M — Atari is RAM-heavy)
    batch_size: int = 128  # minibatch sampled from the buffer per gradient step (fixed, not a slider)
    learning_starts: int = 1_000  # random warmup steps before the first gradient update (budget-scaled)
    train_freq: int = 4  # env steps collected between update phases (gradient_steps set by the trainer)
    target_update_interval: int = 250  # steps between hard copies of the live net into the target net
    exploration_fraction: float = 0.2  # fraction of the budget to anneal ε over (then hold at the final)
    exploration_final_eps: float = 0.05  # the ε value held after annealing (residual random exploration)


class TrainConfig(BaseModel):
    """Full, reproducible description of a training run (echoed back in status).

    ``hyperparams`` configures PPO; ``evolution`` configures neuroevolution; ``q_learning``
    configures tabular Q-learning; ``alphazero`` configures the AlphaZero-lite board trainer;
    ``sac`` configures Soft Actor-Critic; ``td3`` configures Twin Delayed DDPG; ``dqn`` configures
    Deep Q-Network; ``a2c`` configures Advantage Actor-Critic; ``qrdqn`` configures Quantile-Regression
    DQN. Exactly one applies, selected by ``algo``; the others stay None so the recorded config is clean.
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
    # Present only for Soft Actor-Critic runs (algo=="sac", S5a); None otherwise. SAC reuses the PPO
    # total_timesteps budget (it runs that many env steps) — only the param surface differs.
    sac: SACHyperparams | None = None
    # Present only for Twin Delayed DDPG runs (algo=="td3", S5b); None otherwise. Like SAC, TD3 reuses
    # the env-step total_timesteps budget (and the off-policy offpolicy_total_timesteps ★ on the registry).
    td3: TD3Hyperparams | None = None
    # Present only for Deep Q-Network runs (algo=="dqn", S5c); None otherwise. Like SAC/TD3, DQN reuses
    # the env-step total_timesteps budget (and the off-policy offpolicy_total_timesteps ★ on the registry).
    dqn: DQNHyperparams | None = None
    # Present only for Advantage Actor-Critic runs (algo=="a2c", S5d); None otherwise. A2C is on-policy
    # (like PPO), so it reuses the PPO env-step total_timesteps budget (default_total_timesteps) — NOT the
    # off-policy budget.
    a2c: A2CHyperparams | None = None
    # Present only for Quantile-Regression DQN runs (algo=="qrdqn", S5e); None otherwise. QR-DQN is the
    # distributional DQN, so like DQN it reuses the env-step total_timesteps budget (and the off-policy
    # offpolicy_total_timesteps ★ on the registry).
    qrdqn: QRDQNHyperparams | None = None
    # Seed-sweep grouping (X3): runs launched by one seed-sweep share this ``experiment_id`` (minted
    # server-side; None for a plain single run). ``experiment_label`` is an optional human name. Each
    # queued run still records its own ``seed`` + full config — the sweep is just orchestration, so the
    # reproducibility contract is untouched. Written into the run's meta.json for later aggregation (X4).
    experiment_id: str | None = None
    experiment_label: str | None = None


class SweepRequest(BaseModel):
    """Request to launch a seed-sweep (X3): one config trained across N seeds, queued sequentially.

    Provide **either** an explicit ``seeds`` list, **or** ``seed_count`` consecutive seeds starting at
    ``config.seed`` (s, s+1, … s+N−1). ``seeds`` wins when both are given. The manager mints one
    ``experiment_id``, then drains the queue one run at a time (reusing the single-run path), each run
    carrying its own seed + the shared experiment id.
    """

    config: TrainConfig
    seeds: list[int] | None = None
    seed_count: int | None = None


class SweepStatus(BaseModel):
    """Live state of an active seed-sweep (X3), carried on :class:`TrainStatus`; None outside a sweep.

    ``index`` is the 1-based position of the currently-running seed in ``seeds`` (so "seed 2 of 5"),
    ``running_seed`` the seed the active run trains. Cleared once the last seed finishes or the sweep
    is cancelled.
    """

    experiment_id: str
    experiment_label: str | None = None
    total: int  # number of seeds queued in this sweep
    index: int  # 1-based position of the seed currently running
    running_seed: int  # the seed the active run is training
    seeds: list[int]  # the full seed plan (so the UI can show what's queued)


class _CanonicalAxes(BaseModel):
    """The two canonical comparison axes carried by every *persisted* metric frame (X1, ADR-082).

    Cross-run / cross-algorithm comparison is only honest on a **shared** X-axis, but the trainers log
    on different native ones (PPO per-rollout · off-policy per 2 000 env steps · neuroevolution per
    generation · Q-learning per episode-batch · AlphaZero per self-play game). These two fields are the
    algorithm-independent axes the analysis suite (DataLab, Phase X) rebins and overlays runs on:

    * ``env_steps`` — cumulative **environment interactions** (the sample-efficiency axis, the RL
      standard, defined for *every* algorithm). For every trainer whose ``timesteps`` already counts env
      steps (PPO / SAC / TD3 / DQN / board-MaskablePPO / neuroevolution / Q-learning / competitive
      self-play) this equals ``timesteps``. The one exception is **AlphaZero**, whose ``timesteps`` is
      self-play *games* (its progress unit); it sets ``env_steps`` to the cumulative self-play **plies**
      (moves) so it is directly comparable to the board MaskablePPO trainer, which counts moves.
    * ``wall_clock`` — elapsed wall-clock seconds (== the frame's ``elapsed``).

    Both default to a sentinel and are **auto-filled** from the frame's own ``timesteps`` / ``elapsed`` by
    the validator below, so every trainer gets them for free and no future trainer can forget them. A
    trainer whose ``timesteps`` is *not* env steps (AlphaZero) passes ``env_steps`` explicitly and it is
    kept as-is. Legacy runs recorded before this contract are backfilled on read (see services/runs.py).
    """

    env_steps: int = -1  # sentinel <0 → filled from ``timesteps`` unless the trainer sets it (AlphaZero)
    wall_clock: float = -1.0  # sentinel <0 → filled from ``elapsed``

    @model_validator(mode="after")
    def _fill_canonical_axes(self) -> "_CanonicalAxes":
        # getattr (not self.timesteps) keeps the mixin standalone — the subclasses all carry both fields.
        if self.env_steps < 0:
            self.env_steps = int(getattr(self, "timesteps", 0))
        if self.wall_clock < 0:
            self.wall_clock = float(getattr(self, "elapsed", 0.0))
        return self


class TrainingMetrics(_CanonicalAxes):
    """One per-rollout metrics frame, pushed over WS as {type:"metrics", ...}.

    Carries the canonical ``env_steps`` + ``wall_clock`` axes (see :class:`_CanonicalAxes`); for the
    step-based trainers (PPO / SAC / TD3 / DQN / board-MaskablePPO) ``env_steps`` == ``timesteps``.
    """

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


class EvolutionMetrics(_CanonicalAxes):
    """One per-generation frame, pushed over WS as {type:"evolution", ...}.

    ``env_steps`` == ``timesteps`` here (neuroevolution's ``timesteps`` already counts cumulative env
    steps simulated across all generations); ``wall_clock`` == ``elapsed`` (see :class:`_CanonicalAxes`).
    """

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


class QLearningMetrics(_CanonicalAxes):
    """One periodic Q-learning frame, pushed over WS as {type:"q_learning", ...}.

    ``env_steps`` == ``timesteps`` here (Q-learning's ``timesteps`` already counts cumulative env steps
    across all episodes); ``wall_clock`` == ``elapsed`` (see :class:`_CanonicalAxes`).

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


class MultiAgentMetrics(_CanonicalAxes):
    """One competitive self-play frame, pushed over WS as {type:"ma_metrics", ...} (simple_tag, G7b-2).

    ``env_steps`` == ``timesteps`` here (the cumulative env steps across both species); ``wall_clock``
    == ``elapsed`` (see :class:`_CanonicalAxes`).

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
    # Live seed-sweep state (X3): set while a sweep is draining its queue (which seed of N is running),
    # None for a single run or once the sweep completes/cancels. Additive — the WS union is unchanged.
    sweep: SweepStatus | None = None
    error: str | None = None
