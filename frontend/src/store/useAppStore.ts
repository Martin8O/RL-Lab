import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  Algo,
  AlphaZeroHyperparams,
  BoardStrength,
  EnvSkill,
  EnvSpec,
  EvolutionHyperparams,
  EvolutionMetrics,
  HighScore,
  MultiAgentMetrics,
  PlayMode,
  PlayResult,
  PlayScores,
  PlayState,
  PlayStatus,
  PPOHyperparams,
  QLearningHyperparams,
  QLearningMetrics,
  QTableFrame,
  HwStats,
  SweepStatus,
  SACHyperparams,
  TD3Hyperparams,
  DQNHyperparams,
  A2CHyperparams,
  QRDQNHyperparams,
  SelfPlayHyperparams,
  TrainingMetrics,
  TrainingProgress,
  TrainState,
} from '../api/types'
import { boardMetaFor } from '../content/boardGames'

export type Locale        = 'cz' | 'en'
export type Theme         = 'dark' | 'light'
export type BackendStatus = 'connecting' | 'online' | 'offline'
export type ChartTab      = 'reward' | 'loss' | 'fitness'
// #2b: the audience mode. `simple` = the guided "arcade" scene for newcomers (advanced controls
// hidden, the ★ recommended algo forced); `advanced` = the full scientist UI (everything as before).
export type AudienceMode  = 'simple' | 'advanced'

// Cap on retained ~1 Hz progress frames (3 h of training); the chart's window control
// still slices this down for display.
const PROGRESS_CAP = 10_800
// Generations are coarse (one frame each), so a generous cap covers any realistic run.
const EVOLUTION_CAP = 2_000
// Q-learning reports ~300 frames per run; a generous cap covers several runs of history.
const Q_LEARNING_CAP = 2_000
// Self-play (simple_tag) emits a handful of ecosystem frames per round; a generous cap covers runs.
const MA_CAP = 4_000

const DEFAULT_HYPERPARAMS: PPOHyperparams = {
  learning_rate:   3e-4,
  gamma:           0.99,
  clip_range:      0.2,
  ent_coef:        0.0,
  n_steps:         2048,
  batch_size:      64,
  n_epochs:        10,
  n_hidden_layers: 2,
  neurons_per_layer: 64,
  activation:      'tanh',
}

// Matches the registry's ★ recommended neuroevolution block. ``episodes`` is a non-UI knob
// (fitness = mean return over this many episodes); kept here so the sent config is complete.
const DEFAULT_EVOLUTION_PARAMS: EvolutionHyperparams = {
  population_size: 50,
  top_k_parents:   10,
  mutation_rate:   0.1,
  crossover_rate:  0.5,
  generations:     30,
  episodes:        3,
}

// Matches the registry's ★ recommended q_learning block. ``episodes`` is per-env (Toy Text),
// snapped from the registry on env switch like the other budgets.
const DEFAULT_Q_PARAMS: QLearningHyperparams = {
  learning_rate: 0.1,
  gamma:         0.99,
  epsilon_start: 1.0,
  epsilon_end:   0.05,
  epsilon_decay: 0.5,
  episodes:      5000,
}

// Matches the registry's ★ self-play block (simple_tag). `rounds` snaps from the registry on env switch.
const DEFAULT_SELF_PLAY_PARAMS: SelfPlayHyperparams = {
  rounds: 8,
}

// Matches the registry's ★ alphazero block (board games, G6f/G6h). All snap from the registry on env
// switch (TTT recommends more iterations than Connect Four; chess a wider self-play cohort). Self-play
// uses Gumbel search (G6h), so the search dial is gumbel_sims (a low ★16 — Gumbel needs far fewer sims).
const DEFAULT_AZ_PARAMS: AlphaZeroHyperparams = {
  learning_rate:     5e-4,
  gumbel_sims:       16,
  gumbel_considered: 16,
  games_per_iter:    24,
  iterations:        30,
  actor_processes:   1,  // G6i: >1 parallelises self-play across GPU worker processes (★2 for chess)
}

// Matches the registry's ★ SAC block (S5a — off-policy continuous control; SB3's MuJoCo recipe). Snaps
// from the registry on env switch like the other algorithms. ent_coef is a string ("auto" self-tunes).
const DEFAULT_SAC_PARAMS: SACHyperparams = {
  learning_rate: 3e-4,
  gamma:         0.99,
  tau:           0.005,
  buffer_size:   1_000_000,
  train_freq:    1,
  ent_coef:      'auto',
}

// Matches the registry's ★ TD3 block (S5b — off-policy continuous control; SAC's deterministic sibling).
// Snaps from the registry on env switch like the others. learning_rate ★ is 1e-3 (TD3's canonical value);
// instead of an entropy temperature it has train_noise — the exploration-noise std a deterministic policy
// injects to explore.
const DEFAULT_TD3_PARAMS: TD3Hyperparams = {
  learning_rate: 1e-3,
  gamma:         0.99,
  tau:           0.005,
  buffer_size:   1_000_000,
  train_freq:    1,
  train_noise:   0.1,
}

// Matches the registry's ★ DQN block (S5c — off-policy value-based, discrete actions). These are the
// generic classic-control defaults; the real per-env ★ snap from the registry on env switch (CartPole's
// recipe wants a fast target sync + high train_freq, Atari uses the Nature recipe). It explores by
// ε-greedy (exploration_fraction / exploration_final_eps), DQN's distinctive knob vs SAC's ent_coef.
const DEFAULT_DQN_PARAMS: DQNHyperparams = {
  learning_rate:          1e-3,
  gamma:                  0.99,
  buffer_size:            100_000,
  train_freq:             4,
  target_update_interval: 250,
  exploration_fraction:   0.2,
  exploration_final_eps:  0.05,
}

// Matches the registry's ★ A2C block (S5d — on-policy actor-critic, PPO's simpler predecessor). These
// are the generic SB3 defaults; the per-env ★ (n_steps nudged up for the single-env setup) snap from the
// registry on env switch. Same net-arch/lr/ent_coef surface as PPO, minus clip/batch/epochs, plus
// gae_lambda; n_steps is A2C's signature short rollout.
const DEFAULT_A2C_PARAMS: A2CHyperparams = {
  learning_rate:     7e-4,
  gamma:             0.99,
  n_steps:           5,
  gae_lambda:        1.0,
  ent_coef:          0.0,
  n_hidden_layers:   2,
  neurons_per_layer: 64,
  activation:        'tanh',
}

// Matches the registry's ★ QR-DQN block (S5e — distributional DQN). These are the generic classic-control
// defaults; the per-env ★ (the rl-zoo3 recipe + n_quantiles — CartPole uses 10) snap from the registry
// when QR-DQN becomes active. Same knobs as DQN plus n_quantiles (DQN's single mean Q → a distribution).
const DEFAULT_QRDQN_PARAMS: QRDQNHyperparams = {
  learning_rate:          1e-3,
  gamma:                  0.99,
  n_quantiles:            25,
  buffer_size:            100_000,
  train_freq:             4,
  target_update_interval: 250,
  exploration_fraction:   0.2,
  exploration_final_eps:  0.05,
}

// The run-result state that must NOT outlive its run: chart history, the latest stats frame, the
// session-best, and every algorithm's curve. Cleared both between runs (clearMetrics) and when the
// user switches game — otherwise a finished run's chart/stats/skill linger and get silently rescaled
// under the new game's [min_score, solved_score] range (e.g. CartPole's 166 reading "Superhuman" once
// Breakout's scale is applied). Defined once so the two reset paths can't drift apart.
const EMPTY_RUN_RESULTS = {
  metricsHistory:   [] as TrainingMetrics[],
  progressHistory:  [] as TrainingProgress[],
  lastProgress:     null as TrainingProgress | null,
  bestReward:       null as number | null,
  evolutionHistory: [] as EvolutionMetrics[],
  lastEvolution:    null as EvolutionMetrics | null,
  qLearningHistory: [] as QLearningMetrics[],
  lastQLearning:    null as QLearningMetrics | null,
  lastQTable:       null as QTableFrame | null,
  maHistory:        [] as MultiAgentMetrics[],
  lastMa:           null as MultiAgentMetrics | null,
}

// The ★ step budget for the active algorithm: the off-policy methods (SAC + TD3 + DQN) carry their own
// per-env budget (offpolicy_total_timesteps) distinct from PPO's — far smaller for the continuous robots,
// a touch larger for DQN on the trivial classics; every other step-ladder algo (PPO) uses the env's
// default. Atari-DQN has no offpolicy budget set, so it falls back to the PPO image budget. Used to snap
// totalTimesteps on algo / env switch.
function budgetFor(spec: EnvSpec | undefined, algo: Algo): number {
  if (!spec) return 50_000
  if ((algo === 'sac' || algo === 'td3' || algo === 'dqn' || algo === 'qrdqn') && spec.offpolicy_total_timesteps)
    return spec.offpolicy_total_timesteps
  return spec.default_total_timesteps || 50_000
}

// Per-env defaults: when the user picks a different game, the sidebar params + step budget snap
// to *that* env's ★ recommended values from the registry (LunarLander wants very different
// settings than CartPole). Falls back to the previous value for any param the env doesn't define.
function envDefaults(
  spec: EnvSpec | undefined,
  prev: {
    hyperparams: PPOHyperparams
    evolutionParams: EvolutionHyperparams
    qLearningParams: QLearningHyperparams
    selfPlayParams: SelfPlayHyperparams
    alphaZeroParams: AlphaZeroHyperparams
    sacParams: SACHyperparams
    td3Params: TD3Hyperparams
    dqnParams: DQNHyperparams
    a2cParams: A2CHyperparams
    qrdqnParams: QRDQNHyperparams
    totalTimesteps: number
  },
): {
  hyperparams: PPOHyperparams
  evolutionParams: EvolutionHyperparams
  qLearningParams: QLearningHyperparams
  selfPlayParams: SelfPlayHyperparams
  alphaZeroParams: AlphaZeroHyperparams
  sacParams: SACHyperparams
  td3Params: TD3Hyperparams
  dqnParams: DQNHyperparams
  a2cParams: A2CHyperparams
  qrdqnParams: QRDQNHyperparams
  totalTimesteps: number
} | null {
  if (!spec) return null
  const ppo = spec.hyperparams?.ppo ?? {}
  const evo = spec.hyperparams?.neuroevolution ?? {}
  const ql = spec.hyperparams?.q_learning ?? {}
  const az = spec.hyperparams?.alphazero ?? {}
  const sac = spec.hyperparams?.sac ?? {}
  const td3 = spec.hyperparams?.td3 ?? {}
  const dqn = spec.hyperparams?.dqn ?? {}
  const a2c = spec.hyperparams?.a2c ?? {}
  const qrdqn = spec.hyperparams?.qrdqn ?? {}
  const num = (key: string, block: Record<string, { recommended: number | string }>, fb: number) =>
    block[key] !== undefined ? Number(block[key].recommended) : fb
  return {
    hyperparams: {
      learning_rate:     num('learning_rate', ppo, prev.hyperparams.learning_rate),
      gamma:             num('gamma', ppo, prev.hyperparams.gamma),
      clip_range:        num('clip_range', ppo, prev.hyperparams.clip_range),
      ent_coef:          num('ent_coef', ppo, prev.hyperparams.ent_coef),
      n_steps:           num('n_steps', ppo, prev.hyperparams.n_steps),
      batch_size:        num('batch_size', ppo, prev.hyperparams.batch_size),
      n_epochs:          num('n_epochs', ppo, prev.hyperparams.n_epochs),
      n_hidden_layers:   num('n_hidden_layers', ppo, prev.hyperparams.n_hidden_layers),
      neurons_per_layer: num('neurons_per_layer', ppo, prev.hyperparams.neurons_per_layer),
      activation:        (ppo.activation?.recommended as 'tanh' | 'relu') ?? prev.hyperparams.activation,
    },
    evolutionParams: {
      population_size: num('population_size', evo, prev.evolutionParams.population_size),
      top_k_parents:   num('top_k_parents', evo, prev.evolutionParams.top_k_parents),
      mutation_rate:   num('mutation_rate', evo, prev.evolutionParams.mutation_rate),
      crossover_rate:  num('crossover_rate', evo, prev.evolutionParams.crossover_rate),
      generations:     num('generations', evo, prev.evolutionParams.generations),
      episodes:        prev.evolutionParams.episodes,  // non-UI knob — preserve
    },
    qLearningParams: {
      learning_rate: num('learning_rate', ql, prev.qLearningParams.learning_rate),
      gamma:         num('gamma', ql, prev.qLearningParams.gamma),
      epsilon_start: num('epsilon_start', ql, prev.qLearningParams.epsilon_start),
      epsilon_end:   num('epsilon_end', ql, prev.qLearningParams.epsilon_end),
      epsilon_decay: num('epsilon_decay', ql, prev.qLearningParams.epsilon_decay),
      episodes:      num('episodes', ql, prev.qLearningParams.episodes),
    },
    // Self-play rounds rides in the ppo block (algo stays "ppo"); only simple_tag defines it.
    selfPlayParams: {
      rounds: Math.round(num('rounds', ppo, prev.selfPlayParams.rounds)),
    },
    // AlphaZero block (board games, G6f/G6h); the budget is iterations × games_per_iter, computed at submit.
    alphaZeroParams: {
      learning_rate:     num('learning_rate', az, prev.alphaZeroParams.learning_rate),
      gumbel_sims:       Math.round(num('gumbel_sims', az, prev.alphaZeroParams.gumbel_sims)),
      gumbel_considered: Math.round(num('gumbel_considered', az, prev.alphaZeroParams.gumbel_considered)),
      games_per_iter:    Math.round(num('games_per_iter', az, prev.alphaZeroParams.games_per_iter)),
      iterations:        Math.round(num('iterations', az, prev.alphaZeroParams.iterations)),
      actor_processes:   Math.round(num('actor_processes', az, prev.alphaZeroParams.actor_processes)),
    },
    // SAC block (S5a — continuous-Box envs). ent_coef is categorical ("auto"/numeric), so it preserves
    // the recommended string rather than going through num(); SAC reuses the PPO totalTimesteps budget.
    sacParams: {
      learning_rate: num('learning_rate', sac, prev.sacParams.learning_rate),
      gamma:         num('gamma', sac, prev.sacParams.gamma),
      tau:           num('tau', sac, prev.sacParams.tau),
      buffer_size:   Math.round(num('buffer_size', sac, prev.sacParams.buffer_size)),
      train_freq:    Math.round(num('train_freq', sac, prev.sacParams.train_freq)),
      ent_coef:      (sac.ent_coef?.recommended as string) ?? prev.sacParams.ent_coef,
    },
    // TD3 block (S5b — continuous-Box envs, SAC's deterministic sibling). All-numeric (no categorical);
    // train_noise is the exploration-noise std. Reuses the off-policy totalTimesteps budget like SAC.
    td3Params: {
      learning_rate: num('learning_rate', td3, prev.td3Params.learning_rate),
      gamma:         num('gamma', td3, prev.td3Params.gamma),
      tau:           num('tau', td3, prev.td3Params.tau),
      buffer_size:   Math.round(num('buffer_size', td3, prev.td3Params.buffer_size)),
      train_freq:    Math.round(num('train_freq', td3, prev.td3Params.train_freq)),
      train_noise:   num('train_noise', td3, prev.td3Params.train_noise),
    },
    // DQN block (S5c — discrete-action value-based). All-numeric; the per-env recipe (CartPole's fast
    // target sync, Atari's Nature values) snaps from the registry here. Reuses the off-policy budget.
    dqnParams: {
      learning_rate:          num('learning_rate', dqn, prev.dqnParams.learning_rate),
      gamma:                  num('gamma', dqn, prev.dqnParams.gamma),
      buffer_size:            Math.round(num('buffer_size', dqn, prev.dqnParams.buffer_size)),
      train_freq:             Math.round(num('train_freq', dqn, prev.dqnParams.train_freq)),
      target_update_interval: Math.round(num('target_update_interval', dqn, prev.dqnParams.target_update_interval)),
      exploration_fraction:   num('exploration_fraction', dqn, prev.dqnParams.exploration_fraction),
      exploration_final_eps:  num('exploration_final_eps', dqn, prev.dqnParams.exploration_final_eps),
    },
    // A2C block (S5d — on-policy actor-critic). Same net-arch/lr/ent_coef surface as PPO plus gae_lambda;
    // the per-env ★ n_steps (nudged up for the single-env setup) snaps from the registry here.
    a2cParams: {
      learning_rate:     num('learning_rate', a2c, prev.a2cParams.learning_rate),
      gamma:             num('gamma', a2c, prev.a2cParams.gamma),
      n_steps:           Math.round(num('n_steps', a2c, prev.a2cParams.n_steps)),
      gae_lambda:        num('gae_lambda', a2c, prev.a2cParams.gae_lambda),
      ent_coef:          num('ent_coef', a2c, prev.a2cParams.ent_coef),
      n_hidden_layers:   Math.round(num('n_hidden_layers', a2c, prev.a2cParams.n_hidden_layers)),
      neurons_per_layer: Math.round(num('neurons_per_layer', a2c, prev.a2cParams.neurons_per_layer)),
      activation:        (a2c.activation?.recommended as 'tanh' | 'relu') ?? prev.a2cParams.activation,
    },
    // QR-DQN block (S5e — distributional DQN). Same knobs as DQN plus n_quantiles; the per-env recipe
    // (CartPole's fast target sync + 10 quantiles, Atari's Nature values + 200) snaps from the registry
    // here. Reuses the off-policy budget, like DQN.
    qrdqnParams: {
      learning_rate:          num('learning_rate', qrdqn, prev.qrdqnParams.learning_rate),
      gamma:                  num('gamma', qrdqn, prev.qrdqnParams.gamma),
      n_quantiles:            Math.round(num('n_quantiles', qrdqn, prev.qrdqnParams.n_quantiles)),
      buffer_size:            Math.round(num('buffer_size', qrdqn, prev.qrdqnParams.buffer_size)),
      train_freq:             Math.round(num('train_freq', qrdqn, prev.qrdqnParams.train_freq)),
      target_update_interval: Math.round(num('target_update_interval', qrdqn, prev.qrdqnParams.target_update_interval)),
      exploration_fraction:   num('exploration_fraction', qrdqn, prev.qrdqnParams.exploration_fraction),
      exploration_final_eps:  num('exploration_final_eps', qrdqn, prev.qrdqnParams.exploration_final_eps),
    },
    totalTimesteps: spec.default_total_timesteps || prev.totalTimesteps,
  }
}

// #2b (Simple mode): the state patch that forces the env's ★ recommended algorithm — the audited
// `recommended_algo` (ADR-104). Simple mode hides the algo picker, so the algo must be snapped to the
// recommendation on every env switch (and when entering Simple). Mirrors setAlgo: re-point the chart
// tab to the tab that algo feeds, re-snap the step budget to that algo's ★, and (for the per-env-tuned
// value-based algos) snap their sliders to THIS env's ★. AppState is only referenced as a type here
// (hoisted), so declaring this above the interface is fine.
function recommendedPatch(spec: EnvSpec | undefined, s: AppState): Partial<AppState> {
  if (!spec) return {}
  const algo = (spec.recommended_algo || spec.supported_algos[0] || 'ppo') as Algo
  const d = envDefaults(spec, s)
  const perEnvPatch =
    algo === 'dqn'   ? (d ? { dqnParams: d.dqnParams } : {})
    : algo === 'a2c'   ? (d ? { a2cParams: d.a2cParams } : {})
    : algo === 'qrdqn' ? (d ? { qrdqnParams: d.qrdqnParams } : {})
    : {}
  return {
    algo,
    activeTab: (algo === 'neuroevolution' ? 'fitness' : 'reward') as ChartTab,
    totalTimesteps: budgetFor(spec, algo),
    ...perEnvPatch,
  }
}

interface AppState {
  // ─ persisted ───────────────────────────────────────────────
  locale:          Locale
  theme:           Theme
  mode:            AudienceMode   // #2b: simple (guided) ↔ advanced (full UI); persisted
  modeChosen:      boolean        // #2b: has the user picked a mode? false ⇒ show the first-launch chooser
  selectedEnvId:   string | null
  algo:            Algo       // PPO ↔ neuroevolution ↔ Q-learning ↔ AlphaZero
  hyperparams:     PPOHyperparams
  evolutionParams: EvolutionHyperparams
  qLearningParams: QLearningHyperparams
  selfPlayParams:  SelfPlayHyperparams   // G7b-2: competitive self-play round schedule (simple_tag)
  alphaZeroParams: AlphaZeroHyperparams  // G6f: AlphaZero-lite board self-play knobs
  sacParams:       SACHyperparams        // S5a: Soft Actor-Critic (off-policy continuous control)
  td3Params:       TD3Hyperparams        // S5b: Twin Delayed DDPG (off-policy continuous control)
  dqnParams:       DQNHyperparams        // S5c: Deep Q-Network (off-policy value-based, discrete actions)
  a2cParams:       A2CHyperparams        // S5d: Advantage Actor-Critic (on-policy, PPO's predecessor)
  qrdqnParams:     QRDQNHyperparams      // S5e: Quantile-Regression DQN (distributional value-based)
  seed:            number
  totalTimesteps:  number
  sweepCount:      number     // X3: how many seeds a "Run N seeds" sweep launches (★ 3)
  emaAlpha:        number     // 1 = raw; 0.05 = heavy smoothing
  chartWindow:     number     // 0 = all; N = last N rollouts
  activeTab:       ChartTab
  visual:          boolean    // env-preview frame streaming on/off
  speed:           number     // playback speed multiplier (1×–20×)
  attemptMode:     'preview' | 'real'  // "New attempt" counter: preview restarts vs real training steps
  playMode:        PlayMode   // E2: who plays — human at the keyboard ↔ AI watch
  playSpeed:       number     // E2: play-session pacing (0.1×–20×; slow-mo for humans)
  playerName:      string     // E2: last name used for a human leaderboard entry
  boardSide:       number     // G6a: which side the human takes in a board game (0 = first player)
  boardStrength:   BoardStrength  // G6a: the board MCTS opponent strength (easy/medium/hard)

  // ─ ephemeral (not persisted) ───────────────────────────────
  backendStatus:   BackendStatus
  gpuAvailable:    boolean     // G4a: CUDA present? gates GPU-only training (Atari) in the UI
  atariAvailable:  boolean     // R1: optional ale-py installed? gates the Atari family (ADR-101)
  envs:            EnvSpec[]
  trainState:      TrainState
  metricsHistory:  TrainingMetrics[]
  progressHistory: TrainingProgress[]   // ~1 Hz frames — feeds the reward chart
  lastProgress:    TrainingProgress | null
  lastHwStats:     HwStats | null       // G4b: latest CPU/GPU telemetry — feeds the HW panel
  bestReward:      number | null        // best score this session (live high)
  evolutionHistory: EvolutionMetrics[]  // per-generation frames — feeds the Fitness chart
  lastEvolution:   EvolutionMetrics | null
  qLearningHistory: QLearningMetrics[]  // per-report frames — feeds the reward chart (x=episode)
  lastQLearning:   QLearningMetrics | null
  lastQTable:      QTableFrame | null    // latest Q-table snapshot — feeds the heatmap panel
  maHistory:       MultiAgentMetrics[]   // G7b-2: per-round ecosystem frames — the two-line chart
  lastMa:          MultiAgentMetrics | null
  sweep:           SweepStatus | null    // X3: live seed-sweep progress (null outside a sweep)
  highScores:      Record<string, HighScore>  // all-time best per env id
  checkpointsNonce: number                     // bumped on save/delete so other pickers (AI-play) re-fetch
  analysisOpen:    boolean               // X6: the fullscreen DataLab surface is open (over the dashboard)

  // ─ play vs AI (E2) ─────────────────────────────────────────
  playState:        PlayState            // idle until a session starts
  playScore:        number               // live score of the current/last session
  playStep:         number
  playResult:       PlayResult | null    // set once the episode ends (carries the rated band)
  playError:        string | null
  playCheckpointId: string | null        // selected checkpoint for AI watch mode
  playCheckpointLabel: string | null     // its label — the AI's leaderboard identity on finish
  playActiveCheckpoint: string | null    // the ACTIVE session's checkpoint id (from the backend status);
                                         // for board play, non-null ⇒ the opponent is your trained net (G6b)
  envSkill:         EnvSkill | null       // backend skill thresholds for the selected env
  playScores:       PlayScores | null     // Human + AI boards for the selected env

  // ─ actions ────────────────────────────────────────────────
  setLocale:          (l: Locale)                       => void
  setTheme:           (t: Theme)                        => void
  setMode:            (m: AudienceMode)                 => void
  setBackendStatus:   (s: BackendStatus)                => void
  setGpuAvailable:    (v: boolean)                      => void
  setAtariAvailable:  (v: boolean)                      => void
  setEnvs:            (envs: EnvSpec[])                 => void
  setSelectedEnvId:   (id: string | null)               => void
  setAlgo:            (a: Algo)                          => void
  setHyperparams:     (h: Partial<PPOHyperparams>)      => void
  setEvolutionParams: (e: Partial<EvolutionHyperparams>) => void
  setQLearningParams: (q: Partial<QLearningHyperparams>) => void
  setSelfPlayParams:  (s: Partial<SelfPlayHyperparams>) => void
  setAlphaZeroParams: (a: Partial<AlphaZeroHyperparams>) => void
  setSacParams:       (s: Partial<SACHyperparams>)      => void
  setTd3Params:       (s: Partial<TD3Hyperparams>)      => void
  setDqnParams:       (s: Partial<DQNHyperparams>)      => void
  setA2cParams:       (s: Partial<A2CHyperparams>)      => void
  setQrdqnParams:     (s: Partial<QRDQNHyperparams>)    => void
  setSeed:            (s: number)                       => void
  setTotalTimesteps:  (n: number)                       => void
  setSweepCount:      (n: number)                       => void
  setSweep:           (s: SweepStatus | null)           => void
  setEmaAlpha:        (a: number)                       => void
  setChartWindow:     (w: number)                       => void
  setActiveTab:       (t: ChartTab)                     => void
  setVisual:          (v: boolean)                      => void
  setSpeed:           (n: number)                       => void
  setAttemptMode:     (m: 'preview' | 'real')           => void
  setTrainState:      (s: TrainState)                   => void
  addMetrics:         (m: TrainingMetrics)              => void
  setProgress:        (p: TrainingProgress)             => void
  setHwStats:         (s: HwStats | null)               => void
  addEvolution:       (e: EvolutionMetrics)             => void
  seedEvolution:      (e: EvolutionMetrics)             => void
  addQLearning:       (q: QLearningMetrics)             => void
  setQTable:          (t: QTableFrame)                  => void
  seedQLearning:      (q: QLearningMetrics, t: QTableFrame | null) => void
  addMa:              (m: MultiAgentMetrics)            => void
  seedMa:             (m: MultiAgentMetrics)            => void
  setHighScore:       (hs: HighScore)                   => void
  setHighScores:      (list: HighScore[])               => void
  bumpCheckpoints:    ()                                => void
  setAnalysisOpen:    (v: boolean)                      => void
  clearMetrics:       ()                                => void

  // play vs AI (E2)
  setPlayMode:         (m: PlayMode)                    => void
  setPlaySpeed:        (n: number)                      => void
  setPlayerName:       (name: string)                  => void
  setBoardSide:        (side: number)                  => void
  setBoardStrength:    (s: BoardStrength)              => void
  setPlayCheckpointId: (id: string | null)             => void
  setPlayCheckpointLabel: (label: string | null)       => void
  applyPlayStatus:     (s: PlayStatus)                  => void
  setPlayProgress:     (score: number, step: number)   => void
  setPlayResult:       (r: PlayResult)                  => void
  setEnvSkill:         (s: EnvSkill | null)             => void
  setPlayScores:       (s: PlayScores | null)           => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      locale:          'en',
      theme:           'dark',
      mode:            'simple',   // #2b: newcomers land in the guided scene; the chooser lets them switch
      modeChosen:      false,      // #2b: no choice yet → first launch shows the mode chooser
      selectedEnvId:   null,
      algo:            'ppo',
      hyperparams:     DEFAULT_HYPERPARAMS,
      evolutionParams: DEFAULT_EVOLUTION_PARAMS,
      qLearningParams: DEFAULT_Q_PARAMS,
      selfPlayParams:  DEFAULT_SELF_PLAY_PARAMS,
      alphaZeroParams: DEFAULT_AZ_PARAMS,
      sacParams:       DEFAULT_SAC_PARAMS,
      td3Params:       DEFAULT_TD3_PARAMS,
      dqnParams:       DEFAULT_DQN_PARAMS,
      a2cParams:       DEFAULT_A2C_PARAMS,
      qrdqnParams:     DEFAULT_QRDQN_PARAMS,
      seed:            42,
      totalTimesteps:  50_000,
      sweepCount:      3,
      emaAlpha:        0.3,
      chartWindow:     0,
      activeTab:       'reward',
      visual:          true,
      speed:           1,
      attemptMode:     'real',
      playMode:        'human',
      playSpeed:       1,
      playerName:      '',
      boardSide:       0,
      boardStrength:   'medium',

      backendStatus:   'connecting',
      gpuAvailable:    false,
      atariAvailable:  true,   // assume present until /api/system says otherwise (dev + default GPU build have it)
      envs:            [],
      trainState:      'idle',
      metricsHistory:  [],
      progressHistory: [],
      lastProgress:    null,
      lastHwStats:     null,
      bestReward:      null,
      evolutionHistory: [],
      lastEvolution:   null,
      qLearningHistory: [],
      lastQLearning:   null,
      lastQTable:      null,
      maHistory:       [],
      lastMa:          null,
      sweep:           null,
      highScores:      {},
      checkpointsNonce: 0,
      analysisOpen:    false,

      playState:        'idle',
      playScore:        0,
      playStep:         0,
      playResult:       null,
      playError:        null,
      playCheckpointId: null,
      playCheckpointLabel: null,
      playActiveCheckpoint: null,
      envSkill:         null,
      playScores:       null,

      setLocale:         (locale)         => set({ locale }),
      setTheme:          (theme)          => set({ theme }),
      // #2b: switch audience mode (also records that a choice was made, so the first-launch chooser
      // never reappears). Entering Simple snaps the algo to the env's ★ recommendation, since Simple
      // hides the algo picker and must run the audited best default (ADR-104).
      setMode:           (mode)           => set((s) => {
        const patch: Partial<AppState> = { mode, modeChosen: true }
        if (mode === 'simple') {
          const spec = s.envs.find((e) => e.id === s.selectedEnvId)
          Object.assign(patch, recommendedPatch(spec, s))
        }
        return patch
      }),
      setBackendStatus:  (backendStatus)  => set({ backendStatus }),
      setGpuAvailable:   (gpuAvailable)   => set({ gpuAvailable }),
      setAtariAvailable: (atariAvailable) => set({ atariAvailable }),
      setEnvs:           (envs)           => set({ envs }),
      // Switching game also snaps the sidebar params + step budget to the new env's ★ recommended
      // values (CartPole and LunarLander want very different settings). The env selector is disabled
      // during a run, so this never fires mid-training.
      setSelectedEnvId:  (selectedEnvId)  => set((s) => {
        const spec = s.envs.find((e) => e.id === selectedEnvId)
        const defaults = envDefaults(spec, s)
        // #2b: in Simple mode the algo picker is hidden, so every env switch forces that env's ★
        // recommended algo (the audited best default, ADR-104) — algo + chart tab + budget snapped
        // together. In Advanced we only keep the algorithm *valid*: each env lists its supported_algos
        // (an image env may be PPO-only), so if the current algo isn't supported, snap to the first
        // allowed one (and re-point the chart tab, mirroring setAlgo).
        const algoPatch =
          s.mode === 'simple'
            ? recommendedPatch(spec, s)
            : spec && spec.supported_algos.length > 0 && !spec.supported_algos.includes(s.algo)
              ? (() => {
                  const algo = spec.supported_algos[0] as Algo
                  return { algo, activeTab: (algo === 'neuroevolution' ? 'fitness' : 'reward') as ChartTab }
                })()
              : {}
        // Clear the previous run's chart/stats/skill when the game actually changes, so they don't
        // linger and get rescaled under the new game. The env selector is disabled during a run, so
        // this only fires between runs; a no-op re-select (same id) keeps the current results.
        const cleared = selectedEnvId !== s.selectedEnvId ? EMPTY_RUN_RESULTS : {}
        // Snap the step budget to the (possibly newly-snapped) algo's ★ for this env — SAC's budget is
        // much smaller than PPO's, so picking a SAC env (or one where the algo snapped) shows SAC's ★, not
        // PPO's. Overrides envDefaults' totalTimesteps (which is the PPO budget) when spec is known.
        const effectiveAlgo = ((algoPatch as { algo?: Algo }).algo ?? s.algo)
        const budgetPatch = spec ? { totalTimesteps: budgetFor(spec, effectiveAlgo) } : {}
        // Default the human's side to the board's first mover, so pressing Play without touching the
        // side picker gives you the opening move. Usually player 0, but OpenSpiel chess makes white =
        // player 1 the first mover (boardMeta.firstPlayer) — without this, picking chess + Play would
        // silently start you as black with the AI moving first.
        const sidePatch =
          selectedEnvId !== s.selectedEnvId
            ? { boardSide: boardMetaFor(selectedEnvId)?.firstPlayer ?? 0 }
            : {}
        return { selectedEnvId, ...(defaults ?? {}), ...algoPatch, ...cleared, ...sidePatch, ...budgetPatch }
      }),
      // Switching algorithm also jumps to the chart tab that algorithm feeds, so the chart never sits
      // empty after a switch (PPO → Reward, neuroevolution → Fitness), and re-snaps the step budget to the
      // new algo's ★ (SAC's budget is far smaller than PPO's, so PPO↔SAC must not keep the other's 5M/500k).
      setAlgo:           (algo)           => set((s) => {
        const spec = s.envs.find((e) => e.id === s.selectedEnvId)
        // DQN's ★ recommended hyperparameters are **per-env** (rl-zoo3 recipes — CartPole's fast target
        // sync + high train_freq differ sharply from the others), unlike PPO/SAC/TD3 whose ★ are
        // env-independent. So when DQN becomes the active algo, snap its sliders to THIS env's ★ — else
        // picking DQN on an already-selected game would show the generic defaults sitting off the green ★
        // tick. The other algos keep their values across a switch (their defaults already equal their ★).
        const dqnPatch =
          algo === 'dqn' ? (() => { const d = envDefaults(spec, s); return d ? { dqnParams: d.dqnParams } : {} })() : {}
        // A2C (S5d) is the same story: its ★ n_steps is nudged up **per env** for the single-env setup, so
        // snap its sliders to THIS env's ★ when A2C becomes active (else the generic n_steps=5 shows off the
        // green ★ tick). Mirrors dqnPatch.
        const a2cPatch =
          algo === 'a2c' ? (() => { const d = envDefaults(spec, s); return d ? { a2cParams: d.a2cParams } : {} })() : {}
        // QR-DQN (S5e) mirrors DQN: its ★ recipe (CartPole's fast target sync + 10 quantiles, Atari's
        // Nature values + 200) is **per-env**, so snap its sliders to THIS env's ★ when QR-DQN becomes
        // active — else the generic classic-control defaults would sit off the green ★ tick. Mirrors dqnPatch.
        const qrdqnPatch =
          algo === 'qrdqn' ? (() => { const d = envDefaults(spec, s); return d ? { qrdqnParams: d.qrdqnParams } : {} })() : {}
        return {
          algo,
          activeTab: (algo === 'neuroevolution' ? 'fitness' : 'reward') as ChartTab,
          totalTimesteps: spec ? budgetFor(spec, algo) : s.totalTimesteps,
          ...dqnPatch,
          ...a2cPatch,
          ...qrdqnPatch,
        }
      }),
      setHyperparams:    (h)              => set((s) => ({ hyperparams: { ...s.hyperparams, ...h } })),
      setEvolutionParams:(e)              => set((s) => ({ evolutionParams: { ...s.evolutionParams, ...e } })),
      setQLearningParams:(q)              => set((s) => ({ qLearningParams: { ...s.qLearningParams, ...q } })),
      setSelfPlayParams: (sp)             => set((s) => ({ selfPlayParams: { ...s.selfPlayParams, ...sp } })),
      setAlphaZeroParams:(a)              => set((s) => ({ alphaZeroParams: { ...s.alphaZeroParams, ...a } })),
      setSacParams:      (sp)             => set((s) => ({ sacParams: { ...s.sacParams, ...sp } })),
      setTd3Params:      (sp)             => set((s) => ({ td3Params: { ...s.td3Params, ...sp } })),
      setDqnParams:      (sp)             => set((s) => ({ dqnParams: { ...s.dqnParams, ...sp } })),
      setA2cParams:      (sp)             => set((s) => ({ a2cParams: { ...s.a2cParams, ...sp } })),
      setQrdqnParams:    (sp)             => set((s) => ({ qrdqnParams: { ...s.qrdqnParams, ...sp } })),
      setSeed:           (seed)           => set({ seed }),
      setTotalTimesteps: (n)              => set({ totalTimesteps: n }),
      setSweepCount:     (n)              => set({ sweepCount: n }),
      setSweep:          (sweep)          => set({ sweep }),
      setEmaAlpha:       (emaAlpha)       => set({ emaAlpha }),
      setChartWindow:    (chartWindow)    => set({ chartWindow }),
      setActiveTab:      (activeTab)      => set({ activeTab }),
      setVisual:         (visual)         => set({ visual }),
      setSpeed:          (speed)          => set({ speed }),
      setAttemptMode:    (attemptMode)    => set({ attemptMode }),
      setTrainState:     (trainState)     => set({ trainState }),
      setHwStats:        (lastHwStats)    => set({ lastHwStats }),

      addMetrics: (m) =>
        set((s) => {
          const newBest =
            s.bestReward === null
              ? (m.ep_rew_mean ?? null)
              : m.ep_rew_mean !== null && m.ep_rew_mean > s.bestReward
                ? m.ep_rew_mean
                : s.bestReward
          return {
            metricsHistory: [...s.metricsHistory, m],
            bestReward: newBest,
          }
        }),

      // lastProgress always updates (stats refresh ~1 Hz); only append a chart point when
      // timesteps actually advanced, so the step-less update-phase ticks don't pile up
      // duplicate-x points. Capped to keep very long runs from growing without bound.
      setProgress: (p) =>
        set((s) => {
          const last = s.progressHistory.at(-1)
          const advanced = !last || p.timesteps > last.timesteps
          return {
            lastProgress: p,
            progressHistory: advanced
              ? [...s.progressHistory, p].slice(-PROGRESS_CAP)
              : s.progressHistory,
          }
        }),

      // One frame per generation: append (capped), track the latest, and fold best_fitness
      // into the session-best so the same "live high" surface works for both algorithms.
      addEvolution: (e) =>
        set((s) => ({
          evolutionHistory: [...s.evolutionHistory, e].slice(-EVOLUTION_CAP),
          lastEvolution: e,
          bestReward:
            s.bestReward === null ? e.best_fitness : Math.max(s.bestReward, e.best_fitness),
        })),

      // Late-join reconcile (D2.5): seed the latest evolution frame from /api/train/status so
      // the leaderboard / stats / Fitness panels repopulate on reconnect without waiting for the
      // next generation. Only primes history when it's empty (a single point is enough for the
      // leaderboard/stats; the Fitness curve refills as new frames stream in) — so a live frame
      // that already arrived is never double-appended.
      seedEvolution: (e) =>
        set((s) => ({
          lastEvolution: e,
          evolutionHistory: s.evolutionHistory.length === 0 ? [e] : s.evolutionHistory,
          bestReward:
            s.bestReward === null ? e.best_fitness : Math.max(s.bestReward, e.best_fitness),
        })),

      // One frame per Q-learning report: append (capped), track the latest, fold ep_rew_mean into
      // the session-best (so the same "live high" surface works for all three algorithms).
      addQLearning: (q) =>
        set((s) => ({
          qLearningHistory: [...s.qLearningHistory, q].slice(-Q_LEARNING_CAP),
          lastQLearning: q,
          bestReward:
            q.ep_rew_mean === null
              ? s.bestReward
              : s.bestReward === null
                ? q.ep_rew_mean
                : Math.max(s.bestReward, q.ep_rew_mean),
        })),

      // The heatmap snapshot is high-frequency + large, so it only ever overwrites the latest
      // (never accumulates) — the panel always draws the current table.
      setQTable: (t) => set({ lastQTable: t }),

      // Late-join reconcile (mirrors seedEvolution): repopulate the Q-learning chart/stats/heatmap
      // from /api/train/status so a reload mid-run shows the latest report immediately. Only primes
      // history when empty so a live frame that already arrived is never double-appended.
      seedQLearning: (q, t) =>
        set((s) => ({
          lastQLearning: q,
          lastQTable: t ?? s.lastQTable,
          qLearningHistory: s.qLearningHistory.length === 0 ? [q] : s.qLearningHistory,
          bestReward:
            q.ep_rew_mean === null
              ? s.bestReward
              : s.bestReward === null
                ? q.ep_rew_mean
                : Math.max(s.bestReward, q.ep_rew_mean),
        })),

      // One frame per self-play round (simple_tag): append (capped), track the latest, fold the
      // predator headline (ep_rew_mean) into the session-best like the other algorithms.
      addMa: (m) =>
        set((s) => ({
          maHistory: [...s.maHistory, m].slice(-MA_CAP),
          lastMa: m,
          bestReward:
            m.ep_rew_mean === null
              ? s.bestReward
              : s.bestReward === null
                ? m.ep_rew_mean
                : Math.max(s.bestReward, m.ep_rew_mean),
        })),

      // Late-join reconcile (mirrors seedQLearning): repopulate the ecosystem chart from the retained
      // status snapshot. Only primes history when empty so a live frame isn't double-appended.
      seedMa: (m) =>
        set((s) => ({
          lastMa: m,
          maHistory: s.maHistory.length === 0 ? [m] : s.maHistory,
          bestReward:
            m.ep_rew_mean === null
              ? s.bestReward
              : s.bestReward === null
                ? m.ep_rew_mean
                : Math.max(s.bestReward, m.ep_rew_mean),
        })),

      setHighScore:  (hs)   => set((s) => ({ highScores: { ...s.highScores, [hs.env_id]: hs } })),
      setHighScores: (list) =>
        set({ highScores: Object.fromEntries(list.map((hs) => [hs.env_id, hs])) }),

      // A new checkpoint was saved/deleted somewhere — bump so components that fetch the list on their
      // own (the AI-play picker in PlayControls) re-fetch without a page reload.
      bumpCheckpoints: () => set((s) => ({ checkpointsNonce: s.checkpointsNonce + 1 })),

      // X6: open/close the fullscreen DataLab surface. Ephemeral (not persisted) — the dashboard stays
      // mounted underneath so a live run keeps streaming while the overlay is up.
      setAnalysisOpen: (analysisOpen) => set({ analysisOpen }),

      clearMetrics: () => set({ ...EMPTY_RUN_RESULTS }),

      // ─ play vs AI (E2) ────────────────────────────────────────
      setPlayMode:         (playMode)         => set({ playMode }),
      setPlaySpeed:        (playSpeed)        => set({ playSpeed }),
      setPlayerName:       (playerName)       => set({ playerName }),
      setBoardSide:        (boardSide)        => set({ boardSide }),
      setBoardStrength:    (boardStrength)    => set({ boardStrength }),
      setPlayCheckpointId: (playCheckpointId) => set({ playCheckpointId }),
      setPlayCheckpointLabel: (playCheckpointLabel) => set({ playCheckpointLabel }),
      setPlayScores:       (playScores)       => set({ playScores }),

      // Lifecycle snapshot from a {type:"play_status"} frame or REST start/stop response.
      // A fresh 'playing' status arrives with step/score 0 + result null, which resets the meter.
      applyPlayStatus: (s) =>
        set((state) => ({
          playState: s.state,
          playScore: s.score,
          playStep:  s.step,
          playResult: s.result,
          playError: s.error,
          // Mode is authoritative from the backend while a session exists, so the skill-meter label
          // ("Your skill" vs "AI skill") matches the running session even after a reload reconcile;
          // when idle (mode null) keep the selector's choice for the next session.
          playMode: s.mode ?? state.playMode,
          // The active session's checkpoint (board play, G6b): non-null ⇒ the opponent is the trained
          // net, so the result banner says "your trained AI" instead of an MCTS difficulty.
          playActiveCheckpoint: s.checkpoint_id ?? null,
        })),

      // High-frequency per-frame update (throttled by the caller) so the skill meter climbs live.
      setPlayProgress: (playScore, playStep) => set({ playScore, playStep }),

      // Terminal {type:"play_result"} frame — carries the rated band + final score/steps.
      setPlayResult: (r) =>
        set({ playResult: r, playScore: r.score, playStep: r.steps }),

      setEnvSkill: (envSkill) => set({ envSkill }),
    }),
    {
      name: 'rl-app-store',
      partialize: (s) => ({
        locale:          s.locale,
        theme:           s.theme,
        mode:            s.mode,
        modeChosen:      s.modeChosen,
        selectedEnvId:   s.selectedEnvId,
        algo:            s.algo,
        hyperparams:     s.hyperparams,
        evolutionParams: s.evolutionParams,
        qLearningParams: s.qLearningParams,
        selfPlayParams:  s.selfPlayParams,
        alphaZeroParams: s.alphaZeroParams,
        sacParams:       s.sacParams,
        td3Params:       s.td3Params,
        dqnParams:       s.dqnParams,
        a2cParams:       s.a2cParams,
        qrdqnParams:     s.qrdqnParams,
        seed:            s.seed,
        totalTimesteps:  s.totalTimesteps,
        sweepCount:      s.sweepCount,
        emaAlpha:        s.emaAlpha,
        chartWindow:     s.chartWindow,
        activeTab:       s.activeTab,
        visual:          s.visual,
        speed:           s.speed,
        attemptMode:     s.attemptMode,
        playMode:        s.playMode,
        playSpeed:       s.playSpeed,
        playerName:      s.playerName,
        boardSide:       s.boardSide,
        boardStrength:   s.boardStrength,
      }),
    },
  ),
)
