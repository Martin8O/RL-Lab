export type ObsType = 'vector' | 'image'
export type ActionSpace = 'discrete' | 'box'
export type Difficulty = 'beginner' | 'intermediate' | 'advanced'
export type HwRequirement = 'cpu' | 'gpu'
export type HyperparamType = 'float' | 'int' | 'categorical'

export interface Bilingual {
  en: string
  cz: string
}

export interface HyperparamDef {
  type: HyperparamType
  default: number | string
  recommended: number | string
  min: number | null
  max: number | null
  step: number | null
  choices: string[] | null
}

export interface EnvSpec {
  id: string
  gym_id: string
  display_name: Bilingual
  description: Bilingual
  family: string
  obs_type: ObsType
  action_space: ActionSpace
  supported_algos: string[]
  /** algo_id → param_id → definition */
  hyperparams: Record<string, Record<string, HyperparamDef>>
  human_playable: boolean
  competitive: boolean
  difficulty: Difficulty
  hw_requirement: HwRequirement
}

// --- Training (B2) ---------------------------------------------------------
// Mirrors backend/app/schemas/training.py — keep both sides in sync.

export type Algo = 'ppo' | 'neuroevolution'
export type TrainState =
  | 'idle'
  | 'running'
  | 'paused'
  | 'stopping'
  | 'stopped'
  | 'finished'
  | 'error'

export interface PPOHyperparams {
  learning_rate: number
  gamma: number
  clip_range: number
  ent_coef: number
  n_steps: number
  batch_size: number
  n_hidden_layers: number
  neurons_per_layer: number
  activation: 'tanh' | 'relu'
}

export interface EvolutionHyperparams {
  population_size: number
  top_k_parents: number
  mutation_rate: number
  crossover_rate: number
  generations: number
  episodes: number
}

export interface TrainConfig {
  env_id: string
  algo: Algo
  seed: number
  total_timesteps: number
  hyperparams: PPOHyperparams
  /** Present only for neuroevolution runs; null/omitted for PPO. */
  evolution?: EvolutionHyperparams | null
}

/** WS frame: {type:"metrics", ...} pushed once per PPO rollout. */
export interface TrainingMetrics {
  type: 'metrics'
  iteration: number
  timesteps: number
  total_timesteps: number
  ep_rew_mean: number | null
  ep_len_mean: number | null
  loss: number | null
  learning_rate: number | null
  elapsed: number
}

/** WS frame: {type:"progress", ...} pushed at a steady ~1 Hz during a run.
 *  Carries the rolling reward/length means so the reward chart can plot at ~1 Hz. */
export interface TrainingProgress {
  type: 'progress'
  iteration: number
  timesteps: number
  total_timesteps: number
  steps_per_sec: number
  ep_rew_mean: number | null
  ep_len_mean: number | null
  elapsed: number
}

/** One ranked genome in a generation's Top-K leaderboard (C1). */
export interface EvolutionChild {
  id: number
  total_reward: number
  avg_reward: number
  steps: number
  seed: number
}

/** Histogram of the weight perturbations applied to breed a generation's offspring. */
export interface MutationDist {
  /** bin edges; length == counts.length + 1 */
  bins: number[]
  counts: number[]
}

/** WS frame: {type:"evolution", ...} pushed once per neuroevolution generation. */
export interface EvolutionMetrics {
  type: 'evolution'
  generation: number
  total_generations: number
  best_fitness: number
  avg_fitness: number
  worst_fitness: number
  children: EvolutionChild[]
  mutation_dist: MutationDist
  timesteps: number
  elapsed: number
}

// --- High scores (C2) ------------------------------------------------------
// Mirrors backend/app/schemas/highscores.py — keep both sides in sync.

/** How an all-time best was achieved (generation for evolution, iteration for PPO). */
export interface HighScoreMeta {
  algo: Algo
  seed: number
  generation: number | null
  iteration: number | null
  achieved_at: string
}

/** The persisted all-time best for one env. Returned by /api/highscores[/{env_id}]
 *  and pushed as {type:"highscore"} whenever a run beats the stored best. */
export interface HighScore {
  type: 'highscore'
  env_id: string
  score: number
  meta: HighScoreMeta
}

// --- Checkpoints (D1) ------------------------------------------------------
// Mirrors backend/app/schemas/checkpoints.py — keep both sides in sync.

/** One saved checkpoint slot. PPO fills timesteps/total_timesteps/iteration;
 *  neuroevolution fills generation/total_generations. */
export interface CheckpointMeta {
  id: string
  label: string
  env_id: string
  algo: Algo
  seed: number
  created_at: string
  reward: number | null
  timesteps: number
  total_timesteps: number
  iteration: number | null
  generation: number | null
  total_generations: number | null
  /** on-disk model filename: "model.zip" (PPO) | "population.npz" (evolution) */
  artifact: string
}

/** Lifecycle snapshot: returned by /api/train/* and pushed as {type:"status", ...}. */
export interface TrainStatus {
  type: 'status'
  state: TrainState
  env_id: string | null
  algo: Algo | null
  seed: number | null
  timesteps: number
  total_timesteps: number
  config: TrainConfig | null
  last_metrics: TrainingMetrics | null
  error: string | null
}

// --- Preview / frame streaming (B4) ----------------------------------------
// Mirrors backend/app/schemas/preview.py — keep both sides in sync.

/** Current preview settings: returned by /api/preview, pushed as {type:"preview"}. */
export interface PreviewState {
  type: 'preview'
  visual: boolean
  speed: number
  active: boolean
}

/** WS frame: {type:"frame", ...} — one rendered env image. */
export interface PreviewFrame {
  type: 'frame'
  episode: number
  step: number
  reward: number
  width: number
  height: number
  /** base64-encoded JPEG, no data-URI prefix (the client prepends it). */
  image: string
}

/** Partial preview update for POST /api/preview. */
export interface PreviewConfig {
  visual?: boolean
  speed?: number
}

/** Any frame the WS channel can push. */
export type TrainWsFrame =
  | TrainingMetrics
  | TrainingProgress
  | EvolutionMetrics
  | TrainStatus
  | PreviewState
  | PreviewFrame
  | HighScore
