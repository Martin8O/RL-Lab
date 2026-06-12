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

export type Algo = 'ppo'
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

export interface TrainConfig {
  env_id: string
  algo: Algo
  seed: number
  total_timesteps: number
  hyperparams: PPOHyperparams
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

/** WS frame: {type:"progress", ...} pushed ~once per second during a run. */
export interface TrainingProgress {
  type: 'progress'
  iteration: number
  timesteps: number
  total_timesteps: number
  steps_per_sec: number
  elapsed: number
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
  | TrainStatus
  | PreviewState
  | PreviewFrame
