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
  solved_score: number
  /** The score that reads as 0% on the skill meter (0 for CartPole, negative for shaped envs). */
  min_score: number
  /** Recommended PPO training budget (the ★ default); the sidebar builds its step ladder from it. */
  default_total_timesteps: number
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

// --- Run history (D2) ------------------------------------------------------
// Mirrors backend/app/schemas/runs.py — keep both sides in sync.

/** One finished training run. PPO fills timesteps/total_timesteps/iteration;
 *  neuroevolution fills generation/total_generations. */
export interface RunMeta {
  id: string
  label: string
  env_id: string
  algo: Algo
  seed: number
  created_at: string
  finished_at: string
  state: TrainState
  final_reward: number | null
  /** x where the run first hit the solved score: a timestep (PPO) or generation (evolution),
   *  in the same unit as that algorithm's chart. null = never solved. */
  solved_at: number | null
  timesteps: number
  total_timesteps: number
  iteration: number | null
  generation: number | null
  total_generations: number | null
  frames: number
}

/** A run read back in full for the chart overlay: listing row + reproducible config +
 *  every recorded metric frame (each a TrainingMetrics or EvolutionMetrics dump). */
export interface RunDetail {
  meta: RunMeta
  config: TrainConfig
  metrics: (TrainingMetrics | EvolutionMetrics)[]
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
  /** Latest neuroevolution frame, retained so a late-joining client can repopulate the
   *  leaderboard / Evolution Stats / Fitness chart on reconnect. null for PPO runs. */
  last_evolution: EvolutionMetrics | null
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

/** WS frame: {type:"frame", ...} — a rendered env image OR client-render state. */
export interface PreviewFrame {
  type: 'frame'
  episode: number
  step: number
  reward: number
  width?: number
  height?: number
  /** base64-encoded JPEG, no data-URI prefix (the client prepends it). */
  image?: string
  /** Client-side render state (CartPole: [x, theta]); present instead of image. */
  state?: number[]
}

/** Partial preview update for POST /api/preview. */
export interface PreviewConfig {
  visual?: boolean
  speed?: number
}

// --- Play vs AI & skill meter (E1) -----------------------------------------
// Mirrors backend/app/schemas/skill.py + schemas/play.py — keep both sides in sync.

/** The five fixed rating bands, weakest → strongest. The UI maps each id to a localized
 *  label + meter colour; the backend deals only in ids + numeric thresholds. */
export type SkillBandId =
  | 'child'
  | 'below_average'
  | 'average'
  | 'above_average'
  | 'superhuman'

/** One band: its id and the inclusive lower score bound that qualifies for it. */
export interface SkillBand {
  id: SkillBandId
  min_score: number
}

/** Skill-band thresholds for one env — returned by GET /api/skill/{env_id}. */
export interface EnvSkill {
  env_id: string
  max_score: number
  min_score: number
  bands: SkillBand[]
}

/** How a finished session scored. `ratio` (score/max_score, clamped 0..1) is a ready-made
 *  fill fraction for the skill meter. */
export interface SkillRating {
  band: SkillBandId
  score: number
  max_score: number
  ratio: number
}

/** Who controls the agent: a human at the keyboard, or a loaded checkpoint playing itself. */
export type PlayMode = 'human' | 'ai'
export type PlayState = 'idle' | 'playing' | 'finished' | 'stopped' | 'error'

/** Start request for POST /api/play/start. `checkpoint_id` is required for mode "ai". */
export interface PlayConfig {
  env_id: string
  mode: PlayMode
  checkpoint_id?: string | null
  seed?: number | null
  speed: number
}

/** WS frame: {type:"play_result", ...} — the final score + skill rating of a finished episode. */
export interface PlayResult {
  type: 'play_result'
  env_id: string
  mode: PlayMode
  score: number
  steps: number
  rating: SkillRating
}

/** Lifecycle snapshot: returned by /api/play/* and pushed as {type:"play_status", ...}. */
export interface PlayStatus {
  type: 'play_status'
  state: PlayState
  env_id: string | null
  mode: PlayMode | null
  checkpoint_id: string | null
  seed: number | null
  speed: number
  step: number
  score: number
  result: PlayResult | null
  error: string | null
}

/** WS frame: {type:"play_frame", ...} — a rendered episode image OR client-render state. */
export interface PlayFrame {
  type: 'play_frame'
  step: number
  score: number
  width?: number
  height?: number
  image?: string
  /** Client-side render state (CartPole: [x, theta]); present instead of image. */
  state?: number[]
}

/** Outbound human input over WS: {type:"action", action:<int>} (CartPole: 0=left, 1=right). */
export interface PlayActionMessage {
  type: 'action'
  action: number
}

// --- Play leaderboards (E2) -------------------------------------------------
// Mirrors backend/app/schemas/play_scores.py — keep both sides in sync.

/** Which board: the human (you, at the keyboard) or an AI (a checkpoint playing itself). */
export type PlayCategory = 'human' | 'ai'

/** One leaderboard row. `model_id`/`algo` are set only for AI rows (de-dup + badge). */
export interface PlayScoreEntry {
  name: string
  score: number
  steps: number
  achieved_at: string
  model_id?: string | null
  algo?: string | null
}

/** Both boards for one env — returned by GET /api/playscores/{env_id}. */
export interface PlayScores {
  env_id: string
  human: PlayScoreEntry[]
  ai: PlayScoreEntry[]
}

/** Submit a finished session's score (POST /api/playscores/{env_id}). */
export interface PlayScoreSubmit {
  category: PlayCategory
  name: string
  score: number
  steps?: number
  model_id?: string | null
  algo?: string | null
}

/** Result of a submit: updated boards + whether/where the entry landed. */
export interface PlayScoreResult {
  scores: PlayScores
  qualified: boolean
  rank: number | null
}

/** How many rows each board keeps + shows (mirror of backend TOP_N) — used to pre-decide a name prompt. */
export const PLAY_SCORE_TOP_N = 5

/** Any frame the WS channel can push. */
export type TrainWsFrame =
  | TrainingMetrics
  | TrainingProgress
  | EvolutionMetrics
  | TrainStatus
  | PreviewState
  | PreviewFrame
  | HighScore
  | PlayStatus
  | PlayResult
  | PlayFrame
