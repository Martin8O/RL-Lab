export type ObsType = 'vector' | 'image' | 'discrete'
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
  /** Play episodes run this many times longer than training (so a person has time to play). 1 = same. */
  play_step_scale: number
  /** Whether the play skill-meter floor is widened by play_step_scale. True for step-penalty envs
   *  (floor grows with steps); false for shaped/terminal-reward envs like LunarLander (a crash ends
   *  early and doesn't scale), so their floor stays put and a crash isn't rated as a near-success. */
  floor_scales_with_steps: boolean
  /** Sparse 0/1 reward (FrozenLake): the running play score is 0 until the goal, so the play skill
   *  meter shows "measuring…" until the episode ends (the partial score isn't a valid reading). */
  sparse_reward: boolean
  /** Grid-worlds the human plays turn-based (one move per key press) instead of in real time. */
  turn_based: boolean
  human_playable: boolean
  competitive: boolean
  difficulty: Difficulty
  hw_requirement: HwRequirement
  /** Whether this env's training code path exists yet. True for every vector/discrete env (incl. the
   *  GPU-gated vector heavies BipedalWalker + MuJoCo, which train with MlpPolicy). False for image envs
   *  (Atari, CarRacing): their CnnPolicy/GPU trainer isn't built (G4b/G3c-train), so training stays
   *  gated even on a GPU machine until that seam lands. Decouples "needs a GPU" from "trainer missing". */
  train_implemented: boolean
  /** Competitive multi-agent only (simple_tag, G7b-2): the SECOND species' skill scale for the
   *  two-line ecosystem chart. `min_score`/`solved_score` are the predator (headline) scale; these are
   *  the prey scale (returns are negative — a deep "caught" floor up to a near-0 "escapes" good end).
   *  null for every single-species env. */
  prey_min_score?: number | null
  prey_solved_score?: number | null
}

// --- System capabilities (G4a) ---------------------------------------------
// Mirrors backend/app/schemas/system.py — keep both sides in sync.

/** Runtime hardware facts used to gate features. `gpu_available` decides whether GPU-only
 *  training (Atari and other image-obs envs) can run on this machine. */
export interface SystemInfo {
  gpu_available: boolean
}

// --- Training (B2) ---------------------------------------------------------
// Mirrors backend/app/schemas/training.py — keep both sides in sync.

export type Algo = 'ppo' | 'neuroevolution' | 'q_learning'
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

export interface QLearningHyperparams {
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_end: number
  /** Fraction of the episode budget to anneal ε over (start→end), then hold at end. */
  epsilon_decay: number
  /** The episode budget — Q-learning's "Total Steps". */
  episodes: number
}

/** Competitive multi-agent self-play knobs (simple_tag, G7b-2). The per-species PPO nets reuse
 *  PPOHyperparams; this carries only the round schedule (how many times the two species alternate). */
export interface SelfPlayHyperparams {
  rounds: number
}

export interface TrainConfig {
  env_id: string
  algo: Algo
  seed: number
  total_timesteps: number
  hyperparams: PPOHyperparams
  /** Present only for neuroevolution runs; null/omitted otherwise. */
  evolution?: EvolutionHyperparams | null
  /** Present only for tabular Q-learning runs; null/omitted otherwise. */
  q_learning?: QLearningHyperparams | null
  /** Present only for competitive multi-agent self-play runs (simple_tag); null/omitted otherwise. */
  self_play?: SelfPlayHyperparams | null
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

/** Live hardware telemetry (G4b). CPU + RAM always present; GPU fields are null when NVML/pynvml is
 *  unavailable (a non-NVIDIA machine) → the panel shows `—`. `cpu_process_pct` is this process
 *  normalised to 0–100 % of the whole machine; memory in MB. */
export interface HwStats {
  cpu_process_pct: number
  ram_used_mb: number
  ram_total_mb: number
  gpu_util_pct: number | null
  gpu_vram_used_mb: number | null
  gpu_vram_total_mb: number | null
  gpu_temp_c: number | null
  gpu_power_w: number | null
}

/** WS frame: {type:"hwstats", ...} — a 1 Hz HW telemetry sample, broadcast by the manager for any
 *  active run regardless of algorithm (PPO / neuroevolution / Q-learning all feed the HW panel). */
export interface HwStatsFrame {
  type: 'hwstats'
  stats: HwStats
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

/** WS frame: {type:"q_learning", ...} pushed periodically during a tabular Q-learning run.
 *  Episodic, so its chart x-unit is `episode`; `ep_rew_mean` is the headline learning curve
 *  (for FrozenLake this is literally the success rate). The table rides in QTableFrame. */
export interface QLearningMetrics {
  type: 'q_learning'
  iteration: number
  episode: number
  total_episodes: number
  epsilon: number
  ep_rew_mean: number | null
  ep_len_mean: number | null
  timesteps: number
  elapsed: number
}

/** The learned action-value table for the heatmap — row-major [n_states][n_actions]. */
export interface QTable {
  n_states: number
  n_actions: number
  values: number[][]
}

/** WS frame: {type:"qtable", ...} — the current Q-table snapshot for the live heatmap.
 *  Decoupled from QLearningMetrics (and never logged) so it can stream at its own cadence. */
export interface QTableFrame {
  type: 'qtable'
  episode: number
  total_episodes: number
  table: QTable
}

/** One species' current learning stats inside a competitive self-play frame (simple_tag, G7b-2).
 *  `role` is "adversary" (predators) or "agent" (prey); the return is per-agent (the shared net's
 *  mean episode return); `timesteps` only grows during that species' learning turns. */
export interface SpeciesMetrics {
  role: string
  ep_rew_mean: number | null
  ep_len_mean: number | null
  timesteps: number
}

/** WS frame: {type:"ma_metrics", ...} — one competitive self-play frame carrying BOTH species at
 *  once (the two-line "ecosystem" chart). `learning_role` is whichever species is optimising now (the
 *  other plays frozen this round). `ep_rew_mean` mirrors the predator headline (drives high-score). */
export interface MultiAgentMetrics {
  type: 'ma_metrics'
  round: number
  total_rounds: number
  learning_role: string
  species: SpeciesMetrics[]
  ep_rew_mean: number | null
  timesteps: number
  total_timesteps: number
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
  metrics: (TrainingMetrics | EvolutionMetrics | QLearningMetrics)[]
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
  /** Latest tabular Q-learning frame + Q-table snapshot, retained so a late-joining client can
   *  repopulate the chart / stats / heatmap on reconnect. null for PPO / neuroevolution runs. */
  last_q_learning: QLearningMetrics | null
  last_qtable: QTableFrame | null
  /** Latest competitive self-play frame (simple_tag), retained so a late-joining client repopulates
   *  the two-line ecosystem chart on reconnect. null for every single-policy run. */
  last_ma_metrics: MultiAgentMetrics | null
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

/** Static board layout for a client-rendered grid-world (Toy Text), streamed with each frame.
 *  The dynamic agent/passenger/destination position rides in the frame's `state`; this is the fixed
 *  board drawn under it. `cells` is row-major (length rows*cols): "normal" | "start" | "goal" |
 *  "hole" (FrozenLake) | "cliff" (CliffWalking) | "stop" (Taxi pickup/drop-off). */
export interface GridLayout {
  kind: 'frozenlake' | 'cliffwalking' | 'taxi'
  rows: number
  cols: number
  cells: string[]
}

/** One agent's render state for the multi-agent "swarm" canvas (PettingZoo, G7a). World-space
 *  [x, y] (the client autoscales the scene); `role` ("agent" | "adversary") drives the colour,
 *  `size` is the entity's radius in the same world units. */
export interface AgentSprite {
  x: number
  y: number
  role: string
  size: number
}

/** A landmark for the swarm canvas — a coverage "target" (simple_spread) or an "obstacle"
 *  (a collidable landmark). Same world-space coordinates + size as AgentSprite. */
export interface WorldEntity {
  x: number
  y: number
  kind: string
  size: number
}

/** One ply of an OpenSpiel board game (G6a), streamed inside a play frame. Built from the generic
 *  pyspiel.State API (app/services/board_engine.py) so it carries Tic-Tac-Toe today and Connect Four
 *  / chess / go later unchanged; the renderer (content/boardGames.ts + BoardStage) maps the glyphs. */
/** One legal move of a move-based board game (Breakthrough, G6e): the action int + the board cells it
 *  moves from/to (row-major, matching `BoardState.cells`). The renderer maps a clicked (from,to) pair
 *  back to `action`; present only for move-mode games (`BoardState.moves`). Mirrors backend BoardMove. */
export interface BoardMove {
  action: number
  from_cell: number
  to_cell: number
}

export interface BoardState {
  /** Row-major board glyphs: "." empty, "x"/"o" for Tic-Tac-Toe, etc. (the renderer interprets them). */
  cells: string[]
  rows: number
  cols: number
  /** Action indices legal for the player to move now (empty once the game is over); the client
   *  highlights these on the human's turn and rejects clicks elsewhere. */
  legal_actions: number[]
  current_player: number
  last_action: number | null
  is_terminal: boolean
  /** Winning player index, or null for a draw / a game still in progress. */
  winner: number | null
  /** A legal "pass" move mapping to no board cell (Othello when a player has no placement; Go), or
   *  null. The renderer shows a Pass button for it instead of a cell click (G6d). */
  pass_action?: number | null
  /** Move-based games (Breakthrough, G6e): per legal action of the current player, its (from→to) cells.
   *  Absent for placement games (TTT/Connect Four/Othello), which keep the single-click cell/column flow. */
  moves?: BoardMove[] | null
  /** The from/to cells of the last move played, for a move highlight (move-mode games only). */
  last_from?: number | null
  last_to?: number | null
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
  /** The discrete action just applied (lets the client draw the firing thruster, e.g. LunarLander). */
  action?: number | null
  /** Per-episode scene geometry not in the obs — LunarLander's random moon surface as [x, y] points. */
  terrain?: number[][] | null
  /** Static board layout for a grid-world (Toy Text); the client draws the board under the agent. */
  grid?: GridLayout | null
  /** Multi-agent (PettingZoo) render state: per-agent sprites + landmark entities for the swarm
   *  canvas (G7a). Present instead of image/state for the multi-agent family. */
  agents?: AgentSprite[] | null
  world?: WorldEntity[] | null
  /** Board-game state (G6b): the live training preview self-plays the learning net ply by ply, so a
   *  board run streams its board here (rendered on the same BoardStage as play). Null for non-board envs. */
  board?: BoardState | null
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
  /** Action held when the human gives no input (the env's "do nothing"); null = no idle (CartPole
   *  always moves). A number for a discrete/continuous-scalar env, an array for a continuous vector
   *  env. Source of truth: content/playKeymaps.ts. */
  idle_action?: number | number[] | null
  /** Board games (G6a) only: which player the human controls (0 = first to move) and the MCTS
   *  opponent strength. Ignored by every other env; `mode:"ai"` on a board env is an AI-vs-AI watch. */
  side?: number
  ai_strength?: BoardStrength
}

/** Board-game AI opponent strength (G6a) → MCTS simulation count, server-side. */
export type BoardStrength = 'easy' | 'medium' | 'hard'

/** WS frame: {type:"play_result", ...} — the final score + skill rating of a finished episode.
 *  Continuous-score envs carry `rating`; board games (G6a) are 3-valued and carry `outcome` with a
 *  null `rating` (the UI shows a win/draw/loss card, not the misleading continuous skill %). */
export interface PlayResult {
  type: 'play_result'
  env_id: string
  mode: PlayMode
  score: number
  steps: number
  rating: SkillRating | null
  outcome?: 'win' | 'draw' | 'loss' | null
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
  /** The discrete action just applied (lets the client draw the firing thruster, e.g. LunarLander). */
  action?: number | null
  /** Per-episode scene geometry not in the obs — LunarLander's random moon surface as [x, y] points. */
  terrain?: number[][] | null
  /** Static board layout for a grid-world (Toy Text); the client draws the board under the agent. */
  grid?: GridLayout | null
  /** Multi-agent swarm state — never set on a play frame today (the play session is single-agent);
   *  declared so the shared frame handler can read it off the PreviewFrame|PlayFrame union. */
  agents?: AgentSprite[] | null
  world?: WorldEntity[] | null
  /** Board-game state (G6a) — present instead of image/state on a board play frame. */
  board?: BoardState | null
}

/** Outbound human input over WS: {type:"action", action:<number|number[]>}. A discrete action id
 *  (CartPole: 0=left, 1=right), a continuous scalar command (Pendulum: a torque in [-2, 2]), or a
 *  continuous vector. */
export interface PlayActionMessage {
  type: 'action'
  action: number | number[]
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
  | HwStatsFrame
  | EvolutionMetrics
  | QLearningMetrics
  | QTableFrame
  | MultiAgentMetrics
  | TrainStatus
  | PreviewState
  | PreviewFrame
  | HighScore
  | PlayStatus
  | PlayResult
  | PlayFrame
