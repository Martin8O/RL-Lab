import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  Algo,
  EnvSkill,
  EnvSpec,
  EvolutionHyperparams,
  EvolutionMetrics,
  HighScore,
  PlayMode,
  PlayResult,
  PlayScores,
  PlayState,
  PlayStatus,
  PPOHyperparams,
  TrainingMetrics,
  TrainingProgress,
  TrainState,
} from '../api/types'

export type Locale        = 'cz' | 'en'
export type Theme         = 'dark' | 'light'
export type BackendStatus = 'connecting' | 'online' | 'offline'
export type ChartTab      = 'reward' | 'loss' | 'fitness'

// Cap on retained ~1 Hz progress frames (3 h of training); the chart's window control
// still slices this down for display.
const PROGRESS_CAP = 10_800
// Generations are coarse (one frame each), so a generous cap covers any realistic run.
const EVOLUTION_CAP = 2_000

const DEFAULT_HYPERPARAMS: PPOHyperparams = {
  learning_rate:   3e-4,
  gamma:           0.99,
  clip_range:      0.2,
  ent_coef:        0.0,
  n_steps:         2048,
  batch_size:      64,
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

interface AppState {
  // ─ persisted ───────────────────────────────────────────────
  locale:          Locale
  theme:           Theme
  selectedEnvId:   string | null
  algo:            Algo       // PPO ↔ neuroevolution
  hyperparams:     PPOHyperparams
  evolutionParams: EvolutionHyperparams
  seed:            number
  totalTimesteps:  number
  emaAlpha:        number     // 1 = raw; 0.05 = heavy smoothing
  chartWindow:     number     // 0 = all; N = last N rollouts
  activeTab:       ChartTab
  visual:          boolean    // env-preview frame streaming on/off
  speed:           number     // playback speed multiplier (1×–20×)
  playMode:        PlayMode   // E2: who plays — human at the keyboard ↔ AI watch
  playSpeed:       number     // E2: play-session pacing (0.1×–20×; slow-mo for humans)
  playerName:      string     // E2: last name used for a human leaderboard entry

  // ─ ephemeral (not persisted) ───────────────────────────────
  backendStatus:   BackendStatus
  envs:            EnvSpec[]
  trainState:      TrainState
  metricsHistory:  TrainingMetrics[]
  progressHistory: TrainingProgress[]   // ~1 Hz frames — feeds the reward chart
  lastProgress:    TrainingProgress | null
  bestReward:      number | null        // best score this session (live high)
  evolutionHistory: EvolutionMetrics[]  // per-generation frames — feeds the Fitness chart
  lastEvolution:   EvolutionMetrics | null
  highScores:      Record<string, HighScore>  // all-time best per env id

  // ─ play vs AI (E2) ─────────────────────────────────────────
  playState:        PlayState            // idle until a session starts
  playScore:        number               // live score of the current/last session
  playStep:         number
  playResult:       PlayResult | null    // set once the episode ends (carries the rated band)
  playError:        string | null
  playCheckpointId: string | null        // selected checkpoint for AI watch mode
  playCheckpointLabel: string | null     // its label — the AI's leaderboard identity on finish
  envSkill:         EnvSkill | null       // backend skill thresholds for the selected env
  playScores:       PlayScores | null     // Human + AI boards for the selected env

  // ─ actions ────────────────────────────────────────────────
  setLocale:          (l: Locale)                       => void
  setTheme:           (t: Theme)                        => void
  setBackendStatus:   (s: BackendStatus)                => void
  setEnvs:            (envs: EnvSpec[])                 => void
  setSelectedEnvId:   (id: string | null)               => void
  setAlgo:            (a: Algo)                          => void
  setHyperparams:     (h: Partial<PPOHyperparams>)      => void
  setEvolutionParams: (e: Partial<EvolutionHyperparams>) => void
  setSeed:            (s: number)                       => void
  setTotalTimesteps:  (n: number)                       => void
  setEmaAlpha:        (a: number)                       => void
  setChartWindow:     (w: number)                       => void
  setActiveTab:       (t: ChartTab)                     => void
  setVisual:          (v: boolean)                      => void
  setSpeed:           (n: number)                       => void
  setTrainState:      (s: TrainState)                   => void
  addMetrics:         (m: TrainingMetrics)              => void
  setProgress:        (p: TrainingProgress)             => void
  addEvolution:       (e: EvolutionMetrics)             => void
  seedEvolution:      (e: EvolutionMetrics)             => void
  setHighScore:       (hs: HighScore)                   => void
  setHighScores:      (list: HighScore[])               => void
  clearMetrics:       ()                                => void

  // play vs AI (E2)
  setPlayMode:         (m: PlayMode)                    => void
  setPlaySpeed:        (n: number)                      => void
  setPlayerName:       (name: string)                  => void
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
      selectedEnvId:   null,
      algo:            'ppo',
      hyperparams:     DEFAULT_HYPERPARAMS,
      evolutionParams: DEFAULT_EVOLUTION_PARAMS,
      seed:            42,
      totalTimesteps:  50_000,
      emaAlpha:        0.3,
      chartWindow:     0,
      activeTab:       'reward',
      visual:          true,
      speed:           1,
      playMode:        'human',
      playSpeed:       1,
      playerName:      '',

      backendStatus:   'connecting',
      envs:            [],
      trainState:      'idle',
      metricsHistory:  [],
      progressHistory: [],
      lastProgress:    null,
      bestReward:      null,
      evolutionHistory: [],
      lastEvolution:   null,
      highScores:      {},

      playState:        'idle',
      playScore:        0,
      playStep:         0,
      playResult:       null,
      playError:        null,
      playCheckpointId: null,
      playCheckpointLabel: null,
      envSkill:         null,
      playScores:       null,

      setLocale:         (locale)         => set({ locale }),
      setTheme:          (theme)          => set({ theme }),
      setBackendStatus:  (backendStatus)  => set({ backendStatus }),
      setEnvs:           (envs)           => set({ envs }),
      setSelectedEnvId:  (selectedEnvId)  => set({ selectedEnvId }),
      // Switching algorithm also jumps to the chart tab that algorithm feeds, so the chart
      // never sits empty after a switch (PPO → Reward, neuroevolution → Fitness).
      setAlgo:           (algo)           => set({ algo, activeTab: algo === 'neuroevolution' ? 'fitness' : 'reward' }),
      setHyperparams:    (h)              => set((s) => ({ hyperparams: { ...s.hyperparams, ...h } })),
      setEvolutionParams:(e)              => set((s) => ({ evolutionParams: { ...s.evolutionParams, ...e } })),
      setSeed:           (seed)           => set({ seed }),
      setTotalTimesteps: (n)              => set({ totalTimesteps: n }),
      setEmaAlpha:       (emaAlpha)       => set({ emaAlpha }),
      setChartWindow:    (chartWindow)    => set({ chartWindow }),
      setActiveTab:      (activeTab)      => set({ activeTab }),
      setVisual:         (visual)         => set({ visual }),
      setSpeed:          (speed)          => set({ speed }),
      setTrainState:     (trainState)     => set({ trainState }),

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

      setHighScore:  (hs)   => set((s) => ({ highScores: { ...s.highScores, [hs.env_id]: hs } })),
      setHighScores: (list) =>
        set({ highScores: Object.fromEntries(list.map((hs) => [hs.env_id, hs])) }),

      clearMetrics: () =>
        set({
          metricsHistory: [], progressHistory: [], lastProgress: null, bestReward: null,
          evolutionHistory: [], lastEvolution: null,
        }),

      // ─ play vs AI (E2) ────────────────────────────────────────
      setPlayMode:         (playMode)         => set({ playMode }),
      setPlaySpeed:        (playSpeed)        => set({ playSpeed }),
      setPlayerName:       (playerName)       => set({ playerName }),
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
        selectedEnvId:   s.selectedEnvId,
        algo:            s.algo,
        hyperparams:     s.hyperparams,
        evolutionParams: s.evolutionParams,
        seed:            s.seed,
        totalTimesteps:  s.totalTimesteps,
        emaAlpha:        s.emaAlpha,
        chartWindow:     s.chartWindow,
        activeTab:       s.activeTab,
        visual:          s.visual,
        speed:           s.speed,
        playMode:        s.playMode,
        playSpeed:       s.playSpeed,
        playerName:      s.playerName,
      }),
    },
  ),
)
