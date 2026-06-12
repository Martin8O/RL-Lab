import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { EnvSpec, PPOHyperparams, TrainingMetrics, TrainingProgress, TrainState } from '../api/types'

export type Locale        = 'cz' | 'en'
export type Theme         = 'dark' | 'light'
export type BackendStatus = 'connecting' | 'online' | 'offline'
export type ChartTab      = 'reward' | 'loss' | 'fitness'

// Cap on retained ~1 Hz progress frames (3 h of training); the chart's window control
// still slices this down for display.
const PROGRESS_CAP = 10_800

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

interface AppState {
  // ─ persisted ───────────────────────────────────────────────
  locale:          Locale
  theme:           Theme
  selectedEnvId:   string | null
  hyperparams:     PPOHyperparams
  seed:            number
  totalTimesteps:  number
  emaAlpha:        number     // 1 = raw; 0.05 = heavy smoothing
  chartWindow:     number     // 0 = all; N = last N rollouts
  activeTab:       ChartTab
  visual:          boolean    // env-preview frame streaming on/off
  speed:           number     // playback speed multiplier (1×–20×)

  // ─ ephemeral (not persisted) ───────────────────────────────
  backendStatus:   BackendStatus
  envs:            EnvSpec[]
  trainState:      TrainState
  metricsHistory:  TrainingMetrics[]
  progressHistory: TrainingProgress[]   // ~1 Hz frames — feeds the reward chart
  lastProgress:    TrainingProgress | null
  bestReward:      number | null

  // ─ actions ────────────────────────────────────────────────
  setLocale:          (l: Locale)                   => void
  setTheme:           (t: Theme)                    => void
  setBackendStatus:   (s: BackendStatus)            => void
  setEnvs:            (envs: EnvSpec[])             => void
  setSelectedEnvId:   (id: string | null)           => void
  setHyperparams:     (h: Partial<PPOHyperparams>)  => void
  setSeed:            (s: number)                   => void
  setTotalTimesteps:  (n: number)                   => void
  setEmaAlpha:        (a: number)                   => void
  setChartWindow:     (w: number)                   => void
  setActiveTab:       (t: ChartTab)                 => void
  setVisual:          (v: boolean)                  => void
  setSpeed:           (n: number)                   => void
  setTrainState:      (s: TrainState)               => void
  addMetrics:         (m: TrainingMetrics)          => void
  setProgress:        (p: TrainingProgress)         => void
  clearMetrics:       ()                            => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      locale:          'en',
      theme:           'dark',
      selectedEnvId:   null,
      hyperparams:     DEFAULT_HYPERPARAMS,
      seed:            42,
      totalTimesteps:  50_000,
      emaAlpha:        0.3,
      chartWindow:     0,
      activeTab:       'reward',
      visual:          true,
      speed:           1,

      backendStatus:   'connecting',
      envs:            [],
      trainState:      'idle',
      metricsHistory:  [],
      progressHistory: [],
      lastProgress:    null,
      bestReward:      null,

      setLocale:         (locale)         => set({ locale }),
      setTheme:          (theme)          => set({ theme }),
      setBackendStatus:  (backendStatus)  => set({ backendStatus }),
      setEnvs:           (envs)           => set({ envs }),
      setSelectedEnvId:  (selectedEnvId)  => set({ selectedEnvId }),
      setHyperparams:    (h)              => set((s) => ({ hyperparams: { ...s.hyperparams, ...h } })),
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

      clearMetrics: () =>
        set({ metricsHistory: [], progressHistory: [], lastProgress: null, bestReward: null }),
    }),
    {
      name: 'rl-app-store',
      partialize: (s) => ({
        locale:         s.locale,
        theme:          s.theme,
        selectedEnvId:  s.selectedEnvId,
        hyperparams:    s.hyperparams,
        seed:           s.seed,
        totalTimesteps: s.totalTimesteps,
        emaAlpha:       s.emaAlpha,
        chartWindow:    s.chartWindow,
        activeTab:      s.activeTab,
        visual:         s.visual,
        speed:          s.speed,
      }),
    },
  ),
)
