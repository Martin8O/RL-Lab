import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { EnvSpec, PPOHyperparams, TrainingMetrics, TrainState } from '../api/types'

export type Locale        = 'cz' | 'en'
export type Theme         = 'dark' | 'light'
export type BackendStatus = 'connecting' | 'online' | 'offline'
export type ChartTab      = 'reward' | 'loss' | 'fitness'

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

  // ─ ephemeral (not persisted) ───────────────────────────────
  backendStatus:   BackendStatus
  envs:            EnvSpec[]
  trainState:      TrainState
  metricsHistory:  TrainingMetrics[]
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
  setTrainState:      (s: TrainState)               => void
  addMetrics:         (m: TrainingMetrics)          => void
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

      backendStatus:   'connecting',
      envs:            [],
      trainState:      'idle',
      metricsHistory:  [],
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

      clearMetrics: () => set({ metricsHistory: [], bestReward: null }),
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
      }),
    },
  ),
)
