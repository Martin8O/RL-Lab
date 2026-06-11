import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { EnvSpec } from '../api/types'

export type Locale        = 'cz' | 'en'
export type Theme         = 'dark' | 'light'
export type BackendStatus = 'connecting' | 'online' | 'offline'

interface AppState {
  locale:          Locale
  theme:           Theme
  backendStatus:   BackendStatus
  envs:            EnvSpec[]
  selectedEnvId:   string | null
  setLocale:         (l: Locale)        => void
  setTheme:          (t: Theme)         => void
  setBackendStatus:  (s: BackendStatus) => void
  setEnvs:           (envs: EnvSpec[])  => void
  setSelectedEnvId:  (id: string | null) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      locale:          'en',
      theme:           'dark',
      backendStatus:   'connecting',
      envs:            [],
      selectedEnvId:   null,
      setLocale:         (locale)         => set({ locale }),
      setTheme:          (theme)          => set({ theme }),
      setBackendStatus:  (backendStatus)  => set({ backendStatus }),
      setEnvs:           (envs)           => set({ envs }),
      setSelectedEnvId:  (selectedEnvId)  => set({ selectedEnvId }),
    }),
    {
      name:       'rl-app-store',
      partialize: (s) => ({ locale: s.locale, theme: s.theme, selectedEnvId: s.selectedEnvId }),
    },
  ),
)
