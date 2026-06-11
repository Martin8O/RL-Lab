import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Locale        = 'cz' | 'en'
export type Theme         = 'dark' | 'light'
export type BackendStatus = 'connecting' | 'online' | 'offline'

interface AppState {
  locale:        Locale
  theme:         Theme
  backendStatus: BackendStatus
  setLocale:        (l: Locale)        => void
  setTheme:         (t: Theme)         => void
  setBackendStatus: (s: BackendStatus) => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      locale:        'en',
      theme:         'dark',
      backendStatus: 'connecting',
      setLocale:        (locale)  => set({ locale }),
      setTheme:         (theme)   => set({ theme }),
      setBackendStatus: (backendStatus) => set({ backendStatus }),
    }),
    {
      name:        'rl-app-store',
      partialize: (s) => ({ locale: s.locale, theme: s.theme }),
    },
  ),
)
