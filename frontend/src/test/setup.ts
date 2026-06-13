import '@testing-library/jest-dom/vitest'
import '../i18n' // initialise i18next (lng "en") so t() resolves real strings in component tests
import { afterEach, beforeEach, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'

// ── jsdom gaps the app relies on ──────────────────────────────────────────────

// ResizeObserver — RewardChart measures its plot area with one; jsdom has no implementation.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub)

// WebSocket — the live training/play socket. The stub stays silent (never fires onopen), so the
// reconcile fetches in useTrainingWs don't run during a smoke render. jsdom has no WebSocket.
class WebSocketStub {
  static readonly OPEN = 1
  readyState = 0
  onopen: ((e: unknown) => void) | null = null
  onclose: ((e: unknown) => void) | null = null
  onmessage: ((e: unknown) => void) | null = null
  constructor(public url: string) {}
  send() {}
  close() {}
}
vi.stubGlobal('WebSocket', WebSocketStub)

// ── Per-test isolation ────────────────────────────────────────────────────────

// The store's initial snapshot (defaults + action closures), captured once before any test mutates it.
const initialState = useAppStore.getInitialState()

beforeEach(() => {
  localStorage.clear() // drop any persisted state the previous test wrote
  useAppStore.setState(initialState, true) // full replace → defaults, including the actions
  // Default network: every call site catches failures, so a fast reject keeps effects offline
  // with no real I/O. A test can override with vi.mocked(fetch).mockResolvedValue(...).
  vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new Error('network disabled in tests'))))
})

afterEach(() => {
  cleanup()
})
