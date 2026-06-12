import { useEffect } from 'react'
import { useAppStore } from '../store/useAppStore'
import type {
  CheckpointMeta,
  EnvSpec,
  HighScore,
  PreviewConfig,
  PreviewFrame,
  PreviewState,
  TrainConfig,
  TrainStatus,
  TrainWsFrame,
} from './types'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''

// ── Health ──────────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<{ status: string; version: string }> {
  const res = await fetch(`${API_BASE}/api/health`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<{ status: string; version: string }>
}

export function useHealthPoll(intervalMs = 5000): void {
  const setBackendStatus = useAppStore((s) => s.setBackendStatus)

  useEffect(() => {
    let active = true

    async function poll() {
      try {
        await fetchHealth()
        if (active) setBackendStatus('online')
      } catch {
        if (active) setBackendStatus('offline')
      }
    }

    void poll()
    const id = setInterval(() => void poll(), intervalMs)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [setBackendStatus, intervalMs])
}

// ── WebSocket ───────────────────────────────────────────────────────────────

const WS_BASE =
  (import.meta.env.VITE_WS_BASE as string | undefined) ??
  (typeof window !== 'undefined'
    ? `ws://${window.location.host}`
    : 'ws://localhost:8000')

export function createWsClient(
  onMessage: (data: unknown) => void,
  onStatusChange: (connected: boolean) => void,
): { stop: () => void } {
  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let stopped = false

  function connect() {
    ws = new WebSocket(`${WS_BASE}/ws`)
    ws.onopen  = () => onStatusChange(true)
    ws.onclose = () => {
      onStatusChange(false)
      if (!stopped) reconnectTimer = setTimeout(connect, 3000)
    }
    ws.onmessage = (e: MessageEvent<string>) => {
      try { onMessage(JSON.parse(e.data)) } catch { /* skip non-JSON */ }
    }
  }

  function stop() {
    stopped = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    ws?.close()
  }

  connect()
  return { stop }
}

// Env-preview frames are high-frequency (≤30 fps) and large; routing them through the
// store would thrash React. Instead EnvPreview registers a handler that draws straight to
// its canvas, and the WS dispatch calls it directly.
type FrameHandler = (frame: PreviewFrame) => void
let frameHandler: FrameHandler | null = null

export function setFrameHandler(handler: FrameHandler | null): void {
  frameHandler = handler
}

/** Seed the store from a REST status snapshot. Called on every WS (re)connect so the
 *  Run/Stop controls recover the live run after a page reload, Vite HMR, or WS reconnect
 *  that missed the one-shot 'status' frame (the backend only broadcasts status on lifecycle
 *  changes, so without this the controls desync to "idle" while a run is still active). */
function syncStoreFromStatus(status: TrainStatus): void {
  const st = useAppStore.getState()
  st.setTrainState(status.state)
  // Re-adopt the active run's identity so the sidebar + controls match what's training.
  const active = status.state === 'running' || status.state === 'paused' || status.state === 'stopping'
  if (active && status.config) {
    if (status.env_id) st.setSelectedEnvId(status.env_id)
    if (status.algo) st.setAlgo(status.algo)
    st.setSeed(status.config.seed)
    st.setTotalTimesteps(status.config.total_timesteps)
    st.setHyperparams(status.config.hyperparams)
    if (status.config.evolution) st.setEvolutionParams(status.config.evolution)
  }
}

/** React hook: opens the /ws connection and dispatches incoming frames. */
export function useTrainingWs(): void {
  useEffect(() => {
    const { stop } = createWsClient(
      (data) => {
        const frame = data as TrainWsFrame
        if (frame.type === 'metrics') {
          useAppStore.getState().addMetrics(frame)
        } else if (frame.type === 'progress') {
          useAppStore.getState().setProgress(frame)
        } else if (frame.type === 'status') {
          const prev = useAppStore.getState().trainState
          useAppStore.getState().setTrainState(frame.state)
          // Clear chart when a brand-new run starts (timesteps reset to 0)
          if (
            frame.state === 'running' &&
            frame.timesteps === 0 &&
            (prev === 'idle' || prev === 'stopped' || prev === 'finished' || prev === 'error')
          ) {
            useAppStore.getState().clearMetrics()
          }
        } else if (frame.type === 'evolution') {
          useAppStore.getState().addEvolution(frame)
        } else if (frame.type === 'highscore') {
          useAppStore.getState().setHighScore(frame)
        } else if (frame.type === 'frame') {
          frameHandler?.(frame)
        } else if (frame.type === 'preview') {
          // Keep the toggle/slider in sync (e.g. across tabs); echoes our own changes.
          useAppStore.getState().setVisual(frame.visual)
          useAppStore.getState().setSpeed(frame.speed)
        }
      },
      (connected) => {
        // On every (re)connect, reconcile with the backend's authoritative run state so the
        // controls can't get stuck after a reload/HMR that missed the live 'status' frame.
        if (connected) void fetchTrainStatus().then(syncStoreFromStatus).catch(() => {})
      },
    )
    return stop
  }, [])
}

// ── Env catalog ─────────────────────────────────────────────────────────────

export async function fetchEnvs(): Promise<EnvSpec[]> {
  const res = await fetch(`${API_BASE}/api/envs`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<EnvSpec[]>
}

export async function fetchEnv(id: string): Promise<EnvSpec> {
  const res = await fetch(`${API_BASE}/api/envs/${encodeURIComponent(id)}`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<EnvSpec>
}

// ── High scores (C2) ──────────────────────────────────────────────────────────

/** All-time best per env (persisted server-side). Live updates arrive via WS. */
export async function fetchHighScores(): Promise<HighScore[]> {
  const res = await fetch(`${API_BASE}/api/highscores`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<HighScore[]>
}

// ── Training control ────────────────────────────────────────────────────────

async function trainPost(path: string, body?: unknown): Promise<TrainStatus> {
  const res = await fetch(`${API_BASE}/api/train/${path}`, {
    method: 'POST',
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body:    body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const detail = ((await res.json().catch(() => ({}))) as { detail?: string }).detail
    throw new Error(detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<TrainStatus>
}

export const startTraining  = (config: TrainConfig) => trainPost('start', config)
export const stopTraining   = ()                     => trainPost('stop')
export const pauseTraining  = ()                     => trainPost('pause')
export const resumeTraining = ()                     => trainPost('resume')

/** Authoritative run snapshot — fetched on WS (re)connect to reconcile the controls. */
export async function fetchTrainStatus(): Promise<TrainStatus> {
  const res = await fetch(`${API_BASE}/api/train/status`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<TrainStatus>
}

// ── Checkpoints (D1) ──────────────────────────────────────────────────────────

/** Saved checkpoint slots, newest first. */
export async function fetchCheckpoints(): Promise<CheckpointMeta[]> {
  const res = await fetch(`${API_BASE}/api/checkpoints`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<CheckpointMeta[]>
}

/** Save the current run's latest model snapshot into a new slot. */
export async function saveCheckpoint(label?: string): Promise<CheckpointMeta> {
  const res = await fetch(`${API_BASE}/api/checkpoints`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label: label ?? null }),
  })
  if (!res.ok) {
    const detail = ((await res.json().catch(() => ({}))) as { detail?: string }).detail
    throw new Error(detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<CheckpointMeta>
}

/** Resume training from a saved checkpoint; returns the new run's status. */
export async function loadCheckpoint(id: string): Promise<TrainStatus> {
  const res = await fetch(`${API_BASE}/api/checkpoints/${encodeURIComponent(id)}/load`, {
    method: 'POST',
  })
  if (!res.ok) {
    const detail = ((await res.json().catch(() => ({}))) as { detail?: string }).detail
    throw new Error(detail ?? `HTTP ${res.status}`)
  }
  return res.json() as Promise<TrainStatus>
}

export async function deleteCheckpoint(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/checkpoints/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })
  if (!res.ok && res.status !== 404) throw new Error(`HTTP ${res.status}`)
}

/** Browser-navigable URL that streams the slot's zip as a download. */
export function checkpointExportUrl(id: string): string {
  return `${API_BASE}/api/checkpoints/${encodeURIComponent(id)}/export`
}

// ── Preview control (B4) ──────────────────────────────────────────────────────

export async function fetchPreview(): Promise<PreviewState> {
  const res = await fetch(`${API_BASE}/api/preview`)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<PreviewState>
}

export async function setPreview(config: PreviewConfig): Promise<PreviewState> {
  const res = await fetch(`${API_BASE}/api/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json() as Promise<PreviewState>
}
