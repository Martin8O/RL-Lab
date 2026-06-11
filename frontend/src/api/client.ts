import { useEffect } from 'react'
import { useAppStore } from '../store/useAppStore'
import type { EnvSpec } from './types'

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? ''

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
