import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { EVO_GRACE_MS, useShowEvoLeaderboard } from './useShowEvoLeaderboard'

// Locks the behaviour of the evolution-leaderboard grace window after the F2 lint refactor
// (the rule that flagged the old synchronous setState-in-effect). The window is timer-driven,
// so these use fake timers.
describe('useShowEvoLeaderboard', () => {
  beforeEach(() => { vi.useFakeTimers() })
  afterEach(() => { vi.useRealTimers() })

  it('is hidden by default (PPO, idle)', () => {
    const { result } = renderHook(() => useShowEvoLeaderboard())
    expect(result.current).toBe(false)
  })

  it('shows while a neuroevolution run is active', () => {
    const { result } = renderHook(() => useShowEvoLeaderboard())
    act(() => { useAppStore.setState({ algo: 'neuroevolution', trainState: 'running' }) })
    expect(result.current).toBe(true)
  })

  it('stays hidden when neuroevolution is selected but never run', () => {
    const { result } = renderHook(() => useShowEvoLeaderboard())
    act(() => { useAppStore.setState({ algo: 'neuroevolution', trainState: 'idle' }) })
    expect(result.current).toBe(false)
  })

  it('holds through the grace window after the run ends, then reverts', () => {
    const { result } = renderHook(() => useShowEvoLeaderboard())
    act(() => { useAppStore.setState({ algo: 'neuroevolution', trainState: 'running' }) })
    expect(result.current).toBe(true)

    act(() => { useAppStore.setState({ trainState: 'finished' }) })
    expect(result.current).toBe(true) // grace window still open

    act(() => { vi.advanceTimersByTime(EVO_GRACE_MS + 50) })
    expect(result.current).toBe(false)
  })

  it('drops immediately when switching back to PPO during the grace window', () => {
    const { result } = renderHook(() => useShowEvoLeaderboard())
    act(() => { useAppStore.setState({ algo: 'neuroevolution', trainState: 'running' }) })
    act(() => { useAppStore.setState({ trainState: 'finished' }) })
    expect(result.current).toBe(true)

    act(() => { useAppStore.setState({ algo: 'ppo' }) })
    expect(result.current).toBe(false)
  })
})
