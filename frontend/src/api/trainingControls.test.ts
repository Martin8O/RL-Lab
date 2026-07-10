import { beforeEach, describe, expect, it } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { useRunControls } from './trainingControls'
import { cartpoleEnv } from '../test/fixtures'
import type { EnvSpec } from './types'

// An Atari-like spec: image obs, GPU-trained, but usable only when the optional ale-py package is
// installed (R1/ADR-101). Reuses the CartPole fixture's param shape; only the gating fields matter.
const pongEnv: EnvSpec = {
  ...cartpoleEnv,
  id: 'pong',
  gym_id: 'ALE/Pong-v5',
  display_name: { en: 'Pong', cz: 'Pong' },
  family: 'atari',
  obs_type: 'image',
  hw_requirement: 'gpu',
  supported_algos: ['ppo', 'dqn', 'qrdqn'],
  recommended_algo: 'ppo',
}

describe('useRunControls — Atari ale-py gate (R1)', () => {
  // A GPU is present in every case here, so the only variable is whether ale-py is installed —
  // isolates the R1 gate from the pre-existing GPU gate.
  beforeEach(() => {
    useAppStore.setState({ trainState: 'idle', gpuAvailable: true })
  })

  it('gates an Atari env with reason "no_atari" when ale-py is missing', () => {
    const { result } = renderHook(() => useRunControls())
    act(() => { useAppStore.setState({ envs: [pongEnv], selectedEnvId: 'pong', atariAvailable: false }) })
    expect(result.current.trainGated).toBe(true)
    expect(result.current.trainGatedReason).toBe('no_atari')
    expect(result.current.canRun).toBe(false)
  })

  it('un-gates the same Atari env once ale-py is available', () => {
    const { result } = renderHook(() => useRunControls())
    act(() => { useAppStore.setState({ envs: [pongEnv], selectedEnvId: 'pong', atariAvailable: true }) })
    expect(result.current.trainGated).toBe(false)
    expect(result.current.trainGatedReason).toBe(null)
    expect(result.current.canRun).toBe(true)
  })

  it('leaves a non-Atari env unaffected when ale-py is missing', () => {
    const { result } = renderHook(() => useRunControls())
    act(() => { useAppStore.setState({ envs: [cartpoleEnv], selectedEnvId: 'cartpole', atariAvailable: false }) })
    expect(result.current.trainGated).toBe(false)
    expect(result.current.canRun).toBe(true)
  })
})
