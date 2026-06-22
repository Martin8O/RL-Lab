import { beforeEach, describe, expect, it } from 'vitest'
import { useAppStore } from './useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import type { EnvSpec, TrainingProgress } from '../api/types'

// A second game with a different reward scale (Breakout: image obs, PPO-only). Switching to it must
// not inherit CartPole's finished-run chart/stats/skill (the bug: 166 read "Superhuman" rescaled).
const breakoutEnv: EnvSpec = {
  ...cartpoleEnv,
  id: 'breakout',
  gym_id: 'ALE/Breakout-v5',
  obs_type: 'image',
  supported_algos: ['ppo'],
}

const sampleProgress: TrainingProgress = {
  type: 'progress',
  iteration: 5,
  timesteps: 23_000,
  total_timesteps: 50_000,
  steps_per_sec: 1949,
  ep_rew_mean: 166.6,
  ep_len_mean: 200,
  elapsed: 12,
}

describe('useAppStore — run results clear on env switch', () => {
  beforeEach(() => {
    useAppStore.setState({ envs: [cartpoleEnv, breakoutEnv], selectedEnvId: 'cartpole' })
    useAppStore.getState().clearMetrics()
  })

  it('clears the previous run chart/stats/skill when the game changes', () => {
    // Simulate a finished CartPole run leaving its chart + stats + session-best behind.
    useAppStore.getState().setProgress(sampleProgress)
    useAppStore.setState({ bestReward: 166.6 })
    expect(useAppStore.getState().progressHistory).toHaveLength(1)

    useAppStore.getState().setSelectedEnvId('breakout')

    const s = useAppStore.getState()
    expect(s.selectedEnvId).toBe('breakout')
    expect(s.progressHistory).toEqual([])
    expect(s.lastProgress).toBeNull()
    expect(s.bestReward).toBeNull()
  })

  it('keeps results on a no-op re-select of the same game', () => {
    useAppStore.getState().setProgress(sampleProgress)
    useAppStore.getState().setSelectedEnvId('cartpole')

    const s = useAppStore.getState()
    expect(s.progressHistory).toHaveLength(1)
    expect(s.lastProgress).not.toBeNull()
  })
})

describe('useAppStore — checkpoint refresh signal', () => {
  it('bumpCheckpoints increments the nonce so other pickers re-fetch', () => {
    const before = useAppStore.getState().checkpointsNonce
    useAppStore.getState().bumpCheckpoints()
    useAppStore.getState().bumpCheckpoints()
    expect(useAppStore.getState().checkpointsNonce).toBe(before + 2)
  })
})
