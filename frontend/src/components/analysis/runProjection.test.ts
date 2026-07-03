import { describe, expect, it } from 'vitest'
import type { Algo, EnvSpec, RunDetail } from '../../api/types'
import { cartpoleEnv } from '../../test/fixtures'
import { emaPoints, logClamp, niceTicks } from './chartMath'
import { runToPoints } from './runProjection'

// Build a minimal RunDetail carrying only the fields runToPoints reads (env_steps / wall_clock +
// the per-algo reward field), cast through the union — enough to exercise the projection.
function detail(algo: Algo, frames: { x: number; w: number; rew: number | null }[]): RunDetail {
  const metrics = frames.map((f) =>
    algo === 'neuroevolution'
      ? { type: 'evolution', env_steps: f.x, wall_clock: f.w, best_fitness: f.rew }
      : { type: 'metrics', env_steps: f.x, wall_clock: f.w, ep_rew_mean: f.rew },
  )
  return {
    meta: { id: 'r', algo, env_id: 'cartpole', seed: 1 },
    config: { algo, env_id: 'cartpole', seed: 1 },
    metrics,
  } as unknown as RunDetail
}

describe('runToPoints', () => {
  const env: EnvSpec = { ...cartpoleEnv, min_score: 0, solved_score: 500 }

  it('projects ep_rew_mean over env_steps as raw reward', () => {
    const d = detail('ppo', [
      { x: 0, w: 0, rew: 10 },
      { x: 1000, w: 5, rew: 250 },
      { x: 2000, w: 9, rew: 500 },
    ])
    expect(runToPoints(d, 'env_steps', 'reward', env)).toEqual([
      { x: 0, y: 10 },
      { x: 1000, y: 250 },
      { x: 2000, y: 500 },
    ])
  })

  it('normalizes to skill % over [min, solved] for the per-algorithm pivot', () => {
    const d = detail('ppo', [{ x: 1000, w: 5, rew: 250 }]) // 250 of [0,500] → 50%
    expect(runToPoints(d, 'env_steps', 'skill_pct', env)).toEqual([{ x: 1000, y: 50 }])
  })

  it('uses best_fitness for neuroevolution and the wall_clock axis when asked', () => {
    const d = detail('neuroevolution', [
      { x: 0, w: 2, rew: 100 },
      { x: 500, w: 8, rew: 400 },
    ])
    expect(runToPoints(d, 'wall_clock', 'reward', env)).toEqual([
      { x: 2, y: 100 },
      { x: 8, y: 400 },
    ])
  })

  it('drops null-reward frames and returns points sorted by x', () => {
    const d = detail('ppo', [
      { x: 2000, w: 9, rew: 300 },
      { x: 0, w: 0, rew: null },
      { x: 1000, w: 5, rew: 200 },
    ])
    expect(runToPoints(d, 'env_steps', 'reward', env)).toEqual([
      { x: 1000, y: 200 },
      { x: 2000, y: 300 },
    ])
  })
})

describe('chartMath', () => {
  it('emaPoints is a passthrough at alpha=1 and smooths below it', () => {
    const raw = [{ x: 0, y: 0 }, { x: 1, y: 10 }, { x: 2, y: 0 }]
    expect(emaPoints(raw, 1)).toBe(raw)
    const sm = emaPoints(raw, 0.5)
    expect(sm[1].y).toBeCloseTo(5) // 0.5*10 + 0.5*0
    expect(sm[2].y).toBeCloseTo(2.5) // 0.5*0 + 0.5*5
  })

  it('niceTicks yields a clean ascending grid; logClamp floors non-positive input', () => {
    const ticks = niceTicks(0, 100, 4)
    expect(ticks[0]).toBe(0)
    expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(100)
    expect(logClamp(0)).toBeLessThan(-4) // log10 of the tiny floor, not -Infinity
    expect(logClamp(100)).toBeCloseTo(2)
  })
})
