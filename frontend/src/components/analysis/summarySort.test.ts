import { describe, expect, it } from 'vitest'
import type { RunSummary } from '../../api/types'
import { compareNullable, sortRows, type SummaryRow } from './summarySort'

function row(label: string, fields: Partial<RunSummary>): SummaryRow {
  return {
    label,
    summary: {
      run_id: label, env_id: 'cartpole', algo: 'ppo', seed: 0, n_frames: 10,
      min_score: 0, solved_score: 500,
      final_reward: null, final_skill_pct: null, solved_env_steps: null, solved_wall_clock: null,
      auc_normalized: null, late_reward_std: null, across_seed_std: null,
      final_env_steps: 0, final_wall_clock: 0, mean_steps_per_sec: null,
      peak_reward: null, peak_env_steps: null, peak_skill_pct: null, collapse_pct: null,
      ...fields,
    },
  }
}

describe('compareNullable', () => {
  it('orders real numbers by direction', () => {
    expect(compareNullable(1, 2, 'asc')).toBeLessThan(0)
    expect(compareNullable(1, 2, 'desc')).toBeGreaterThan(0)
    expect(compareNullable(5, 5, 'asc')).toBe(0)
  })

  it('sinks nulls last regardless of direction', () => {
    expect(compareNullable(null, 3, 'asc')).toBe(1)
    expect(compareNullable(null, 3, 'desc')).toBe(1)
    expect(compareNullable(3, null, 'asc')).toBe(-1)
    expect(compareNullable(3, null, 'desc')).toBe(-1)
    expect(compareNullable(null, null, 'asc')).toBe(0)
  })

  it('treats NaN as null', () => {
    expect(compareNullable(NaN, 1, 'asc')).toBe(1)
    expect(compareNullable(1, NaN, 'desc')).toBe(-1)
  })
})

describe('sortRows', () => {
  const rows = [
    row('b', { auc_normalized: 0.5, solved_env_steps: 3000 }),
    row('a', { auc_normalized: 0.9, solved_env_steps: null }), // never solved
    row('c', { auc_normalized: 0.1, solved_env_steps: 1000 }),
  ]

  it('ranks a numeric field descending', () => {
    expect(sortRows(rows, 'auc_normalized', 'desc').map((r) => r.label)).toEqual(['a', 'b', 'c'])
  })

  it('ranks a numeric field ascending', () => {
    expect(sortRows(rows, 'auc_normalized', 'asc').map((r) => r.label)).toEqual(['c', 'b', 'a'])
  })

  it('keeps never-solved runs at the bottom when ranking time-to-solve, both directions', () => {
    // asc: fastest first, the null ('a') last — not first.
    expect(sortRows(rows, 'solved_env_steps', 'asc').map((r) => r.label)).toEqual(['c', 'b', 'a'])
    // desc: slowest first, the null still last.
    expect(sortRows(rows, 'solved_env_steps', 'desc').map((r) => r.label)).toEqual(['b', 'c', 'a'])
  })

  it('sorts by label case-insensitively and numerically', () => {
    const labelled = [row('PPO · s10', {}), row('ppo · s2', {}), row('ppo · s1', {})]
    expect(sortRows(labelled, 'label', 'asc').map((r) => r.label)).toEqual(['ppo · s1', 'ppo · s2', 'PPO · s10'])
  })

  it('does not mutate the input array', () => {
    const input = rows.slice()
    sortRows(input, 'auc_normalized', 'desc')
    expect(input.map((r) => r.label)).toEqual(['b', 'a', 'c'])
  })
})
