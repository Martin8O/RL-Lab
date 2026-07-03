import { describe, expect, it } from 'vitest'
import type { EnvSpec, RunMeta } from '../../api/types'
import { cartpoleEnv } from '../../test/fixtures'
import {
  DEFAULT_RUN_FILTERS,
  organizeRuns,
  runFacets,
  type RunFilters,
} from './analysisPicker'

const pendulum: EnvSpec = {
  ...cartpoleEnv,
  id: 'pendulum',
  display_name: { en: 'Pendulum', cz: 'Kyvadlo' },
  family: 'classic_control',
  min_score: -1600,
  solved_score: -200,
}
const breakout: EnvSpec = {
  ...cartpoleEnv,
  id: 'breakout',
  display_name: { en: 'Breakout', cz: 'Breakout' },
  family: 'atari',
  min_score: 0,
  solved_score: 100,
}
const ENVS: EnvSpec[] = [cartpoleEnv, pendulum, breakout]

function run(p: Partial<RunMeta>): RunMeta {
  return {
    id: p.id ?? 'x',
    label: p.label ?? 'lbl',
    env_id: p.env_id ?? 'cartpole',
    algo: p.algo ?? 'ppo',
    seed: p.seed ?? 0,
    created_at: p.created_at ?? '2026-06-01T00:00:00Z',
    finished_at: p.finished_at ?? '2026-06-01T00:10:00Z',
    state: p.state ?? 'finished',
    final_reward: p.final_reward ?? null,
    solved_at: p.solved_at ?? null,
    timesteps: p.timesteps ?? 0,
    total_timesteps: p.total_timesteps ?? 0,
    iteration: p.iteration ?? null,
    generation: p.generation ?? null,
    total_generations: p.total_generations ?? null,
    frames: p.frames ?? 0,
    experiment_id: p.experiment_id ?? null,
  }
}

// cartpole [0,500] reward 250 → 50%; breakout [0,100] reward 90 → 90%; pendulum [-1600,-200] reward
// -900 → 50%; cartpole 500 → 100%. Distinct created_at so newest/oldest is unambiguous.
const RUNS: RunMeta[] = [
  run({ id: 'a', env_id: 'cartpole', algo: 'ppo', final_reward: 250, created_at: '2026-06-01T10:00:00Z', seed: 1 }),
  run({ id: 'b', env_id: 'breakout', algo: 'dqn', final_reward: 90, created_at: '2026-06-03T10:00:00Z', seed: 2 }),
  run({ id: 'c', env_id: 'pendulum', algo: 'sac', final_reward: -900, created_at: '2026-06-02T10:00:00Z', seed: 3 }),
  run({ id: 'd', env_id: 'cartpole', algo: 'neuroevolution', final_reward: 500, created_at: '2026-06-04T10:00:00Z', seed: 4 }),
]

const flat = (f: Partial<RunFilters>) =>
  organizeRuns(RUNS, ENVS, 'en', { ...DEFAULT_RUN_FILTERS, ...f })[0].items.map((r) => r.id)

describe('runFacets', () => {
  it('lists distinct categories in roadmap order + the present algos', () => {
    const { categories, algos } = runFacets(RUNS, ENVS)
    expect(categories).toEqual(['classic_control', 'atari'])
    expect([...algos].sort()).toEqual(['dqn', 'neuroevolution', 'ppo', 'sac'])
  })
})

describe('organizeRuns — filtering', () => {
  it('defaults to newest-first across one ungrouped bucket', () => {
    const out = organizeRuns(RUNS, ENVS, 'en', DEFAULT_RUN_FILTERS)
    expect(out).toHaveLength(1)
    expect(out[0].key).toBe('')
    expect(out[0].items.map((r) => r.id)).toEqual(['d', 'b', 'c', 'a'])
  })

  it('filters by category, algorithm, and search (game name / algo / seed)', () => {
    expect(flat({ category: 'atari' })).toEqual(['b'])
    expect(flat({ algo: 'sac' })).toEqual(['c'])
    expect(flat({ search: 'pendulum' })).toEqual(['c'])
    expect(flat({ search: 'dqn' })).toEqual(['b'])
    expect(flat({ search: '4' })).toEqual(['d'])
  })

  it('returns an empty single bucket when nothing matches', () => {
    expect(organizeRuns(RUNS, ENVS, 'en', { ...DEFAULT_RUN_FILTERS, search: 'zzz' })).toEqual([
      { key: '', items: [] },
    ])
  })
})

describe('organizeRuns — sorting', () => {
  it('sorts newest / oldest by created_at', () => {
    expect(flat({ sort: 'newest' })).toEqual(['d', 'b', 'c', 'a'])
    expect(flat({ sort: 'oldest' })).toEqual(['a', 'c', 'b', 'd'])
  })

  it('sorts by best % solved (final_reward over [min, solved]); 50% ties break newest-first', () => {
    // d=100%, b=90%, then a & c both 50% → tie broken by newest (c saved after a).
    expect(flat({ sort: 'best' })).toEqual(['d', 'b', 'c', 'a'])
  })

  it('sorts by game name A–Z', () => {
    // Breakout, CartPole (d newest, then a), Pendulum.
    expect(flat({ sort: 'game' })).toEqual(['b', 'd', 'a', 'c'])
  })
})

describe('organizeRuns — grouping', () => {
  it('groups by category in roadmap order, each internally sorted', () => {
    const out = organizeRuns(RUNS, ENVS, 'en', { ...DEFAULT_RUN_FILTERS, group: 'category' })
    expect(out.map((g) => g.key)).toEqual(['classic_control', 'atari'])
    expect(out[0].items.map((r) => r.id)).toEqual(['d', 'c', 'a'])
    expect(out[1].items.map((r) => r.id)).toEqual(['b'])
  })

  it('groups by game and by algo', () => {
    const byGame = organizeRuns(RUNS, ENVS, 'en', { ...DEFAULT_RUN_FILTERS, group: 'game' })
    expect(byGame.map((g) => g.key)).toEqual(['breakout', 'cartpole', 'pendulum'])
    expect(byGame[1].items.map((r) => r.id)).toEqual(['d', 'a'])
    const byAlgo = organizeRuns(RUNS, ENVS, 'en', { ...DEFAULT_RUN_FILTERS, group: 'algo' })
    expect(byAlgo.map((g) => g.key)).toEqual(['dqn', 'neuroevolution', 'ppo', 'sac'])
  })
})
