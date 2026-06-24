import { describe, expect, it } from 'vitest'
import type { CheckpointMeta, EnvSpec } from '../api/types'
import { cartpoleEnv } from '../test/fixtures'
import {
  DEFAULT_CKPT_FILTERS,
  checkpointFacets,
  organizeCheckpoints,
  solvedPct,
  type CkptFilters,
} from './checkpointBrowser'

// Three envs across two families + different reward scales, so the category/game grouping and the
// "best %" sort have something to bite on.
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

function slot(p: Partial<CheckpointMeta>): CheckpointMeta {
  return {
    id: p.id ?? 'x',
    label: p.label ?? 'lbl',
    env_id: p.env_id ?? 'cartpole',
    algo: p.algo ?? 'ppo',
    seed: p.seed ?? 0,
    created_at: p.created_at ?? '2026-06-01T00:00:00Z',
    reward: p.reward ?? null,
    timesteps: 0,
    total_timesteps: 0,
    iteration: null,
    generation: null,
    total_generations: null,
    artifact: 'model.zip',
  }
}

// cartpole [0,500] reward 250 → 50%; breakout [0,100] reward 90 → 90%; pendulum [-1600,-200] reward
// -900 → 50%. Distinct created_at so the newest/oldest sort is unambiguous.
const SLOTS: CheckpointMeta[] = [
  slot({ id: 'a', env_id: 'cartpole', algo: 'ppo', reward: 250, created_at: '2026-06-01T10:00:00Z', seed: 1 }),
  slot({ id: 'b', env_id: 'breakout', algo: 'dqn', reward: 90, created_at: '2026-06-03T10:00:00Z', seed: 2 }),
  slot({ id: 'c', env_id: 'pendulum', algo: 'sac', reward: -900, created_at: '2026-06-02T10:00:00Z', seed: 3 }),
  slot({ id: 'd', env_id: 'cartpole', algo: 'neuroevolution', reward: 500, created_at: '2026-06-04T10:00:00Z', seed: 4 }),
]

const flat = (f: Partial<CkptFilters>) =>
  organizeCheckpoints(SLOTS, ENVS, 'en', { ...DEFAULT_CKPT_FILTERS, ...f })[0].items.map((s) => s.id)

describe('solvedPct', () => {
  it('maps reward onto the [min, solved] range and clamps to 0–100', () => {
    expect(solvedPct(250, 0, 500)).toBe(50)
    expect(solvedPct(-900, -1600, -200)).toBe(50)
    expect(solvedPct(9999, 0, 500)).toBe(100)
    expect(solvedPct(-9999, 0, 500)).toBe(0)
    expect(solvedPct(null, 0, 500)).toBeNull()
    expect(solvedPct(5, 500, 500)).toBeNull() // degenerate range
  })
})

describe('checkpointFacets', () => {
  it('lists distinct categories in roadmap order + the present algos', () => {
    const { categories, algos } = checkpointFacets(SLOTS, ENVS)
    // classic_control comes before atari in the roadmap order, regardless of save order.
    expect(categories).toEqual(['classic_control', 'atari'])
    expect([...algos].sort()).toEqual(['dqn', 'neuroevolution', 'ppo', 'sac'])
  })
})

describe('organizeCheckpoints — filtering', () => {
  it('defaults to newest-first across one ungrouped bucket', () => {
    const out = organizeCheckpoints(SLOTS, ENVS, 'en', DEFAULT_CKPT_FILTERS)
    expect(out).toHaveLength(1)
    expect(out[0].key).toBe('')
    expect(out[0].items.map((s) => s.id)).toEqual(['d', 'b', 'c', 'a'])
  })

  it('filters by category', () => {
    expect(flat({ category: 'atari' })).toEqual(['b'])
    expect(flat({ category: 'classic_control' })).toEqual(['d', 'c', 'a'])
  })

  it('filters by algorithm', () => {
    expect(flat({ algo: 'sac' })).toEqual(['c'])
  })

  it('searches across game name, algo, label and seed', () => {
    expect(flat({ search: 'pendulum' })).toEqual(['c']) // game display name
    expect(flat({ search: 'dqn' })).toEqual(['b']) // algo
    expect(flat({ search: '4' })).toEqual(['d']) // seed
  })

  it('returns an empty single bucket when nothing matches', () => {
    const out = organizeCheckpoints(SLOTS, ENVS, 'en', { ...DEFAULT_CKPT_FILTERS, search: 'zzz' })
    expect(out).toEqual([{ key: '', items: [] }])
  })
})

describe('organizeCheckpoints — sorting', () => {
  it('sorts newest / oldest by created_at', () => {
    expect(flat({ sort: 'newest' })).toEqual(['d', 'b', 'c', 'a'])
    expect(flat({ sort: 'oldest' })).toEqual(['a', 'c', 'b', 'd'])
  })

  it('sorts by best % solved (cartpole 500→100%, breakout 90%, then the two 50% ties newest-first)', () => {
    // d=100%, b=90%, then a & c both 50% → tie broken by newest (c saved after a).
    expect(flat({ sort: 'best' })).toEqual(['d', 'b', 'c', 'a'])
  })

  it('sorts by game name A–Z', () => {
    // Breakout, CartPole, CartPole, Pendulum → b, [a or d], pendulum; cartpole tie newest-first (d before a).
    expect(flat({ sort: 'game' })).toEqual(['b', 'd', 'a', 'c'])
  })
})

describe('organizeCheckpoints — grouping', () => {
  it('groups by category in roadmap order, each internally sorted', () => {
    const out = organizeCheckpoints(SLOTS, ENVS, 'en', { ...DEFAULT_CKPT_FILTERS, group: 'category' })
    expect(out.map((g) => g.key)).toEqual(['classic_control', 'atari'])
    expect(out[0].items.map((s) => s.id)).toEqual(['d', 'c', 'a']) // newest-first within
    expect(out[1].items.map((s) => s.id)).toEqual(['b'])
  })

  it('groups by game', () => {
    const out = organizeCheckpoints(SLOTS, ENVS, 'en', { ...DEFAULT_CKPT_FILTERS, group: 'game' })
    // Header order = game name A–Z: Breakout, CartPole, Pendulum.
    expect(out.map((g) => g.key)).toEqual(['breakout', 'cartpole', 'pendulum'])
    expect(out[1].items.map((s) => s.id)).toEqual(['d', 'a'])
  })

  it('groups by algorithm', () => {
    const out = organizeCheckpoints(SLOTS, ENVS, 'en', { ...DEFAULT_CKPT_FILTERS, group: 'algo' })
    expect(out.map((g) => g.key)).toEqual(['dqn', 'neuroevolution', 'ppo', 'sac'])
  })

  it('applies filters before grouping', () => {
    const out = organizeCheckpoints(SLOTS, ENVS, 'en', {
      ...DEFAULT_CKPT_FILTERS,
      group: 'game',
      category: 'classic_control',
    })
    expect(out.map((g) => g.key)).toEqual(['cartpole', 'pendulum'])
  })
})
