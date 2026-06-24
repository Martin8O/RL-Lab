// Pure filter/sort/group logic for the checkpoint manager (save/load v2). The Load + Manage modals
// used to render a flat newest-first list; once many saves pile up (≥3 algos × dozens of games) that
// stops being browseable, so the toolbar in SaveLoadControls lets the user **search, filter by
// category/algorithm, sort, and group**. This module is deliberately i18n-free + side-effect-free so
// it's unit-testable: the component owns all label rendering; here we only key + order the data.

import type { Algo, CheckpointMeta, EnvSpec } from '../api/types'
import type { Locale } from '../store/useAppStore'
import { categoryOrder } from '../content/envCategories'

export type CkptSort = 'newest' | 'oldest' | 'best' | 'game'
export type CkptGroup = 'none' | 'category' | 'game' | 'algo'

export interface CkptFilters {
  search: string
  category: string // family id, or '' = all
  algo: string // Algo, or '' = all
  sort: CkptSort
  group: CkptGroup
}

export const DEFAULT_CKPT_FILTERS: CkptFilters = {
  search: '',
  category: '',
  algo: '',
  sort: 'newest',
  group: 'none',
}

/** One rendered group of cards. `key` is '' for the ungrouped (group === 'none') single bucket; for a
 *  real group it's the family id / env id / algo so the component can resolve a localized header. */
export interface CkptGroupResult {
  key: string
  items: CheckpointMeta[]
}

// % of the env's [min_score, solved_score] range the saved model reached (the same skill-% the chart
// uses), so a shaped env that starts negative still reads a meaningful fraction. null when unknown.
export function solvedPct(reward: number | null, min: number, solved: number): number | null {
  if (reward == null || solved <= min) return null
  return Math.max(0, Math.min(100, ((reward - min) / (solved - min)) * 100))
}

function envOf(slot: CheckpointMeta, envs: EnvSpec[]): EnvSpec | undefined {
  return envs.find((e) => e.id === slot.env_id)
}

function gameName(slot: CheckpointMeta, envs: EnvSpec[], locale: Locale): string {
  return envOf(slot, envs)?.display_name[locale] ?? slot.env_id
}

function bestPct(slot: CheckpointMeta, envs: EnvSpec[]): number {
  const env = envOf(slot, envs)
  // Unknown env (or no reward) sorts to the bottom of a "best" sort — never above a real score.
  if (!env) return -Infinity
  return solvedPct(slot.reward, env.min_score, env.solved_score) ?? -Infinity
}

/** The distinct categories + algorithms present across the saved slots, in display order — drives the
 *  two filter dropdowns (we only offer facets that actually match something, so no dead options). */
export function checkpointFacets(
  slots: CheckpointMeta[],
  envs: EnvSpec[],
): { categories: string[]; algos: Algo[] } {
  const families = new Set<string>()
  const algos = new Set<Algo>()
  for (const s of slots) {
    families.add(envOf(s, envs)?.family ?? s.env_id)
    algos.add(s.algo)
  }
  return {
    categories: [...families].sort((a, b) => categoryOrder(a) - categoryOrder(b)),
    algos: [...algos],
  }
}

function matchesSearch(slot: CheckpointMeta, envs: EnvSpec[], locale: Locale, q: string): boolean {
  if (!q) return true
  const hay = [
    slot.label,
    gameName(slot, envs, locale),
    slot.env_id,
    slot.algo,
    String(slot.seed),
  ]
    .join(' ')
    .toLowerCase()
  return hay.includes(q.toLowerCase())
}

function sortSlots(
  slots: CheckpointMeta[],
  envs: EnvSpec[],
  locale: Locale,
  sort: CkptSort,
): CheckpointMeta[] {
  const out = [...slots]
  switch (sort) {
    case 'newest':
      out.sort((a, b) => b.created_at.localeCompare(a.created_at))
      break
    case 'oldest':
      out.sort((a, b) => a.created_at.localeCompare(b.created_at))
      break
    case 'best':
      // Highest % solved first; ties + unknowns fall back to newest so the order is stable.
      out.sort(
        (a, b) =>
          bestPct(b, envs) - bestPct(a, envs) || b.created_at.localeCompare(a.created_at),
      )
      break
    case 'game':
      out.sort(
        (a, b) =>
          gameName(a, envs, locale).localeCompare(gameName(b, envs, locale), locale) ||
          b.created_at.localeCompare(a.created_at),
      )
      break
  }
  return out
}

function groupKey(slot: CheckpointMeta, envs: EnvSpec[], group: CkptGroup): string {
  switch (group) {
    case 'category':
      return envOf(slot, envs)?.family ?? slot.env_id
    case 'game':
      return slot.env_id
    case 'algo':
      return slot.algo
    case 'none':
      return ''
  }
}

// Order the group headers: categories in roadmap order, games + algos by label, so the headers read
// in a predictable order regardless of how the slots happened to be saved.
function groupOrder(
  keys: string[],
  envs: EnvSpec[],
  locale: Locale,
  group: CkptGroup,
): string[] {
  if (group === 'category') return [...keys].sort((a, b) => categoryOrder(a) - categoryOrder(b))
  if (group === 'game') {
    const name = (k: string) => envs.find((e) => e.id === k)?.display_name[locale] ?? k
    return [...keys].sort((a, b) => name(a).localeCompare(name(b), locale))
  }
  if (group === 'algo') return [...keys].sort((a, b) => a.localeCompare(b))
  return keys
}

/** Filter → sort → group the saved slots for the manager. Returns one bucket (`key === ''`) when the
 *  group is 'none'; otherwise one bucket per distinct group key, in header order, each internally
 *  sorted by the chosen sort. The component resolves each `key` to a localized header + count. */
export function organizeCheckpoints(
  slots: CheckpointMeta[],
  envs: EnvSpec[],
  locale: Locale,
  filters: CkptFilters,
): CkptGroupResult[] {
  const filtered = slots.filter(
    (s) =>
      matchesSearch(s, envs, locale, filters.search) &&
      (filters.category === '' || (envOf(s, envs)?.family ?? s.env_id) === filters.category) &&
      (filters.algo === '' || s.algo === filters.algo),
  )
  const sorted = sortSlots(filtered, envs, locale, filters.sort)

  if (filters.group === 'none') {
    // Always return the single bucket — even when empty — so the caller can show a "no matches" note.
    return [{ key: '', items: sorted }]
  }

  const buckets = new Map<string, CheckpointMeta[]>()
  for (const s of sorted) {
    const k = groupKey(s, envs, filters.group)
    const arr = buckets.get(k)
    if (arr) arr.push(s)
    else buckets.set(k, [s])
  }
  return groupOrder([...buckets.keys()], envs, locale, filters.group).map((key) => ({
    key,
    items: buckets.get(key) ?? [],
  }))
}
