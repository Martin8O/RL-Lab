// Pure filter/sort/group logic for the DataLab source picker (Zone 1) — the run-history counterpart of
// checkpointBrowser.ts. Runs pile up across dozens of games × 7 algorithms, so the picker lets the user
// search, filter by category/algorithm, sort, and group before multi-selecting runs to overlay. Kept
// i18n-free + side-effect-free so it's unit-testable: the component owns label rendering; here we only
// key + order the data. Reuses `solvedPct` (the shared skill-% clamp) + `categoryOrder` so a run's
// "best %" and category ordering read identically to the checkpoint manager.

import type { Algo, EnvSpec, RunMeta } from '../../api/types'
import type { Locale } from '../../store/useAppStore'
import { categoryOrder } from '../../content/envCategories'
import { solvedPct } from '../checkpointBrowser'

export type RunSort = 'newest' | 'oldest' | 'best' | 'game'
export type RunGroup = 'none' | 'category' | 'game' | 'algo'
// Curation view (X7): whether runs the user marked `excluded` (curated out of analysis) show. 'hide' is
// the default so an excluded run leaves the overlay/export by default; 'only' surfaces the excluded set
// for review/restore. This is what makes exclude "honoured by the Data Lab" — a hidden run can't be
// selected, so it can't reach the chart or an export.
export type RunExcludedView = 'hide' | 'show' | 'only'

export interface RunFilters {
  search: string
  category: string // family id, or '' = all
  algo: string // Algo, or '' = all
  minPct: number // hide runs whose best skill-% is below this (0 = show all, incl. 0%-skill runs)
  sort: RunSort
  group: RunGroup
  excluded: RunExcludedView
}

export const DEFAULT_RUN_FILTERS: RunFilters = {
  search: '',
  category: '',
  algo: '',
  minPct: 0,
  sort: 'newest',
  group: 'none',
  excluded: 'hide',
}

/** One rendered group of runs. `key` is '' for the ungrouped (group === 'none') single bucket; for a
 *  real group it's the family / env / algo id so the component can resolve a localized header. */
export interface RunGroupResult {
  key: string
  items: RunMeta[]
}

/** A stable experiment id derived from a human name (X7), so tagging two runs with the same name groups
 *  them under one id. Empty name → null (ungrouped). The `manual:` prefix distinguishes a hand-made group
 *  from an X3 sweep id or an `auto:` config-hash group. */
export function experimentIdFromLabel(label: string): string | null {
  const slug = label.trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return slug ? `manual:${slug}` : null
}

function envOf(run: RunMeta, envs: EnvSpec[]): EnvSpec | undefined {
  return envs.find((e) => e.id === run.env_id)
}

function gameName(run: RunMeta, envs: EnvSpec[], locale: Locale): string {
  return envOf(run, envs)?.display_name[locale] ?? run.env_id
}

/** % of the env's [min, solved] range the run's final reward reached — the same skill-% the chart +
 *  meter use, so a shaped env that starts negative still ranks meaningfully. -Infinity sorts unknowns
 *  to the bottom of a "best" sort. */
function bestPct(run: RunMeta, envs: EnvSpec[]): number {
  const env = envOf(run, envs)
  if (!env) return -Infinity
  return solvedPct(run.final_reward, env.min_score, env.solved_score) ?? -Infinity
}

/** The distinct categories + algorithms present across the runs, in display order — drives the two
 *  filter dropdowns (only facets that actually match something, so no dead options). */
export function runFacets(runs: RunMeta[], envs: EnvSpec[]): { categories: string[]; algos: Algo[] } {
  const families = new Set<string>()
  const algos = new Set<Algo>()
  for (const r of runs) {
    families.add(envOf(r, envs)?.family ?? r.env_id)
    algos.add(r.algo)
  }
  return {
    categories: [...families].sort((a, b) => categoryOrder(a) - categoryOrder(b)),
    algos: [...algos],
  }
}

function matchesSearch(run: RunMeta, envs: EnvSpec[], locale: Locale, q: string): boolean {
  if (!q) return true
  const hay = [run.label, gameName(run, envs, locale), run.env_id, run.algo, String(run.seed)]
    .join(' ')
    .toLowerCase()
  return hay.includes(q.toLowerCase())
}

function sortRuns(runs: RunMeta[], envs: EnvSpec[], locale: Locale, sort: RunSort): RunMeta[] {
  const out = [...runs]
  switch (sort) {
    case 'newest':
      out.sort((a, b) => b.created_at.localeCompare(a.created_at))
      break
    case 'oldest':
      out.sort((a, b) => a.created_at.localeCompare(b.created_at))
      break
    case 'best':
      out.sort(
        (a, b) => bestPct(b, envs) - bestPct(a, envs) || b.created_at.localeCompare(a.created_at),
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

function groupKey(run: RunMeta, envs: EnvSpec[], group: RunGroup): string {
  switch (group) {
    case 'category':
      return envOf(run, envs)?.family ?? run.env_id
    case 'game':
      return run.env_id
    case 'algo':
      return run.algo
    case 'none':
      return ''
  }
}

function groupOrder(keys: string[], envs: EnvSpec[], locale: Locale, group: RunGroup): string[] {
  if (group === 'category') return [...keys].sort((a, b) => categoryOrder(a) - categoryOrder(b))
  if (group === 'game') {
    const name = (k: string) => envs.find((e) => e.id === k)?.display_name[locale] ?? k
    return [...keys].sort((a, b) => name(a).localeCompare(name(b), locale))
  }
  if (group === 'algo') return [...keys].sort((a, b) => a.localeCompare(b))
  return keys
}

/** Filter → sort → group the run history for the picker. Returns one bucket (`key === ''`) when the
 *  group is 'none'; otherwise one bucket per distinct group key, in header order, each internally
 *  sorted. The component resolves each `key` to a localized header + count. */
export function organizeRuns(
  runs: RunMeta[],
  envs: EnvSpec[],
  locale: Locale,
  filters: RunFilters,
): RunGroupResult[] {
  const filtered = runs.filter(
    (r) =>
      // Excluded-view gate first: 'hide' drops curated-out runs, 'only' keeps just them, 'show' keeps all.
      (filters.excluded === 'show' ||
        (filters.excluded === 'only' ? r.excluded === true : r.excluded !== true)) &&
      matchesSearch(r, envs, locale, filters.search) &&
      (filters.category === '' || (envOf(r, envs)?.family ?? r.env_id) === filters.category) &&
      (filters.algo === '' || r.algo === filters.algo) &&
      // minPct 0 shows everything (incl. runs at the idle floor); a positive floor hides low-skill runs
      // by the same skill-% the meter/chart use. Unknown-env runs (-Infinity) drop once minPct > 0.
      (filters.minPct <= 0 || bestPct(r, envs) >= filters.minPct),
  )
  const sorted = sortRuns(filtered, envs, locale, filters.sort)

  if (filters.group === 'none') return [{ key: '', items: sorted }]

  const buckets = new Map<string, RunMeta[]>()
  for (const r of sorted) {
    const k = groupKey(r, envs, filters.group)
    const arr = buckets.get(k)
    if (arr) arr.push(r)
    else buckets.set(k, [r])
  }
  return groupOrder([...buckets.keys()], envs, locale, filters.group).map((key) => ({
    key,
    items: buckets.get(key) ?? [],
  }))
}
