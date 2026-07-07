// DataLab source picker (Zone 1) — a filterable / sortable / groupable list of finished runs the user
// multi-selects to overlay. Mirrors the checkpoint manager's browser (SaveLoadControls) for runs, so it
// reuses the same saveload.* filter/sort/group labels. Filter state is local + unpersisted, so it opens
// fresh newest-first. Pure organization lives in analysisPicker.ts; this component only renders + selects.

import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { EnvSpec, RunMeta } from '../../api/types'
import type { Locale } from '../../store/useAppStore'
import { categoryLabel } from '../../content/envCategories'
import { solvedPct } from '../checkpointBrowser'
import { formatCount } from '../../format'
import { algoLabel } from './chartMath'
import RunConfigModal from './RunConfigModal'
import LabSelect from '../LabSelect'
import {
  DEFAULT_RUN_FILTERS,
  organizeRuns,
  runFacets,
  type RunFilters,
} from './analysisPicker'

// Trigger layout for the LabSelect filters (visuals come from `.lab-trigger` in index.css).
const selectStyle: React.CSSProperties = {
  height: 30,
  borderRadius: 'var(--radius-sm)',
  color: 'var(--text-default)',
}
const searchStyle: React.CSSProperties = {
  height: 30,
  padding: '0 8px',
  background: 'var(--surface-inset)',
  border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-sm)',
  color: 'var(--text-default)',
  fontSize: 'var(--fs-label)',
  minWidth: 0,
}

function runBudget(run: RunMeta): string {
  return run.algo === 'neuroevolution' && run.generation != null
    ? `g${Math.round(run.generation)}`
    : formatCount(run.timesteps)
}

export default function SourcePicker({
  runs,
  envs,
  locale,
  selectedIds,
  colorFor,
  onToggle,
  onRunsChanged,
  onRunDeleted,
}: {
  runs: RunMeta[]
  envs: EnvSpec[]
  locale: Locale
  selectedIds: Set<string>
  colorFor: (id: string) => string | null
  onToggle: (id: string) => void
  /** A curation edit persisted (X7) — refetch the run list so the picker reflects the updated meta. */
  onRunsChanged?: (updated: RunMeta) => void
  /** A run was deleted (X7) — refetch + drop it from any selection. */
  onRunDeleted?: (id: string) => void
}) {
  const { t } = useTranslation()
  const [filters, setFilters] = useState<RunFilters>(DEFAULT_RUN_FILTERS)
  const patch = (p: Partial<RunFilters>) => setFilters((f) => ({ ...f, ...p }))
  const [infoRun, setInfoRun] = useState<RunMeta | null>(null) // the run whose config modal is open

  const facets = useMemo(() => runFacets(runs, envs), [runs, envs])
  const groups = useMemo(() => organizeRuns(runs, envs, locale, filters), [runs, envs, locale, filters])
  const total = groups.reduce((n, g) => n + g.items.length, 0)

  const gameName = (id: string) => envs.find((e) => e.id === id)?.display_name[locale] ?? id
  const groupHeader = (key: string): string => {
    if (filters.group === 'category') return categoryLabel(key)[locale]
    if (filters.group === 'game') return gameName(key)
    if (filters.group === 'algo') return algoLabel(t, key as RunMeta['algo'])
    return ''
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      {/* filter toolbar */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '0 2px 8px' }}>
        <input
          type="text"
          value={filters.search}
          onChange={(e) => patch({ search: e.target.value })}
          placeholder={t('saveload.filter_search')}
          aria-label={t('saveload.filter_search')}
          style={{ ...searchStyle, width: '100%' }}
        />
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
          <LabSelect ariaLabel={t('saveload.filter_category')} value={filters.category}
            onChange={(v) => patch({ category: v })} style={selectStyle}
            options={[
              { value: '', label: t('saveload.filter_all_categories') },
              ...facets.categories.map((c) => ({ value: c, label: categoryLabel(c)[locale] })),
            ]} />
          <LabSelect ariaLabel={t('saveload.filter_algo')} value={filters.algo}
            onChange={(v) => patch({ algo: v })} style={selectStyle}
            options={[
              { value: '', label: t('saveload.filter_all_algos') },
              ...facets.algos.map((a) => ({ value: a, label: algoLabel(t, a) })),
            ]} />
          <LabSelect ariaLabel={t('saveload.sort_label')} value={filters.sort}
            onChange={(v) => patch({ sort: v as RunFilters['sort'] })} style={selectStyle}
            options={[
              { value: 'newest', label: t('saveload.sort_newest') },
              { value: 'oldest', label: t('saveload.sort_oldest') },
              { value: 'best', label: t('saveload.sort_best') },
              { value: 'game', label: t('saveload.sort_game') },
            ]} />
          <LabSelect ariaLabel={t('saveload.group_label')} value={filters.group}
            onChange={(v) => patch({ group: v as RunFilters['group'] })} style={selectStyle}
            options={[
              { value: 'none', label: t('saveload.group_none') },
              { value: 'category', label: t('saveload.group_category') },
              { value: 'game', label: t('saveload.group_game') },
              { value: 'algo', label: t('saveload.group_algo') },
            ]} />
          <LabSelect ariaLabel={t('analysis.filter_min_skill')} value={String(filters.minPct)}
            onChange={(v) => patch({ minPct: Number(v) })} style={selectStyle}
            options={[
              { value: '0', label: t('analysis.filter_skill_all') },
              { value: '25', label: '≥ 25 %' },
              { value: '50', label: '≥ 50 %' },
              { value: '75', label: '≥ 75 %' },
              { value: '100', label: t('analysis.filter_skill_solved') },
            ]} />
          <LabSelect ariaLabel={t('analysis.filter_excluded')} value={filters.excluded}
            onChange={(v) => patch({ excluded: v as RunFilters['excluded'] })} style={selectStyle}
            options={[
              { value: 'hide', label: t('analysis.excluded_hide') },
              { value: 'show', label: t('analysis.excluded_show') },
              { value: 'only', label: t('analysis.excluded_only') },
            ]} />
        </div>
      </div>

      {/* run list */}
      <div style={{ flex: 1, overflowY: 'auto', minHeight: 0, display: 'flex', flexDirection: 'column', gap: 2 }}>
        {total === 0 && (
          <div style={{ padding: '16px 4px', color: 'var(--text-muted)', fontSize: 'var(--fs-label)', textAlign: 'center' }}>
            {runs.length === 0 ? t('analysis.no_runs') : t('saveload.no_match')}
          </div>
        )}
        {groups.map((g) => (
          <div key={g.key || '_'}>
            {g.key !== '' && (
              <div style={{
                position: 'sticky', top: 0, zIndex: 1, padding: '6px 4px 3px',
                background: 'var(--surface-1)', color: 'var(--text-muted)',
                fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)',
                letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase',
              }}>
                {groupHeader(g.key)} · {g.items.length}
              </div>
            )}
            {g.items.map((run) => {
              const env = envs.find((e) => e.id === run.env_id)
              const pct = env ? solvedPct(run.final_reward, env.min_score, env.solved_score) : null
              const selected = selectedIds.has(run.id)
              const color = colorFor(run.id)
              return (
                <div key={run.id} style={{
                  display: 'flex', alignItems: 'stretch', gap: 3,
                  background: selected ? 'var(--accent-surface)' : 'var(--surface-2)',
                  border: `1px solid ${selected ? 'var(--accent)' : 'var(--border-default)'}`,
                  borderRadius: 'var(--radius-sm)', transition: 'var(--t-colors)', overflow: 'hidden',
                }}>
                  <button
                    onClick={() => onToggle(run.id)}
                    aria-pressed={selected}
                    aria-label={`${gameName(run.env_id)} · ${algoLabel(t, run.algo)} · ${t('sidebar.seed')} ${run.seed}`}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 9, flex: 1, minWidth: 0, textAlign: 'left',
                      padding: '7px 9px', cursor: 'pointer', background: 'transparent', border: 'none',
                    }}
                  >
                    {/* selection swatch: filled with the assigned overlay colour when selected */}
                    <span style={{
                      width: 12, height: 12, flexShrink: 0, borderRadius: 3,
                      background: selected && color ? color : 'transparent',
                      border: `1.5px solid ${selected && color ? color : 'var(--border-strong)'}`,
                    }} />
                    <div style={{ flex: 1, minWidth: 0, opacity: run.excluded ? 0.55 : 1 }}>
                      <div style={{
                        fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)', color: 'var(--text-strong)',
                        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        display: 'flex', alignItems: 'center', gap: 6,
                      }}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{gameName(run.env_id)}</span>
                        {run.excluded && (
                          <span style={{
                            flexShrink: 0, fontSize: 'var(--fs-micro)', fontWeight: 'var(--fw-semibold)',
                            letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase',
                            color: 'var(--text-muted)', border: '1px solid var(--border-default)',
                            borderRadius: 'var(--radius-pill)', padding: '0 5px',
                          }}>
                            {t('analysis.excluded_badge')}
                          </span>
                        )}
                      </div>
                      <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {algoLabel(t, run.algo)} · {t('sidebar.seed')} {run.seed} · {runBudget(run)}
                      </div>
                    </div>
                    <span style={{
                      fontSize: 'var(--fs-label)', fontFamily: 'var(--font-mono)',
                      fontFeatureSettings: 'var(--ff-tabular)', color: 'var(--text-default)', flexShrink: 0,
                    }}>
                      {pct != null ? `${pct.toFixed(0)}%` : '—'}
                    </span>
                  </button>
                  {/* per-run parameters (full config) — opens the read-only detail modal */}
                  <button
                    onClick={() => setInfoRun(run)}
                    aria-label={t('analysis.run_params_open', { label: gameName(run.env_id) })}
                    title={t('analysis.run_params')}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'center', width: 28, flexShrink: 0,
                      background: 'transparent', border: 'none', borderLeft: '1px solid var(--border-default)',
                      color: 'var(--text-faint)', cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent)')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-faint)')}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden role="img">
                      <circle cx="12" cy="12" r="9.25" stroke="currentColor" strokeWidth="1.7" />
                      <circle cx="12" cy="7.75" r="1.15" fill="currentColor" />
                      <path d="M12 11.25v5.25" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" />
                    </svg>
                  </button>
                </div>
              )
            })}
          </div>
        ))}
      </div>

      {infoRun && (
        <RunConfigModal
          key={infoRun.id}
          run={infoRun}
          envName={gameName(infoRun.env_id)}
          onClose={() => setInfoRun(null)}
          onChanged={(u) => onRunsChanged?.(u)}
          onDeleted={(id) => onRunDeleted?.(id)}
        />
      )}
    </div>
  )
}
