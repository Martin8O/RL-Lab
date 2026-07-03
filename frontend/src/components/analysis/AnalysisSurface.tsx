// DataLab — the fullscreen research surface (X6a). A portal overlay peer to the dashboard: the user
// picks finished runs from history (Zone 1), overlays them as interactive curves (Zone 2), switches
// between the two pivots (per-game raw reward / per-algorithm normalized skill-%) and the two X axes
// (env_steps / wall-clock), and collapses a set of same-config seeds into a mean ± CI band (X4). The
// dashboard stays mounted underneath (a live run keeps streaming) — closing returns to it instantly.
// Zones 3–5 (rliable panel, ranking table, export) land in X6b.

import { useCallback, useEffect, useMemo, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../../store/useAppStore'
import {
  fetchRuns, fetchRun, fetchAggregate, fetchSummary, fetchRliable, groupRuns, deleteRuns,
} from '../../api/client'
import { experimentIdFromLabel } from './analysisPicker'
import type { Algo, AggregateResponse, RliableResult, RunDetail, RunMeta, RunSummary } from '../../api/types'
import { formatCount } from '../../format'
import { algoLabel, fmtTick } from './chartMath'
import { runToPoints, type Axis, type Metric } from './runProjection'
import AnalysisChart, { type ChartBand, type ChartSeries } from './AnalysisChart'
import AnalysisTable from './AnalysisTable'
import RliablePanel from './RliablePanel'
import ExportZone from './ExportZone'
import SourcePicker from './SourcePicker'
import ModeSwitch from '../ModeSwitch'
import LangThemeToggle from '../LangThemeToggle'
import ParamInfo from '../ParamInfo'

// Overlay palette (the run-compare hues + two viz colours) — assigned by selection order.
const COLORS = [
  'var(--cmp-1)', 'var(--cmp-2)', 'var(--cmp-3)', 'var(--cmp-4)',
  'var(--cmp-5)', 'var(--cmp-6)', 'var(--viz-5)', 'var(--viz-6)',
]
const MAX_SELECT = COLORS.length

function fmtElapsed(s: number): string {
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`
  return `${Math.floor(s / 3600)}h${Math.round((s % 3600) / 60)}m`
}

// Compact selection-curation button (X7) — filled when it's the active/primary action.
function curateBtn(active: boolean): React.CSSProperties {
  return {
    height: 28, padding: '0 10px', cursor: 'pointer', whiteSpace: 'nowrap',
    borderRadius: 'var(--radius-sm)', fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-medium)',
    border: `1px solid ${active ? 'var(--accent)' : 'var(--border-default)'}`,
    background: active ? 'var(--accent)' : 'transparent',
    color: active ? 'var(--text-on-accent)' : 'var(--text-muted)',
  }
}

function Segmented<T extends string>({ value, options, onChange, ariaLabel }: {
  value: T
  options: { id: T; label: string }[]
  onChange: (v: T) => void
  ariaLabel: string
}) {
  return (
    <div role="group" aria-label={ariaLabel} style={{
      display: 'inline-flex', padding: 2, gap: 2,
      background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
      borderRadius: 'var(--radius-md)',
    }}>
      {options.map((o) => {
        const on = o.id === value
        return (
          <button key={o.id} onClick={() => onChange(o.id)} aria-pressed={on} style={{
            height: 26, padding: '0 12px', cursor: 'pointer', whiteSpace: 'nowrap',
            border: 'none', borderRadius: 'var(--radius-sm)',
            background: on ? 'var(--accent)' : 'transparent',
            color: on ? 'var(--text-on-accent)' : 'var(--text-muted)',
            fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)', transition: 'var(--t-colors)',
          }}>
            {o.label}
          </button>
        )
      })}
    </div>
  )
}

export default function AnalysisSurface() {
  const { t } = useTranslation()
  const open = useAppStore((s) => s.analysisOpen)
  const setOpen = useAppStore((s) => s.setAnalysisOpen)
  const envs = useAppStore((s) => s.envs)
  const locale = useAppStore((s) => s.locale)
  const backendStatus = useAppStore((s) => s.backendStatus)
  const trainState = useAppStore((s) => s.trainState)

  const [runs, setRuns] = useState<RunMeta[]>([])
  const [selected, setSelected] = useState<string[]>([]) // ordered → stable colours
  const [details, setDetails] = useState<Record<string, RunDetail>>({})
  const [hidden, setHidden] = useState<Set<string>>(new Set()) // legend toggles

  const [mode, setMode] = useState<'game' | 'algo'>('game')
  const [axis, setAxis] = useState<Axis>('env_steps')
  const [logY, setLogY] = useState(false)
  const [emaAlpha, setEmaAlpha] = useState(0.4)
  const [collapse, setCollapse] = useState(false)
  const [excludedSeeds, setExcludedSeeds] = useState<Set<number>>(new Set())
  const [band, setBand] = useState<AggregateResponse | null>(null)
  const [bandKey, setBandKey] = useState('') // the signature `band` was fetched for (staleness guard)

  // Zone 3/4 data (X6b): the X2 summary rows (ranking table) + the rliable aggregate for the selection.
  const [summaries, setSummaries] = useState<RunSummary[]>([])
  const [rliable, setRliable] = useState<RliableResult | null>(null)
  const [statsKey, setStatsKey] = useState('') // the activeIds signature the two above were fetched for

  const metric: Metric = mode === 'game' ? 'reward' : 'skill_pct'
  const colorOf = useCallback((id: string) => {
    const i = selected.indexOf(id)
    return i >= 0 ? COLORS[i % COLORS.length] : null
  }, [selected])

  const loadRuns = useCallback(() => {
    if (backendStatus !== 'online') return
    void fetchRuns().then(setRuns).catch(() => {})
  }, [backendStatus])

  // Load the run history when the surface opens (and on reconnect while open).
  useEffect(() => {
    if (!open) return
    loadRuns()
  }, [open, loadRuns])

  // A run persists to history the instant before it broadcasts a terminal state, so refetch whenever
  // training settles (X3: each seed of a sweep lands 'finished' in turn, then the last one clears the
  // sweep) — otherwise a list fetched mid-sweep is missing every seed that finished after it opened.
  useEffect(() => {
    if (!open) return
    if (trainState === 'finished' || trainState === 'stopped') loadRuns()
  }, [open, trainState, loadRuns])

  // Fetch (and cache) the full metrics of each newly-selected run for client-side projection.
  useEffect(() => {
    for (const id of selected) {
      if (details[id]) continue
      void fetchRun(id).then((d) => setDetails((prev) => ({ ...prev, [id]: d }))).catch(() => {})
    }
  }, [selected, details])

  // Esc closes the surface.
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  const toggle = useCallback((id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : prev.length >= MAX_SELECT ? prev : [...prev, id],
    )
  }, [])

  // Clear the whole selection + its view state (legend toggles, excluded seeds) in one click.
  const clearAll = useCallback(() => {
    setSelected([])
    setHidden(new Set())
    setExcludedSeeds(new Set())
  }, [])

  // ── X7 curation of the current selection ────────────────────────────────────
  const [groupOpen, setGroupOpen] = useState(false)
  const [groupName, setGroupName] = useState('')
  const [confirmDeleteSel, setConfirmDeleteSel] = useState(false)

  // A single run was edited (label/note/tag/exclude) — refetch so the picker + zones reflect it. If it was
  // excluded, drop it from the selection so it also leaves the chart/table/export (exclude "honoured").
  const handleRunChanged = useCallback((updated: RunMeta) => {
    if (updated.excluded) setSelected((prev) => prev.filter((x) => x !== updated.id))
    loadRuns()
  }, [loadRuns])

  const handleRunDeleted = useCallback((id: string) => {
    setSelected((prev) => prev.filter((x) => x !== id))
    loadRuns()
  }, [loadRuns])

  // Tag every selected run into one named experiment (or ungroup when the name is blank).
  const applyGroup = useCallback(() => {
    const name = groupName.trim()
    void groupRuns([...selected], experimentIdFromLabel(name), name || null).then(() => {
      setGroupOpen(false)
      setGroupName('')
      loadRuns()
    })
  }, [groupName, selected, loadRuns])

  // Bulk-delete the whole selection (with confirm), then clear the now-dangling selection state.
  const deleteSelected = useCallback(() => {
    void deleteRuns([...selected]).then(() => {
      clearAll()
      setConfirmDeleteSel(false)
      loadRuns()
    })
  }, [selected, clearAll, loadRuns])

  const envById = useCallback((id: string) => envs.find((e) => e.id === id), [envs])

  // One run's display label, shared by the chart series, legend, hover readout and summary table so they
  // read identically. Per-algorithm mode always names the game (the pivot spans games); per-game mode
  // shows just "algo · seed" — UNLESS the selection mixes games (`withGame`), where the game name is
  // added so the hover/legend stays unambiguous. `withGame` is passed in (not closed over) so this
  // callback can be defined before `mixedGames` without a temporal-dead-zone reference.
  const runLabel = useCallback((envId: string, algo: Algo, seed: number, withGame: boolean): string => {
    const gname = envById(envId)?.display_name[locale] ?? envId
    if (mode === 'algo') return `${gname} · ${algoLabel(t, algo)}`
    return withGame ? `${gname} · ${algoLabel(t, algo)} · s${seed}` : `${algoLabel(t, algo)} · s${seed}`
  }, [envById, locale, mode, t])

  // The selected runs that have loaded their detail, in selection order.
  const loaded = useMemo(
    () => selected.map((id) => details[id]).filter((d): d is RunDetail => !!d),
    [selected, details],
  )
  const selectedEnvIds = useMemo(
    () => [...new Set(loaded.map((d) => d.config.env_id))],
    [loaded],
  )
  // The selection spans more than one game (only meaningful in per-game mode, where reward scales differ).
  const mixedGames = mode === 'game' && selectedEnvIds.length > 1
  // Row label for the summary table, from the summary's own env/algo/seed (needs no loaded RunDetail).
  const labelOfRun = useCallback((rid: string): string => {
    const s = summaries.find((x) => x.run_id === rid)
    return s ? runLabel(s.env_id, s.algo, s.seed, mixedGames) : rid
  }, [summaries, runLabel, mixedGames])
  // A collapsible seed set = all selected runs share one (env, algo) and there are ≥2 of them.
  const sameGroup = useMemo(() => {
    const keys = new Set(loaded.map((d) => `${d.config.env_id}·${d.config.algo}`))
    return keys.size === 1 && loaded.length >= 2
  }, [loaded])
  const canCollapse = sameGroup
  const bandActive = collapse && canCollapse

  const seedsInSelection = useMemo(
    () => [...new Set(loaded.map((d) => d.config.seed))].sort((a, b) => a - b),
    [loaded],
  )

  // The selected runs whose seed isn't excluded — the members the band aggregates.
  const includedIds = useMemo(
    () => selected.filter((id) => {
      const d = details[id]
      return !!d && !excludedSeeds.has(d.config.seed)
    }),
    [selected, details, excludedSeeds],
  )
  // The runs the analysis zones (table / rliable / export) reflect: when a seed set is collapsed into a
  // band, a dropped seed leaves the whole analysis (chart, stats, export) — otherwise every selected run
  // counts. Keeps the three zones coherent with what the chart shows.
  const activeIds = useMemo(() => (bandActive ? includedIds : selected), [bandActive, includedIds, selected])
  // Signature of the band the current controls WANT ('' when none applies — not collapsed, or fewer
  // than two seeds left). The chart only shows a fetched band whose signature still matches this, so a
  // stale band never lingers after a param change — no synchronous state-clear in the effect needed.
  const bandKeyWanted = useMemo(
    () => (bandActive && includedIds.length >= 2 ? `${axis}|${metric}|${includedIds.join(',')}` : ''),
    [bandActive, includedIds, axis, metric],
  )

  useEffect(() => {
    if (bandKeyWanted === '') return
    let cancelled = false
    void fetchAggregate({ runIds: includedIds, axis, metric, points: 120 })
      .then((r) => { if (!cancelled) { setBand(r); setBandKey(bandKeyWanted) } })
      .catch(() => {})
    return () => { cancelled = true }
  }, [bandKeyWanted, includedIds, axis, metric])

  // Fetch the summary rows + rliable aggregate for the active selection (X6b). Both are cheap server-side
  // reductions over on-disk history; the cancelled guard keeps a slow response for a stale selection from
  // clobbering the current one (axis/mode don't affect these — they read the run's own recorded curve).
  const activeKey = useMemo(() => activeIds.join(','), [activeIds])
  useEffect(() => {
    if (activeIds.length === 0) return // empty selection → the zones derive their empty state from activeKey
    let cancelled = false
    void Promise.all([fetchSummary(activeIds), fetchRliable(activeIds)])
      .then(([s, r]) => { if (!cancelled) { setSummaries(s); setRliable(r); setStatsKey(activeKey) } })
      .catch(() => {})
    return () => { cancelled = true }
  }, [activeIds, activeKey])
  // Derive the display state (no setState in the effect): the fetched data is "ready" only while its key
  // still matches the live selection — so a just-changed selection shows a loading state, not stale rows.
  const statsReady = activeIds.length > 0 && statsKey === activeKey
  const statsLoading = activeIds.length > 0 && !statsReady
  const shownSummaries = statsReady ? summaries : []
  const shownRliable = statsReady ? rliable : null

  // ── build chart inputs ───────────────────────────────────────────────────────
  const chartBand: ChartBand | null = useMemo(() => {
    if (!bandActive || bandKey !== bandKeyWanted || !band?.band) return null
    const b = band.band
    return { color: 'var(--viz-1)', x: b.x, mean: b.mean, lo: b.ci_low ?? b.lo, hi: b.ci_high ?? b.hi }
  }, [bandActive, bandKey, bandKeyWanted, band])
  // A valid band replaces the individual lines; otherwise (mid-fetch, or <2 seeds left) the lines stay.
  const showingBand = !!chartBand

  const series: ChartSeries[] = useMemo(() => {
    if (showingBand) return []
    const out: ChartSeries[] = []
    for (const id of selected) {
      if (hidden.has(id)) continue
      const d = details[id]
      if (!d) continue
      const env = envById(d.config.env_id)
      const points = runToPoints(d, axis, metric, env)
      if (points.length === 0) continue
      const label = runLabel(d.config.env_id, d.config.algo, d.config.seed, mixedGames)
      out.push({ id, label, color: colorOf(id) ?? 'var(--accent)', points })
    }
    return out
  }, [showingBand, selected, hidden, details, envById, axis, metric, runLabel, mixedGames, colorOf])

  const singleEnv = selectedEnvIds.length === 1 ? envById(selectedEnvIds[0]) : undefined
  const goal = mode === 'algo'
    ? { value: 100, label: t('chart.goal') }
    : singleEnv ? { value: singleEnv.solved_score, label: t('chart.goal') } : null
  // Log-Y only makes sense for strictly-positive data.
  const allPositive = useMemo(() => {
    if (showingBand) return (chartBand?.lo ?? []).every((v) => v > 0)
    return series.length > 0 && series.every((s) => s.points.every((p) => p.y > 0))
  }, [showingBand, chartBand, series])
  const effectiveLogY = logY && allPositive && (series.length > 0 || !!chartBand)

  const xFmt = axis === 'env_steps' ? formatCount : fmtElapsed
  const yFmt = metric === 'skill_pct' ? (v: number) => `${fmtTick(v)}%` : fmtTick
  const hoverFmt = metric === 'skill_pct'
    ? (v: number) => `${v.toFixed(1)}%`
    : (v: number) => (Math.abs(v) >= 1000 ? fmtTick(v) : v.toFixed(1))

  if (!open) return null

  const legendItems = showingBand
    ? [{ id: '_band', label: t('analysis.mean_band'), color: 'var(--viz-1)', hideable: false }]
    : selected.map((id) => {
        const d = details[id]
        const label = d ? runLabel(d.config.env_id, d.config.algo, d.config.seed, mixedGames) : id
        return { id, label, color: colorOf(id) ?? 'var(--accent)', hideable: true }
      })

  return createPortal(
    <div style={{
      position: 'fixed', inset: 0, zIndex: 'var(--z-modal)',
      background: 'var(--surface-1)', display: 'flex', flexDirection: 'column',
    }} role="dialog" aria-modal="true" aria-label={t('analysis.title')}>
      {/* ── header ── */}
      <header style={{
        height: 'var(--topbar-h)', flexShrink: 0, display: 'flex', alignItems: 'center',
        gap: 'var(--space-4)', padding: '0 var(--space-5)',
        background: 'var(--surface-1)', borderBottom: '2px solid var(--border-default)',
      }}>
        {/* View switcher: the greyed RL Lab tab is the way back to the dashboard. */}
        <ModeSwitch />
        <div style={{ width: 1, height: 26, background: 'var(--border-default)' }} />
        <span style={{
          fontSize: 'var(--fs-micro)', fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-eyebrow)',
          textTransform: 'uppercase', color: 'var(--text-faint)', whiteSpace: 'nowrap',
        }}>
          {t('analysis.subtitle')}
        </span>
        <div style={{ flex: 1 }} />
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <Segmented ariaLabel={t('analysis.mode')} value={mode} onChange={setMode} options={[
            { id: 'game', label: t('analysis.mode_game') },
            { id: 'algo', label: t('analysis.mode_algo') },
          ]} />
          <ParamInfo paramId="analysis_mode" label={t('analysis.mode')} />
        </div>
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <Segmented ariaLabel={t('analysis.axis')} value={axis} onChange={setAxis} options={[
            { id: 'env_steps', label: t('analysis.axis_steps') },
            { id: 'wall_clock', label: t('analysis.axis_time') },
          ]} />
          <ParamInfo paramId="analysis_axis" label={t('analysis.axis')} />
        </div>
        <div style={{ width: 1, height: 26, background: 'var(--border-default)' }} />
        <LangThemeToggle />
      </header>

      {/* ── body ── */}
      <div style={{ display: 'flex', flex: 1, minHeight: 0 }}>
        {/* left: source picker */}
        <aside style={{
          width: 306, flexShrink: 0, display: 'flex', flexDirection: 'column', minHeight: 0,
          padding: '12px 12px 8px', borderRight: '2px solid var(--border-default)', background: 'var(--surface-1)',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 8 }}>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 'var(--fs-heading)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
                {t('analysis.sources')}
              </span>
              <button onClick={loadRuns} aria-label={t('analysis.refresh')} title={t('analysis.refresh')}
                style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 22, height: 22, padding: 0, border: 'none', background: 'transparent', color: 'var(--text-faint)', cursor: 'pointer', borderRadius: 'var(--radius-sm)' }}
                onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent)')}
                onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-faint)')}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden role="img">
                  <path d="M20 11a8 8 0 1 0-.5 3.5M20 5v6h-6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </span>
            <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-muted)' }}>
              {t('analysis.selected_count', { n: selected.length, max: MAX_SELECT })}
              {selected.length > 0 && (
                <button onClick={clearAll}
                  style={{ marginLeft: 8, background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: 'var(--fs-meta)' }}>
                  {t('runs.clear')}
                </button>
              )}
            </span>
          </div>
          {/* X7 — curate the current selection: group into a named experiment, or bulk-delete. */}
          {selected.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 8 }}>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <button onClick={() => { setGroupOpen((o) => !o); setConfirmDeleteSel(false) }}
                  aria-expanded={groupOpen} style={curateBtn(groupOpen)}>
                  {t('analysis.group_selected')}
                </button>
                {confirmDeleteSel ? (
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--danger)' }}>
                      {t('analysis.delete_selected_confirm', { n: selected.length })}
                    </span>
                    <button onClick={deleteSelected} style={{ ...curateBtn(false), color: '#fff', background: 'var(--danger)', borderColor: 'var(--danger)' }}>
                      {t('analysis.manage_confirm_yes')}
                    </button>
                    <button onClick={() => setConfirmDeleteSel(false)} style={curateBtn(false)}>
                      {t('analysis.manage_confirm_no')}
                    </button>
                  </span>
                ) : (
                  <button onClick={() => { setConfirmDeleteSel(true); setGroupOpen(false) }}
                    style={{ ...curateBtn(false), color: 'var(--danger)' }}>
                    {t('analysis.delete_selected')}
                  </button>
                )}
              </div>
              {groupOpen && (
                <div style={{ display: 'flex', gap: 6 }}>
                  <input value={groupName} onChange={(e) => setGroupName(e.target.value)}
                    placeholder={t('analysis.group_placeholder')} aria-label={t('analysis.group_placeholder')}
                    onKeyDown={(e) => { if (e.key === 'Enter') applyGroup() }}
                    style={{ flex: 1, minWidth: 0, height: 28, padding: '0 8px', background: 'var(--surface-inset)', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-sm)', color: 'var(--text-default)', fontSize: 'var(--fs-label)' }} />
                  <button onClick={applyGroup} style={{ ...curateBtn(true) }}>{t('analysis.group_apply')}</button>
                </div>
              )}
            </div>
          )}
          <div style={{ flex: 1, minHeight: 0 }}>
            <SourcePicker runs={runs} envs={envs} locale={locale} selectedIds={new Set(selected)}
              colorFor={colorOf} onToggle={toggle}
              onRunsChanged={handleRunChanged} onRunDeleted={handleRunDeleted} />
          </div>
        </aside>

        {/* center: chart + controls */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, padding: '12px 16px' }}>
          {/* control bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap', marginBottom: 8, minHeight: 30 }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 'var(--fs-label)', color: 'var(--text-muted)' }}>
              {t('chart.ema_alpha')}
              <input type="range" min={0.02} max={1} step={0.02} value={emaAlpha}
                onChange={(e) => setEmaAlpha(Number(e.target.value))} aria-label={t('chart.ema_alpha')}
                style={{ width: 96 }} />
            </label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fs-label)', color: allPositive ? 'var(--text-muted)' : 'var(--text-faint)', cursor: allPositive ? 'pointer' : 'not-allowed' }}>
              <input type="checkbox" checked={effectiveLogY} disabled={!allPositive}
                onChange={(e) => setLogY(e.target.checked)} aria-label={t('analysis.log_y')} />
              {t('analysis.log_y')}
            </label>
            {canCollapse && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fs-label)', color: 'var(--text-default)', cursor: 'pointer' }}>
                  <input type="checkbox" checked={collapse} onChange={(e) => setCollapse(e.target.checked)}
                    aria-label={t('analysis.collapse_seeds')} />
                  {t('analysis.collapse_seeds')}
                </label>
                <ParamInfo paramId="analysis_collapse" label={t('analysis.collapse_seeds')} />
              </span>
            )}
            {/* seed include/exclude chips (band only) */}
            {bandActive && seedsInSelection.length > 0 && (
              <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-muted)' }}>{t('analysis.seeds')}</span>
                {seedsInSelection.map((sd) => {
                  const on = !excludedSeeds.has(sd)
                  return (
                    <button key={sd} onClick={() => setExcludedSeeds((prev) => {
                      const next = new Set(prev)
                      if (next.has(sd)) next.delete(sd); else next.add(sd)
                      return next
                    })} aria-pressed={on} style={{
                      height: 22, padding: '0 8px', cursor: 'pointer', borderRadius: 'var(--radius-pill)',
                      fontFamily: 'var(--font-mono)', fontSize: 'var(--fs-meta)',
                      background: on ? 'var(--accent-surface)' : 'transparent',
                      border: `1px solid ${on ? 'var(--accent)' : 'var(--border-default)'}`,
                      color: on ? 'var(--text-strong)' : 'var(--text-faint)',
                      textDecoration: on ? 'none' : 'line-through',
                    }}>{sd}</button>
                  )
                })}
              </div>
            )}
            <div style={{ flex: 1 }} />
            <span style={{ fontSize: 'var(--fs-meta)', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)' }}>
              {axis === 'env_steps' ? t('analysis.axis_steps_full') : t('analysis.axis_time_full')}
            </span>
          </div>

          {/* the chart */}
          <div style={{
            flex: 1, minHeight: 0, position: 'relative',
            border: '1px solid var(--border-default)', borderRadius: 'var(--radius-lg)',
            background: 'var(--chart-plot-bg)', padding: 4,
          }}>
            {selected.length === 0 ? (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 'var(--fs-body)', textAlign: 'center', padding: 24 }}>
                {t('analysis.empty_hint')}
              </div>
            ) : (
              <AnalysisChart series={series} band={chartBand} goal={goal} logY={effectiveLogY}
                emaAlpha={emaAlpha} xFmt={xFmt} yFmt={yFmt} hoverFmt={hoverFmt}
                ariaLabel={t('analysis.chart_aria')} />
            )}
          </div>

          {/* mixed-games honesty hint */}
          {mixedGames && (
            <div style={{ marginTop: 8, fontSize: 'var(--fs-meta)', color: 'var(--warning)' }}>
              {t('analysis.mixed_games_hint')}
            </div>
          )}

          {/* legend (per-series toggle) */}
          {legendItems.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 10 }}>
              {legendItems.map((it) => {
                const off = it.hideable && hidden.has(it.id)
                return (
                  <button key={it.id} disabled={!it.hideable}
                    onClick={() => it.hideable && setHidden((prev) => {
                      const next = new Set(prev)
                      if (next.has(it.id)) next.delete(it.id); else next.add(it.id)
                      return next
                    })}
                    aria-pressed={it.hideable ? !off : undefined}
                    style={{
                      display: 'inline-flex', alignItems: 'center', gap: 7, padding: '3px 9px',
                      background: 'var(--surface-2)', border: '1px solid var(--border-default)',
                      borderRadius: 'var(--radius-pill)', cursor: it.hideable ? 'pointer' : 'default',
                      opacity: off ? 0.4 : 1, transition: 'var(--t-base)',
                    }}>
                    <span style={{ width: 11, height: 11, borderRadius: 3, background: it.color, flexShrink: 0 }} />
                    <span style={{ fontSize: 'var(--fs-label)', color: 'var(--text-default)', whiteSpace: 'nowrap' }}>{it.label}</span>
                  </button>
                )
              })}
              {/* clear the whole selection without un-clicking each run in the picker */}
              {selected.length > 0 && (
                <button onClick={clearAll} aria-label={t('analysis.clear_all')} title={t('analysis.clear_all')}
                  style={{
                    display: 'inline-flex', alignItems: 'center', gap: 5, padding: '3px 10px',
                    background: 'transparent', border: '1px solid var(--border-default)',
                    borderRadius: 'var(--radius-pill)', cursor: 'pointer', color: 'var(--text-muted)',
                    fontSize: 'var(--fs-label)', transition: 'var(--t-colors)',
                  }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--danger)'; e.currentTarget.style.borderColor = 'var(--danger)' }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border-default)' }}>
                  ✕ {t('analysis.clear_all')}
                </button>
              )}
            </div>
          )}

          {/* Zone 4 — sortable summary table (X2 stats per selected run) */}
          {selected.length > 0 && (
            <div style={{ marginTop: 12, flexShrink: 0, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5 }}>
                <span style={{ fontSize: 'var(--fs-heading)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
                  {t('analysis.table_title')}
                </span>
                <ParamInfo paramId="analysis_table" label={t('analysis.table_title')} />
              </div>
              <div style={{ maxHeight: 232, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
                <AnalysisTable
                  summaries={shownSummaries}
                  labelOf={labelOfRun}
                  colorOf={colorOf}
                  aggregate={showingBand ? band?.summary ?? null : null}
                  seedCount={includedIds.length}
                />
              </div>
            </div>
          )}
        </main>

        {/* right: aggregate (rliable) + export */}
        <aside style={{
          width: 344, flexShrink: 0, display: 'flex', flexDirection: 'column', minHeight: 0,
          borderLeft: '2px solid var(--border-default)', background: 'var(--surface-1)',
        }}>
          {/* Zone 3 — rliable aggregate panel */}
          <div style={{ flex: 1, minHeight: 0, overflowY: 'auto', padding: '12px 12px 10px', display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 'var(--fs-heading)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
                {t('analysis.rliable_title')}
              </span>
              <ParamInfo paramId="analysis_rliable" label={t('analysis.rliable_title')} />
            </div>
            <RliablePanel result={shownRliable} loading={statsLoading} />
          </div>
          {/* Zone 5 — export */}
          <div style={{ flexShrink: 0, borderTop: '2px solid var(--border-default)', padding: '10px 12px 12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8 }}>
              <span style={{ fontSize: 'var(--fs-heading)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
                {t('analysis.export_title')}
              </span>
              <ParamInfo paramId="analysis_export" label={t('analysis.export_title')} />
            </div>
            <ExportZone runIds={activeIds} pivot={mode} />
          </div>
        </aside>
      </div>
    </div>,
    document.body,
  )
}
