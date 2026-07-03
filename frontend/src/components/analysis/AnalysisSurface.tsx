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
import { fetchRuns, fetchRun, fetchAggregate } from '../../api/client'
import type { AggregateResponse, RunDetail, RunMeta } from '../../api/types'
import { formatCount } from '../../format'
import { algoLabel, fmtTick } from './chartMath'
import { runToPoints, type Axis, type Metric } from './runProjection'
import AnalysisChart, { type ChartBand, type ChartSeries } from './AnalysisChart'
import SourcePicker from './SourcePicker'
import ModeSwitch from '../ModeSwitch'
import LangThemeToggle from '../LangThemeToggle'

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

  const metric: Metric = mode === 'game' ? 'reward' : 'skill_pct'
  const colorOf = useCallback((id: string) => {
    const i = selected.indexOf(id)
    return i >= 0 ? COLORS[i % COLORS.length] : null
  }, [selected])

  // Load the run history when the surface opens (and on reconnect while open).
  useEffect(() => {
    if (!open || backendStatus !== 'online') return
    void fetchRuns().then(setRuns).catch(() => {})
  }, [open, backendStatus])

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

  const envById = useCallback((id: string) => envs.find((e) => e.id === id), [envs])

  // The selected runs that have loaded their detail, in selection order.
  const loaded = useMemo(
    () => selected.map((id) => details[id]).filter((d): d is RunDetail => !!d),
    [selected, details],
  )
  const selectedEnvIds = useMemo(
    () => [...new Set(loaded.map((d) => d.config.env_id))],
    [loaded],
  )
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
      const gname = env?.display_name[locale] ?? d.config.env_id
      const label = mode === 'game'
        ? `${algoLabel(t, d.config.algo)} · s${d.config.seed}`
        : `${gname} · ${algoLabel(t, d.config.algo)}`
      out.push({ id, label, color: colorOf(id) ?? 'var(--accent)', points })
    }
    return out
  }, [showingBand, selected, hidden, details, envById, axis, metric, mode, locale, t, colorOf])

  const singleEnv = selectedEnvIds.length === 1 ? envById(selectedEnvIds[0]) : undefined
  const goal = mode === 'algo'
    ? { value: 100, label: t('chart.goal') }
    : singleEnv ? { value: singleEnv.solved_score, label: t('chart.goal') } : null
  const mixedGames = mode === 'game' && selectedEnvIds.length > 1
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
        const env = d ? envById(d.config.env_id) : undefined
        const gname = env?.display_name[locale] ?? d?.config.env_id ?? id
        const label = d
          ? (mode === 'game' ? `${algoLabel(t, d.config.algo)} · s${d.config.seed}` : `${gname} · ${algoLabel(t, d.config.algo)}`)
          : id
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
        <Segmented ariaLabel={t('analysis.mode')} value={mode} onChange={setMode} options={[
          { id: 'game', label: t('analysis.mode_game') },
          { id: 'algo', label: t('analysis.mode_algo') },
        ]} />
        <Segmented ariaLabel={t('analysis.axis')} value={axis} onChange={setAxis} options={[
          { id: 'env_steps', label: t('analysis.axis_steps') },
          { id: 'wall_clock', label: t('analysis.axis_time') },
        ]} />
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
            <span style={{ fontSize: 'var(--fs-heading)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
              {t('analysis.sources')}
            </span>
            <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-muted)' }}>
              {t('analysis.selected_count', { n: selected.length, max: MAX_SELECT })}
              {selected.length > 0 && (
                <button onClick={() => { setSelected([]); setHidden(new Set()); setExcludedSeeds(new Set()) }}
                  style={{ marginLeft: 8, background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: 'var(--fs-meta)' }}>
                  {t('runs.clear')}
                </button>
              )}
            </span>
          </div>
          <div style={{ flex: 1, minHeight: 0 }}>
            <SourcePicker runs={runs} envs={envs} locale={locale} selectedIds={new Set(selected)}
              colorFor={colorOf} onToggle={toggle} />
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
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 'var(--fs-label)', color: 'var(--text-default)', cursor: 'pointer' }}>
                <input type="checkbox" checked={collapse} onChange={(e) => setCollapse(e.target.checked)}
                  aria-label={t('analysis.collapse_seeds')} />
                {t('analysis.collapse_seeds')}
              </label>
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
            </div>
          )}

          {/* X6b marker — the surface is intentionally partial for now */}
          <div style={{ marginTop: 10, fontSize: 'var(--fs-meta)', color: 'var(--text-faint)' }}>
            {t('analysis.wave3_note')}
          </div>
        </main>
      </div>
    </div>,
    document.body,
  )
}
