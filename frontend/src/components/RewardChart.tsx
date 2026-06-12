import { useRef, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { ChartTab } from '../store/useAppStore'
import { skillScaleFor } from '../content/skill'
import SkillMeter from './SkillMeter'
import ParamInfo from './ParamInfo'

// ── EMA ─────────────────────────────────────────────────────────────────────

function computeEma(values: (number | null)[], alpha: number): (number | null)[] {
  const out: (number | null)[] = []
  let prev: number | null = null
  for (const v of values) {
    if (v === null) {
      out.push(prev)
    } else {
      prev = prev === null ? v : alpha * v + (1 - alpha) * prev
      out.push(prev)
    }
  }
  return out
}

// ── Format helpers ───────────────────────────────────────────────────────────

function fmtSteps(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000)     return `${Math.round(n / 1_000)}k`
  return String(n)
}

function fmtGen(n: number): string {
  return String(Math.round(n))
}

function fmtElapsed(s: number): string {
  if (s < 60) return `${Math.round(s)}s`
  const m = Math.floor(s / 60)
  return `${m}m ${Math.round(s % 60)}s`
}

// ── Nice tick values ─────────────────────────────────────────────────────────

function niceTicks(min: number, max: number, count = 4): number[] {
  if (min === max) return [min]
  const range = max - min
  const rough = range / count
  const mag   = Math.pow(10, Math.floor(Math.log10(rough)))
  const mult  = ([1, 2, 5, 10] as const).find((s) => s * mag >= rough) ?? 10
  const step  = mult * mag
  const start = Math.floor(min / step) * step
  const ticks: number[] = []
  for (let t = start; t <= max + step * 0.5; t = +(t + step).toFixed(12)) {
    if (t >= min - step * 0.01) ticks.push(+(t.toFixed(10)))
    if (ticks.length > count * 2 + 2) break
  }
  return ticks
}

function fmtTick(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(0)}k`
  if (v % 1 !== 0)         return v.toFixed(1)
  return String(v)
}

// ── SVG chart (multi-line) ────────────────────────────────────────────────────

const PAD = { t: 10, r: 12, b: 30, l: 48 }

interface Line {
  values: (number | null)[]  // aligned with the shared x[]
  color: string
  width: number
  opacity?: number
  dot?: boolean              // draw a dot at the latest value
}

function buildSvgPath(x: number[], values: (number | null)[], toX: (v: number) => number, toY: (v: number) => number): string {
  let d = ''
  let pen = false
  for (let i = 0; i < x.length; i++) {
    const y = values[i]
    if (y === null || y === undefined) { pen = false; continue }
    const px = toX(x[i]).toFixed(1)
    const py = toY(y).toFixed(1)
    d += pen ? `L${px},${py}` : `M${px},${py}`
    pen = true
  }
  return d
}

function lastDefined(values: (number | null)[]): number | null {
  for (let i = values.length - 1; i >= 0; i--) {
    if (values[i] !== null && values[i] !== undefined) return values[i]
  }
  return null
}

function LineChart({ x, lines, width, height, xFmt }: {
  x: number[]; lines: Line[]; width: number; height: number; xFmt: (v: number) => string
}) {
  if (x.length === 0 || width < 10 || height < 10) return null

  const chartW = width  - PAD.l - PAD.r
  const chartH = height - PAD.t - PAD.b

  const all: number[] = []
  for (const ln of lines) for (const v of ln.values) if (v !== null && v !== undefined) all.push(v)
  const yMin = all.length ? Math.min(0, ...all) : 0
  const yMax = all.length ? Math.max(1, ...all) : 1
  const yRange = yMax - yMin || 1

  const xMin = x[0]
  const xMax = x[x.length - 1]
  const xRange = xMax - xMin || 1

  const toX = (v: number) => PAD.l + ((v - xMin) / xRange) * chartW
  const toY = (v: number) => PAD.t + (1 - (v - yMin) / yRange) * chartH

  const yTicks = niceTicks(yMin, yMax, 4)
  const xTicks = niceTicks(xMin, xMax, 3)

  return (
    <svg
      width={width} height={height}
      style={{ display: 'block', overflow: 'visible' }}
      aria-label="Training chart"
    >
      {/* Horizontal grid */}
      {yTicks.map((v) => (
        <line key={v} x1={PAD.l} y1={toY(v)} x2={PAD.l + chartW} y2={toY(v)} stroke="var(--border)" strokeWidth={1} />
      ))}

      {/* Y axis labels */}
      {yTicks.map((v) => (
        <text key={v} x={PAD.l - 5} y={toY(v) + 4} textAnchor="end" fontSize={9} fill="var(--text-muted)">
          {fmtTick(v)}
        </text>
      ))}

      {/* X axis labels */}
      {xTicks.map((v) => (
        <text key={v} x={toX(v)} y={PAD.t + chartH + 18} textAnchor="middle" fontSize={9} fill="var(--text-muted)">
          {xFmt(v)}
        </text>
      ))}

      {/* Lines */}
      {lines.map((ln, i) => {
        const d = buildSvgPath(x, ln.values, toX, toY)
        if (!d) return null
        return (
          <path
            key={i}
            d={d} fill="none" stroke={ln.color} strokeWidth={ln.width}
            opacity={ln.opacity ?? 1} strokeLinejoin="round" strokeLinecap="round"
          />
        )
      })}

      {/* Latest-value dots */}
      {lines.map((ln, i) => {
        if (!ln.dot) return null
        const y = lastDefined(ln.values)
        if (y === null) return null
        return <circle key={`dot-${i}`} cx={toX(x[x.length - 1])} cy={toY(y)} r={3.5} fill={ln.color} />
      })}
    </svg>
  )
}

// ── StatChip ─────────────────────────────────────────────────────────────────

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{label}</span>
      <span style={{
        fontSize: 13, fontWeight: 600, color: 'var(--text-h)',
        fontVariantNumeric: 'tabular-nums',
      }}>{value}</span>
    </div>
  )
}

// ── Fitness legend ─────────────────────────────────────────────────────────────

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
      <span style={{ width: 8, height: 8, borderRadius: 2, background: color, display: 'inline-block' }} />
      {label}
    </span>
  )
}

// ── RewardChart ───────────────────────────────────────────────────────────────

const TABS: ChartTab[] = ['reward', 'loss', 'fitness']

export default function RewardChart() {
  const { t } = useTranslation()

  const algo            = useAppStore((s) => s.algo)
  const metricsHistory  = useAppStore((s) => s.metricsHistory)
  const progressHistory = useAppStore((s) => s.progressHistory)
  const lastProgress    = useAppStore((s) => s.lastProgress)
  const evolutionHistory = useAppStore((s) => s.evolutionHistory)
  const lastEvolution   = useAppStore((s) => s.lastEvolution)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const emaAlpha        = useAppStore((s) => s.emaAlpha)
  const chartWindow     = useAppStore((s) => s.chartWindow)
  const activeTab       = useAppStore((s) => s.activeTab)
  const setEmaAlpha     = useAppStore((s) => s.setEmaAlpha)
  const setChartWindow  = useAppStore((s) => s.setChartWindow)
  const setActiveTab    = useAppStore((s) => s.setActiveTab)

  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 400, h: 200 })

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const measure = () => setSize({ w: el.offsetWidth, h: el.offsetHeight })
    measure()
    const obs = new ResizeObserver(measure)
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  const win = <T,>(arr: T[]): T[] => (chartWindow > 0 ? arr.slice(-chartWindow) : arr)
  const accent = 'var(--accent)'

  // Build the active tab's shared x[] + lines.
  let chartX: number[] = []
  let lines: Line[] = []
  let xFmt = fmtSteps

  if (activeTab === 'reward') {
    const v = win(progressHistory)
    chartX = v.map((p) => p.timesteps)
    const raw = v.map((p) => p.ep_rew_mean)
    lines = [
      { values: raw, color: accent, width: 1, opacity: 0.28 },
      { values: computeEma(raw, emaAlpha), color: accent, width: 2, dot: true },
    ]
  } else if (activeTab === 'loss') {
    const v = win(metricsHistory)
    chartX = v.map((m) => m.timesteps)
    const raw = v.map((m) => m.loss)
    lines = [
      { values: raw, color: accent, width: 1, opacity: 0.28 },
      { values: computeEma(raw, emaAlpha), color: accent, width: 2, dot: true },
    ]
  } else {
    // Fitness: best / gen-avg / worst across generations.
    const v = win(evolutionHistory)
    chartX = v.map((e) => e.generation)
    xFmt = fmtGen
    const best  = v.map((e) => e.best_fitness)
    const avg   = v.map((e) => e.avg_fitness)
    const worst = v.map((e) => e.worst_fitness)
    lines = [
      { values: worst, color: 'var(--err)', width: 1, opacity: 0.18 },
      { values: computeEma(worst, emaAlpha), color: 'var(--err)', width: 2 },
      { values: avg, color: accent, width: 1, opacity: 0.18 },
      { values: computeEma(avg, emaAlpha), color: accent, width: 2 },
      { values: best, color: 'var(--ok)', width: 1, opacity: 0.18 },
      { values: computeEma(best, emaAlpha), color: 'var(--ok)', width: 2, dot: true },
    ]
  }

  const hasChart = chartX.length > 0
  const emptyMsg = activeTab === 'fitness' ? t('chart.fitness_stub') : t('chart.placeholder')

  // Live stats are algorithm-aware: PPO reads the ~1 Hz progress frame (with the last
  // metrics frame as a fallback); neuroevolution reads the per-generation frame.
  const isEvo = algo === 'neuroevolution'
  const lastMetrics = metricsHistory.at(-1)
  const compactControls = size.w > 0 && size.w < 380

  let hasStats: boolean
  let trainPct: number | null
  let score: number | null
  let steps: number | undefined
  let sps: number | undefined
  let elapsed: number | undefined

  if (isEvo) {
    const e = lastEvolution
    hasStats = !!e
    trainPct = e ? Math.min(100, (e.generation / e.total_generations) * 100) : null
    score = e ? e.best_fitness : null
    steps = e?.timesteps
    elapsed = e?.elapsed
    // steps/s from the two most recent generation frames (no intra-generation frames).
    if (evolutionHistory.length >= 2) {
      const a = evolutionHistory[evolutionHistory.length - 2]
      const b = evolutionHistory[evolutionHistory.length - 1]
      const dt = b.elapsed - a.elapsed
      if (dt > 0) sps = (b.timesteps - a.timesteps) / dt
    }
  } else {
    hasStats = !!lastProgress || !!lastMetrics
    const total = lastProgress?.total_timesteps ?? lastMetrics?.total_timesteps ?? 0
    steps = lastProgress?.timesteps ?? lastMetrics?.timesteps
    trainPct = total > 0 && steps != null ? Math.min(100, (steps / total) * 100) : null
    score = lastProgress?.ep_rew_mean ?? lastMetrics?.ep_rew_mean ?? null
    sps = lastProgress?.steps_per_sec
    elapsed = lastProgress?.elapsed ?? lastMetrics?.elapsed
  }

  // Solve progress: Score relative to the env's solved score (CartPole = 500). Same scale
  // for PPO reward and evolution fitness, so the skill meter + % reads consistently.
  const solveMax = skillScaleFor(selectedEnvId).max
  const solvePct =
    score != null && solveMax > 0 ? Math.max(0, Math.min(100, (score / solveMax) * 100)) : null

  return (
    <section style={{
      flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
      background: 'var(--bg)',
    }}>
      {/* Tab bar + chart controls (Smooth / Window) */}
      <div style={{
        display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--border)',
        background: 'var(--surface)', flexShrink: 0, padding: '0 10px 0 4px', minHeight: 35,
      }}>
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '7px 10px', border: 'none', cursor: 'pointer',
              background: 'transparent', fontSize: 12,
              fontWeight: activeTab === tab ? 600 : 400,
              color: activeTab === tab ? 'var(--accent-h)' : 'var(--text-muted)',
              borderBottom: `2px solid ${activeTab === tab ? 'var(--accent)' : 'transparent'}`,
            }}
          >
            {t(`chart.tab_${tab}`)}
          </button>
        ))}

        {/* Info for the currently-selected tab (Reward / Loss / Fitness) */}
        <span style={{ display: 'inline-flex', alignItems: 'center', marginLeft: 2 }}>
          <ParamInfo paramId={activeTab} label={t(`chart.tab_${activeTab}`)} />
        </span>

        <div style={{ flex: 1, minWidth: 8 }} />

        {/* EMA smoothing */}
        <label
          title={t('chart.ema_alpha')}
          style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, marginRight: 8 }}
        >
          {!compactControls && (
            <span style={{ color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 3 }}>
              {t('chart.ema_alpha')}
              <ParamInfo paramId="smooth" label={t('chart.ema_alpha')} />
            </span>
          )}
          <input
            type="range" min={0.05} max={1} step={0.05}
            value={emaAlpha}
            onChange={(e) => setEmaAlpha(parseFloat(e.target.value))}
            style={{ width: 44, cursor: 'pointer', accentColor: 'var(--accent)' }}
            aria-label={t('chart.ema_alpha')}
          />
        </label>

        {/* Window */}
        <label
          title={t('chart.window')}
          style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11 }}
        >
          {!compactControls && <span style={{ color: 'var(--text-muted)' }}>{t('chart.window')}</span>}
          <select
            value={chartWindow}
            onChange={(e) => setChartWindow(parseInt(e.target.value, 10))}
            style={{
              background: 'var(--surface-2)', color: 'var(--text)',
              border: '1px solid var(--border)', borderRadius: 4,
              fontSize: 11, padding: '2px 4px', cursor: 'pointer',
            }}
            aria-label={t('chart.window')}
          >
            <option value={0}>{t('chart.window_all')}</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </label>
      </div>

      {/* Chart area */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {hasChart ? (
          <>
            <LineChart x={chartX} lines={lines} width={size.w} height={size.h} xFmt={xFmt} />
            {activeTab === 'fitness' && (
              <div style={{
                position: 'absolute', top: 6, left: PAD.l, display: 'flex', gap: 10,
                fontSize: 10, color: 'var(--text-muted)',
              }}>
                <LegendDot color="var(--ok)"     label={t('chart.series_best')} />
                <LegendDot color="var(--accent)" label={t('chart.series_avg')} />
                <LegendDot color="var(--err)"    label={t('chart.series_worst')} />
              </div>
            )}
          </>
        ) : (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 16,
          }}>
            {emptyMsg}
          </div>
        )}
      </div>

      {/* Stats row */}
      <div style={{
        flexShrink: 0, borderTop: '1px solid var(--border)',
        background: 'var(--surface)', padding: '6px 12px',
        display: 'flex', alignItems: 'center',
        minHeight: 46,
      }}>
        {hasStats ? (
          <>
            <StatChip
              label={t('stats.progress')}
              value={trainPct != null ? `${Math.round(trainPct)}%` : '—'}
            />
            <StatChip
              label={t('stats.score')}
              value={
                score != null
                  ? solvePct != null
                    ? `${score.toFixed(1)} (${Math.round(solvePct)}%)`
                    : score.toFixed(1)
                  : '—'
              }
            />
            <StatChip label={t('stats.steps')}        value={steps != null ? fmtSteps(steps) : '—'} />
            <StatChip label={t('stats.steps_per_sec')} value={sps != null ? String(Math.round(sps)) : '—'} />
            <StatChip label={t('stats.elapsed')}      value={elapsed != null ? fmtElapsed(elapsed) : '—'} />
          </>
        ) : (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t('stats.no_data')}</span>
        )}
      </div>

      {/* AI skill meter */}
      <SkillMeter score={score ?? null} />
    </section>
  )
}
