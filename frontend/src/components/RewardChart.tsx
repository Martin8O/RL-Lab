import { useRef, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { ChartTab } from '../store/useAppStore'
import SkillMeter from './SkillMeter'

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

// ── SVG chart ────────────────────────────────────────────────────────────────

const PAD = { t: 10, r: 12, b: 30, l: 48 }

interface ChartPoint { x: number; raw: number | null; ema: number | null }

function buildSvgPath(points: { x: number; y: number | null }[], toX: (v: number) => number, toY: (v: number) => number): string {
  let d = ''
  let pen = false
  for (const p of points) {
    if (p.y === null) { pen = false; continue }
    const px = toX(p.x).toFixed(1)
    const py = toY(p.y).toFixed(1)
    d += pen ? `L${px},${py}` : `M${px},${py}`
    pen = true
  }
  return d
}

function SvgChart({ data, width, height }: { data: ChartPoint[]; width: number; height: number }) {
  if (data.length === 0 || width < 10 || height < 10) return null

  const chartW = width  - PAD.l - PAD.r
  const chartH = height - PAD.t - PAD.b

  const allRaw = data.map((d) => d.raw).filter((v): v is number => v !== null)
  const yMin = allRaw.length ? Math.min(0, ...allRaw) : 0
  const yMax = allRaw.length ? Math.max(1, ...allRaw) : 1
  const yRange = yMax - yMin || 1

  const xMin = data[0].x
  const xMax = data[data.length - 1].x
  const xRange = xMax - xMin || 1

  const toX = (v: number) => PAD.l + ((v - xMin) / xRange) * chartW
  const toY = (v: number) => PAD.t + (1 - (v - yMin) / yRange) * chartH

  const rawPath = buildSvgPath(data.map((d) => ({ x: d.x, y: d.raw })), toX, toY)
  const emaPath = buildSvgPath(data.map((d) => ({ x: d.x, y: d.ema })), toX, toY)

  const yTicks = niceTicks(yMin, yMax, 4)
  const xTicks = niceTicks(xMin, xMax, 3)

  const latest = data[data.length - 1]

  return (
    <svg
      width={width} height={height}
      style={{ display: 'block', overflow: 'visible' }}
      aria-label="Training reward chart"
    >
      {/* Horizontal grid */}
      {yTicks.map((v) => (
        <line
          key={v}
          x1={PAD.l} y1={toY(v)} x2={PAD.l + chartW} y2={toY(v)}
          stroke="var(--border)" strokeWidth={1}
        />
      ))}

      {/* Y axis labels */}
      {yTicks.map((v) => (
        <text key={v} x={PAD.l - 5} y={toY(v) + 4}
          textAnchor="end" fontSize={9} fill="var(--text-muted)"
        >
          {fmtTick(v)}
        </text>
      ))}

      {/* X axis labels */}
      {xTicks.map((v) => (
        <text key={v} x={toX(v)} y={PAD.t + chartH + 18}
          textAnchor="middle" fontSize={9} fill="var(--text-muted)"
        >
          {fmtSteps(v)}
        </text>
      ))}

      {/* Raw data (faded) */}
      {rawPath && (
        <path d={rawPath} fill="none" stroke="var(--accent)" strokeWidth={1} opacity={0.28} />
      )}

      {/* EMA line */}
      {emaPath && (
        <path d={emaPath} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      )}

      {/* Latest-value dot */}
      {latest?.ema !== null && (
        <circle
          cx={toX(latest.x)} cy={toY(latest.ema!)}
          r={3.5} fill="var(--accent)"
        />
      )}
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

// ── RewardChart ───────────────────────────────────────────────────────────────

const TABS: ChartTab[] = ['reward', 'loss', 'fitness']

export default function RewardChart() {
  const { t } = useTranslation()

  const metricsHistory = useAppStore((s) => s.metricsHistory)
  const lastProgress   = useAppStore((s) => s.lastProgress)
  const emaAlpha       = useAppStore((s) => s.emaAlpha)
  const chartWindow    = useAppStore((s) => s.chartWindow)
  const activeTab      = useAppStore((s) => s.activeTab)
  const setEmaAlpha    = useAppStore((s) => s.setEmaAlpha)
  const setChartWindow = useAppStore((s) => s.setChartWindow)
  const setActiveTab   = useAppStore((s) => s.setActiveTab)

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

  const visible = chartWindow > 0 ? metricsHistory.slice(-chartWindow) : metricsHistory

  const rawY: (number | null)[] = visible.map((m) =>
    activeTab === 'reward' ? m.ep_rew_mean :
    activeTab === 'loss'   ? m.loss :
    null
  )
  const emaY = computeEma(rawY, emaAlpha)

  const chartData: ChartPoint[] = visible.map((m, i) => ({
    x: m.timesteps,
    raw: rawY[i],
    ema: emaY[i],
  }))

  // Live stats: prefer the ~1 Hz progress frame for timesteps/throughput/elapsed/%;
  // Score is a per-rollout figure so it comes from the latest metrics frame.
  // On a narrow chart panel, drop the control text labels (slider/select stay usable via
  // their tooltips) so the relocated Smooth/Window controls never clip out of the tab row.
  const compactControls = size.w > 0 && size.w < 380

  const lastMetrics = metricsHistory.at(-1)
  const hasStats = !!lastProgress || !!lastMetrics
  const ep      = lastProgress?.iteration ?? lastMetrics?.iteration
  const total   = lastProgress?.total_timesteps ?? lastMetrics?.total_timesteps ?? 0
  const steps   = lastProgress?.timesteps ?? lastMetrics?.timesteps
  const pct     = total > 0 && steps != null ? (steps / total) * 100 : null
  const score   = lastMetrics?.ep_rew_mean
  const sps     = lastProgress?.steps_per_sec
  const elapsed = lastProgress?.elapsed ?? lastMetrics?.elapsed

  return (
    <section style={{
      flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
      background: 'var(--bg)',
    }}>
      {/* Tab bar + chart controls (Smooth / Window moved here from the bottom) */}
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

        <div style={{ flex: 1, minWidth: 8 }} />

        {/* EMA smoothing */}
        <label
          title={t('chart.ema_alpha')}
          style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, marginRight: 8 }}
        >
          {!compactControls && <span style={{ color: 'var(--text-muted)' }}>{t('chart.ema_alpha')}</span>}
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
        {activeTab === 'fitness' || chartData.length === 0 ? (
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 16,
          }}>
            {activeTab === 'fitness' ? t('chart.fitness_stub') : t('chart.placeholder')}
          </div>
        ) : (
          <SvgChart data={chartData} width={size.w} height={size.h} />
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
              label={t('stats.episode')}
              value={ep != null ? (pct != null ? `${ep} (${Math.round(pct)}%)` : String(ep)) : '—'}
            />
            <StatChip label={t('stats.score')}        value={score != null ? score.toFixed(1) : '—'} />
            <StatChip label={t('stats.steps')}        value={steps != null ? fmtSteps(steps) : '—'} />
            <StatChip label={t('stats.steps_per_sec')} value={sps != null ? String(Math.round(sps)) : '—'} />
            <StatChip label={t('stats.elapsed')}      value={elapsed != null ? fmtElapsed(elapsed) : '—'} />
          </>
        ) : (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t('stats.no_data')}</span>
        )}
      </div>

      {/* AI skill meter (replaces the old chart controls, which moved to the tab row) */}
      <SkillMeter score={score ?? null} />
    </section>
  )
}
