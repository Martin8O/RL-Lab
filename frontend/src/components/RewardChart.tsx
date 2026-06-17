import { useRef, useEffect, useState, useMemo, useCallback, type CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { ChartTab } from '../store/useAppStore'
import { skillScaleFor } from '../content/skill'
import { formatCount } from '../format'
import { fetchRuns, fetchRun, deleteRun } from '../api/client'
import type { RunDetail, RunMeta } from '../api/types'
import SkillMeter from './SkillMeter'
import ParamInfo from './ParamInfo'
import HwStats from './HwStats'

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

const fmtSteps = formatCount  // shared "use M past 1000k" formatter

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
  if (Math.abs(v) >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(0)}k`
  if (v % 1 !== 0)         return v.toFixed(1)
  return String(v)
}

// ── SVG chart (multi-line) ────────────────────────────────────────────────────

const PAD = { t: 10, r: 12, b: 30, l: 48 }

interface Series {
  x: number[]                // this line's own x values (parallel to values)
  values: (number | null)[]
  color: string
  width: number
  opacity?: number
  dot?: boolean              // draw a dot at the latest value
  dash?: boolean             // dashed stroke (used for overlaid past runs)
  area?: boolean             // soft gradient fill under the line (main smoothed series)
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

function lastPoint(s: Series): { x: number; y: number } | null {
  for (let i = s.values.length - 1; i >= 0; i--) {
    const y = s.values[i]
    if (y !== null && y !== undefined) return { x: s.x[i], y }
  }
  return null
}

// A vertical "solved" marker: where a compared run first hit 100% of the goal.
interface SolvedMarker { x: number; color: string; label: string }

// The horizontal "goal" line (the env's solved score): a gold dashed line so you can see how far
// the curve still has to climb. Included in the Y-domain so it's always visible, even before the
// curve reaches it.
interface GoalLine { value: number; label: string }

// The chart's y-domain: always spans at least [0, 1], stretched to fit the data and the goal line
// (so the goal is always on-screen). Shared by LineChart and the parent's HTML "Goal" label overlay,
// so the label sits exactly on the gold line.
function chartYDomain(series: Series[], goalY?: number): { min: number; max: number } {
  const ys: number[] = []
  for (const s of series) for (const v of s.values) if (v !== null && v !== undefined) ys.push(v)
  const min = Math.min(0, ...(ys.length ? ys : [0]), ...(goalY != null ? [goalY] : []))
  const max = Math.max(1, ...(ys.length ? ys : [1]), ...(goalY != null ? [goalY] : []))
  return { min, max }
}

// Multi-series chart: each series carries its own x[], so live data and overlaid past runs
// (with different step/generation ranges) share one auto-scaled domain.
function LineChart({ series, markers = [], goal, width, height, xFmt, ariaLabel }: {
  series: Series[]; markers?: SolvedMarker[]; goal?: GoalLine; width: number; height: number; xFmt: (v: number) => string; ariaLabel: string
}) {
  const allX: number[] = []
  const allY: number[] = []
  for (const s of series) {
    for (const xv of s.x) allX.push(xv)
    for (const v of s.values) if (v !== null && v !== undefined) allY.push(v)
  }
  if (allX.length === 0 || width < 10 || height < 10) return null

  const chartW = width  - PAD.l - PAD.r
  const chartH = height - PAD.t - PAD.b

  // Fold the goal value into the domain so the gold line (and the headroom up to it) always shows.
  const goalY = goal?.value
  const { min: yMin, max: yMax } = chartYDomain(series, goalY)
  const yRange = yMax - yMin || 1

  const xMin = Math.min(...allX)
  const xMax = Math.max(...allX)
  const xRange = xMax - xMin || 1

  const toX = (v: number) => PAD.l + ((v - xMin) / xRange) * chartW
  const toY = (v: number) => PAD.t + (1 - (v - yMin) / yRange) * chartH

  const yTicks = niceTicks(yMin, yMax, 4)
  const xTicks = niceTicks(xMin, xMax, 3)

  return (
    <svg
      width={width} height={height}
      style={{ display: 'block', overflow: 'visible' }}
      role="img"
      aria-label={ariaLabel}
    >
      <defs>
        {series.map((s, i) => (s.area ? (
          <linearGradient key={i} id={`rc-area-${i}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={s.color} stopOpacity={0.22} />
            <stop offset="100%" stopColor={s.color} stopOpacity={0} />
          </linearGradient>
        ) : null))}
      </defs>

      {/* Horizontal grid */}
      {yTicks.map((v) => (
        <line key={v} x1={PAD.l} y1={toY(v)} x2={PAD.l + chartW} y2={toY(v)} stroke="var(--chart-grid)" strokeWidth={1} />
      ))}

      {/* Y axis labels */}
      {yTicks.map((v) => (
        <text key={v} x={PAD.l - 6} y={toY(v) + 4} textAnchor="end" fontSize={10}
          fontFamily="var(--font-mono)" fill="var(--chart-axis)">
          {fmtTick(v)}
        </text>
      ))}

      {/* X axis labels */}
      {xTicks.map((v) => (
        <text key={v} x={toX(v)} y={PAD.t + chartH + 18} textAnchor="middle" fontSize={10}
          fontFamily="var(--font-mono)" fill="var(--chart-axis)">
          {xFmt(v)}
        </text>
      ))}

      {/* Goal line: the env's "solved" score as a gold dashed rule, so you can see how far the curve
          still has to climb. Drawn above the grid, below the data lines. Its label is an HTML overlay
          (positioned by the parent) so it sits on the Y axis and is never clipped. */}
      {goal && goalY != null && (
        <line x1={PAD.l} y1={toY(goalY)} x2={PAD.l + chartW} y2={toY(goalY)}
          stroke="var(--goal)" strokeWidth={1.5} strokeDasharray="6 4" opacity={0.9} />
      )}

      {/* Area fill under the main smoothed line (behind the strokes) */}
      {series.map((s, i) => {
        if (!s.area) return null
        const pts: [number, number][] = []
        for (let j = 0; j < s.x.length; j++) {
          const y = s.values[j]
          if (y !== null && y !== undefined) pts.push([toX(s.x[j]), toY(y)])
        }
        if (pts.length < 2) return null
        const baseY = PAD.t + chartH
        let d = `M${pts[0][0].toFixed(1)},${baseY.toFixed(1)}`
        for (const [px, py] of pts) d += `L${px.toFixed(1)},${py.toFixed(1)}`
        d += `L${pts[pts.length - 1][0].toFixed(1)},${baseY.toFixed(1)}Z`
        return <path key={`area-${i}`} d={d} fill={`url(#rc-area-${i})`} />
      })}

      {/* Lines */}
      {series.map((s, i) => {
        const d = buildSvgPath(s.x, s.values, toX, toY)
        if (!d) return null
        return (
          <path
            key={i}
            d={d} fill="none" stroke={s.color} strokeWidth={s.width}
            opacity={s.opacity ?? 1} strokeLinejoin="round" strokeLinecap="round"
            strokeDasharray={s.dash ? '5 4' : undefined}
          />
        )
      })}

      {/* Latest-value dots */}
      {series.map((s, i) => {
        if (!s.dot) return null
        const p = lastPoint(s)
        if (p === null) return null
        return <circle key={`dot-${i}`} cx={toX(p.x)} cy={toY(p.y)} r={3.5} fill={s.color} />
      })}

      {/* Solved markers: a coloured vertical line + the x-value labelled on the axis, so a
          compared run's "steps/generations to solve" reads straight off the chart. */}
      {markers.map((m, i) => {
        if (m.x < xMin || m.x > xMax) return null
        const mx = toX(m.x)
        return (
          <g key={`mk-${i}`}>
            <line x1={mx} y1={PAD.t} x2={mx} y2={PAD.t + chartH} stroke={m.color} strokeWidth={1.5} strokeDasharray="2 3" opacity={0.65} />
            <text x={mx} y={PAD.t + chartH + 28} textAnchor="middle" fontSize={9} fontWeight={700} fill={m.color}>
              ✓ {m.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// Categorical palette for overlaid past runs — the viz comparison hues (amber / violet /
// cyan), which recolour per theme via CSS vars so overlays read on dark + light.
const OVERLAY_COLORS = ['var(--viz-4)', 'var(--viz-5)', 'var(--viz-6)']

// Project a saved run's recorded frames onto the active tab's (x, y), or null if the run has
// no data for that tab. The Reward tab's x-unit depends on the *live* algorithm (timesteps for
// PPO, episodes for Q-learning), so a run is only overlaid there when its frames match that unit
// — keeping the shared x-axis coherent (an evolution run lives on the Fitness tab instead).
function runSeries(run: RunDetail, tab: ChartTab, color: string, liveAlgo: string): Series | null {
  const x: number[] = []
  const values: (number | null)[] = []
  for (const f of run.metrics) {
    if (tab === 'fitness') {
      if (f.type !== 'evolution') continue
      x.push(f.generation); values.push(f.best_fitness)
    } else if (tab === 'loss') {
      if (f.type !== 'metrics') continue
      x.push(f.timesteps); values.push(f.loss)
    } else if (liveAlgo === 'q_learning') {  // reward tab, Q-learning context → x = episode
      if (f.type !== 'q_learning') continue
      x.push(f.episode); values.push(f.ep_rew_mean)
    } else {  // reward tab, PPO context → x = timesteps
      if (f.type !== 'metrics') continue
      x.push(f.timesteps); values.push(f.ep_rew_mean)
    }
  }
  if (x.length === 0) return null
  return { x, values, color, width: 1.5, opacity: 0.9, dash: true }
}

// ── StatChip ─────────────────────────────────────────────────────────────────

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, minWidth: 0 }}>
      <span style={{
        fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-eyebrow)',
        textTransform: 'uppercase', color: 'var(--text-muted)', whiteSpace: 'nowrap',
      }}>{label}</span>
      <span style={{
        fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
        fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-medium)', letterSpacing: 'var(--ls-tight)',
        color: 'var(--text-strong)', whiteSpace: 'nowrap', maxWidth: '100%',
        overflow: 'hidden', textOverflow: 'ellipsis',
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

// ── Run-history compare (D2) ────────────────────────────────────────────────────

// A run row in the compare popover: toggles its overlay, shows its final reward, deletes it.
// The swatch colour matches the overlay line so the chart and list read together.
function RunRow({ run, color, selected, atCap, onToggle, onDelete }: {
  run: RunMeta
  color: string | null
  selected: boolean
  atCap: boolean
  onToggle: (id: string) => void
  onDelete: (run: RunMeta) => void
}) {
  const { t } = useTranslation()
  const disabled = !selected && atCap
  const ellipsis: CSSProperties = { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
      <button
        onClick={() => onToggle(run.id)}
        disabled={disabled}
        title={disabled ? t('runs.max_hint') : run.label}
        style={{
          flex: 1, minWidth: 0, display: 'flex', alignItems: 'center', gap: 6,
          padding: '4px 6px', borderRadius: 4, border: 'none', textAlign: 'left',
          background: selected ? 'var(--accent-soft)' : 'transparent',
          cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.45 : 1,
        }}
      >
        <span style={{
          width: 10, height: 10, flexShrink: 0, borderRadius: 2,
          background: selected && color ? color : 'transparent',
          border: `1px solid ${selected && color ? color : 'var(--border)'}`,
        }} />
        <span style={{ ...ellipsis, flex: 1, minWidth: 0, fontSize: 11, color: 'var(--text-h)' }}>
          {run.label}
        </span>
        {run.solved_at != null && (
          <span
            title={t('runs.solved')}
            style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--accent-h)', flexShrink: 0 }}
          >
            ✓ {run.algo === 'neuroevolution' ? `g${fmtGen(run.solved_at)}` : fmtSteps(run.solved_at)}
          </span>
        )}
        <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--ok)', flexShrink: 0 }}>
          {run.final_reward != null ? run.final_reward.toFixed(1) : '—'}
        </span>
      </button>
      <button
        onClick={() => onDelete(run)}
        title={t('runs.delete')}
        aria-label={t('runs.delete')}
        style={{
          flexShrink: 0, width: 20, height: 20, lineHeight: '18px', textAlign: 'center',
          background: 'transparent', border: 'none', cursor: 'pointer',
          color: 'var(--text-muted)', fontSize: 12, borderRadius: 4,
        }}
      >✕</button>
    </div>
  )
}

function ComparePopover({ runs, selectedOrder, onToggle, onDelete, onClear, onClose }: {
  runs: RunMeta[]
  selectedOrder: string[]   // selected ids in overlay order (drives swatch colour)
  onToggle: (id: string) => void
  onDelete: (run: RunMeta) => void
  onClear: () => void
  onClose: () => void
}) {
  const { t } = useTranslation()
  const atCap = selectedOrder.length >= OVERLAY_COLORS.length
  // Esc closes the popover (keyboard parity with the click-away backdrop).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])
  return (
    <>
      {/* Click-away backdrop */}
      <div onClick={onClose} style={{ position: 'fixed', inset: 0, zIndex: 30 }} />
      <div role="dialog" aria-label={t('runs.title')} style={{
        position: 'absolute', top: 'calc(100% + 4px)', right: 0, zIndex: 31,
        width: 290, maxHeight: 280, display: 'flex', flexDirection: 'column',
        background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 6,
        boxShadow: 'var(--shadow-popover)',
      }}>
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '6px 10px', borderBottom: '1px solid var(--border)',
          fontSize: 12, fontWeight: 600, color: 'var(--text-h)',
        }}>
          <span>{t('runs.title')}</span>
          {selectedOrder.length > 0 && (
            <button onClick={onClear} style={{
              background: 'transparent', border: 'none', cursor: 'pointer',
              color: 'var(--text-muted)', fontSize: 11,
            }}>{t('runs.clear')}</button>
          )}
        </div>
        {runs.length === 0 ? (
          <div style={{ padding: 16, fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
            {t('runs.empty')}
          </div>
        ) : (
          <>
            <div style={{ padding: '5px 10px 2px', fontSize: 10, color: 'var(--text-muted)' }}>
              {t('runs.max_hint')}
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '2px 6px 6px' }}>
              {runs.map((run) => {
                const idx = selectedOrder.indexOf(run.id)
                return (
                  <RunRow
                    key={run.id}
                    run={run}
                    selected={idx >= 0}
                    color={idx >= 0 ? OVERLAY_COLORS[idx % OVERLAY_COLORS.length] : null}
                    atCap={atCap}
                    onToggle={onToggle}
                    onDelete={onDelete}
                  />
                )
              })}
            </div>
          </>
        )}
      </div>
    </>
  )
}

// ── RewardChart ───────────────────────────────────────────────────────────────

const TABS: ChartTab[] = ['reward', 'loss', 'fitness']

// Which chart tabs each algorithm actually produces — the single source for the empty-state
// message, so every tab tells the truth per algorithm: if the active tab IS the algo's output, the
// hint is "start training" (data is coming); otherwise it's "this algorithm doesn't use this chart
// — see <its tabs>". Adding a new algorithm = add one row here and the messaging stays correct
// everywhere (bake-in for the S5 algorithm-breadth roadmap — DQN, SAC, …).
const ALGO_CHART_TABS: Record<string, ChartTab[]> = {
  ppo: ['reward', 'loss'],
  neuroevolution: ['fitness'],
  q_learning: ['reward'],
}

function algoLabel(t: (k: string) => string, algo: string): string {
  switch (algo) {
    case 'ppo': return t('sidebar.algo_ppo')
    case 'neuroevolution': return t('sidebar.algo_evo')
    case 'q_learning': return t('sidebar.algo_q')
    default: return algo
  }
}

export default function RewardChart() {
  const { t } = useTranslation()

  const algo            = useAppStore((s) => s.algo)
  const metricsHistory  = useAppStore((s) => s.metricsHistory)
  const progressHistory = useAppStore((s) => s.progressHistory)
  const lastProgress    = useAppStore((s) => s.lastProgress)
  const evolutionHistory = useAppStore((s) => s.evolutionHistory)
  const lastEvolution   = useAppStore((s) => s.lastEvolution)
  const qLearningHistory = useAppStore((s) => s.qLearningHistory)
  const lastQLearning   = useAppStore((s) => s.lastQLearning)
  const maHistory       = useAppStore((s) => s.maHistory)
  const lastMa          = useAppStore((s) => s.lastMa)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const envs            = useAppStore((s) => s.envs)
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

  // ── Run-history overlay (D2) ───────────────────────────────────────────────
  const trainState = useAppStore((s) => s.trainState)
  const [runs, setRuns] = useState<RunMeta[]>([])
  const [selected, setSelected] = useState<RunDetail[]>([])  // insertion order = overlay order
  const [showCompare, setShowCompare] = useState(false)

  const refreshRuns = useCallback(() => {
    void fetchRuns().then(setRuns).catch(() => {})
  }, [])
  useEffect(() => { refreshRuns() }, [refreshRuns])
  // A finishing/stopped run is archived server-side; pull the freshened list in.
  useEffect(() => {
    if (trainState === 'finished' || trainState === 'stopped') refreshRuns()
  }, [trainState, refreshRuns])

  // Only compare runs from the *same* env — a CartPole run and a LunarLander run live on
  // different reward scales, so mixing them on one chart is meaningless. We filter both the
  // pickable list and the active overlays to the current env (selections from another game stay
  // remembered and simply don't render until you switch back to it).
  const envRuns = useMemo(() => runs.filter((r) => r.env_id === selectedEnvId), [runs, selectedEnvId])
  const envSelected = useMemo(
    () => selected.filter((r) => r.meta.env_id === selectedEnvId), [selected, selectedEnvId],
  )
  const selectedOrder = useMemo(() => envSelected.map((r) => r.meta.id), [envSelected])

  const toggleRun = useCallback(async (id: string) => {
    const exists = selected.some((r) => r.meta.id === id)
    if (exists) {
      setSelected((sel) => sel.filter((r) => r.meta.id !== id))
      return
    }
    // Cap overlays per env at the palette size (selections from other envs don't count).
    if (envSelected.length >= OVERLAY_COLORS.length) return
    try {
      const detail = await fetchRun(id)
      setSelected((sel) =>
        sel.filter((r) => r.meta.env_id === detail.meta.env_id).length >= OVERLAY_COLORS.length
          ? sel
          : [...sel, detail],
      )
    } catch { /* ignore fetch failure */ }
  }, [selected, envSelected])

  const removeRun = useCallback(async (run: RunMeta) => {
    if (!window.confirm(t('runs.confirm_delete', { label: run.label }))) return
    try {
      await deleteRun(run.id)
      setSelected((sel) => sel.filter((r) => r.meta.id !== run.id))
      refreshRuns()
    } catch { /* ignore */ }
  }, [t, refreshRuns])

  const win = <T,>(arr: T[]): T[] => (chartWindow > 0 ? arr.slice(-chartWindow) : arr)
  const accent = 'var(--accent)'

  // Competitive multi-agent self-play (simple_tag, G7b-2): the reward tab draws TWO species lines
  // (predators vs. prey) from the ecosystem frames instead of one. Colours match the swarm canvas
  // (predators --danger red, prey --accent blue).
  const chartEnv = envs.find((e) => e.id === selectedEnvId)
  const isMa = algo === 'ppo' && chartEnv?.family === 'petting_zoo' && chartEnv?.competitive === true
  const PRED_COLOR = 'var(--danger)'
  const speciesVal = (m: typeof maHistory[number], role: string): number | null =>
    m.species.find((s) => s.role === role)?.ep_rew_mean ?? null

  // Build the active tab's live series (each shares this tab's x[]). Assigned in every branch
  // of the exhaustive if/else below, so no initializer is needed.
  let liveSeries: Series[]
  let xFmt = fmtSteps

  if (activeTab === 'reward' && isMa) {
    // Two co-evolving species: predator (headline) + prey, both x = cumulative env steps.
    const v = win(maHistory)
    const lx = v.map((m) => m.timesteps)
    const pred = v.map((m) => speciesVal(m, 'adversary'))
    const prey = v.map((m) => speciesVal(m, 'agent'))
    liveSeries = [
      { x: lx, values: pred, color: PRED_COLOR, width: 1, opacity: 0.25 },
      { x: lx, values: computeEma(pred, emaAlpha), color: PRED_COLOR, width: 2, dot: true },
      { x: lx, values: prey, color: accent, width: 1, opacity: 0.25 },
      { x: lx, values: computeEma(prey, emaAlpha), color: accent, width: 2, dot: true },
    ]
  } else if (activeTab === 'reward' && algo === 'q_learning') {
    // Q-learning's learning curve: mean episode return per report, x = episode (its own unit).
    const v = win(qLearningHistory)
    const lx = v.map((q) => q.episode)
    const raw = v.map((q) => q.ep_rew_mean)
    liveSeries = [
      { x: lx, values: raw, color: accent, width: 1, opacity: 0.28 },
      { x: lx, values: computeEma(raw, emaAlpha), color: accent, width: 2, dot: true, area: true },
    ]
  } else if (activeTab === 'reward') {
    const v = win(progressHistory)
    const lx = v.map((p) => p.timesteps)
    const raw = v.map((p) => p.ep_rew_mean)
    liveSeries = [
      { x: lx, values: raw, color: accent, width: 1, opacity: 0.28 },
      { x: lx, values: computeEma(raw, emaAlpha), color: accent, width: 2, dot: true, area: true },
    ]
  } else if (activeTab === 'loss') {
    const v = win(metricsHistory)
    const lx = v.map((m) => m.timesteps)
    const raw = v.map((m) => m.loss)
    liveSeries = [
      { x: lx, values: raw, color: accent, width: 1, opacity: 0.28 },
      { x: lx, values: computeEma(raw, emaAlpha), color: accent, width: 2, dot: true, area: true },
    ]
  } else {
    // Fitness: best / gen-avg / worst across generations.
    const v = win(evolutionHistory)
    const lx = v.map((e) => e.generation)
    xFmt = fmtGen
    const best  = v.map((e) => e.best_fitness)
    const avg   = v.map((e) => e.avg_fitness)
    const worst = v.map((e) => e.worst_fitness)
    liveSeries = [
      { x: lx, values: worst, color: 'var(--viz-3)', width: 1, opacity: 0.18 },
      { x: lx, values: computeEma(worst, emaAlpha), color: 'var(--viz-3)', width: 2 },
      { x: lx, values: avg, color: 'var(--viz-1)', width: 1, opacity: 0.18 },
      { x: lx, values: computeEma(avg, emaAlpha), color: 'var(--viz-1)', width: 2, area: true },
      { x: lx, values: best, color: 'var(--viz-2)', width: 1, opacity: 0.18 },
      { x: lx, values: computeEma(best, emaAlpha), color: 'var(--viz-2)', width: 2, dot: true },
    ]
  }

  // Overlaid past runs: dashed, behind the live lines. A run with no data for this tab
  // (e.g. an evolution run on the Reward tab) is skipped but stays selected for other tabs.
  const overlays: { id: string; label: string; color: string }[] = []
  const overlaySeries: Series[] = []
  const markers: SolvedMarker[] = []
  envSelected.forEach((run, i) => {
    const color = OVERLAY_COLORS[i % OVERLAY_COLORS.length]
    const s = runSeries(run, activeTab, color, algo)
    if (s) {
      overlaySeries.push(s)
      overlays.push({ id: run.meta.id, label: run.meta.label, color })
      // solved_at is in this tab's x-unit (timesteps for PPO, generation for evolution),
      // because a run only overlays on the tab matching its algorithm.
      if (run.meta.solved_at != null) {
        markers.push({ x: run.meta.solved_at, color, label: xFmt(run.meta.solved_at) })
      }
    }
  })

  const series = [...overlaySeries, ...liveSeries]
  const hasChart = series.some((s) => s.x.length > 0)
  // Algorithm-aware empty state: if this tab is one the current algorithm produces, the data is
  // just not in yet → "start training"; otherwise the tab doesn't apply to this algorithm → say so
  // and point to the tab(s) it does use. Driven by ALGO_CHART_TABS so new algorithms stay correct.
  const applicableTabs = ALGO_CHART_TABS[algo] ?? ['reward']
  const emptyMsg = applicableTabs.includes(activeTab)
    ? t('chart.placeholder')
    : t('chart.na_for_algo', {
        algo: algoLabel(t, algo),
        tab: t(`chart.tab_${activeTab}`),
        tabs: applicableTabs.map((tt) => t(`chart.tab_${tt}`)).join(' / '),
      })

  // Live stats are algorithm-aware: PPO reads the ~1 Hz progress frame (with the last
  // metrics frame as a fallback); neuroevolution reads the per-generation frame; Q-learning
  // reads the per-report frame (episode-based).
  const isEvo = algo === 'neuroevolution'
  const isQ   = algo === 'q_learning'
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
  } else if (isQ) {
    const q = lastQLearning
    hasStats = !!q
    trainPct = q ? Math.min(100, (q.episode / q.total_episodes) * 100) : null
    score = q ? q.ep_rew_mean : null
    steps = q?.timesteps
    elapsed = q?.elapsed
    // steps/s from the two most recent report frames (no intra-report frames).
    if (qLearningHistory.length >= 2) {
      const a = qLearningHistory[qLearningHistory.length - 2]
      const b = qLearningHistory[qLearningHistory.length - 1]
      const dt = b.elapsed - a.elapsed
      if (dt > 0) sps = (b.timesteps - a.timesteps) / dt
    }
  } else if (isMa) {
    // Competitive self-play: progress = cumulative steps; the headline "score" is the predator return
    // (its scale is the env's min/solved). The prey line + round caption live in the chart legend.
    const m = lastMa
    hasStats = !!m
    trainPct = m && m.total_timesteps > 0 ? Math.min(100, (m.timesteps / m.total_timesteps) * 100) : null
    score = m ? speciesVal(m, 'adversary') : null
    steps = m?.timesteps
    elapsed = m?.elapsed
    if (maHistory.length >= 2) {
      const a = maHistory[maHistory.length - 2]
      const b = maHistory[maHistory.length - 1]
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

  // Skill %: Score across the env's [min_score, solved_score] range, so it reads consistently
  // with the skill meter (a shaped env like LunarLander starts negative — -54 is ~37%, not 0%).
  const solveEnv = envs.find((e) => e.id === selectedEnvId)
  const solveMin = solveEnv?.min_score ?? 0
  const solveMax = solveEnv?.solved_score ?? skillScaleFor(selectedEnvId).max
  const solvePct =
    score != null && solveMax > solveMin
      ? Math.max(0, Math.min(100, ((score - solveMin) / (solveMax - solveMin)) * 100))
      : null

  // The gold "goal" line = the env's solved score, on the tabs whose y-axis IS a reward/fitness scale
  // (not Loss, which has no solved target). Shows the user where the curve is heading.
  const chartGoal =
    (activeTab === 'reward' || activeTab === 'fitness') && solveMax > solveMin
      ? { value: solveMax, label: t('chart.goal') }
      : undefined
  // Where the gold goal line sits in pixels (same y-scale as LineChart), so the HTML "Goal" label can
  // be pinned to the Y axis just under it — readable and never clipped by the SVG edge.
  const goalTopPx = (() => {
    if (!chartGoal || size.h <= 0) return null
    const chartH = size.h - PAD.t - PAD.b
    if (chartH <= 0) return null
    const { min, max } = chartYDomain(series, chartGoal.value)
    return PAD.t + (1 - (chartGoal.value - min) / ((max - min) || 1)) * chartH
  })()

  return (
    <section style={{
      flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
      background: 'var(--chart-plot-bg)',
    }}>
      {/* Tab bar + chart controls (Smooth / Window) */}
      <div style={{
        display: 'flex', alignItems: 'center', borderBottom: '1px solid var(--border-default)',
        background: 'var(--surface-1)', flexShrink: 0, padding: '0 var(--space-3) 0 var(--space-1)',
        minHeight: 'var(--panel-head-h)',
      }}>
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '9px 12px', border: 'none', cursor: 'pointer',
              background: 'transparent', fontSize: 'var(--fs-sm)',
              fontWeight: activeTab === tab ? 'var(--fw-semibold)' : 'var(--fw-medium)',
              color: activeTab === tab ? 'var(--text-strong)' : 'var(--text-muted)',
              borderBottom: `2px solid ${activeTab === tab ? 'var(--accent)' : 'transparent'}`,
              transition: 'var(--t-colors)',
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

        {/* Compare past runs (D2) */}
        <div style={{ position: 'relative', marginLeft: 8 }}>
          <button
            onClick={() => setShowCompare((v) => !v)}
            title={t('runs.title')}
            aria-label={t('runs.compare')}
            aria-expanded={showCompare}
            style={{
              display: 'flex', alignItems: 'center', gap: 4, padding: '2px 8px',
              borderRadius: 4, border: '1px solid var(--border)', cursor: 'pointer',
              fontSize: 11, fontWeight: 600,
              background: showCompare || envSelected.length > 0 ? 'var(--accent-soft)' : 'var(--surface-2)',
              color: envSelected.length > 0 ? 'var(--accent-h)' : 'var(--text-muted)',
            }}
          >
            ⊕ {compactControls ? '' : t('runs.compare')}{envSelected.length > 0 ? ` (${envSelected.length})` : ''}
          </button>
          {showCompare && (
            <ComparePopover
              runs={envRuns}
              selectedOrder={selectedOrder}
              onToggle={(id) => void toggleRun(id)}
              onDelete={(run) => void removeRun(run)}
              onClear={() => setSelected((sel) => sel.filter((r) => r.meta.env_id !== selectedEnvId))}
              onClose={() => setShowCompare(false)}
            />
          )}
        </div>
      </div>

      {/* Chart area + skill meter — the meter reserves a strip BELOW the plot (not an overlay), so
          the chart shrinks to sit above it and the meter never covers the curve / x-axis / solved markers. */}
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div ref={containerRef} style={{ flex: 1, minHeight: 0, overflow: 'hidden', position: 'relative' }}>
        <HwStats />
        {hasChart ? (
          <>
            <LineChart series={series} markers={markers} goal={chartGoal} width={size.w} height={size.h} xFmt={xFmt} ariaLabel={t('chart.aria_label')} />
            {/* "Goal" label pinned to the Y axis, just under the gold line, with an info popup that
                explains what "solved" means (board games read on a −1…+1 scale, etc.). */}
            {chartGoal && goalTopPx != null && (
              <div style={{
                position: 'absolute', top: goalTopPx + 3, left: PAD.l + 3,
                display: 'inline-flex', alignItems: 'center', gap: 2, pointerEvents: 'none',
              }}>
                <span style={{
                  fontSize: 10, fontWeight: 600, color: 'var(--goal)', fontFamily: 'var(--font-mono)',
                }}>{chartGoal.label}</span>
                <span style={{ pointerEvents: 'auto' }}>
                  <ParamInfo paramId="goal" label={chartGoal.label} />
                </span>
              </div>
            )}
            {activeTab === 'fitness' && (
              <div style={{
                position: 'absolute', top: 6, left: PAD.l, display: 'flex', gap: 10,
                fontSize: 10, color: 'var(--text-muted)',
              }}>
                <LegendDot color="var(--viz-2)" label={t('chart.series_best')} />
                <LegendDot color="var(--viz-1)" label={t('chart.series_avg')} />
                <LegendDot color="var(--viz-3)" label={t('chart.series_worst')} />
              </div>
            )}
            {/* Competitive self-play (simple_tag): predator/prey legend + which species is learning now. */}
            {isMa && activeTab === 'reward' && (
              <div style={{
                position: 'absolute', top: 6, left: PAD.l, display: 'flex', gap: 10,
                fontSize: 10, color: 'var(--text-muted)', alignItems: 'center', flexWrap: 'wrap',
                maxWidth: '70%',
              }}>
                <LegendDot color={PRED_COLOR} label={t('species.predator')} />
                <LegendDot color={accent} label={t('species.prey')} />
                {lastMa && (
                  <span style={{ fontFamily: 'var(--font-mono)' }}>
                    {t('chart.ma_round', { n: lastMa.round, total: lastMa.total_rounds })}
                    {' · '}
                    {t('chart.ma_learning', {
                      species: t(lastMa.learning_role === 'adversary' ? 'species.predator' : 'species.prey'),
                    })}
                  </span>
                )}
              </div>
            )}
            {/* Overlaid past-run legend (dashed lines) */}
            {overlays.length > 0 && (
              <div style={{
                position: 'absolute', top: 6, right: PAD.r, display: 'flex', flexDirection: 'column',
                gap: 2, fontSize: 10, color: 'var(--text-muted)', alignItems: 'flex-end',
                maxWidth: '55%', pointerEvents: 'none',
              }}>
                {overlays.map((o) => (
                  <span key={o.id} style={{
                    display: 'inline-flex', alignItems: 'center', gap: 4, maxWidth: '100%',
                  }}>
                    <span style={{
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                    }}>{o.label}</span>
                    <span style={{ width: 14, height: 0, flexShrink: 0, borderTop: `2px dashed ${o.color}` }} />
                  </span>
                ))}
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
      {/* Skill meter as a reserved strip below the plot (shown only while training is the live
          context; otherwise it self-gates to null and the chart uses the full height). */}
      <SkillMeter slot="train" />
      </div>

      {/* Stats row — fixed height so the panel's bottom line never shifts between PPO and
          neuroevolution (different stat content) and stays aligned with the env panel. */}
      <div style={{
        flexShrink: 0, borderTop: '1px solid var(--border-default)',
        background: 'var(--surface-1)', padding: '0 var(--space-3)',
        display: 'flex', alignItems: 'center', gap: 6,
        height: 52, overflow: 'hidden',
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
          <span style={{ flex: 1, textAlign: 'center', fontSize: 'var(--fs-label)', color: 'var(--text-muted)' }}>
            {t('stats.no_data')}
          </span>
        )}
      </div>

    </section>
  )
}
