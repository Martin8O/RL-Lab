// The DataLab compare chart (Zone 2) — a purpose-built, interactive SVG chart for overlaying many run
// curves (and a multi-seed mean ± band) at once. A new component, NOT an extension of the live
// RewardChart (the X6 constraint keeps that untouched), but it reuses RewardChart's proven maths via
// chartMath.ts. Interactions the live chart lacks: wheel-zoom + drag-pan on X, a log-Y toggle, an EMA
// smoothing slider (applied per series), a hover readout across all series, and a shaded confidence
// band. Display curves are LTTB-downsampled to a budget so a wide overlay stays smooth; the raw data is
// never mutated. Theme-aware (CSS vars), tabular-mono numerics, role="img" for a11y.

import { useMemo, useRef, useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import type { Pt } from './lttb'
import { lttb } from './lttb'
import { emaPoints, fmtTick, logClamp, niceTicks } from './chartMath'

export interface ChartSeries {
  id: string
  label: string
  color: string
  points: Pt[]
}

/** A multi-seed band: a mean line with a shaded [lo, hi] region (CI or ± std) across seeds (X4). */
export interface ChartBand {
  color: string
  x: number[]
  mean: number[]
  lo: number[]
  hi: number[]
}

const PAD = { t: 14, r: 16, b: 34, l: 56 }
const LTTB_BUDGET = 800 // display points per series after windowing

function ptsInWindow(points: Pt[], x0: number, x1: number): Pt[] {
  // Keep one point beyond each edge so the line enters/leaves the viewport cleanly.
  const out: Pt[] = []
  for (let i = 0; i < points.length; i++) {
    const inside = points[i].x >= x0 && points[i].x <= x1
    const straddleL = i + 1 < points.length && points[i].x < x0 && points[i + 1].x >= x0
    const straddleR = i > 0 && points[i].x > x1 && points[i - 1].x <= x1
    if (inside || straddleL || straddleR) out.push(points[i])
  }
  return out
}

function nearestY(points: Pt[], x: number): number | null {
  if (points.length === 0) return null
  let bestI = 0
  let bestD = Infinity
  for (let i = 0; i < points.length; i++) {
    const d = Math.abs(points[i].x - x)
    if (d < bestD) {
      bestD = d
      bestI = i
    }
  }
  return points[bestI].y
}

export default function AnalysisChart({
  series,
  band,
  goal,
  logY,
  emaAlpha,
  xFmt,
  yFmt = fmtTick,
  ariaLabel,
  hoverFmt,
}: {
  series: ChartSeries[]
  band?: ChartBand | null
  goal?: { value: number; label: string } | null
  logY: boolean
  emaAlpha: number
  xFmt: (v: number) => string
  yFmt?: (v: number) => string
  ariaLabel: string
  hoverFmt: (v: number) => string
}) {
  const { t } = useTranslation()
  const wrapRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })
  // The x-zoom window (null = fit all). Pan/zoom mutate it; a double-click resets.
  const [view, setView] = useState<{ x0: number; x1: number } | null>(null)
  const [hoverX, setHoverX] = useState<number | null>(null) // pixel x of the cursor over the plot
  const [dragging, setDragging] = useState(false) // reflect drag in the cursor without reading the ref in render
  const drag = useRef<{ startPx: number; startX0: number; startX1: number } | null>(null)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    // Measure from the element itself, not the ResizeObserver's contentRect — that can report a stale 0
    // on its first callback for a flex child, leaving the chart sized 0×0 and never drawn. An initial
    // rAF measure guarantees a real size even if the observer's first callback is delayed or missed.
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight })
    const raf = requestAnimationFrame(measure)
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [])

  // EMA-smooth each series once (raw → smoothed) before any windowing; band is pre-aggregated (no EMA).
  const smoothed = useMemo(
    () => series.map((s) => ({ ...s, points: emaPoints(s.points, emaAlpha) })),
    [series, emaAlpha],
  )

  // Full data x-range across every series + the band — the "reset" domain the zoom clamps within.
  const fullX = useMemo(() => {
    let lo = Infinity
    let hi = -Infinity
    for (const s of smoothed) for (const p of s.points) {
      if (p.x < lo) lo = p.x
      if (p.x > hi) hi = p.x
    }
    for (const x of band?.x ?? []) {
      if (x < lo) lo = x
      if (x > hi) hi = x
    }
    if (!Number.isFinite(lo)) return null
    return { lo, hi: hi > lo ? hi : lo + 1 }
  }, [smoothed, band])

  const { w, h } = size
  const chartW = w - PAD.l - PAD.r
  const chartH = h - PAD.t - PAD.b

  // The zoom window is in the current axis's units, so a window kept from a *previous* axis/selection
  // (e.g. an env_steps range after switching to wall-clock) no longer applies: if it doesn't overlap the
  // live data range, drop it (show full); otherwise clamp it into range. Pure render-time correction — no
  // effect/setState needed, so the stale `view` self-heals and the next zoom starts from the real domain.
  const effView = useMemo(() => {
    if (!view || !fullX) return null
    if (view.x1 <= fullX.lo || view.x0 >= fullX.hi) return null
    return { x0: Math.max(view.x0, fullX.lo), x1: Math.min(view.x1, fullX.hi) }
  }, [view, fullX])

  const domX0 = effView?.x0 ?? fullX?.lo ?? 0
  const domX1 = effView?.x1 ?? fullX?.hi ?? 1
  const xSpan = domX1 - domX0 || 1

  // Window + LTTB-downsample each series for display, then collect the visible y's for the y-domain.
  const drawn = useMemo(
    () => smoothed.map((s) => ({ ...s, points: lttb(ptsInWindow(s.points, domX0, domX1), LTTB_BUDGET) })),
    [smoothed, domX0, domX1],
  )

  const yDomain = useMemo(() => {
    const ys: number[] = []
    for (const s of drawn) for (const p of s.points) ys.push(p.y)
    if (band) {
      for (let i = 0; i < band.x.length; i++) {
        if (band.x[i] >= domX0 && band.x[i] <= domX1) {
          ys.push(band.lo[i], band.hi[i])
        }
      }
    }
    if (goal) ys.push(goal.value)
    if (ys.length === 0) return { lo: 0, hi: 1 }
    let lo = Math.min(...ys)
    let hi = Math.max(...ys)
    if (logY) {
      lo = Math.max(lo, 1e-6)
      // Pad to clean decades in log space.
      lo = Math.pow(10, Math.floor(Math.log10(lo)))
      hi = Math.pow(10, Math.ceil(Math.log10(hi)))
      return { lo, hi: hi > lo ? hi : lo * 10 }
    }
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.08
    return { lo: lo - pad, hi: hi + pad }
  }, [drawn, band, goal, logY, domX0, domX1])

  const tY = useCallback((v: number) => (logY ? logClamp(v) : v), [logY])
  const yLo = tY(yDomain.lo)
  const yHi = tY(yDomain.hi)
  const ySpan = yHi - yLo || 1

  const toX = useCallback((v: number) => PAD.l + ((v - domX0) / xSpan) * chartW, [domX0, xSpan, chartW])
  const toY = useCallback(
    (v: number) => PAD.t + (1 - (tY(v) - yLo) / ySpan) * chartH,
    [tY, yLo, ySpan, chartH],
  )

  const yTicks = useMemo(() => {
    if (logY) {
      const ticks: number[] = []
      const p0 = Math.round(Math.log10(yDomain.lo))
      const p1 = Math.round(Math.log10(yDomain.hi))
      for (let p = p0; p <= p1; p++) ticks.push(Math.pow(10, p))
      return ticks
    }
    return niceTicks(yDomain.lo, yDomain.hi, 4)
  }, [logY, yDomain])
  const xTicks = useMemo(() => niceTicks(domX0, domX1, 4), [domX0, domX1])

  // ── interaction ────────────────────────────────────────────────────────────
  const clampView = useCallback(
    (x0: number, x1: number) => {
      if (!fullX) return null
      const min = fullX.lo
      const max = fullX.hi
      // Don't zoom past the data or below a 1% window.
      if (x0 <= min && x1 >= max) return null
      const span = Math.max(x1 - x0, (max - min) * 0.01)
      let a = Math.max(min, x0)
      let b = Math.min(max, a + span)
      if (b > max) { b = max; a = Math.max(min, b - span) }
      return { x0: a, x1: b }
    },
    [fullX],
  )

  const onWheel = useCallback(
    (e: React.WheelEvent) => {
      if (!fullX || chartW <= 0) return
      e.preventDefault()
      const px = e.nativeEvent.offsetX
      const cursorX = domX0 + ((px - PAD.l) / chartW) * xSpan
      const factor = e.deltaY < 0 ? 0.8 : 1.25 // in / out
      const nx0 = cursorX - (cursorX - domX0) * factor
      const nx1 = cursorX + (domX1 - cursorX) * factor
      setView(clampView(nx0, nx1))
    },
    [fullX, chartW, domX0, domX1, xSpan, clampView],
  )

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const px = e.nativeEvent.offsetX
      setHoverX(px >= PAD.l && px <= PAD.l + chartW ? px : null)
      if (drag.current && chartW > 0) {
        const dxPx = px - drag.current.startPx
        const dxData = (dxPx / chartW) * (drag.current.startX1 - drag.current.startX0)
        setView(clampView(drag.current.startX0 - dxData, drag.current.startX1 - dxData))
      }
    },
    [chartW, clampView],
  )

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      drag.current = { startPx: e.nativeEvent.offsetX, startX0: domX0, startX1: domX1 }
      setDragging(true)
    },
    [domX0, domX1],
  )
  const endDrag = useCallback(() => { drag.current = null; setDragging(false) }, [])

  const zoomed = effView !== null
  const hoverDataX = hoverX !== null ? domX0 + ((hoverX - PAD.l) / chartW) * xSpan : null

  return (
    <div ref={wrapRef} style={{ position: 'relative', width: '100%', height: '100%', minHeight: 0 }}>
      {w > 20 && h > 20 && fullX && (
        <svg
          width={w}
          height={h}
          style={{ display: 'block', cursor: dragging ? 'grabbing' : 'crosshair', touchAction: 'none' }}
          role="img"
          aria-label={ariaLabel}
          onWheel={onWheel}
          onMouseMove={onMouseMove}
          onMouseDown={onMouseDown}
          onMouseUp={endDrag}
          onMouseLeave={() => { setHoverX(null); endDrag() }}
          onDoubleClick={() => setView(null)}
        >
          {/* horizontal grid */}
          {yTicks.map((v) => (
            <line key={`gy${v}`} x1={PAD.l} y1={toY(v)} x2={PAD.l + chartW} y2={toY(v)}
              stroke="var(--chart-grid)" strokeWidth={1} />
          ))}
          {/* axis spines */}
          <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={PAD.t + chartH} stroke="var(--chart-axis)" strokeWidth={1.25} />
          <line x1={PAD.l} y1={PAD.t + chartH} x2={PAD.l + chartW} y2={PAD.t + chartH} stroke="var(--chart-axis)" strokeWidth={1.25} />

          {/* y-axis labels */}
          {yTicks.map((v) => (
            <text key={`ty${v}`} x={PAD.l - 7} y={toY(v) + 3.5} textAnchor="end" fontSize={10}
              fontFamily="var(--font-mono)" fill="var(--chart-axis)">{yFmt(v)}</text>
          ))}
          {/* x-axis labels */}
          {xTicks.map((v) => (
            <text key={`tx${v}`} x={toX(v)} y={PAD.t + chartH + 18} textAnchor="middle" fontSize={10}
              fontFamily="var(--font-mono)" fill="var(--chart-axis)">{xFmt(v)}</text>
          ))}

          {/* goal / solved threshold line */}
          {goal && goal.value >= yDomain.lo && goal.value <= yDomain.hi && (
            <line x1={PAD.l} y1={toY(goal.value)} x2={PAD.l + chartW} y2={toY(goal.value)}
              stroke="var(--goal)" strokeWidth={1.5} strokeDasharray="6 4" opacity={0.9} />
          )}

          {/* multi-seed band: shaded [lo, hi] region + the mean line */}
          {band && band.x.length > 1 && (() => {
            const idx = band.x.map((_, i) => i).filter((i) => band.x[i] >= domX0 && band.x[i] <= domX1)
            if (idx.length < 2) return null
            const top = idx.map((i) => `${toX(band.x[i]).toFixed(1)},${toY(band.hi[i]).toFixed(1)}`)
            const bot = idx.slice().reverse().map((i) => `${toX(band.x[i]).toFixed(1)},${toY(band.lo[i]).toFixed(1)}`)
            const mean = idx.map((i, k) => `${k ? 'L' : 'M'}${toX(band.x[i]).toFixed(1)},${toY(band.mean[i]).toFixed(1)}`).join('')
            return (
              <g>
                <polygon points={[...top, ...bot].join(' ')} fill={band.color} opacity={0.16} />
                <path d={mean} fill="none" stroke={band.color} strokeWidth={2.25} strokeLinejoin="round" />
              </g>
            )
          })()}

          {/* series lines */}
          {drawn.map((s) => {
            let d = ''
            let pen = false
            for (const p of s.points) {
              const px = toX(p.x).toFixed(1)
              const py = toY(p.y).toFixed(1)
              d += pen ? `L${px},${py}` : `M${px},${py}`
              pen = true
            }
            return d ? <path key={s.id} d={d} fill="none" stroke={s.color} strokeWidth={1.75}
              strokeLinejoin="round" strokeLinecap="round" /> : null
          })}

          {/* hover guide + readout */}
          {hoverX !== null && hoverDataX !== null && (() => {
            const rows = [
              ...(band && band.x.length > 1
                ? [{ label: t('analysis.mean'), color: band.color, y: nearestY(band.x.map((x, i) => ({ x, y: band.mean[i] })), hoverDataX) }]
                : []),
              ...drawn.map((s) => ({ label: s.label, color: s.color, y: nearestY(s.points, hoverDataX) })),
            ].filter((r) => r.y !== null)
            if (rows.length === 0) return null
            // Size the box to its content and truncate long "<game> · <algo>" labels, so the label never
            // overruns the right-aligned value (a fixed-width box clipped them together).
            const header = xFmt(hoverDataX)
            const labels = rows.map((r) => (r.label.length > 40 ? `${r.label.slice(0, 39)}…` : r.label))
            const vals = rows.map((r) => hoverFmt(r.y as number))
            const labelPx = Math.max(...labels.map((l) => l.length)) * 5.9
            const valPx = Math.max(...vals.map((v) => v.length)) * 6.3
            const boxW = Math.min(360, Math.max(132, Math.ceil(29 + labelPx + 12 + valPx), Math.ceil(16 + header.length * 6)))
            const flip = hoverX > PAD.l + chartW - boxW - 8
            const bx = flip ? hoverX - boxW - 10 : hoverX + 10
            // Header (the x-position, e.g. "248k") gets its own band above a hairline so it never reads
            // as another legend row crowding the first run — hence the +8 lead before the rows start.
            const rowTop = 28
            const boxH = rowTop + rows.length * 14
            return (
              <g pointerEvents="none">
                <line x1={hoverX} y1={PAD.t} x2={hoverX} y2={PAD.t + chartH} stroke="var(--chart-axis)" strokeWidth={1} strokeDasharray="3 3" opacity={0.7} />
                <g transform={`translate(${bx}, ${PAD.t + 6})`}>
                  <rect width={boxW} height={boxH} rx={6} fill="var(--surface-2)" stroke="var(--border-strong)" strokeWidth={1} opacity={0.98} />
                  <text x={8} y={14} fontSize={9.5} fontFamily="var(--font-mono)" fill="var(--text-muted)">
                    {header}
                  </text>
                  <line x1={8} y1={20} x2={boxW - 8} y2={20} stroke="var(--border-default)" strokeWidth={1} opacity={0.8} />
                  {rows.map((r, i) => (
                    <g key={i} transform={`translate(8, ${rowTop + i * 14})`}>
                      <rect width={8} height={8} y={-7} rx={2} fill={r.color} />
                      <text x={13} y={0} fontSize={10} fill="var(--text-default)">{labels[i]}</text>
                      <text x={boxW - 16} y={0} textAnchor="end" fontSize={10} fontFamily="var(--font-mono)"
                        style={{ fontFeatureSettings: 'var(--ff-tabular)' }} fill="var(--text-strong)">
                        {vals[i]}
                      </text>
                    </g>
                  ))}
                </g>
              </g>
            )
          })()}

          {zoomed && (
            <text x={PAD.l + chartW} y={PAD.t + 2} textAnchor="end" fontSize={9} fill="var(--text-faint)">
              {t('analysis.zoom_reset')}
            </text>
          )}
        </svg>
      )}
    </div>
  )
}
