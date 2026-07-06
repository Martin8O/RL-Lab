// Pure chart maths for the DataLab compare chart — nice-tick generation, EMA smoothing, a log-safe
// transform, and the algo label. Deliberately i18n-free / side-effect-free (unit-tested); the chart
// component owns all rendering. These mirror the proven helpers in RewardChart.tsx rather than
// importing them, so the live chart stays untouched (the X6 constraint) while DataLab can extend them
// (log-Y, per-series EMA over its own points).

import type { Algo } from '../../api/types'
import type { Pt } from './lttb'

export type { Pt } from './lttb'

// ── EMA smoothing ──────────────────────────────────────────────────────────

/** Exponential moving average over a point series' y (x is carried through unchanged). `alpha` = 1
 *  is raw (no smoothing); smaller = smoother. Matches RewardChart's `computeEma`. */
export function emaPoints(points: Pt[], alpha: number): Pt[] {
  if (alpha >= 1 || points.length === 0) return points
  const out: Pt[] = []
  let prev: number | null = null
  for (const p of points) {
    prev = prev === null ? p.y : alpha * p.y + (1 - alpha) * prev
    out.push({ x: p.x, y: prev })
  }
  return out
}

// ── Nice tick values (1 / 2 / 5 × 10ⁿ) ───────────────────────────────────────

/** The "nice" tick step for a range, so ticks + the domain floor round to the same clean grid. */
export function niceStep(range: number, count = 4): number {
  if (range <= 0) return 1
  const rough = range / count
  const mag = Math.pow(10, Math.floor(Math.log10(rough)))
  const mult = ([1, 2, 5, 10] as const).find((s) => s * mag >= rough) ?? 10
  return mult * mag
}

export function niceTicks(min: number, max: number, count = 4): number[] {
  if (min === max) return [min]
  const step = niceStep(max - min, count)
  const start = Math.floor(min / step) * step
  const ticks: number[] = []
  for (let t = start; t <= max + step * 0.5; t = +(t + step).toFixed(12)) {
    if (t >= min - step * 0.01) ticks.push(+t.toFixed(10))
    if (ticks.length > count * 2 + 2) break
  }
  return ticks
}

/** Compact tick label: 1.5k / 2M, one decimal only when it disambiguates adjacent ticks. Matches
 *  RewardChart's `fmtTick`. */
export function fmtTick(v: number): string {
  const short = (x: number, suffix: string) => `${x % 1 === 0 ? x.toFixed(0) : x.toFixed(1)}${suffix}`
  if (Math.abs(v) >= 1_000_000) return short(v / 1_000_000, 'M')
  if (Math.abs(v) >= 1000) return short(v / 1000, 'k')
  if (v % 1 !== 0) return v.toFixed(1)
  return String(v)
}

// ── Log-Y transform ──────────────────────────────────────────────────────────

/** Log-scale a value for a log-Y axis. Rewards can be ≤ 0 (shaped/penalty envs), and log is undefined
 *  there, so we clamp to a small positive floor — log-Y is offered as an *option* for wide positive
 *  ranges (a PPO reward climbing 1 → 500), and the UI only enables it when the data is all-positive. */
export function logClamp(v: number, floor = 1e-6): number {
  return Math.log10(Math.max(v, floor))
}

// ── Labels ───────────────────────────────────────────────────────────────────

/** Localized algorithm name — reuses the sidebar's algo keys so DataLab reads identically to the rest
 *  of the app. Pure (takes `t`), so it lives in this i18n-free module without a react-refresh warning. */
export function algoLabel(t: (k: string) => string, algo: Algo): string {
  switch (algo) {
    case 'neuroevolution': return t('sidebar.algo_evo')
    case 'q_learning': return t('sidebar.algo_q')
    case 'alphazero': return t('sidebar.algo_az')
    case 'sac': return t('sidebar.algo_sac')
    case 'td3': return t('sidebar.algo_td3')
    case 'dqn': return t('sidebar.algo_dqn')
    case 'a2c': return t('sidebar.algo_a2c')
    default: return t('sidebar.algo_ppo')
  }
}
