// The DataLab aggregate (rliable) panel (Zone 3, X6b) — the "hold up in a methods section" view. Renders
// X4's rliable analysis (Agarwal et al., NeurIPS 2021): per-algorithm IQM / mean / median / optimality-gap
// point estimates each with a stratified-bootstrap 95 % CI (drawn as a whisker on a [0, 1] normalized-score
// track), a performance profile (fraction of run-scores above each threshold τ — one line per algorithm),
// and, when the selection has ≥2 algorithms sharing a task, the probability that the first improves on the
// second. Scores are each run's final skill % ÷ 100. On few / non-overlapping seeds the CIs are honestly
// wide (or the estimate is withheld) — the panel says so rather than faking confidence. Theme-token SVG.

import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import type { MethodRliable, RliableEstimate, RliableResult } from '../../api/types'
import { algoLabel } from './chartMath'

// Per-method (algorithm) colours — the run-compare palette, assigned by order.
const METHOD_COLORS = ['var(--cmp-1)', 'var(--cmp-2)', 'var(--cmp-3)', 'var(--cmp-4)', 'var(--cmp-5)', 'var(--cmp-6)']

const clamp01 = (v: number) => Math.max(0, Math.min(1, v))
const f2 = (v: number) => v.toFixed(2)

/** One estimate as a labelled whisker on a [0, 1] track: the CI as a bar, the point as a dot, the value
 *  + CI in mono to the right. All rliable scores live on the same normalized [0, 1] scale, so every
 *  metric shares the track — visually comparable at a glance. */
function EstimateBar({ label, est, color }: { label: string; est: RliableEstimate; color: string }) {
  const lo = clamp01(est.ci_low) * 100
  const hi = clamp01(est.ci_high) * 100
  const val = clamp01(est.value) * 100
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '58px 1fr auto', alignItems: 'center', gap: 8 }}>
      <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>{label}</span>
      <div style={{ position: 'relative', height: 10 }}>
        <div style={{ position: 'absolute', top: 4, left: 0, right: 0, height: 2, background: 'var(--border-default)', borderRadius: 1 }} />
        <div style={{ position: 'absolute', top: 3.5, left: `${lo}%`, width: `${Math.max(hi - lo, 0)}%`, height: 3, background: color, opacity: 0.4, borderRadius: 2 }} />
        <div style={{ position: 'absolute', top: 1, left: `calc(${val}% - 3px)`, width: 6, height: 8, borderRadius: 2, background: color }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)', fontSize: 'var(--fs-meta)', color: 'var(--text-strong)', whiteSpace: 'nowrap' }}>
        {f2(est.value)} <span style={{ color: 'var(--text-faint)' }}>[{f2(est.ci_low)}, {f2(est.ci_high)}]</span>
      </span>
    </div>
  )
}

function MethodCard({ m, color, t }: { m: MethodRliable; color: string; t: TFunction }) {
  return (
    <div style={{ border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)', padding: '9px 10px', display: 'flex', flexDirection: 'column', gap: 7 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 8 }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: color, flexShrink: 0 }} />
          <span style={{ fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
            {algoLabel(t, m.algo)}
          </span>
        </span>
        <span style={{ fontSize: 'var(--fs-micro)', fontFamily: 'var(--font-mono)', color: 'var(--text-faint)' }}>
          {t('analysis.rliable_method_meta', { runs: m.n_runs, tasks: m.tasks.length })}
        </span>
      </div>
      <EstimateBar label={t('analysis.metric_iqm')} est={m.iqm} color={color} />
      <EstimateBar label={t('analysis.metric_mean')} est={m.mean} color={color} />
      <EstimateBar label={t('analysis.metric_median')} est={m.median} color={color} />
      <EstimateBar label={t('analysis.metric_optgap')} est={m.optimality_gap} color={color} />
    </div>
  )
}

// ── Performance profile plot ───────────────────────────────────────────────────
const PROF = { w: 300, h: 150, l: 30, r: 8, t: 8, b: 22 }

function ProfilePlot({ methods, colors, t }: { methods: MethodRliable[]; colors: string[]; t: TFunction }) {
  const withData = methods.filter((m) => m.profile.taus.length > 1)
  if (withData.length === 0) return null
  let xMin = Infinity
  let xMax = -Infinity
  for (const m of withData) for (const tau of m.profile.taus) { if (tau < xMin) xMin = tau; if (tau > xMax) xMax = tau }
  const xSpan = xMax - xMin || 1
  const plotW = PROF.w - PROF.l - PROF.r
  const plotH = PROF.h - PROF.t - PROF.b
  const toX = (tau: number) => PROF.l + ((tau - xMin) / xSpan) * plotW
  const toY = (frac: number) => PROF.t + (1 - clamp01(frac)) * plotH
  const yTicks = [0, 0.5, 1]
  const xTicks = [xMin, xMin + xSpan / 2, xMax]

  return (
    <svg viewBox={`0 0 ${PROF.w} ${PROF.h}`} width="100%" role="img"
      aria-label={t('analysis.profile_aria')} style={{ display: 'block' }}>
      {yTicks.map((v) => (
        <g key={`y${v}`}>
          <line x1={PROF.l} y1={toY(v)} x2={PROF.w - PROF.r} y2={toY(v)} stroke="var(--chart-grid)" strokeWidth={1} />
          <text x={PROF.l - 5} y={toY(v) + 3} textAnchor="end" fontSize={8} fontFamily="var(--font-mono)" fill="var(--chart-axis)">{v}</text>
        </g>
      ))}
      {xTicks.map((v, i) => (
        <text key={`x${i}`} x={toX(v)} y={PROF.h - 8} textAnchor="middle" fontSize={8} fontFamily="var(--font-mono)" fill="var(--chart-axis)">{v.toFixed(1)}</text>
      ))}
      <line x1={PROF.l} y1={PROF.t} x2={PROF.l} y2={PROF.t + plotH} stroke="var(--chart-axis)" strokeWidth={1} />
      <line x1={PROF.l} y1={PROF.t + plotH} x2={PROF.w - PROF.r} y2={PROF.t + plotH} stroke="var(--chart-axis)" strokeWidth={1} />
      {withData.map((m, mi) => {
        const color = colors[methods.indexOf(m) % colors.length]
        const d = m.profile.taus.map((tau, i) => `${i ? 'L' : 'M'}${toX(tau).toFixed(1)},${toY(m.profile.fractions[i]).toFixed(1)}`).join('')
        return <path key={mi} d={d} fill="none" stroke={color} strokeWidth={1.75} strokeLinejoin="round" />
      })}
    </svg>
  )
}

export default function RliablePanel({ result, loading }: { result: RliableResult | null; loading: boolean }) {
  const { t } = useTranslation()

  if (!result) {
    return <p style={{ margin: 0, fontSize: 'var(--fs-meta)', color: 'var(--text-faint)' }}>
      {loading ? t('analysis.rliable_computing') : t('analysis.rliable_pick')}
    </p>
  }
  if (result.methods.length === 0) {
    return <p style={{ margin: 0, fontSize: 'var(--fs-meta)', color: 'var(--text-muted)', lineHeight: 1.5 }}>
      {t('analysis.rliable_need_more')}
    </p>
  }

  const colorFor = (i: number) => METHOD_COLORS[i % METHOD_COLORS.length]
  const poi = result.prob_of_improvement

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* aggregate estimate cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {result.methods.map((m, i) => (
          <MethodCard key={m.algo} m={m} color={colorFor(i)} t={t} />
        ))}
      </div>

      {/* performance profile */}
      <div>
        <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-muted)', marginBottom: 4 }}>
          {t('analysis.profile_title')}
        </div>
        <ProfilePlot methods={result.methods} colors={METHOD_COLORS} t={t} />
        <div style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-faint)', textAlign: 'center', marginTop: 2 }}>
          {t('analysis.profile_axes')}
        </div>
      </div>

      {/* probability of improvement */}
      {poi ? (
        <div>
          <div style={{ fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-muted)', marginBottom: 4 }}>
            {t('analysis.poi_title')}
          </div>
          <div style={{ position: 'relative', height: 12, marginBottom: 4 }}>
            <div style={{ position: 'absolute', top: 5, left: 0, right: 0, height: 2, background: 'var(--border-default)' }} />
            {/* 0.5 = a coin-flip reference */}
            <div style={{ position: 'absolute', top: 0, left: '50%', width: 1, height: 12, background: 'var(--text-faint)' }} />
            <div style={{ position: 'absolute', top: 4, left: `${clamp01(poi.ci_low) * 100}%`, width: `${Math.max((clamp01(poi.ci_high) - clamp01(poi.ci_low)) * 100, 0)}%`, height: 4, background: 'var(--accent)', opacity: 0.4, borderRadius: 2 }} />
            <div style={{ position: 'absolute', top: 1, left: `calc(${clamp01(poi.value) * 100}% - 3px)`, width: 6, height: 10, borderRadius: 2, background: 'var(--accent)' }} />
          </div>
          <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-default)', fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)' }}>
            {t('analysis.poi_value', { x: algoLabel(t, poi.algo_x), y: algoLabel(t, poi.algo_y), v: f2(poi.value) })}
            <span style={{ color: 'var(--text-faint)' }}> [{f2(poi.ci_low)}, {f2(poi.ci_high)}]</span>
          </div>
        </div>
      ) : result.methods.length >= 2 ? (
        <p style={{ margin: 0, fontSize: 'var(--fs-micro)', color: 'var(--text-faint)' }}>{t('analysis.poi_no_shared')}</p>
      ) : (
        <p style={{ margin: 0, fontSize: 'var(--fs-micro)', color: 'var(--text-faint)' }}>{t('analysis.poi_need_two')}</p>
      )}

      <p style={{ margin: 0, fontSize: 'var(--fs-micro)', color: 'var(--text-faint)', lineHeight: 1.5 }}>
        {t('analysis.rliable_norm')}
      </p>
    </div>
  )
}
