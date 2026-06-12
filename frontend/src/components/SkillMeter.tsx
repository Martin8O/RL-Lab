import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { currentBand, scaleFromEnvSkill, skillScaleFor } from '../content/skill'

// Fixed low→high skill gradient (red = beginner, green = superhuman). Theme-agnostic.
const GRADIENT =
  'linear-gradient(to right, #e2453c 0%, #f0883e 28%, #e3c000 50%, #86c440 75%, #3fae4f 100%)'

/** Record markers (E2): best-human + best-AI scores drawn as starred lines on the play meter. */
export interface SkillMarkers {
  human?: number | null
  ai?: number | null
}
const MARKER_COLORS = { human: '#4aa3ff', ai: '#c07cf0' } as const

/** Compact skill gauge: maps the live Score to a band (Child → Superhuman) with a needle.
 *  Height matches the env-preview speed row (34px) so the two panels stay aligned.
 *  `titleKey` lets callers relabel it ("AI skill" for the trained agent, "Your skill" for play).
 *  `markers` overlays starred record lines (the play meter passes best human/AI here). */
export default function SkillMeter({
  score,
  titleKey = 'skill.title',
  markers,
}: { score: number | null; titleKey?: string; markers?: SkillMarkers }) {
  const { t } = useTranslation()
  const envId    = useAppStore((s) => s.selectedEnvId)
  const envSkill = useAppStore((s) => s.envSkill)
  // Prefer the backend's per-env thresholds (single source of truth with the play rating);
  // fall back to the local table until that fetch lands or for an unknown env.
  const scale = envSkill ? scaleFromEnvSkill(envSkill) : skillScaleFor(envId)

  const hasScore = score !== null
  const value = hasScore ? score : 0
  const frac = Math.max(0, Math.min(1, value / scale.max))
  const band = currentBand(value, scale)
  const ticks = scale.bands.slice(1).map((b) => b.min / scale.max) // band boundaries

  const recordMarks = (['human', 'ai'] as const)
    .map((key) => ({ key, value: markers?.[key] }))
    .filter((m): m is { key: 'human' | 'ai'; value: number } => typeof m.value === 'number' && m.value > 0)
    .map((m) => ({
      ...m,
      frac: Math.max(0, Math.min(1, m.value / scale.max)),
      color: MARKER_COLORS[m.key],
      label: t(m.key === 'human' ? 'playscore.best_human' : 'playscore.best_ai', { score: Math.round(m.value) }),
    }))

  return (
    <div style={{
      flexShrink: 0, borderTop: '1px solid var(--border)',
      background: 'var(--surface)', padding: '6px 12px', minHeight: 34,
      display: 'flex', alignItems: 'center', gap: 10,
    }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
        {t(titleKey)}
      </span>

      {/* Gradient bar with band-boundary ticks + a needle at the current skill */}
      <div style={{ position: 'relative', flex: 1, height: 14 }}>
        <div style={{
          position: 'absolute', inset: 0, borderRadius: 7,
          background: GRADIENT, opacity: hasScore ? 1 : 0.3,
        }} />
        {ticks.map((tk) => (
          <div key={tk} style={{
            position: 'absolute', left: `${tk * 100}%`, top: 0, bottom: 0,
            width: 1, background: 'rgba(255,255,255,0.5)',
          }} />
        ))}
        {recordMarks.map((m) => (
          <div key={m.key} title={m.label} style={{
            position: 'absolute', left: `${m.frac * 100}%`, top: -3, bottom: -3,
            width: 2, marginLeft: -1, background: m.color, borderRadius: 1,
          }}>
            <span style={{
              position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)',
              fontSize: 9, lineHeight: 1, color: m.color, textShadow: '0 0 2px rgba(0,0,0,0.6)',
            }}>
              ★
            </span>
          </div>
        ))}
        {hasScore && (
          <div style={{
            position: 'absolute', left: `${frac * 100}%`, top: -3, bottom: -3,
            width: 2, marginLeft: -1, background: 'var(--text-h)',
            borderRadius: 1, boxShadow: '0 0 3px rgba(0,0,0,0.55)',
          }}>
            <div style={{
              position: 'absolute', left: '50%', top: -4, transform: 'translateX(-50%)',
              width: 8, height: 8, borderRadius: '50%',
              background: 'var(--text-h)', border: '1.5px solid var(--surface)',
            }} />
          </div>
        )}
      </div>

      <span style={{
        fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap',
        color: hasScore ? 'var(--text-h)' : 'var(--text-muted)',
        minWidth: 110, textAlign: 'right',
      }}>
        {hasScore ? `${t(`skill.${band.key}`)} · ${Math.round(value)}` : t('skill.no_data')}
      </span>
    </div>
  )
}
