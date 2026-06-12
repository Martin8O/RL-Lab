import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { currentBand, skillScaleFor } from '../content/skill'

// Fixed low→high skill gradient (red = beginner, green = superhuman). Theme-agnostic.
const GRADIENT =
  'linear-gradient(to right, #e2453c 0%, #f0883e 28%, #e3c000 50%, #86c440 75%, #3fae4f 100%)'

/** Compact skill gauge: maps the live Score to a band (Child → Superhuman) with a needle.
 *  Height matches the env-preview speed row (34px) so the two panels stay aligned. */
export default function SkillMeter({ score }: { score: number | null }) {
  const { t } = useTranslation()
  const envId = useAppStore((s) => s.selectedEnvId)
  const scale = skillScaleFor(envId)

  const hasScore = score !== null
  const value = hasScore ? score : 0
  const frac = Math.max(0, Math.min(1, value / scale.max))
  const band = currentBand(value, scale)
  const ticks = scale.bands.slice(1).map((b) => b.min / scale.max) // band boundaries

  return (
    <div style={{
      flexShrink: 0, borderTop: '1px solid var(--border)',
      background: 'var(--surface)', padding: '6px 12px', minHeight: 34,
      display: 'flex', alignItems: 'center', gap: 10,
    }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
        {t('skill.title')}
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
