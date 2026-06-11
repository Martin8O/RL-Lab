import { useTranslation } from 'react-i18next'

export default function RewardChart() {
  const { t } = useTranslation()
  return (
    <section style={{
      flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{
        padding: '8px 12px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 13, color: 'var(--text-h)', flexShrink: 0,
      }}>
        {t('chart.title')}
      </div>
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg)',
      }}>
        {/* Live reward/fitness chart will be added in B3 */}
        <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
          {t('chart.placeholder')}
        </span>
      </div>
    </section>
  )
}
