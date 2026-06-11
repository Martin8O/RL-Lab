import { useTranslation } from 'react-i18next'

export default function EnvPreview() {
  const { t } = useTranslation()
  return (
    <section style={{
      flex: '0 0 55%', display: 'flex', flexDirection: 'column',
      borderRight: '1px solid var(--border)', overflow: 'hidden',
    }}>
      <div style={{
        padding: '8px 12px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 13, color: 'var(--text-h)', flexShrink: 0,
      }}>
        {t('envpreview.title')}
      </div>
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg)',
      }}>
        {/* Canvas for live environment frames will be added in B4 */}
        <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
          {t('envpreview.placeholder')}
        </span>
      </div>
    </section>
  )
}
