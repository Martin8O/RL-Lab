import { useTranslation } from 'react-i18next'

export default function Sidebar() {
  const { t } = useTranslation()
  return (
    <aside style={{
      width: 260, flexShrink: 0,
      background: 'var(--surface)', borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      <div style={{
        padding: '10px 14px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 13, color: 'var(--text-h)',
      }}>
        {t('sidebar.title')}
      </div>
      <div style={{
        flex: 1, padding: 14, display: 'flex', alignItems: 'center',
        justifyContent: 'center', color: 'var(--text-muted)', fontSize: 12,
        textAlign: 'center',
      }}>
        {/* Parameter controls (sliders, dropdowns) will appear here in B3 */}
        <span>—</span>
      </div>
    </aside>
  )
}
