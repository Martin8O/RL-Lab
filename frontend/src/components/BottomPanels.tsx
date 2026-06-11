import { useTranslation } from 'react-i18next'

function Panel({ titleKey, placeholderKey, borderRight = true }: {
  titleKey: string
  placeholderKey: string
  borderRight?: boolean
}) {
  const { t } = useTranslation()
  return (
    <div style={{
      flex: 1, display: 'flex', flexDirection: 'column',
      background: 'var(--surface)',
      borderRight: borderRight ? '1px solid var(--border)' : undefined,
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '6px 12px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 12, color: 'var(--text-h)', flexShrink: 0,
      }}>
        {t(titleKey)}
      </div>
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 8,
      }}>
        {t(placeholderKey)}
      </div>
    </div>
  )
}

export default function BottomPanels() {
  return (
    <div style={{
      height: 140, flexShrink: 0, display: 'flex',
      borderTop: '1px solid var(--border)',
    }}>
      <Panel titleKey="leaderboard.title" placeholderKey="leaderboard.placeholder" />
      <Panel titleKey="evolution.title"   placeholderKey="evolution.placeholder" />
      <Panel titleKey="saveload.title"    placeholderKey="saveload.placeholder" borderRight={false} />
    </div>
  )
}
