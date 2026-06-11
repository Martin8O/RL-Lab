import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { BackendStatus } from '../store/useAppStore'

const DOT_COLOR: Record<BackendStatus, string> = {
  online:     'var(--ok)',
  connecting: 'var(--warn)',
  offline:    'var(--err)',
}

function StatusDot() {
  const { t } = useTranslation()
  const status = useAppStore((s) => s.backendStatus)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span
        style={{
          width: 8, height: 8, borderRadius: '50%',
          background: DOT_COLOR[status],
          boxShadow: `0 0 6px ${DOT_COLOR[status]}`,
          display: 'inline-block',
          animation: status === 'connecting' ? 'pulse 1.4s ease-in-out infinite' : undefined,
        }}
      />
      <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{t(`status.${status}`)}</span>
    </div>
  )
}

function Chip({ label, value }: { label: string; value: string }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 4,
      background: 'var(--surface-2)', border: '1px solid var(--border)',
      borderRadius: 6, padding: '2px 8px', fontSize: 12,
    }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: 'var(--text-h)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

export default function TopBar() {
  const { t }  = useTranslation()
  const locale  = useAppStore((s) => s.locale)
  const theme   = useAppStore((s) => s.theme)
  const setLocale = useAppStore((s) => s.setLocale)
  const setTheme  = useAppStore((s) => s.setTheme)

  const metricsHistory = useAppStore((s) => s.metricsHistory)
  const bestReward     = useAppStore((s) => s.bestReward)

  const lastIter = metricsHistory.at(-1)?.iteration

  return (
    <header style={{
      height: 48, flexShrink: 0,
      display: 'flex', alignItems: 'center', gap: 12, padding: '0 16px',
      background: 'var(--surface)', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-h)', letterSpacing: '-0.2px', marginRight: 4 }}>
        {t('app.title')}
      </span>

      <StatusDot />

      <div style={{ flex: 1 }} />

      <Chip label={t('topbar.chips.gen')}  value="—" />
      <Chip label={t('topbar.chips.iter')} value={lastIter !== undefined ? String(lastIter) : '—'} />
      <Chip label={t('topbar.chips.best')} value={bestReward !== null ? bestReward.toFixed(1) : '—'} />
      <Chip label={t('topbar.chips.pop')}  value="—" />

      <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px' }} />

      <button
        onClick={() => setLocale(locale === 'en' ? 'cz' : 'en')}
        style={{
          background: 'var(--surface-2)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '3px 10px', cursor: 'pointer',
          color: 'var(--text-h)', fontSize: 12, fontWeight: 500,
        }}
        aria-label="Toggle language"
      >
        {locale === 'en' ? 'CZ' : 'EN'}
      </button>

      <button
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        style={{
          background: 'var(--surface-2)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '3px 8px', cursor: 'pointer',
          color: 'var(--text-h)', fontSize: 14, lineHeight: 1,
        }}
        aria-label="Toggle theme"
      >
        {theme === 'dark' ? '☀' : '🌙'}
      </button>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.35; }
        }
      `}</style>
    </header>
  )
}
