// The CZ/EN language + dark/light theme toggles, top-right. Shared so they sit in the same corner in
// both the dashboard TopBar and the Data Lab header (both read/write the same store), so the two
// controls are always in reach whichever view you're in.

import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'

function IconBtn({ onClick, label, children, text }: {
  onClick: () => void; label: string; children?: React.ReactNode; text?: string
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        height: 34, minWidth: 34, padding: text ? '0 11px' : 0,
        background: 'transparent', border: '1px solid transparent',
        borderRadius: 'var(--radius-md)', color: 'var(--text-muted)',
        fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)',
        letterSpacing: 'var(--ls-wide)', cursor: 'pointer',
        transition: 'var(--t-colors)',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--surface-2)'; e.currentTarget.style.color = 'var(--text-default)' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)' }}
    >
      {text ?? children}
    </button>
  )
}

const SunIcon = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
    <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="2" />
    <path d="M12 2.5v2.2M12 19.3v2.2M21.5 12h-2.2M4.7 12H2.5M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6M18.4 18.4l-1.6-1.6M7.2 7.2L5.6 5.6"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </svg>
)
const MoonIcon = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
  </svg>
)

export default function LangThemeToggle() {
  const { t } = useTranslation()
  const locale = useAppStore((s) => s.locale)
  const theme = useAppStore((s) => s.theme)
  const setLocale = useAppStore((s) => s.setLocale)
  const setTheme = useAppStore((s) => s.setTheme)
  return (
    <>
      <IconBtn onClick={() => setLocale(locale === 'en' ? 'cz' : 'en')} label={t('topbar.toggle_language')} text={locale === 'en' ? 'CZ' : 'EN'} />
      <IconBtn onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} label={t('topbar.toggle_theme')}>
        {theme === 'dark' ? MoonIcon : SunIcon}
      </IconBtn>
    </>
  )
}
