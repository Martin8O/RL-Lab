// "About" — a short, bilingual blurb about the app + its author, reachable from both the dashboard
// TopBar and the Data Lab header (drop `<AboutButton />` anywhere; it owns its open state + modal).
// Mirrors the ParamInfo / RunConfigModal chrome (portal overlay, sticky header, Esc to close). The
// GitHub link points at Martin's *profile* (not this repo) and opens in the default browser — works
// the same in dev and in the bundled standalone exe.

import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'

// Shown in the modal footer. Bumped per shared build so friends can tell which one they're running
// (and, later, so an update check can compare against the latest GitHub release).
export const APP_VERSION = '1.0.0'

const GITHUB_URL = 'https://github.com/Martin8O'
const WEBSITE_URL = 'http://svobodamartin.dev/'
// A quiet, optional support link (kept out of the main app UI — the tool stays clean; funding CTAs
// live on the marketing site). Goes live once GitHub Sponsors is enabled on the account.
const SPONSOR_URL = 'https://github.com/sponsors/Martin8O'

const InfoIcon = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
    <path d="M12 11v5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    <circle cx="12" cy="7.75" r="1.15" fill="currentColor" />
  </svg>
)

const GithubIcon = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
    <path d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.85 9.73.5.09.68-.22.68-.49 0-.24-.01-.87-.01-1.71-2.79.62-3.38-1.37-3.38-1.37-.46-1.18-1.11-1.5-1.11-1.5-.91-.63.07-.62.07-.62 1 .07 1.53 1.05 1.53 1.05.89 1.56 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.71 0 0 .84-.28 2.75 1.05a9.4 9.4 0 015 0c1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.45.1 2.71.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.81-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.48-.01 2.82 0 .27.18.59.69.49A10.26 10.26 0 0022 12.25C22 6.58 17.52 2 12 2z" />
  </svg>
)

const GlobeIcon = (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden>
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" />
    <path d="M3 12h18M12 3c2.5 2.5 3.5 5.8 3.5 9s-1 6.5-3.5 9c-2.5-2.5-3.5-5.8-3.5-9s1-6.5 3.5-9z"
      stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
  </svg>
)

// A pill link to an external destination (GitHub profile / personal site), opened in the default
// browser — identical behavior in dev and the bundled exe.
function LinkPill({ href, icon, label, sub }: { href: string; icon: React.ReactNode; label: string; sub: string }) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 8, alignSelf: 'flex-start',
        padding: '6px 12px',
        background: 'var(--surface-2)', border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-pill)', color: 'var(--text-strong)',
        fontSize: 13, fontWeight: 'var(--fw-medium)', textDecoration: 'none',
      }}
    >
      {icon}
      <span>{label}</span>
      <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{sub}</span>
    </a>
  )
}

function AboutModal({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation()
  const closeRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return createPortal(
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 'var(--z-modal, 1000)',
        background: 'var(--backdrop)', backdropFilter: 'blur(7px)', WebkitBackdropFilter: 'blur(7px)',
        animation: 'lab-fade-in var(--dur-3) var(--ease-out)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('about.title')}
        onClick={(e) => e.stopPropagation()}
        className="glass"
        style={{
          width: '100%', maxWidth: 440, maxHeight: '82vh', overflowY: 'auto',
          background: 'var(--surface-glass)', color: 'var(--text-default)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-popover)',
          animation: 'lab-rise var(--dur-3) var(--ease-out)',
        }}
      >
        {/* header — frosted: its own backdrop blur keeps scrolled content readable behind it */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: '1px solid var(--border-default)',
          position: 'sticky', top: 0, background: 'var(--surface-glass)',
          backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
        }}>
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-strong)' }}>
            {t('about.title')}
          </span>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label={t('info.close')}
            style={{
              width: 24, height: 24, padding: 0, lineHeight: 1,
              border: '1px solid var(--border-default)', borderRadius: 6,
              background: 'var(--surface-2)', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 14,
            }}
          >
            ✕
          </button>
        </div>

        {/* body */}
        <div style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          <p style={{ margin: 0, fontSize: 14, fontWeight: 'var(--fw-medium)', color: 'var(--text-strong)', lineHeight: 1.5 }}>
            {t('about.tagline')}
          </p>
          <p style={{ margin: 0, fontSize: 13, color: 'var(--text-default)', lineHeight: 1.6 }}>
            {t('about.app_body')}
          </p>

          <div style={{ height: 1, background: 'var(--border-default)' }} />

          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{
              fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)',
            }}>
              {t('about.author_title')}
            </div>
            <p style={{ margin: 0, fontSize: 13, color: 'var(--text-default)', lineHeight: 1.6 }}>
              {t('about.author_body')}
            </p>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 2 }}>
              <LinkPill href={WEBSITE_URL} icon={GlobeIcon} label={t('about.website')} sub="svobodamartin.dev" />
              <LinkPill href={GITHUB_URL} icon={GithubIcon} label={t('about.github')} sub="github.com/Martin8O" />
            </div>

            {/* A light, optional support line — a muted note with an inline sponsor link (not a button, so
                the app stays clean). The link goes live once GitHub Sponsors is enabled (R3). */}
            <p style={{ margin: '4px 0 0', fontSize: 12, color: 'var(--text-muted)', lineHeight: 1.6 }}>
              {t('about.support_note')}{' '}
              <a
                href={SPONSOR_URL}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: 'var(--text-strong)', fontWeight: 'var(--fw-medium)', textDecoration: 'none', whiteSpace: 'nowrap' }}
              >
                {t('about.sponsor')} →
              </a>
            </p>
          </div>

          <div style={{ height: 1, background: 'var(--border-default)' }} />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
            <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>{t('about.version')}</span>
            <span style={{
              fontSize: 12.5, fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
              color: 'var(--text-strong)',
            }}>
              {APP_VERSION}
            </span>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  )
}

export default function AboutButton() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={t('about.button_aria')}
        title={t('about.button_aria')}
        style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          height: 34, minWidth: 34, padding: 0,
          background: 'transparent', border: '1px solid transparent',
          borderRadius: 'var(--radius-md)', color: 'var(--text-muted)', cursor: 'pointer',
          transition: 'var(--t-colors)',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--surface-2)'; e.currentTarget.style.color = 'var(--text-default)' }}
        onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)' }}
      >
        {InfoIcon}
      </button>
      {open && <AboutModal onClose={() => setOpen(false)} />}
    </>
  )
}
