import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { watchTipFor } from '../content/playGuides'

// Watch-only "What am I watching?" affordance (G7a follow-up). Watch-and-train envs with no single
// human driver (the multi-agent swarm) hide the Play / How-to-play bar — taking the env explanation
// with it. This restores it in the same footer slot: a button opening a modal that explains the env
// (its registry description) and what to look for while it trains. Same modal chrome as PlayInstructions.
export default function WatchInfo() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  return (
    <div style={{
      flexShrink: 0, borderTop: '1px solid var(--border-default)',
      background: 'var(--surface-1)', padding: '0 var(--space-3)', minHeight: 52,
      display: 'flex', alignItems: 'center', gap: 8,
    }}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={t('watch.about')}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 7,
          height: 'var(--control-sm)', padding: '0 12px', cursor: 'pointer',
          background: 'var(--surface-2)', color: 'var(--text-strong)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
          fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)', transition: 'var(--t-colors)',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-border)' }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-default)' }}
      >
        <span aria-hidden style={{ display: 'inline-flex', color: 'var(--accent)' }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
            <circle cx="12" cy="12" r="9.25" stroke="currentColor" strokeWidth="1.7" />
            <path d="M9.5 9.3a2.6 2.6 0 015.05.85c0 1.7-2.55 2.3-2.55 2.3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="12" cy="16.7" r="1.05" fill="currentColor" />
          </svg>
        </span>
        {t('watch.about')}
      </button>
      {open && <WatchModal onClose={() => setOpen(false)} />}
    </div>
  )
}

function WatchModal({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation()
  const locale        = useAppStore((s) => s.locale)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const envs          = useAppStore((s) => s.envs)
  const closeRef      = useRef<HTMLButtonElement>(null)

  const env     = envs.find((e) => e.id === selectedEnvId)
  const envName = env?.display_name[locale] ?? ''
  const tip     = watchTipFor(selectedEnvId)

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const title = t('watch.title', { env: envName })

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.5)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 460, maxHeight: '80vh', overflowY: 'auto',
          background: 'var(--surface-1)', color: 'var(--text-default)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-popover)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: '1px solid var(--border)',
          position: 'sticky', top: 0, background: 'var(--surface)',
        }}>
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-h)' }}>{title}</span>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label={t('info.close')}
            style={{
              width: 24, height: 24, padding: 0, lineHeight: 1,
              border: '1px solid var(--border)', borderRadius: 6,
              background: 'var(--surface-2)', color: 'var(--text-muted)',
              cursor: 'pointer', fontSize: 14,
            }}
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {env && (
            <Section title={t('watch.what_is_this')}>
              <p style={bodyText}>{env.description[locale]}</p>
            </Section>
          )}
          {tip && (
            <Section title={t('watch.what_to_look_for')}>
              <p style={bodyText}>{tip[locale]}</p>
            </Section>
          )}
        </div>
      </div>
    </div>
  )
}

const bodyText: CSSProperties = { margin: 0, fontSize: 13, lineHeight: 1.5, color: 'var(--text)' }

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div style={{
        fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em',
        color: 'var(--text-muted)', marginBottom: 4,
      }}>
        {title}
      </div>
      {children}
    </div>
  )
}
