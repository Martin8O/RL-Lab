import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { playGuideFor } from '../content/playGuides'

// "How to play" affordance + modal (E2). Same modal chrome as ParamInfo, but the body is the
// per-env play guide (goal / controls / tips) from content/playGuides.ts. Bilingual + themed.

/** Small "How to play" button that opens the instructions modal for the selected env. */
export default function PlayInstructions() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={t('play.how_to_play')}
        aria-label={t('play.how_to_play')}
        style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
          width: 'var(--control-sm)', height: 'var(--control-sm)', padding: 0, cursor: 'pointer',
          background: 'var(--surface-2)', color: 'var(--text-muted)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
          transition: 'var(--t-colors)',
        }}
        onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent-border)' }}
        onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border-default)' }}
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
          <circle cx="12" cy="12" r="9.25" stroke="currentColor" strokeWidth="1.7" />
          <path d="M9.5 9.3a2.6 2.6 0 015.05.85c0 1.7-2.55 2.3-2.55 2.3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="12" cy="16.7" r="1.05" fill="currentColor" />
        </svg>
      </button>
      {open && <InstructionsModal onClose={() => setOpen(false)} />}
    </>
  )
}

function InstructionsModal({ onClose }: { onClose: () => void }) {
  const { t } = useTranslation()
  const locale        = useAppStore((s) => s.locale)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const envs          = useAppStore((s) => s.envs)
  const closeRef      = useRef<HTMLButtonElement>(null)

  const guide   = playGuideFor(selectedEnvId)
  const envName = envs.find((e) => e.id === selectedEnvId)?.display_name[locale] ?? ''

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const title = t('play.instructions_title', { env: envName })

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 1000,
        background: 'rgba(0,0,0,0.5)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        padding: 20,
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
          <Section title={t('play.goal')}>
            <p style={bodyText}>{guide.goal[locale]}</p>
          </Section>

          <Section title={t('play.controls')}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {guide.controls.map((c) => (
                <div key={c.keys} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <kbd style={{
                    fontFamily: 'var(--font-mono)', fontSize: 12, fontWeight: 700,
                    padding: '2px 8px', borderRadius: 5, whiteSpace: 'nowrap',
                    background: 'var(--surface-2)', color: 'var(--text-h)',
                    border: '1px solid var(--border)',
                  }}>
                    {c.keys}
                  </kbd>
                  <span style={{ fontSize: 13, color: 'var(--text)' }}>{c.action[locale]}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title={t('play.tips')}>
            <p style={bodyText}>{guide.tips[locale]}</p>
          </Section>
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
