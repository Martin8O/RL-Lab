import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { PARAM_INFO } from '../content/parameters'

// ── Info affordance (ⓘ) + popup ──────────────────────────────────────────────
// Data-driven: each control passes its param id; the popup reads the bilingual
// general + per-environment explanation from content/parameters.ts. Adding a new
// parameter is a content-only change — no edits here.

const infoBtnStyle: CSSProperties = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
  width: 14, height: 14, padding: 0,
  border: 'none', borderRadius: '50%', background: 'transparent',
  color: 'var(--text-muted)', cursor: 'pointer',
  fontSize: 11, lineHeight: 1, fontStyle: 'italic', fontWeight: 700,
}

export default function ParamInfo({ paramId, label }: { paramId: string; label: string }) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const info = PARAM_INFO[paramId]
  if (!info) return null

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={t('info.aria_open', { param: label })}
        title={t('info.aria_open', { param: label })}
        style={infoBtnStyle}
        onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent)')}
        onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-muted)')}
      >
        ⓘ
      </button>
      {open && <InfoModal paramId={paramId} label={label} onClose={() => setOpen(false)} />}
    </>
  )
}

// ── Modal ─────────────────────────────────────────────────────────────────────

function InfoModal({ paramId, label, onClose }: { paramId: string; label: string; onClose: () => void }) {
  const { t } = useTranslation()
  const locale         = useAppStore((s) => s.locale)
  const selectedEnvId  = useAppStore((s) => s.selectedEnvId)
  const envs           = useAppStore((s) => s.envs)
  const closeRef       = useRef<HTMLButtonElement>(null)

  const info = PARAM_INFO[paramId]
  const selectedEnv = envs.find((e) => e.id === selectedEnvId)
  const envNote = (selectedEnvId && info.perEnv?.[selectedEnvId]) || null
  const envName = selectedEnv?.display_name[locale] ?? ''

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

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
        aria-label={label}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 440, maxHeight: '80vh', overflowY: 'auto',
          background: 'var(--surface)', color: 'var(--text)',
          border: '1px solid var(--border)', borderRadius: 10,
          boxShadow: '0 12px 40px rgba(0,0,0,0.4)',
        }}
      >
        {/* Header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: '1px solid var(--border)',
          position: 'sticky', top: 0, background: 'var(--surface)',
        }}>
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-h)' }}>{label}</span>
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
          <Section title={t('info.general')}>
            <p style={bodyText}>{info.general[locale]}</p>
          </Section>

          {info.recommended && (
            <Section title={t('info.recommended')}>
              <p style={{ ...bodyText, color: 'var(--ok)' }}>★ {info.recommended[locale]}</p>
            </Section>
          )}

          {info.range && (
            <Section title={t('info.range')}>
              <p style={{ ...bodyText, fontFamily: 'monospace' }}>{info.range}</p>
            </Section>
          )}

          {envNote && (
            <Section title={t('info.for_env', { env: envName })}>
              <p style={bodyText}>{envNote[locale]}</p>
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
