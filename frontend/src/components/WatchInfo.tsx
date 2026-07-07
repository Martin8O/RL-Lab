import { useEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { watchTipFor } from '../content/playGuides'
import { formatCount } from '../format'
import LabSelect from './LabSelect'
import type { CheckpointMeta } from '../api/types'

// Footer for a multi-agent env (no single human driver, so no Play bar). It carries **Watch AI** —
// pick a saved model and watch the trained ecosystem play itself (both species' brains) — plus the
// "What am I watching?" explainer (the env description + the colour-coded "who's who" legend). For an
// env with no saves yet it just shows a "train + save first" hint beside the explainer (G7b-2).
interface WatchInfoProps {
  checkpoints: CheckpointMeta[]
  selected: string
  onSelect: (id: string) => void
  watching: boolean
  onWatch: () => void
  onStop: () => void
}

// Compact label for the save picker: amount of training + when it was saved (MM-DD HH:MM).
function ckptOptionLabel(c: CheckpointMeta): string {
  const when = c.created_at.length >= 16 ? c.created_at.slice(5, 16).replace('T', ' ') : ''
  return `${formatCount(c.timesteps)}${when ? ` · ${when}` : ''}`
}

export default function WatchInfo({ checkpoints, selected, onSelect, watching, onWatch, onStop }: WatchInfoProps) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const hasCkpt = checkpoints.length > 0

  return (
    <div style={{
      flexShrink: 0, borderTop: '1px solid var(--border-default)',
      background: 'var(--surface-1)', padding: '0 var(--space-3)', minHeight: 52,
      display: 'flex', alignItems: 'center', gap: 8,
    }}>
      {/* Watch AI: a saved swarm playing itself (or a hint to train + save first) */}
      {hasCkpt ? (
        <>
          <LabSelect
            ariaLabel={t('watch.pick_save')}
            value={selected}
            onChange={onSelect}
            disabled={watching}
            style={{ maxWidth: 170 }}
            options={checkpoints.map((c) => ({ value: c.id, label: ckptOptionLabel(c) }))}
          />
          <button
            type="button"
            onClick={watching ? onStop : onWatch}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              height: 'var(--control-sm)', padding: '0 14px', cursor: 'pointer',
              border: '1px solid transparent', borderRadius: 'var(--radius-md)',
              fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)', transition: 'var(--t-colors)',
              background: watching ? 'var(--danger-surface)' : 'var(--accent)',
              color: watching ? 'var(--danger)' : 'var(--accent-contrast)',
            }}
          >
            {watching ? `■ ${t('watch.stop')}` : `▶ ${t('watch.watch_ai')}`}
          </button>
        </>
      ) : (
        <span style={{ fontSize: 'var(--fs-label)', color: 'var(--text-muted)' }}>{t('watch.train_first')}</span>
      )}

      <div style={{ flex: 1, minWidth: 4 }} />

      <button
        type="button"
        onClick={() => setOpen(true)}
        title={t('watch.about')}
        aria-label={t('watch.about')}
        style={{
          display: 'inline-flex', alignItems: 'center', gap: 7,
          height: 'var(--control-sm)', padding: '0 12px', cursor: 'pointer',
          background: 'var(--surface-2)', color: 'var(--text-strong)',
          borderWidth: 1, borderStyle: 'solid', borderColor: 'var(--border-default)', borderRadius: 'var(--radius-md)',
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
        background: 'var(--backdrop)', backdropFilter: 'blur(7px)', WebkitBackdropFilter: 'blur(7px)',
        animation: 'lab-fade-in var(--dur-3) var(--ease-out)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        className="glass"
        style={{
          width: '100%', maxWidth: 460, maxHeight: '80vh', overflowY: 'auto',
          background: 'var(--surface-glass)', color: 'var(--text-default)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-popover)',
          animation: 'lab-rise var(--dur-3) var(--ease-out)',
        }}
      >
        {/* Header — frosted: its own backdrop blur keeps scrolled content readable behind it */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: '1px solid var(--border)',
          position: 'sticky', top: 0, background: 'var(--surface-glass)',
          backdropFilter: 'blur(12px)', WebkitBackdropFilter: 'blur(12px)',
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
          {/* Colour-coded "who's who" for a competitive predator–prey world — markers match the swarm
              canvas (red predators / blue prey / grey obstacles), one species per line (visual-labels rule). */}
          {env?.competitive && (
            <Section title={t('watch.who_is_who')}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
                <LegendRow color="var(--danger)" term={t('species.predator')} desc={t('species.predator_desc')} />
                <LegendRow color="var(--accent)" term={t('species.prey')} desc={t('species.prey_desc')} />
                <LegendRow color="var(--border-strong)" term={t('species.obstacles')} desc={t('species.obstacles_desc')} />
              </div>
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

// One colour-coded legend line: a swatch matching the render + a bold term + its description, on its
// own line — the "make labels visually attractive" rule (predators red / prey blue / obstacles grey).
function LegendRow({ color, term, desc }: { color: string; term: string; desc: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
      <span aria-hidden style={{
        flexShrink: 0, width: 11, height: 11, borderRadius: '50%', background: color,
        display: 'inline-block', transform: 'translateY(1px)',
      }} />
      <span style={{ fontSize: 13, lineHeight: 1.5, color: 'var(--text)' }}>
        <span style={{ fontWeight: 700, color: 'var(--text-strong)' }}>{term}</span>
        {' — '}{desc}
      </span>
    </div>
  )
}

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
