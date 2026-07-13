// #2b — the first-launch mode chooser. Instead of silently defaulting, a one-time welcome card lets a
// new user pick how they want to use the app, with a short "who it's for / what it enables" read on
// each. Shown only until a choice is made (store `modeChosen`); the choice persists and the same
// switch stays available in the TopBar. Portalled to <body> so no `transform` ancestor collapses the
// fixed overlay ([[reference_transform_breaks_fixed_modal]]).
//
// Each card carries its own hue so the two read as distinct, lively choices (not two dark tiles):
// Simple = warm amber (the arcade/play feel), Advanced = violet (the technical/tuning feel). The hue
// colours the icon tile, a top accent bar, a soft background wash, the tagline, the ✓ ticks, and the
// hover ring.

import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { AudienceMode } from '../store/useAppStore'
import { SimpleMark, AdvancedMark } from './modeMarks'
import { HUES } from '../content/modeHues'

function Feature({ text, color }: { text: string; color: string }) {
  return (
    <li style={{ display: 'flex', alignItems: 'flex-start', gap: 10, fontSize: 'var(--fs-body)', color: 'var(--text-default)', lineHeight: 1.45 }}>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden style={{ flexShrink: 0, marginTop: 1 }}>
        <path d="M5 12.5l4.5 4.5L19 7.5" stroke={color} strokeWidth="2.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      <span>{text}</span>
    </li>
  )
}

function Card({ mode, mark, onPick }: { mode: AudienceMode; mark: React.ReactNode; onPick: (m: AudienceMode) => void }) {
  const { t } = useTranslation()
  const hue = HUES[mode]
  return (
    <button
      type="button"
      onClick={() => onPick(mode)}
      style={{
        position: 'relative', overflow: 'hidden',
        flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 16,
        textAlign: 'left', padding: '26px 22px 26px', cursor: 'pointer',
        // A pronounced hue wash + a slight white lift over the raised surface, so each card reads as its
        // own colourful, lighter tile — not a dark button on a dark background.
        background: `linear-gradient(150deg, ${hue.tint}, transparent 82%), linear-gradient(rgba(255,255,255,0.035), rgba(255,255,255,0.035)), var(--surface-3)`,
        border: '1px solid var(--border-strong)',
        borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-xs)', transition: 'var(--t-base)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = hue.main
        e.currentTarget.style.transform = 'translateY(-3px)'
        e.currentTarget.style.boxShadow = `0 0 0 1px ${hue.main}, 0 14px 34px -12px ${hue.ring}`
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = 'var(--border-strong)'
        e.currentTarget.style.transform = 'none'
        e.currentTarget.style.boxShadow = 'var(--shadow-xs)'
      }}
    >
      {/* Hue accent bar across the top */}
      <div aria-hidden style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 4, background: hue.main }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 13 }}>
        {mark}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 0 }}>
          <span style={{ fontSize: 24, fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)', letterSpacing: 'var(--ls-tight)' }}>
            {t(`mode.${mode}`)}
          </span>
          <span style={{ fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)', color: hue.main }}>
            {t(`mode.${mode}_tagline`)}
          </span>
        </div>
      </div>
      <ul style={{ display: 'flex', flexDirection: 'column', gap: 10, margin: 0, padding: 0, listStyle: 'none' }}>
        <Feature text={t(`mode.${mode}_f1`)} color={hue.main} />
        <Feature text={t(`mode.${mode}_f2`)} color={hue.main} />
        <Feature text={t(`mode.${mode}_f3`)} color={hue.main} />
      </ul>
    </button>
  )
}

export default function ModeChooser() {
  const { t } = useTranslation()
  const modeChosen = useAppStore((s) => s.modeChosen)
  const setMode = useAppStore((s) => s.setMode)
  if (modeChosen) return null

  return createPortal(
    <div
      role="dialog"
      aria-modal="true"
      aria-label={t('mode.chooser_title')}
      style={{
        position: 'fixed', inset: 0, zIndex: 2000,
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16,
        background: 'var(--scrim, rgba(10, 8, 24, 0.72))', backdropFilter: 'blur(4px)',
      }}
    >
      <div
        className="glass"
        style={{
          width: 'min(680px, 100%)', display: 'flex', flexDirection: 'column', gap: 22,
          padding: 'var(--space-6)', background: 'var(--surface-glass)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-popover)', animation: 'lab-rise var(--dur-2) var(--ease-out)',
        }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <h2 style={{ margin: 0, fontSize: 26, fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)', letterSpacing: 'var(--ls-tight)' }}>
            {t('mode.chooser_title')}
          </h2>
          <p style={{ margin: 0, fontSize: 'var(--fs-body)', color: 'var(--text-muted)' }}>
            {t('mode.chooser_sub')}
          </p>
        </div>

        <div style={{ display: 'flex', gap: 14 }}>
          <Card mode="simple" mark={<SimpleMark hue={HUES.simple} />} onPick={setMode} />
          <Card mode="advanced" mark={<AdvancedMark hue={HUES.advanced} />} onPick={setMode} />
        </div>

        <p style={{ margin: 0, fontSize: 'var(--fs-label)', color: 'var(--text-muted)', textAlign: 'center' }}>
          {t('mode.chooser_hint')}
        </p>
      </div>
    </div>,
    document.body,
  )
}
