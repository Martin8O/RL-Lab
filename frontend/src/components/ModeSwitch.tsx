// The top-left view switcher: two continuously-shown tabs — RL Lab (the dashboard) and DataLab (the
// analysis surface, X6). Rendered in BOTH the dashboard TopBar and the DataLab header (both read the
// same `analysisOpen` store flag, so they stay in sync). The active view is highlighted; the other
// greys out but stays visible + clickable, so it's always obvious how to go back. Each tab carries a
// brand "mark" (a rounded tile in the logo's dark-purple with white + amber glyphs) — the RL Lab
// agent↔environment loop, and a DataLab rising-curve-with-reward chart in the same palette.

import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'

function RlLabMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden style={{ display: 'block', borderRadius: 7 }}>
      <defs>
        <linearGradient id="ms-rl-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#322A63" />
          <stop offset="1" stopColor="#16122C" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#ms-rl-grad)" />
      <path d="M20.3 24.4 A14 14 0 0 1 43.7 24.4" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" />
      <path d="M43.7 39.6 A14 14 0 0 1 20.3 39.6" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" />
      <path d="M44.6 27.5 L46 23.7 L41.3 25.1 Z" fill="#F0A93A" />
      <path d="M19.4 36.5 L18 40.3 L22.7 38.9 Z" fill="#F0A93A" />
      <circle cx="32" cy="32" r="3.6" fill="#F0A93A" />
      <circle cx="30.95" cy="30.83" r="1.3" fill="#FFE0A6" />
      <path d="M46.9 15.1 L48 18 L51 19.2 L48 20.3 L46.9 23.25 L45.7 20.3 L42.8 19.2 L45.7 18 Z" fill="#FAC775" />
    </svg>
  )
}

function DataLabMark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden style={{ display: 'block', borderRadius: 7 }}>
      <defs>
        <linearGradient id="ms-dl-grad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#322A63" />
          <stop offset="1" stopColor="#16122C" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="14" fill="url(#ms-dl-grad)" />
      {/* white axes */}
      <path d="M19 16 V46 H49" stroke="#fff" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" />
      {/* amber rising trend */}
      <path d="M23 40 L31 30 L38 34 L46.5 21" stroke="#F0A93A" strokeWidth="3.6" strokeLinecap="round" strokeLinejoin="round" />
      {/* the amber reward, reached at the peak (echoes the RL Lab star) */}
      <circle cx="46.5" cy="21" r="3.8" fill="#FAC775" />
      <circle cx="45.4" cy="19.9" r="1.4" fill="#FFF3D6" />
    </svg>
  )
}

function Tab({ active, onClick, mark, label, ariaLabel }: {
  active: boolean
  onClick: () => void
  mark: React.ReactNode
  label: string
  ariaLabel: string
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      aria-label={ariaLabel}
      onClick={onClick}
      className="btn-press"
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 9, height: 40, padding: '0 13px 0 9px',
        border: 'none', borderRadius: 'var(--radius-md)', cursor: 'pointer',
        background: active ? 'linear-gradient(180deg, var(--surface-3), var(--surface-2))' : 'transparent',
        boxShadow: active ? 'var(--ring-inset), var(--shadow-xs)' : 'none',
        // The inactive tab greys out (dimmed mark + muted text) so the current view is unmistakable
        // and the other reads as the "go there / go back" affordance.
        opacity: active ? 1 : 0.5,
        filter: active ? 'none' : 'grayscale(0.35)',
        transition: 'var(--t-base)',
      }}
      onMouseEnter={(e) => { if (!active) { e.currentTarget.style.opacity = '0.8'; e.currentTarget.style.background = 'var(--surface-2)' } }}
      onMouseLeave={(e) => { if (!active) { e.currentTarget.style.opacity = '0.5'; e.currentTarget.style.background = 'transparent' } }}
    >
      {mark}
      <span style={{
        fontSize: 15, fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-tight)',
        color: active ? 'var(--text-strong)' : 'var(--text-muted)',
      }}>
        {label}
      </span>
    </button>
  )
}

// `simple` (#2b): Simple mode hides the DataLab (a researcher surface, off the newcomer spine), so the
// switcher collapses to just the RL Lab brand — the logo + title, non-interactive — keeping the header
// identity without offering a view that isn't part of Simple.
export default function ModeSwitch({ simple = false }: { simple?: boolean }) {
  const { t } = useTranslation()
  const open = useAppStore((s) => s.analysisOpen)
  const setOpen = useAppStore((s) => s.setAnalysisOpen)

  if (simple) {
    return (
      <div style={{ display: 'inline-flex', alignItems: 'center', gap: 9, padding: '0 4px', flexShrink: 0 }}>
        <RlLabMark />
        <span style={{ fontSize: 15, fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-tight)', color: 'var(--text-strong)' }}>
          {t('app.title')}
        </span>
      </div>
    )
  }

  return (
    <div
      role="tablist"
      aria-label={t('nav.switch_aria')}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 3, padding: 3,
        background: 'var(--surface-1)', border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-lg)', flexShrink: 0,
      }}
    >
      <Tab active={!open} onClick={() => setOpen(false)} mark={<RlLabMark />} label={t('app.title')} ariaLabel={t('nav.dashboard')} />
      <div style={{ width: 1, height: 22, background: 'var(--border-default)' }} />
      <span data-tour="datalab" style={{ display: 'inline-flex' }}>
        <Tab active={open} onClick={() => setOpen(true)} mark={<DataLabMark />} label={t('analysis.title')} ariaLabel={t('nav.datalab')} />
      </span>
    </div>
  )
}
