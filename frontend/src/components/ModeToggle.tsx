// #2b — the Simple ⇆ Advanced audience switch. A small segmented control in the TopBar (next to the
// language/theme toggles) so it's always in reach whichever mode you're in. Simple = the guided
// "arcade" scene for newcomers; Advanced = the full scientist UI. The choice persists (store).

import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { AudienceMode } from '../store/useAppStore'
import { MarkFor } from './modeMarks'

export default function ModeToggle() {
  const { t } = useTranslation()
  const mode = useAppStore((s) => s.mode)
  const setMode = useAppStore((s) => s.setMode)

  const opts: { id: AudienceMode; label: string; title: string }[] = [
    { id: 'simple',   label: t('mode.simple'),   title: t('mode.simple_desc') },
    { id: 'advanced', label: t('mode.advanced'), title: t('mode.advanced_desc') },
  ]

  return (
    <div
      role="tablist"
      aria-label={t('mode.toggle_aria')}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 2, padding: 3,
        background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-pill)', flexShrink: 0,
      }}
    >
      {opts.map((o) => {
        const active = mode === o.id
        return (
          <button
            key={o.id}
            role="tab"
            aria-selected={active}
            title={o.title}
            onClick={() => setMode(o.id)}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              height: 28, padding: '0 12px 0 8px', border: 'none', borderRadius: 'var(--radius-pill)',
              cursor: 'pointer', whiteSpace: 'nowrap',
              fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-label)',
              fontWeight: active ? 'var(--fw-semibold)' : 'var(--fw-medium)',
              background: active ? 'var(--surface-2)' : 'transparent',
              color: active ? 'var(--text-strong)' : 'var(--text-muted)',
              boxShadow: active ? 'var(--ring-inset), var(--shadow-xs)' : 'none',
              // The inactive icon greys back so the active mode's mark reads as the current one.
              transition: 'var(--t-colors)',
            }}
          >
            <span style={{ display: 'inline-flex', opacity: active ? 1 : 0.6, transition: 'var(--t-colors)' }}>
              <MarkFor mode={o.id} size={18} />
            </span>
            {o.label}
          </button>
        )
      })}
    </div>
  )
}
