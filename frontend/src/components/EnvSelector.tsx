import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { categoryLabel, categoryOrder } from '../content/envCategories'

// Game picker as a category → games flyout: the trigger lists the current game; clicking it opens a
// panel of *categories* (the registry `family`), and hovering a category reveals its games to the
// right. Keeps the picker browseable as the catalogue grows to dozens of games. Rendered through a
// portal so the sidebar's `overflow:hidden` can't clip the flyout. (A search box on top is the next
// step once there are many games.)
export default function EnvSelector({ disabled }: { disabled?: boolean }) {
  const { t } = useTranslation()
  const envs           = useAppStore((s) => s.envs)
  const selectedEnvId  = useAppStore((s) => s.selectedEnvId)
  const locale         = useAppStore((s) => s.locale)
  const setSelectedEnvId = useAppStore((s) => s.setSelectedEnvId)

  const [open, setOpen] = useState(false)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const popoverRef = useRef<HTMLDivElement | null>(null)

  const selected = envs.find((e) => e.id === selectedEnvId)

  // Games grouped by family, families in roadmap order, only non-empty ones.
  const groups = useMemo(() => {
    const byFam = new Map<string, typeof envs>()
    for (const e of envs) byFam.set(e.family, [...(byFam.get(e.family) ?? []), e])
    return [...byFam.keys()]
      .sort((a, b) => categoryOrder(a) - categoryOrder(b))
      .map((id) => ({ id, items: byFam.get(id)! }))
  }, [envs])

  const [activeCat, setActiveCat] = useState<string | null>(null)
  const activeItems = groups.find((g) => g.id === activeCat)?.items ?? []

  function openMenu() {
    if (disabled || envs.length === 0) return
    setActiveCat(selected?.family ?? groups[0]?.id ?? null)
    setRect(triggerRef.current?.getBoundingClientRect() ?? null)
    setOpen(true)
  }

  // Keep the flyout glued to the trigger on scroll/resize; close on outside click / Esc.
  useLayoutEffect(() => {
    if (!open) return
    const reposition = () => setRect(triggerRef.current?.getBoundingClientRect() ?? null)
    const onDoc = (e: MouseEvent) => {
      const target = e.target as Node
      if (!triggerRef.current?.contains(target) && !popoverRef.current?.contains(target)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('resize', reposition)
    window.addEventListener('scroll', reposition, true)
    document.addEventListener('mousedown', onDoc)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('resize', reposition)
      window.removeEventListener('scroll', reposition, true)
      document.removeEventListener('mousedown', onDoc)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <label style={labelStyle}>
        <span>{t('sidebar.game_selector')}</span>
        {envs.length > 0 && (
          <span
            style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}
            title={t('sidebar.game_count')}
          >
            <span>{t('sidebar.total')}:</span>
            <span style={countStyle} aria-label={t('sidebar.game_count')}>{envs.length}</span>
          </span>
        )}
      </label>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={disabled || envs.length === 0}
        onClick={() => (open ? setOpen(false) : openMenu())}
        onMouseEnter={(e) => { if (!disabled && envs.length > 0) { e.currentTarget.style.borderColor = 'var(--border-strong)'; e.currentTarget.style.background = 'var(--surface-3)' } }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border-default)'; e.currentTarget.style.background = 'var(--surface-2)' }}
        style={{ ...triggerStyle, cursor: disabled || envs.length === 0 ? 'default' : 'pointer' }}
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {envs.length === 0
            ? t('sidebar.loading_envs')
            : selected?.display_name[locale] ?? t('sidebar.game_selector')}
        </span>
        <span aria-hidden style={{
          color: 'var(--text-faint)', flexShrink: 0,
          display: 'inline-flex',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform var(--dur-2) var(--ease-out)',
        }}>▾</span>
      </button>

      {open && rect && createPortal(
        <div
          ref={popoverRef}
          role="menu"
          className="glass"
          style={{
            position: 'fixed', top: rect.bottom + 4, left: rect.left, zIndex: 1000,
            display: 'flex', maxWidth: 'calc(100vw - 24px)',
            background: 'var(--surface-glass)', border: '1px solid var(--border-default)',
            borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-popover)', overflow: 'hidden',
            animation: 'lab-rise var(--dur-2) var(--ease-out)',
          }}
        >
          {/* Categories */}
          <div style={{ display: 'flex', flexDirection: 'column', minWidth: 168, padding: 5, borderRight: '1px solid var(--border-default)' }}>
            {groups.map((g) => (
              <button
                key={g.id}
                role="menuitem"
                onMouseEnter={() => setActiveCat(g.id)}
                onFocus={() => setActiveCat(g.id)}
                onClick={() => setActiveCat(g.id)}
                className={g.id === activeCat ? 'menu-row is-active' : 'menu-row'}
                style={rowStyle(g.id === activeCat)}
              >
                <span style={ellipsis}>{categoryLabel(g.id)[locale]}</span>
                <span aria-hidden style={{ color: 'var(--text-faint)', fontSize: 11, flexShrink: 0 }}>{g.items.length} ›</span>
              </button>
            ))}
          </div>
          {/* Games in the hovered category */}
          <div style={{ display: 'flex', flexDirection: 'column', minWidth: 196, maxHeight: 340, overflowY: 'auto', padding: 5 }}>
            {activeItems.map((e) => (
              <button
                key={e.id}
                role="menuitem"
                onClick={() => { setSelectedEnvId(e.id); setOpen(false) }}
                className={e.id === selectedEnvId ? 'menu-row is-selected' : 'menu-row'}
                style={rowStyle(e.id === selectedEnvId, true)}
              >
                <span style={ellipsis}>{e.display_name[locale]}</span>
                {e.id === selectedEnvId && <span aria-hidden style={{ color: 'var(--accent)', fontSize: 11, flexShrink: 0 }}>✓</span>}
              </button>
            ))}
          </div>
        </div>,
        document.body,
      )}
    </div>
  )
}

const labelStyle: CSSProperties = {
  fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)', color: 'var(--text-muted)',
  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6,
}
// Total number of registered games/envs, shown beside the label as a quiet mono count pill. The
// algorithm picker mirrors this exact style for its own "Total: N" pill (Sidebar.countStyle).
const countStyle: CSSProperties = {
  fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
  fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-faint)',
  background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-pill)', padding: '0 7px', lineHeight: '16px',
}
const triggerStyle: CSSProperties = {
  width: '100%', height: 'var(--control-md)', padding: '0 12px',
  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
  background: 'var(--surface-2)', color: 'var(--text-strong)',
  border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
  fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-sm)', textAlign: 'left',
  transition: 'var(--t-colors)',
}
const ellipsis: CSSProperties = { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }

// Background + border live in the `.menu-row` CSS classes (index.css) so :hover can win — an
// inline background would override it (CSS specificity: inline beats stylesheet pseudo-classes).
function rowStyle(active: boolean, accentWhenActive = false): CSSProperties {
  return {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
    height: 'var(--control-sm)', padding: '0 10px', borderRadius: 'var(--radius-sm)',
    cursor: 'pointer', textAlign: 'left',
    fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-label)',
    color: active && accentWhenActive ? 'var(--accent)' : active ? 'var(--text-strong)' : 'var(--text-muted)',
    transition: 'var(--t-colors)',
  }
}
