import { useId, useLayoutEffect, useRef, useState } from 'react'
import type { CSSProperties } from 'react'
import { createPortal } from 'react-dom'

// Custom dropdown replacing the native <select>: the OS draws a native select's OPEN list, so it
// can never match the Laboratory theme — this renders its own frosted-glass menu instead (same
// look as the EnvSelector flyout: --surface-glass + .glass + .menu-row rows).
//
// Accessibility mirrors the ARIA 1.2 combobox/listbox pattern the native control provided:
// the trigger is a `role="combobox"` button (aria-expanded / aria-haspopup), the menu a portal
// `role="listbox"` with `role="option"` rows, and keyboard focus STAYS on the trigger while
// `aria-activedescendant` tracks the highlighted option (ArrowUp/Down · Home/End · Enter/Space
// select · Esc closes). Tests keep addressing it as getByRole('combobox').

export interface LabOption {
  value: string
  label: string
  title?: string
}

export default function LabSelect({ value, options, onChange, ariaLabel, disabled, style, menuStyle }: {
  value: string
  options: LabOption[]
  onChange: (value: string) => void
  ariaLabel: string
  disabled?: boolean
  /** Layout overrides for the trigger (width/height/font); visuals come from `.lab-trigger`. */
  style?: CSSProperties
  /** Optional overrides for the menu panel (e.g. a wider fixed width). */
  menuStyle?: CSSProperties
}) {
  const menuId = useId()
  const [open, setOpen] = useState(false)
  const [rect, setRect] = useState<DOMRect | null>(null)
  const [activeIdx, setActiveIdx] = useState(0)
  const triggerRef = useRef<HTMLButtonElement | null>(null)
  const menuRef = useRef<HTMLDivElement | null>(null)

  const selectedIdx = Math.max(0, options.findIndex((o) => o.value === value))
  const selected = options[selectedIdx]

  function openMenu() {
    if (disabled || options.length === 0) return
    setActiveIdx(selectedIdx)
    setRect(triggerRef.current?.getBoundingClientRect() ?? null)
    setOpen(true)
  }

  function pick(idx: number) {
    const opt = options[idx]
    if (opt && opt.value !== value) onChange(opt.value)
    setOpen(false)
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (disabled) return
    if (!open) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
        e.preventDefault()
        openMenu()
      }
      return
    }
    switch (e.key) {
      case 'ArrowDown': e.preventDefault(); setActiveIdx((i) => Math.min(options.length - 1, i + 1)); break
      case 'ArrowUp':   e.preventDefault(); setActiveIdx((i) => Math.max(0, i - 1)); break
      case 'Home':      e.preventDefault(); setActiveIdx(0); break
      case 'End':       e.preventDefault(); setActiveIdx(options.length - 1); break
      case 'Enter': case ' ': e.preventDefault(); pick(activeIdx); break
      case 'Escape':    e.preventDefault(); setOpen(false); break
      case 'Tab':       setOpen(false); break
    }
  }

  // Keep the menu glued to the trigger on scroll/resize; close on outside click (same pattern as
  // the EnvSelector flyout). Keyboard focus never leaves the trigger (activedescendant pattern).
  useLayoutEffect(() => {
    if (!open) return
    const reposition = () => setRect(triggerRef.current?.getBoundingClientRect() ?? null)
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node
      if (!triggerRef.current?.contains(t) && !menuRef.current?.contains(t)) setOpen(false)
    }
    window.addEventListener('resize', reposition)
    window.addEventListener('scroll', reposition, true)
    document.addEventListener('mousedown', onDoc)
    return () => {
      window.removeEventListener('resize', reposition)
      window.removeEventListener('scroll', reposition, true)
      document.removeEventListener('mousedown', onDoc)
    }
  }, [open])

  // Keep the highlighted option in view while arrowing through a long (scrollable) list.
  useLayoutEffect(() => {
    if (!open) return
    menuRef.current?.querySelector(`[data-idx="${activeIdx}"]`)?.scrollIntoView?.({ block: 'nearest' })
  }, [open, activeIdx])

  // Flip upward when the viewport space below the trigger can't fit the menu.
  const MENU_MAX_H = 280
  const openUp = rect ? rect.bottom + MENU_MAX_H + 8 > window.innerHeight && rect.top > MENU_MAX_H : false

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        role="combobox"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        aria-activedescendant={open ? `${menuId}-${activeIdx}` : undefined}
        disabled={disabled || options.length === 0}
        onClick={() => (open ? setOpen(false) : openMenu())}
        onKeyDown={onKeyDown}
        className="lab-trigger"
        style={{
          display: 'inline-flex', alignItems: 'center', justifyContent: 'space-between', gap: 7,
          height: 'var(--control-sm)', padding: '0 9px 0 10px',
          borderRadius: 'var(--radius-md)', color: 'var(--text-strong)',
          fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-label)', textAlign: 'left',
          cursor: disabled || options.length === 0 ? 'default' : 'pointer',
          // Dim when locked (e.g. a run holds the config) so a disabled dropdown reads as inert,
          // matching the dimmed sliders/seed — otherwise env/algo/steps stay bright while sliders fade.
          opacity: disabled || options.length === 0 ? 0.5 : 1,
          minWidth: 0,
          ...style,
        }}
      >
        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {selected?.label ?? ''}
        </span>
        <span aria-hidden style={{
          color: 'var(--text-faint)', flexShrink: 0, display: 'inline-flex',
          transform: open ? 'rotate(180deg)' : 'none',
          transition: 'transform var(--dur-2) var(--ease-out)',
        }}>
          <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
            <path d="M1 1l4 4 4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
      </button>

      {open && rect && createPortal(
        <div
          ref={menuRef}
          id={menuId}
          role="listbox"
          aria-label={ariaLabel}
          className="glass"
          // Selecting an option must not blur the trigger first (focus stays for the
          // activedescendant pattern), so suppress the menu's default mousedown behaviour.
          onMouseDown={(e) => e.preventDefault()}
          style={{
            position: 'fixed', zIndex: 1200,
            ...(openUp
              ? { bottom: window.innerHeight - rect.top + 4 }
              : { top: rect.bottom + 4 }),
            left: Math.min(rect.left, Math.max(8, window.innerWidth - Math.max(rect.width, 160) - 8)),
            minWidth: Math.max(rect.width, 120), maxWidth: 320,
            maxHeight: MENU_MAX_H, overflowY: 'auto', padding: 5,
            display: 'flex', flexDirection: 'column',
            background: 'var(--surface-glass)', border: '1px solid var(--border-default)',
            borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-popover)',
            animation: 'lab-rise var(--dur-2) var(--ease-out)',
            ...menuStyle,
          }}
        >
          {options.map((o, i) => {
            const isSel = o.value === value
            const isActive = i === activeIdx
            return (
              <div
                key={o.value}
                id={`${menuId}-${i}`}
                data-idx={i}
                role="option"
                aria-selected={isSel}
                title={o.title}
                onMouseEnter={() => setActiveIdx(i)}
                onClick={() => pick(i)}
                className={`menu-row${isSel ? ' is-selected' : isActive ? ' is-active' : ''}`}
                style={{
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8,
                  minHeight: 'var(--control-sm)', padding: '0 10px', borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-label)',
                  color: isSel ? 'var(--accent)' : isActive ? 'var(--text-strong)' : 'var(--text-default)',
                  whiteSpace: 'nowrap', flexShrink: 0, transition: 'var(--t-colors)',
                }}
              >
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{o.label}</span>
                {isSel && <span aria-hidden style={{ fontSize: 11, flexShrink: 0 }}>✓</span>}
              </div>
            )
          })}
        </div>,
        document.body,
      )}
    </>
  )
}
