// #1 — the guided tour. A hand-rolled driver-style overlay (no new dependency): it spotlights one
// real UI anchor at a time and floats a copy card beside it, with Back / Next / Skip and a step
// counter. It auto-opens once on first launch (after a mode is picked) and can be re-opened anytime
// from the TopBar tour button. Portalled to <body> so a `transform` ancestor can't collapse the
// fixed overlay ([[reference_transform_breaks_fixed_modal]]).
//
// Mode-aware: the visible step list is (steps allowed in the current mode) ∩ (anchors actually
// rendered right now), computed when the tour opens — so Simple mode gets a shorter tour for free
// (no hyperparameters / Data Lab anchors exist), and a slot that self-gated to null (e.g. the skill
// meter while idle) is skipped rather than spotlighting empty space.
//
// Layers (all fixed): a transparent click-catcher swallows app clicks so the layout can't shift
// under the tour; a spotlight box darkens everything except the padded anchor rect via a huge
// box-shadow (the classic "hole" trick) and rings it in the accent; the card sits on top.

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { stepsForFlow, stepInMode, type TourStep } from '../content/tourSteps'

const SCRIM = 'rgba(6, 7, 15, 0.62)'
const PAD = 8      // spotlight padding around the anchor
const GAP = 14     // card ↔ anchor gap
const MARGIN = 12  // viewport clamp
const CARD_W = 340
const Z = 1800

type Pos = { top: number; left: number }
type Side = 'top' | 'bottom' | 'left' | 'right'

// Is this anchor actually on screen right now (rendered + non-zero size)? Centered steps (no target)
// always qualify.
function anchorRect(sel?: string): DOMRect | null {
  if (!sel) return null
  const el = document.querySelector(sel) as HTMLElement | null
  if (!el) return null
  const r = el.getBoundingClientRect()
  return r.width > 0 && r.height > 0 ? r : null
}
function isPresent(step: TourStep): boolean {
  return !step.target || anchorRect(step.target) !== null
}

// Choose a card position: preferred side if it fits, else the first side that does, then clamp to
// the viewport. Centered (no rect) → middle of the screen.
function place(rect: DOMRect | null, cardW: number, cardH: number, prefer: Side): Pos {
  const vw = window.innerWidth
  const vh = window.innerHeight
  if (!rect) return { top: Math.max(MARGIN, (vh - cardH) / 2), left: Math.max(MARGIN, (vw - cardW) / 2) }

  const fits: Record<Side, boolean> = {
    bottom: rect.bottom + GAP + cardH <= vh - MARGIN,
    top:    rect.top - GAP - cardH >= MARGIN,
    right:  rect.right + GAP + cardW <= vw - MARGIN,
    left:   rect.left - GAP - cardW >= MARGIN,
  }
  const side: Side = fits[prefer] ? prefer : (['bottom', 'top', 'right', 'left'] as Side[]).find((s) => fits[s]) ?? prefer

  let top: number
  let left: number
  if (side === 'bottom')      { top = rect.bottom + GAP; left = rect.left }
  else if (side === 'top')    { top = rect.top - GAP - cardH; left = rect.left }
  else if (side === 'right')  { left = rect.right + GAP; top = rect.top }
  else                        { left = rect.left - GAP - cardW; top = rect.top }

  return {
    left: Math.max(MARGIN, Math.min(left, vw - cardW - MARGIN)),
    top:  Math.max(MARGIN, Math.min(top, vh - cardH - MARGIN)),
  }
}

export default function Tour() {
  const { t } = useTranslation()
  const mode          = useAppStore((s) => s.mode)
  const tourOpen      = useAppStore((s) => s.tourOpen)
  const tourFlow      = useAppStore((s) => s.tourFlow)
  const modeChosen    = useAppStore((s) => s.modeChosen)
  const tourSeen      = useAppStore((s) => s.tourSeen)
  const datalabTourSeen = useAppStore((s) => s.datalabTourSeen)
  const analysisOpen  = useAppStore((s) => s.analysisOpen)
  const envsLoaded    = useAppStore((s) => s.envs.length > 0)
  const startTour     = useAppStore((s) => s.startTour)
  const closeTour     = useAppStore((s) => s.closeTour)
  const idx           = useAppStore((s) => s.tourStep)
  const setTourStep   = useAppStore((s) => s.setTourStep)

  const [rect, setRect] = useState<DOMRect | null>(null)
  const [pos, setPos] = useState<Pos>({ top: 80, left: 80 })
  const cardRef = useRef<HTMLDivElement>(null)
  const nextRef = useRef<HTMLButtonElement>(null)

  // Auto-open the DASHBOARD tour once: after the user has picked a mode (chooser dismissed) and the
  // dashboard has rendered its anchors, open it a single time. `tourSeen` (persisted, set on close)
  // keeps it from ever re-opening on its own; the TopBar button re-opens it on demand. Skip while the
  // Data Lab is open (that surface has its own tour).
  const autoTried = useRef(false)
  useEffect(() => {
    if (!autoTried.current && modeChosen && !tourSeen && envsLoaded && !tourOpen && !analysisOpen) {
      autoTried.current = true
      startTour('dashboard')
    }
  }, [modeChosen, tourSeen, envsLoaded, tourOpen, analysisOpen, startTour])

  // Auto-open the DATA LAB tour once, the first time the analysis surface is opened (its anchors only
  // exist while it's open). Its own `datalabTourSeen` flag gates the auto-open; the header button
  // re-opens it on demand.
  const autoTriedDl = useRef(false)
  useEffect(() => {
    if (!autoTriedDl.current && analysisOpen && !datalabTourSeen && !tourOpen) {
      autoTriedDl.current = true
      startTour('datalab')
    }
  }, [analysisOpen, datalabTourSeen, tourOpen, startTour])

  // The step list for THIS run — the active flow's steps, allowed in the current mode and currently on
  // screen. Frozen while the tour is open (the click-catcher blocks interaction, so the DOM can't
  // shift under it).
  const steps = useMemo(
    () => (tourOpen ? stepsForFlow(tourFlow).filter((s) => stepInMode(s, mode) && isPresent(s)) : []),
    [tourOpen, tourFlow, mode],
  )
  const step = steps[idx]

  // Measure the current anchor + position the card. Runs before paint (useLayoutEffect) so the card
  // lands correctly on the first frame; re-runs on step change and on resize.
  const measure = useCallback(() => {
    if (!step) return
    const el = step.target ? (document.querySelector(step.target) as HTMLElement | null) : null
    if (el) el.scrollIntoView({ block: 'nearest', inline: 'nearest' })
    const r = el ? el.getBoundingClientRect() : null
    setRect(r)
    const card = cardRef.current
    const cardH = card?.offsetHeight ?? 180
    const cardW = card?.offsetWidth ?? CARD_W
    setPos(place(r, cardW, cardH, step.placement ?? 'bottom'))
  }, [step])

  useLayoutEffect(() => {
    if (!tourOpen) return
    // Imperative DOM measurement → card position: the intended "sync from an external system" use of
    // an effect (mirrors PlayScoreGate / the popover positioners), not derivable during render.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    measure()
  }, [tourOpen, idx, measure])

  useEffect(() => {
    if (!tourOpen) return
    const onResize = () => measure()
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [tourOpen, measure])

  const atFirst = idx <= 0
  const atLast = idx >= steps.length - 1

  const next = useCallback(() => {
    if (atLast) closeTour()
    else setTourStep(Math.min(idx + 1, steps.length - 1))
  }, [atLast, closeTour, setTourStep, idx, steps.length])
  const back = useCallback(() => setTourStep(Math.max(0, idx - 1)), [setTourStep, idx])

  // Keyboard: Esc closes, arrows navigate. Focus the primary button on each step for keyboard flow.
  // Capture phase + stopImmediatePropagation so that, when the Data Lab tour is running, Esc closes
  // the tour without ALSO reaching the analysis surface's own Esc-to-close handler (which would
  // unmount the tour's anchors mid-run).
  useEffect(() => {
    if (!tourOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); closeTour() }
      else if (e.key === 'ArrowRight') { e.preventDefault(); next() }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); back() }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [tourOpen, next, back, closeTour])

  useEffect(() => { if (tourOpen) nextRef.current?.focus() }, [tourOpen, idx])

  if (!tourOpen || steps.length === 0 || !step) return null

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={t(tourFlow === 'datalab' ? 'tour.datalab_title' : 'tour.title')}>
      {/* Click-catcher — swallows clicks so the app underneath can't be changed mid-tour. */}
      <div style={{ position: 'fixed', inset: 0, zIndex: Z, cursor: 'default' }} />

      {/* Spotlight (anchored) or a plain scrim (centered welcome/finish). */}
      {rect ? (
        <div
          aria-hidden
          style={{
            position: 'fixed',
            top: rect.top - PAD, left: rect.left - PAD,
            width: rect.width + PAD * 2, height: rect.height + PAD * 2,
            borderRadius: 'var(--radius-lg)',
            boxShadow: `0 0 0 9999px ${SCRIM}, 0 0 0 2px var(--accent), 0 10px 30px -6px rgba(0,0,0,0.5)`,
            pointerEvents: 'none', zIndex: Z + 1,
            transition: 'top var(--dur-2) var(--ease-out), left var(--dur-2) var(--ease-out), width var(--dur-2) var(--ease-out), height var(--dur-2) var(--ease-out)',
          }}
        />
      ) : (
        <div aria-hidden style={{ position: 'fixed', inset: 0, background: SCRIM, pointerEvents: 'none', zIndex: Z + 1 }} />
      )}

      {/* Card */}
      <div
        ref={cardRef}
        className="glass"
        style={{
          position: 'fixed', top: pos.top, left: pos.left, zIndex: Z + 2,
          width: `min(${CARD_W}px, calc(100vw - 24px))`,
          display: 'flex', flexDirection: 'column', gap: 12,
          padding: 'var(--space-5)',
          background: 'var(--surface-glass)', border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-popover)',
          animation: 'lab-rise var(--dur-2) var(--ease-out)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
          <span style={{
            fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-eyebrow)',
            textTransform: 'uppercase', color: 'var(--accent)',
          }}>
            {t('tour.step_counter', { n: idx + 1, total: steps.length })}
          </span>
          <button
            type="button"
            onClick={closeTour}
            aria-label={t('tour.close_aria')}
            style={{
              width: 24, height: 24, padding: 0, lineHeight: 1, fontSize: 14,
              border: '1px solid var(--border-default)', borderRadius: 6,
              background: 'var(--surface-2)', color: 'var(--text-muted)', cursor: 'pointer',
            }}
          >
            ✕
          </button>
        </div>

        <h3 style={{ margin: 0, fontSize: 'var(--fs-h3, 17px)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)', letterSpacing: 'var(--ls-tight)' }}>
          {t(`tour.step_${step.id}_title`)}
        </h3>
        <p style={{ margin: 0, fontSize: 'var(--fs-body)', color: 'var(--text-default)', lineHeight: 1.55 }}>
          {t(`tour.step_${step.id}_body`)}
        </p>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 2 }}>
          <button
            type="button"
            onClick={closeTour}
            style={{
              background: 'none', border: 'none', padding: 0, cursor: 'pointer',
              fontSize: 'var(--fs-sm)', color: 'var(--text-muted)', fontWeight: 'var(--fw-medium)',
            }}
          >
            {t('tour.skip')}
          </button>
          <div style={{ flex: 1 }} />
          {!atFirst && (
            <button
              type="button"
              onClick={back}
              style={{
                height: 32, padding: '0 14px', cursor: 'pointer',
                background: 'var(--surface-2)', border: '1px solid var(--border-default)',
                borderRadius: 'var(--radius-md)', color: 'var(--text-default)',
                fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-medium)',
              }}
            >
              {t('tour.back')}
            </button>
          )}
          <button
            ref={nextRef}
            type="button"
            onClick={next}
            className="btn-cta"
            style={{
              height: 32, padding: '0 16px', cursor: 'pointer',
              background: 'var(--accent-grad)', color: 'var(--accent-contrast)', border: '1px solid transparent',
              borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-cta)',
              fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)',
            }}
          >
            {atLast ? t('tour.done') : t('tour.next')}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
