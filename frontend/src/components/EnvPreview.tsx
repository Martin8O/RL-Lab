import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { sendPlayAction, setFrameHandler, setPlayFrameHandler, setPreview } from '../api/client'
import type { PlayFrame, PreviewFrame } from '../api/types'
import PlayControls from './PlayControls'
import SkillMeter from './SkillMeter'

const MIN_SPEED = 1
const MAX_SPEED = 20

// Envs the frontend draws itself from raw physics state (keep in sync with the backend's
// app/services/client_render.py). For these, the server streams state instead of an image.
const CLIENT_RENDER_ENVS = new Set(['cartpole'])
const CART_X_LIMIT = 2.4   // CartPole fails at |x| ≈ 2.4
const CART_X_SCALE = 250   // px of horizontal travel from ±x-limit (wide track 30..570, centre 300)

const EyeOn = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
  </svg>
)
const EyeOff = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M3 3l18 18M10.6 10.6A3 3 0 0014 14M6.9 6.9C4.2 8.5 2 12 2 12s3.5 7 10 7c2 0 3.7-.5 5.2-1.3M9.9 5.2A10 10 0 0112 5c6.5 0 10 7 10 7a17.7 17.7 0 01-2.2 3"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

export default function EnvPreview() {
  const { t } = useTranslation()

  const visual        = useAppStore((s) => s.visual)
  const speed         = useAppStore((s) => s.speed)
  const setVisual     = useAppStore((s) => s.setVisual)
  const setSpeed      = useAppStore((s) => s.setSpeed)
  const trainState    = useAppStore((s) => s.trainState)
  const backendStatus = useAppStore((s) => s.backendStatus)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const playState     = useAppStore((s) => s.playState)
  const playMode      = useAppStore((s) => s.playMode)

  const canvasRef     = useRef<HTMLCanvasElement | null>(null)
  const cartGroupRef  = useRef<SVGGElement | null>(null)   // CartPole: horizontal cart travel
  const poleGroupRef  = useRef<SVGGElement | null>(null)   // CartPole: pole angle
  // True once a frame has actually arrived this session — lets a finished session linger, but
  // falls back to the idle (centred) cart after a reload that reconciled a finished session.
  const [hasFrame, setHasFrame] = useState(false)

  const runLive      = trainState === 'running' || trainState === 'paused' || trainState === 'stopping'
  const playVisible  = playState !== 'idle'
  // "live" = frames are (or just were) flowing for this env.
  const live         = (visual && runLive) || playState === 'playing' || (playVisible && hasFrame)
  const clientRender = !!selectedEnvId && CLIENT_RENDER_ENVS.has(selectedEnvId)

  // Sync persisted toggle/speed to the backend whenever it comes online (UI is source of truth).
  useEffect(() => {
    if (backendStatus !== 'online') return
    const { visual: v, speed: s } = useAppStore.getState()
    void setPreview({ visual: v, speed: s }).catch(() => {})
  }, [backendStatus])

  // One frame sink for both training-preview and play frames. A frame carries either client
  // -render state (CartPole → drive the SVG cart, no React render) or a JPEG (→ canvas).
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      canvas.getContext('2d')?.drawImage(img, 0, 0)
    }
    const drawCart = (x: number, theta: number) => {
      const cg = cartGroupRef.current
      const pg = poleGroupRef.current
      if (cg) {
        const tx = Math.max(-CART_X_SCALE, Math.min(CART_X_SCALE, (x / CART_X_LIMIT) * CART_X_SCALE))
        cg.setAttribute('transform', `translate(${tx.toFixed(1)} 0)`)
      }
      if (pg) pg.setAttribute('transform', `rotate(${((theta * 180) / Math.PI).toFixed(2)} 300 190)`)
    }
    const onFrame = (frame: PreviewFrame | PlayFrame) => {
      if (frame.state && frame.state.length >= 2) {
        setHasFrame(true)
        drawCart(frame.state[0], frame.state[1])
      } else if (frame.image) {
        setHasFrame(true)
        img.src = `data:image/jpeg;base64,${frame.image}`
      }
    }
    setFrameHandler(onFrame)
    setPlayFrameHandler(onFrame)
    return () => { setFrameHandler(null); setPlayFrameHandler(null) }
  }, [])

  // When nothing is live, re-centre the cart (drop the last streamed transform).
  useEffect(() => {
    if (!live) {
      cartGroupRef.current?.removeAttribute('transform')
      poleGroupRef.current?.removeAttribute('transform')
    }
  }, [live])

  // Keyboard control for human play: ← / A = left (0), → / D = right (1). Latency-tolerant —
  // the backend holds the last action, so a single keydown suffices (auto-repeat is harmless).
  useEffect(() => {
    if (!(playState === 'playing' && playMode === 'human')) return
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') { sendPlayAction(0); e.preventDefault() }
      else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') { sendPlayAction(1); e.preventDefault() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [playState, playMode])

  function toggleVisual() {
    const next = !visual
    setVisual(next)
    void setPreview({ visual: next }).catch(() => {})
  }

  function changeSpeed(next: number) {
    setSpeed(next)
    void setPreview({ speed: next }).catch(() => {})
  }

  const hint = visual ? t('envpreview.idle_hint') : t('envpreview.visual_off_hint')

  return (
    <section style={{
      flex: '0 0 55%', display: 'flex', flexDirection: 'column',
      borderRight: '2px solid var(--border-default)', overflow: 'hidden',
    }}>
      {/* Header: title + visual toggle + speed (moved up so the stage grows to align with chart) */}
      <div style={{
        minHeight: 'var(--panel-head-h)', padding: '0 var(--space-4)',
        borderBottom: '1px solid var(--border-default)',
        background: 'var(--surface-1)', flexShrink: 0,
        display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
      }}>
        <span style={{ fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase', color: 'var(--text-faint)' }}>
          {t('envpreview.title')}
        </span>

        <div style={{ flex: 1 }} />

        {/* Visual on/off — quiet (surface, never a bright fill); the eye carries the accent */}
        <button
          onClick={toggleVisual}
          aria-pressed={visual}
          title={visual ? t('envpreview.visual_on_hint') : t('envpreview.visual_off_hint')}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            height: 'var(--control-sm)', padding: '0 10px',
            borderRadius: 'var(--radius-md)', cursor: 'pointer',
            fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)',
            background: 'var(--surface-2)', border: '1px solid var(--border-default)',
            color: visual ? 'var(--text-strong)' : 'var(--text-muted)', transition: 'var(--t-colors)',
          }}
        >
          <span aria-hidden style={{ display: 'inline-flex', color: visual ? 'var(--accent)' : 'var(--text-faint)' }}>
            {visual ? EyeOn : EyeOff}
          </span>
          {t('envpreview.visual')}
        </button>

        {/* Speed — compact, fixed-width track so it reads as a setting, not a control bar */}
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, opacity: visual ? 1 : 0.45 }}>
          <span style={{ fontSize: 'var(--fs-label)', color: 'var(--text-muted)' }}>{t('envpreview.speed')}</span>
          <input
            type="range"
            min={MIN_SPEED} max={MAX_SPEED} step={1}
            value={speed}
            disabled={!visual}
            onChange={(e) => changeSpeed(parseInt(e.target.value, 10))}
            style={{ width: 88, cursor: visual ? 'pointer' : 'default' }}
            aria-label={t('envpreview.speed')}
          />
          <span style={{
            fontSize: 'var(--fs-label)', fontFamily: 'var(--font-mono)',
            fontFeatureSettings: 'var(--ff-tabular)', color: 'var(--text-strong)',
            minWidth: 28, textAlign: 'right',
          }}>
            {speed}×
          </span>
        </div>
      </div>

      {/* Stage: the live SVG cart (CartPole), or a JPEG canvas for image-rendered envs. */}
      <div style={{
        position: 'relative',
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--chart-plot-bg)', overflow: 'hidden', padding: 'var(--space-5)',
      }}>
        {clientRender ? (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
            <svg viewBox="0 0 600 260" preserveAspectRatio="xMidYMid meet"
              style={{ width: '100%', maxWidth: 820, maxHeight: '100%' }} aria-label="CartPole">
              <line x1="30" y1="210" x2="570" y2="210" stroke="var(--border-strong)" strokeWidth="2.5" />
              <g ref={cartGroupRef}>
                <g ref={poleGroupRef}>
                  <line x1="300" y1="190" x2="300" y2="70" stroke="var(--accent)" strokeWidth="7" strokeLinecap="round" />
                  <circle cx="300" cy="66" r="9" fill="var(--accent)" />
                </g>
                <rect x="268" y="184" width="64" height="26" rx="5" fill="var(--surface-3)" stroke="var(--border-strong)" strokeWidth="2.5" />
                <circle cx="282" cy="214" r="6" fill="var(--text-faint)" />
                <circle cx="318" cy="214" r="6" fill="var(--text-faint)" />
              </g>
            </svg>
            {!live && (
              <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', maxWidth: 360 }}>
                {hint}
              </span>
            )}
          </div>
        ) : live ? (
          <canvas
            ref={canvasRef}
            style={{
              maxWidth: '100%', maxHeight: '100%', objectFit: 'contain',
              borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-md)',
            }}
          />
        ) : (
          <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '0 16px' }}>
            {hint}
          </span>
        )}
        {playState === 'playing' && playMode === 'human' && (
          <div style={{
            position: 'absolute', top: 12, left: 12,
            padding: '4px 10px', borderRadius: 'var(--radius-pill)',
            background: 'var(--surface-3)', border: '1px solid var(--border-default)',
            color: 'var(--text-strong)', boxShadow: 'var(--shadow-sm)',
            fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', pointerEvents: 'none',
          }}>
            ⌨ {t('play.playing_hint')}
          </div>
        )}

        {/* Skill meter floats as an overlay at the bottom of the stage (no footer row) — shown only
            while a play session is the live context, so it doesn't steal space from the panels below. */}
        <SkillMeter slot="play" overlay />
      </div>

      {/* Play vs AI (E2): controls. */}
      <PlayControls />
      <SkillMeter slot="play" />
    </section>
  )
}
