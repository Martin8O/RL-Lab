import { useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { setFrameHandler, setPreview } from '../api/client'

const MIN_SPEED = 1
const MAX_SPEED = 20

export default function EnvPreview() {
  const { t } = useTranslation()

  const visual        = useAppStore((s) => s.visual)
  const speed         = useAppStore((s) => s.speed)
  const setVisual     = useAppStore((s) => s.setVisual)
  const setSpeed      = useAppStore((s) => s.setSpeed)
  const trainState    = useAppStore((s) => s.trainState)
  const backendStatus = useAppStore((s) => s.backendStatus)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imgRef    = useRef<HTMLImageElement | null>(null)

  const runLive    = trainState === 'running' || trainState === 'paused' || trainState === 'stopping'
  const showCanvas = visual && runLive

  // Sync persisted toggle/speed to the backend whenever it comes online (UI is source of truth).
  useEffect(() => {
    if (backendStatus !== 'online') return
    const { visual: v, speed: s } = useAppStore.getState()
    void setPreview({ visual: v, speed: s }).catch(() => {})
  }, [backendStatus])

  // Register the frame sink once: draw incoming frames straight to the canvas (bypass React).
  useEffect(() => {
    const img = new Image()
    imgRef.current = img
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      canvas.getContext('2d')?.drawImage(img, 0, 0)
    }
    setFrameHandler((frame) => {
      img.src = `data:image/jpeg;base64,${frame.image}`
    })
    return () => setFrameHandler(null)
  }, [])

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
      borderRight: '1px solid var(--border)', overflow: 'hidden',
    }}>
      {/* Header (title only — mirrors the chart's tab-bar height for top alignment) */}
      <div style={{
        padding: '7px 14px', borderBottom: '1px solid var(--border)',
        background: 'var(--surface)', flexShrink: 0,
        fontWeight: 600, fontSize: 13, color: 'var(--text-h)',
      }}>
        {t('envpreview.title')}
      </div>

      {/* Live canvas, or a graceful hint */}
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--bg)', overflow: 'hidden', padding: 12,
      }}>
        {showCanvas ? (
          <canvas
            ref={canvasRef}
            style={{
              maxWidth: '100%', maxHeight: '100%', objectFit: 'contain',
              borderRadius: 6, boxShadow: '0 1px 6px rgba(0,0,0,0.25)',
            }}
          />
        ) : (
          <span style={{ color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: '0 16px' }}>
            {hint}
          </span>
        )}
      </div>

      {/* Bottom controls — two bars mirroring the chart's stats + controls rows, so the
          preview window and the chart window line up. Always rendered for a stable height. */}
      <div style={{
        flexShrink: 0, borderTop: '1px solid var(--border)',
        background: 'var(--surface)', padding: '6px 12px',
        display: 'flex', alignItems: 'center', gap: 10, minHeight: 46,
      }}>
        <button
          onClick={toggleVisual}
          aria-pressed={visual}
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            padding: '5px 12px', borderRadius: 6, cursor: 'pointer',
            fontSize: 11, fontWeight: 600,
            background: visual ? 'var(--accent)' : 'var(--surface-2)',
            color: visual ? '#fff' : 'var(--text-muted)',
            border: `1px solid ${visual ? 'var(--accent)' : 'var(--border)'}`,
          }}
        >
          <span aria-hidden>{visual ? '👁' : '🚫'}</span>
          {t('envpreview.visual')}: {visual ? t('envpreview.on') : t('envpreview.off')}
        </button>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {visual ? t('envpreview.visual_on_hint') : t('envpreview.visual_off_hint')}
        </span>
      </div>

      <div style={{
        flexShrink: 0, borderTop: '1px solid var(--border)',
        background: 'var(--surface)', padding: '5px 12px', minHeight: 34,
        display: 'flex', alignItems: 'center', gap: 10, opacity: visual ? 1 : 0.5,
      }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t('envpreview.speed')}</span>
        <input
          type="range"
          min={MIN_SPEED} max={MAX_SPEED} step={1}
          value={speed}
          disabled={!visual}
          onChange={(e) => changeSpeed(parseInt(e.target.value, 10))}
          style={{ flex: 1, cursor: visual ? 'pointer' : 'default', accentColor: 'var(--accent)' }}
          aria-label={t('envpreview.speed')}
        />
        <span style={{
          fontSize: 11, fontFamily: 'monospace', color: 'var(--text)',
          minWidth: 30, textAlign: 'right',
        }}>
          {speed}×
        </span>
      </div>
    </section>
  )
}
