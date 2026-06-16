import { useState, type CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'

// Live CPU/GPU telemetry, floated bottom-right inside the chart panel (G4b). Reads the 1 Hz
// `hwstats` frame the training manager broadcasts for *any* active run (store.lastHwStats) — so it
// shows for every algorithm (PPO / neuroevolution / Q-learning), not just PPO. GPU column is hidden
// for CPU-only envs (scrub, not zero); unavailable numeric fields render `—`, never 0. The panel is
// translucent (the chart shows through) with opaque readings. Pure UI — no zustand state for the toggle.

const mono: CSSProperties = {
  fontFamily: 'var(--font-mono)',
  fontFeatureSettings: 'var(--ff-tabular)',
  letterSpacing: 'var(--ls-tight)',
  color: 'var(--text-default)',
}

function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? '—' : `${Math.round(v)}%`
}
function gb(usedMb: number | null | undefined, totalMb: number | null | undefined): string {
  if (usedMb === null || usedMb === undefined || totalMb === null || totalMb === undefined) return '—'
  return `${(usedMb / 1000).toFixed(1)}/${(totalMb / 1000).toFixed(1)} GB`
}
function unit(v: number | null | undefined, suffix: string): string {
  return v === null || v === undefined ? '—' : `${Math.round(v)}${suffix}`
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, whiteSpace: 'nowrap' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={mono}>{value}</span>
    </div>
  )
}

function Column({ header, children }: { header: string; children: React.ReactNode }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 96 }}>
      <span style={{
        color: 'var(--text-muted)', fontSize: 'var(--fs-micro)', fontWeight: 'var(--fw-semibold)',
        letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase',
      }}>
        {header}
      </span>
      {children}
    </div>
  )
}

const EyeIcon = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
  </svg>
)
const EyeOffIcon = (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M9.9 5.2A10.6 10.6 0 0112 5c6.5 0 10 7 10 7a17 17 0 01-3 3.8M6.2 6.2A17 17 0 002 12s3.5 7 10 7a10.6 10.6 0 004.1-.8M3 3l18 18"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

export default function HwStats() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(true)
  const trainState = useAppStore((s) => s.trainState)
  const hw = useAppStore((s) => s.lastHwStats)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const envs = useAppStore((s) => s.envs)

  // Only while a run is live (matches the EnvPreview badge: the manager streams telemetry through
  // the 'stopping' wind-down too, until the run actually ends).
  const active = trainState === 'running' || trainState === 'paused' || trainState === 'stopping'
  if (!active || !hw) return null

  const showGpu = envs.find((e) => e.id === selectedEnvId)?.hw_requirement === 'gpu'

  const toggle = (
    <button
      onClick={() => setOpen((v) => !v)}
      aria-label={open ? t('hwstats.toggle_hide') : t('hwstats.toggle_show')}
      title={open ? t('hwstats.toggle_hide') : t('hwstats.toggle_show')}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        width: 20, height: 20, padding: 0, flexShrink: 0,
        background: 'transparent', border: 'none', borderRadius: 'var(--radius-sm)',
        color: 'var(--text-muted)', cursor: 'pointer',
      }}
    >
      {open ? EyeOffIcon : EyeIcon}
    </button>
  )

  return (
    <div style={{
      position: 'absolute', bottom: '1rem', right: '1rem', zIndex: 2,
      display: 'flex', alignItems: 'flex-start', gap: 10,
      // Translucent panel (lets the chart show through) but fully opaque numbers on top — a
      // semi-transparent *background colour* (not opacity, which would fade the text) + a light
      // blur so the readings stay crisp over the curve.
      background: 'color-mix(in srgb, var(--surface-2) 55%, transparent)',
      backdropFilter: 'blur(3px)', WebkitBackdropFilter: 'blur(3px)',
      border: '1px solid color-mix(in srgb, var(--border-default) 60%, transparent)',
      borderRadius: 'var(--radius-md)', padding: '0.5rem 0.75rem',
      fontSize: 'var(--fs-label)', boxShadow: 'var(--shadow-sm)',
      pointerEvents: 'auto',
    }}>
      {open && (
        <div style={{ display: 'flex', gap: 18 }}>
          <Column header={t('hwstats.cpu')}>
            <Row label={t('hwstats.util')} value={pct(hw.cpu_process_pct)} />
            <Row label={t('hwstats.ram')} value={gb(hw.ram_used_mb, hw.ram_total_mb)} />
          </Column>
          {showGpu && (
            <Column header={t('hwstats.gpu')}>
              <Row label={t('hwstats.util')} value={pct(hw.gpu_util_pct)} />
              <Row label={t('hwstats.vram')} value={gb(hw.gpu_vram_used_mb, hw.gpu_vram_total_mb)} />
              <Row label={t('hwstats.temp')} value={unit(hw.gpu_temp_c, '°C')} />
              <Row label={t('hwstats.power')} value={unit(hw.gpu_power_w, ' W')} />
            </Column>
          )}
        </div>
      )}
      {toggle}
    </div>
  )
}
