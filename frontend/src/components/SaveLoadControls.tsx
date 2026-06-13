import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import {
  checkpointExportUrl,
  deleteCheckpoint,
  fetchCheckpoints,
  loadCheckpoint,
  saveCheckpoint,
} from '../api/client'
import type { CheckpointMeta } from '../api/types'

// Compact Save / Load / Manage controls for the sidebar (under Run). The checkpoint *slots*
// are no longer shown inline in the dashboard — Load opens a quick picker, Manage opens a full
// list with export/delete. Replaces the old BottomPanels "Save / Load" panel (D1, ADR-012).

// A run has a saveable model once it has started; the backend still validates + rejects with a
// clear message if no snapshot exists yet.
const SAVEABLE = new Set(['running', 'paused', 'stopped', 'finished'])

function slotProgress(s: CheckpointMeta): { frac: number; text: string } {
  if (s.algo === 'neuroevolution') {
    const total = s.total_generations ?? 0
    const frac = total > 0 ? (s.generation ?? 0) / total : 0
    return { frac, text: `gen ${s.generation ?? 0}/${total}` }
  }
  const frac = s.total_timesteps > 0 ? s.timesteps / s.total_timesteps : 0
  const k = (n: number) => (n >= 1000 ? `${Math.round(n / 1000)}k` : String(n))
  return { frac, text: `${k(s.timesteps)}/${k(s.total_timesteps)}` }
}

function ctrlBtn(primary: boolean): CSSProperties {
  return {
    flex: 1, height: 'var(--control-sm)', display: 'inline-flex', alignItems: 'center',
    justifyContent: 'center', gap: 5, borderRadius: 'var(--radius-md)',
    border: '1px solid var(--border-default)', cursor: 'pointer',
    fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)',
    background: primary ? 'var(--accent)' : 'var(--surface-2)',
    color: primary ? 'var(--accent-contrast)' : 'var(--text-muted)',
    transition: 'var(--t-colors)',
  }
}

function SlotAction({ label, color, onClick, href }: {
  label: string
  color?: string
  onClick?: () => void
  href?: string
}) {
  const style: CSSProperties = {
    flex: 1, padding: '4px 0', textAlign: 'center',
    background: 'var(--surface-2)', color: color ?? 'var(--text-muted)',
    border: '1px solid var(--border-default)', borderRadius: 4,
    fontSize: 10, fontWeight: 600, cursor: 'pointer', textDecoration: 'none',
  }
  return href
    ? <a href={href} style={style}>{label}</a>
    : <button onClick={onClick} style={style}>{label}</button>
}

function SlotCard({ slot, manage, onLoad, onDelete }: {
  slot: CheckpointMeta
  manage: boolean
  onLoad: (s: CheckpointMeta) => void
  onDelete: (s: CheckpointMeta) => void
}) {
  const { t } = useTranslation()
  const { frac, text } = slotProgress(slot)

  const meta = (
    <>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 6 }}>
        <span style={{
          fontSize: 12, fontWeight: 600, color: 'var(--text-strong)',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }} title={slot.label}>
          {slot.label}
        </span>
        {slot.reward != null && (
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--success)', flexShrink: 0 }}>
            {slot.reward.toFixed(1)}
          </span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 6, marginTop: 3 }}>
        <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
          {slot.algo === 'neuroevolution' ? t('sidebar.algo_evo') : t('sidebar.algo_ppo')} · {t('sidebar.seed')} {slot.seed}
        </span>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>{text}</span>
      </div>
      <div style={{ height: 2, background: 'var(--border-default)', borderRadius: 1, marginTop: 5 }}>
        <div style={{ width: `${Math.min(1, frac) * 100}%`, height: '100%', background: 'var(--accent)', borderRadius: 1 }} />
      </div>
    </>
  )

  const cardStyle: CSSProperties = {
    border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
    padding: '8px 10px', marginTop: 6, background: 'var(--surface-2)',
    display: 'block', width: '100%', textAlign: 'left',
  }

  // Quick-load mode: the whole card is one big button that loads on click.
  if (!manage) {
    return (
      <button onClick={() => onLoad(slot)} style={{ ...cardStyle, cursor: 'pointer' }} title={t('saveload.load')}>
        {meta}
      </button>
    )
  }
  return (
    <div style={cardStyle}>
      {meta}
      <div style={{ display: 'flex', gap: 5, marginTop: 7 }}>
        <SlotAction label={t('saveload.load')} color="var(--accent)" onClick={() => onLoad(slot)} />
        <SlotAction label={t('saveload.export')} href={checkpointExportUrl(slot.id)} />
        <SlotAction label={t('saveload.delete')} color="var(--danger)" onClick={() => onDelete(slot)} />
      </div>
    </div>
  )
}

function Modal({ title, hint, onClose, children }: {
  title: string
  hint?: string
  onClose: () => void
  children: ReactNode
}) {
  const { t } = useTranslation()
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 50, background: 'rgba(2, 6, 23, 0.5)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        role="dialog" aria-modal="true" aria-label={title}
        style={{
          width: 'min(440px, 92vw)', maxHeight: '74vh', display: 'flex', flexDirection: 'column',
          background: 'var(--surface-1)', border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-xl)', boxShadow: 'var(--shadow-popover)', overflow: 'hidden',
        }}
      >
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 14px', borderBottom: '1px solid var(--border-default)',
        }}>
          <span style={{ fontWeight: 'var(--fw-semibold)', fontSize: 'var(--fs-body)', color: 'var(--text-strong)' }}>
            {title}
          </span>
          <button onClick={onClose} aria-label={t('info.close')} style={{
            background: 'transparent', border: 'none', cursor: 'pointer',
            color: 'var(--text-muted)', fontSize: 18, lineHeight: 1,
          }}>✕</button>
        </div>
        {hint && (
          <div style={{ padding: '8px 14px 0', fontSize: 'var(--fs-label)', color: 'var(--text-muted)' }}>{hint}</div>
        )}
        <div style={{ overflowY: 'auto', padding: '6px 14px 14px' }}>{children}</div>
      </div>
    </div>
  )
}

export default function SaveLoadControls() {
  const { t } = useTranslation()
  const trainState = useAppStore((s) => s.trainState)

  const [slots, setSlots] = useState<CheckpointMeta[]>([])
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [modal, setModal] = useState<null | 'load' | 'manage'>(null)
  const [toast, setToast] = useState<string | null>(null)

  const refresh = useCallback(() => {
    void fetchCheckpoints().then(setSlots).catch(() => {})
  }, [])
  useEffect(() => { refresh() }, [refresh])

  // Brief confirmation popup ("Saved" / "Loaded") so the action is visibly acknowledged.
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const flashToast = useCallback((msg: string) => {
    setToast(msg)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 1200)
  }, [])
  useEffect(() => () => { if (toastTimer.current) clearTimeout(toastTimer.current) }, [])

  const canSave = SAVEABLE.has(trainState)

  async function handleSave() {
    setSaving(true)
    setError(null)
    try {
      await saveCheckpoint()
      refresh()
      flashToast(t('saveload.saved'))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }

  async function handleLoad(slot: CheckpointMeta) {
    setError(null)
    try {
      const status = await loadCheckpoint(slot.id)
      // Mirror the resumed run into the sidebar so the controls match what's training.
      const st = useAppStore.getState()
      st.clearMetrics()
      st.setSelectedEnvId(slot.env_id)
      st.setAlgo(slot.algo)
      if (status.config) {
        st.setSeed(status.config.seed)
        st.setTotalTimesteps(status.config.total_timesteps)
        st.setHyperparams(status.config.hyperparams)
        if (status.config.evolution) st.setEvolutionParams(status.config.evolution)
      }
      setModal(null)
      flashToast(t('saveload.loaded'))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleDelete(slot: CheckpointMeta) {
    if (!window.confirm(t('saveload.confirm_delete', { label: slot.label }))) return
    setError(null)
    try {
      await deleteCheckpoint(slot.id)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const open = (m: 'load' | 'manage') => { refresh(); setError(null); setModal(m) }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
        <button
          onClick={handleSave}
          disabled={!canSave || saving}
          title={canSave ? undefined : t('saveload.nothing_to_save')}
          style={{ ...ctrlBtn(canSave && !saving), opacity: canSave && !saving ? 1 : 0.65, cursor: canSave && !saving ? 'pointer' : 'not-allowed' }}
        >
          {saving ? t('saveload.saving') : `＋ ${t('saveload.save')}`}
        </button>
        <button
          onClick={() => open('load')}
          disabled={slots.length === 0}
          style={{ ...ctrlBtn(false), opacity: slots.length === 0 ? 0.6 : 1, cursor: slots.length === 0 ? 'not-allowed' : 'pointer' }}
        >
          {t('saveload.load')}
        </button>
        <button onClick={() => open('manage')} style={ctrlBtn(false)}>
          {t('saveload.manage')}
        </button>
      </div>
      {error && <div style={{ fontSize: 'var(--fs-meta)', color: 'var(--danger)' }}>{error}</div>}

      {toast && (
        <div style={{
          position: 'fixed', left: '50%', bottom: 28, transform: 'translateX(-50%)', zIndex: 60,
          display: 'flex', alignItems: 'center', gap: 8, pointerEvents: 'none',
          background: 'var(--surface-1)', border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-popover)',
          padding: '8px 16px', fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)',
        }}>
          <span style={{ color: 'var(--success)' }}>✓</span> {toast}
        </div>
      )}

      {modal && (
        <Modal
          title={modal === 'load' ? t('saveload.load_title') : t('saveload.manage_title')}
          hint={modal === 'load' ? t('saveload.pick_hint') : undefined}
          onClose={() => setModal(null)}
        >
          {slots.length === 0 ? (
            <div style={{ padding: '16px 4px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 'var(--fs-label)' }}>
              {t('saveload.empty')}
            </div>
          ) : (
            slots.map((s) => (
              <SlotCard key={s.id} slot={s} manage={modal === 'manage'} onLoad={handleLoad} onDelete={handleDelete} />
            ))
          )}
        </Modal>
      )}
    </div>
  )
}
