// Run details + curation modal (X6a read-only config → X7 editable curation). Every run persists its full
// TrainConfig (data/runs/<id>/config.json), so the whole reproducible recipe is recoverable below; X7 adds
// an editable header — rename/label, free-text note, experiment tag, exclude-from-analysis, and delete —
// all sidecar-only (PATCH/DELETE /api/runs/{id}). Mirrors the ParamInfo modal's chrome.

import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../../store/useAppStore'
import { deleteRun, fetchRun, patchRun } from '../../api/client'
import type { RunDetail, RunMeta, RunMetaPatch, TrainConfig } from '../../api/types'
import { formatCount } from '../../format'
import { algoLabel } from './chartMath'
import { experimentIdFromLabel } from './analysisPicker'

// The algo-specific hyperparameter block on TrainConfig (PPO/board/tag all use the shared `hyperparams`).
const HYPER_KEY: Record<string, keyof TrainConfig> = {
  neuroevolution: 'evolution',
  q_learning: 'q_learning',
  sac: 'sac',
  td3: 'td3',
  dqn: 'dqn',
  a2c: 'a2c',
  qrdqn: 'qrdqn',
  alphazero: 'alphazero',
}

function fmtVal(v: unknown): string {
  if (v == null) return '—'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') {
    if (Number.isInteger(v)) return String(v)
    return Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(1) : String(v)
  }
  if (Array.isArray(v)) return `[${v.map(fmtVal).join(', ')}]`
  return String(v)
}

const inputStyle: React.CSSProperties = {
  width: '100%', boxSizing: 'border-box', padding: '6px 8px',
  background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-sm)', color: 'var(--text-default)', fontSize: 13,
  fontFamily: 'var(--font-sans)',
}

export default function RunConfigModal({
  run,
  envName,
  onClose,
  onChanged,
  onDeleted,
}: {
  run: RunMeta
  envName: string
  onClose: () => void
  /** A curation edit persisted (X7) — the parent refreshes its run list from the returned meta. */
  onChanged?: (updated: RunMeta) => void
  /** The run was deleted (X7) — the parent drops it from the list + any selection. */
  onDeleted?: (id: string) => void
}) {
  const { t } = useTranslation()
  const locale = useAppStore((s) => s.locale)
  const closeRef = useRef<HTMLButtonElement>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [failed, setFailed] = useState(false)

  // ── X7 curation edit state (seeded from the run; a partial PATCH sends only what changed) ──
  const [label, setLabel] = useState(run.label)
  const [note, setNote] = useState(run.note ?? '')
  const [experiment, setExperiment] = useState(run.experiment_label ?? '')
  const [excluded, setExcluded] = useState(run.excluded ?? false)
  const [busy, setBusy] = useState(false)
  const [saveError, setSaveError] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)

  const dirty =
    label !== run.label ||
    note !== (run.note ?? '') ||
    experiment !== (run.experiment_label ?? '') ||
    excluded !== (run.excluded ?? false)

  const save = async () => {
    // Diff each editable field against the run; send only what changed (server applies a partial patch).
    const patch: RunMetaPatch = {}
    if (label !== run.label) patch.label = label.trim() || run.label
    if (note !== (run.note ?? '')) patch.note = note.trim() || null
    if (experiment !== (run.experiment_label ?? '')) {
      const name = experiment.trim()
      patch.experiment_label = name || null
      patch.experiment_id = experimentIdFromLabel(name) // same name → same id, so runs group together
    }
    if (excluded !== (run.excluded ?? false)) patch.excluded = excluded
    setBusy(true)
    setSaveError(false)
    try {
      const updated = await patchRun(run.id, patch)
      onChanged?.(updated)
      onClose()
    } catch {
      setSaveError(true)
      setBusy(false)
    }
  }

  const doDelete = async () => {
    setBusy(true)
    try {
      await deleteRun(run.id)
      onDeleted?.(run.id)
      onClose()
    } catch {
      setSaveError(true)
      setBusy(false)
    }
  }

  // Mounted fresh per run (the parent keys this on run.id), so no in-effect reset is needed — just fetch.
  useEffect(() => {
    let cancelled = false
    void fetchRun(run.id)
      .then((d) => { if (!cancelled) setDetail(d) })
      .catch(() => { if (!cancelled) setFailed(true) })
    return () => { cancelled = true }
  }, [run.id])

  useEffect(() => {
    closeRef.current?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  // The reproducible recipe: fixed run facts + the active algorithm's hyperparameter block (plus the
  // self-play block when a competitive run carries one), flattened to label/value rows.
  const hyperGroups = useMemo(() => {
    const cfg = detail?.config
    if (!cfg) return []
    const groups: { key: string; rows: [string, unknown][] }[] = []
    const push = (key: keyof TrainConfig) => {
      const obj = cfg[key]
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
        groups.push({ key: String(key), rows: Object.entries(obj as unknown as Record<string, unknown>) })
      }
    }
    push(HYPER_KEY[cfg.algo] ?? 'hyperparams')
    if (cfg.self_play) push('self_play')
    return groups
  }, [detail])

  const runRows: [string, string][] = useMemo(() => {
    const rows: [string, string][] = [
      [t('analysis.param_game'), envName],
      [t('analysis.param_algo'), algoLabel(t, run.algo)],
      [t('sidebar.seed'), String(run.seed)],
      [t('analysis.param_budget'), formatCount(run.total_timesteps || run.timesteps)],
    ]
    if (run.experiment_label) rows.push([t('analysis.param_experiment'), run.experiment_label])
    if (run.final_reward != null) rows.push([t('analysis.param_final'), run.final_reward.toFixed(1)])
    rows.push([t('analysis.param_finished'), new Date(run.finished_at).toLocaleString(locale)])
    return rows
  }, [t, envName, run, locale])

  return createPortal(
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 'var(--z-modal, 1000)',
        background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('analysis.run_params')}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 460, maxHeight: '82vh', overflowY: 'auto',
          background: 'var(--surface-1)', color: 'var(--text-default)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-popover)',
        }}
      >
        {/* header */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '12px 16px', borderBottom: '1px solid var(--border-default)',
          position: 'sticky', top: 0, background: 'var(--surface-1)',
        }}>
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-strong)' }}>
            {t('analysis.run_details')} · {envName}
          </span>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label={t('info.close')}
            style={{
              width: 24, height: 24, padding: 0, lineHeight: 1,
              border: '1px solid var(--border-default)', borderRadius: 6,
              background: 'var(--surface-2)', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 14,
            }}
          >
            ✕
          </button>
        </div>

        {/* body */}
        <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* X7 — editable curation */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
              {t('analysis.manage_title')}
            </div>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{t('analysis.manage_label')}</span>
              <input value={label} onChange={(e) => setLabel(e.target.value)} style={inputStyle}
                aria-label={t('analysis.manage_label')} />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{t('analysis.manage_note')}</span>
              <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2}
                placeholder={t('analysis.manage_note_ph')} aria-label={t('analysis.manage_note')}
                style={{ ...inputStyle, resize: 'vertical', minHeight: 40 }} />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{t('analysis.manage_experiment')}</span>
              <input value={experiment} onChange={(e) => setExperiment(e.target.value)} style={inputStyle}
                placeholder={t('analysis.manage_experiment_ph')} aria-label={t('analysis.manage_experiment')} />
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 13, color: 'var(--text-default)' }}>
              <input type="checkbox" checked={excluded} onChange={(e) => setExcluded(e.target.checked)}
                aria-label={t('analysis.manage_exclude')} />
              <span>
                {t('analysis.manage_exclude')}
                <span style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)' }}>{t('analysis.manage_exclude_hint')}</span>
              </span>
            </label>
            {saveError && (
              <p style={{ margin: 0, fontSize: 12.5, color: 'var(--danger)' }}>{t('analysis.manage_failed')}</p>
            )}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <button type="button" onClick={() => void save()} disabled={busy || !dirty}
                style={{
                  padding: '6px 14px', borderRadius: 'var(--radius-sm)', border: 'none', cursor: busy || !dirty ? 'not-allowed' : 'pointer',
                  background: dirty ? 'var(--accent)' : 'var(--surface-3)', color: dirty ? 'var(--text-on-accent)' : 'var(--text-muted)',
                  fontSize: 13, fontWeight: 'var(--fw-semibold)', opacity: busy ? 0.6 : 1,
                }}>
                {t('analysis.manage_save')}
              </button>
              <div style={{ flex: 1 }} />
              {confirmDelete ? (
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 12.5, color: 'var(--danger)' }}>{t('analysis.manage_delete_confirm')}</span>
                  <button type="button" onClick={() => void doDelete()} disabled={busy}
                    style={{ padding: '5px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--danger)', background: 'var(--danger)', color: '#fff', fontSize: 12.5, cursor: 'pointer' }}>
                    {t('analysis.manage_confirm_yes')}
                  </button>
                  <button type="button" onClick={() => setConfirmDelete(false)} disabled={busy}
                    style={{ padding: '5px 10px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--text-muted)', fontSize: 12.5, cursor: 'pointer' }}>
                    {t('analysis.manage_confirm_no')}
                  </button>
                </span>
              ) : (
                <button type="button" onClick={() => setConfirmDelete(true)} disabled={busy}
                  style={{ padding: '6px 12px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-default)', background: 'transparent', color: 'var(--danger)', fontSize: 13, cursor: 'pointer' }}>
                  {t('analysis.manage_delete')}
                </button>
              )}
            </div>
          </div>

          <div style={{ height: 1, background: 'var(--border-default)' }} />

          <KVSection title={t('analysis.param_run_section')} rows={runRows} />

          {failed && (
            <p style={{ margin: 0, fontSize: 13, color: 'var(--danger)' }}>{t('analysis.run_params_failed')}</p>
          )}
          {!detail && !failed && (
            <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>{t('analysis.run_params_loading')}</p>
          )}

          {hyperGroups.map((g) => (
            <KVSection
              key={g.key}
              title={t('analysis.param_hyper_section')}
              rows={g.rows.filter(([, v]) => typeof v !== 'object' || v === null).map(([k, v]) => [k, fmtVal(v)])}
            />
          ))}
        </div>
      </div>
    </div>,
    document.body,
  )
}

function KVSection({ title, rows }: { title: string; rows: [string, string][] }) {
  if (rows.length === 0) return null
  return (
    <div>
      <div style={{
        fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.06em',
        color: 'var(--text-muted)', marginBottom: 6,
      }}>
        {title}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {rows.map(([k, v]) => (
          <div key={k} style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12 }}>
            <span style={{ fontSize: 12.5, color: 'var(--text-muted)', wordBreak: 'break-word' }}>{k}</span>
            <span style={{
              fontSize: 12.5, fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
              color: 'var(--text-strong)', textAlign: 'right', wordBreak: 'break-word',
            }}>
              {v}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
