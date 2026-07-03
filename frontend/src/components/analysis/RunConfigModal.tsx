// Read-only "run parameters" detail (X6a review) — every run persists its full TrainConfig
// (data/runs/<id>/config.json), so the whole reproducible recipe (algorithm hyperparameters, seed,
// budget, experiment) is recoverable, not just the seed shown in the picker row. This modal fetches
// that config on demand and lists it, giving the same numbers the XLSX `Config` sheet + repro-card
// export already carry. Purely presentational; mirrors the ParamInfo modal's chrome.

import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../../store/useAppStore'
import { fetchRun } from '../../api/client'
import type { RunDetail, RunMeta, TrainConfig } from '../../api/types'
import { formatCount } from '../../format'
import { algoLabel } from './chartMath'

// The algo-specific hyperparameter block on TrainConfig (PPO/board/tag all use the shared `hyperparams`).
const HYPER_KEY: Record<string, keyof TrainConfig> = {
  neuroevolution: 'evolution',
  q_learning: 'q_learning',
  sac: 'sac',
  td3: 'td3',
  dqn: 'dqn',
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

export default function RunConfigModal({
  run,
  envName,
  onClose,
}: {
  run: RunMeta
  envName: string
  onClose: () => void
}) {
  const { t } = useTranslation()
  const locale = useAppStore((s) => s.locale)
  const closeRef = useRef<HTMLButtonElement>(null)
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [failed, setFailed] = useState(false)

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
            {t('analysis.run_params')} · {envName}
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
