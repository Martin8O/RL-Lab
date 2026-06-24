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
import { buildTrainConfig } from '../api/trainingControls'
import type { Algo, CheckpointMeta } from '../api/types'
import { categoryLabel } from '../content/envCategories'
import { formatCount } from '../format'
import {
  DEFAULT_CKPT_FILTERS,
  checkpointFacets,
  organizeCheckpoints,
  solvedPct,
  type CkptFilters,
  type CkptGroup,
  type CkptSort,
} from './checkpointBrowser'

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
  if (s.algo === 'q_learning') {
    // Q-learning is episodic: iteration = episodes elapsed, total_generations = episode budget.
    const total = s.total_generations ?? 0
    const frac = total > 0 ? (s.iteration ?? 0) / total : 0
    return { frac, text: `${s.iteration ?? 0}/${total} ep` }
  }
  // The actual steps can slightly exceed the budget (a resumed run, or self-play's per-round rollout
  // granularity), so the denominator is at least the numerator — never show "797k/500k".
  const denom = Math.max(s.total_timesteps, s.timesteps)
  const frac = denom > 0 ? s.timesteps / denom : 0
  return { frac, text: `${formatCount(s.timesteps)}/${formatCount(denom)}` }
}

function algoLabel(t: (k: string) => string, algo: Algo): string {
  if (algo === 'neuroevolution') return t('sidebar.algo_evo')
  if (algo === 'q_learning') return t('sidebar.algo_q')
  if (algo === 'alphazero') return t('sidebar.algo_az')
  if (algo === 'sac') return t('sidebar.algo_sac')
  if (algo === 'td3') return t('sidebar.algo_td3')
  if (algo === 'dqn') return t('sidebar.algo_dqn')
  return t('sidebar.algo_ppo')  // ppo (and any unrecognised algo) reads as PPO
}

// Compact local save timestamp ("2026-06-17 14:32") from the slot's ISO created_at.
function fmtSaveDate(iso: string): string {
  const d = new Date(iso)
  if (isNaN(d.getTime())) return ''
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
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
  const locale = useAppStore((s) => s.locale)
  const envs = useAppStore((s) => s.envs)
  const { frac, text } = slotProgress(slot)

  // Headline = "Category · Game · Algo" (from the live env registry). The training amount lives only
  // in the bottom-right "steps/total" now (no longer duplicated here). When the game name already
  // contains its category (e.g. "MiniGrid Door Key" in the MiniGrid family), the category is dropped
  // so it isn't shown twice.
  const env = envs.find((e) => e.id === slot.env_id)
  const category = env ? categoryLabel(env.family)[locale] : ''
  const game = env ? env.display_name[locale] : slot.env_id
  const showCategory = !!category && !game.toLowerCase().includes(category.toLowerCase())
  const headline = [showCategory ? category : null, game, algoLabel(t, slot.algo)].filter(Boolean).join(' · ')
  const pct = env ? solvedPct(slot.reward, env.min_score, env.solved_score) : null
  const date = fmtSaveDate(slot.created_at)
  const progressPct = Math.round(Math.min(1, frac) * 100)

  const meta = (
    <>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 6 }}>
        <span style={{
          fontSize: 12, fontWeight: 600, color: 'var(--text-strong)', minWidth: 0,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }} title={headline}>
          {headline}
        </span>
        {pct != null && (
          <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--success)', flexShrink: 0 }}>
            {Math.round(pct)}% {t('saveload.solved')}
          </span>
        )}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 6, marginTop: 3 }}>
        <span style={{ fontSize: 10, color: 'var(--text-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', minWidth: 0 }}>
          {t('sidebar.seed')} {slot.seed}{date && <span style={{ opacity: 0.7 }}> · {date}</span>}
        </span>
        {/* steps/total (small, like the date) — % progress (blue, like the bar, sized like % solved) */}
        <span style={{ flexShrink: 0, fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{text}</span>
          <span style={{ fontSize: 11, color: 'var(--accent)' }}> – {progressPct}% {t('saveload.progress')}</span>
        </span>
      </div>
      <div style={{ height: 2, background: 'var(--border-default)', borderRadius: 1, marginTop: 5 }}>
        <div style={{ width: `${progressPct}%`, height: '100%', background: 'var(--accent)', borderRadius: 1 }} />
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
          width: 'min(560px, 94vw)', maxHeight: '74vh', display: 'flex', flexDirection: 'column',
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

const browserCtrl: CSSProperties = {
  height: 'var(--control-sm)', padding: '0 8px', borderRadius: 'var(--radius-md)',
  fontSize: 'var(--fs-label)', fontFamily: 'var(--font-sans)',
  background: 'var(--surface-2)', color: 'var(--text-strong)',
  border: '1px solid var(--border-default)', cursor: 'pointer', transition: 'var(--t-colors)',
}

// save/load v2 — the filter / sort / group toolbar + grouped slot list rendered inside the Load and
// Manage modals. Local, unpersisted state (the modal mounts fresh each open, so it resets to the
// newest-first default = the pre-v2 behaviour). All the data work lives in ./checkpointBrowser.
function CheckpointBrowser({ slots, manage, onLoad, onDelete }: {
  slots: CheckpointMeta[]
  manage: boolean
  onLoad: (s: CheckpointMeta) => void
  onDelete: (s: CheckpointMeta) => void
}) {
  const { t } = useTranslation()
  const locale = useAppStore((s) => s.locale)
  const envs = useAppStore((s) => s.envs)
  const [filters, setFilters] = useState<CkptFilters>(DEFAULT_CKPT_FILTERS)
  const set = <K extends keyof CkptFilters>(k: K, v: CkptFilters[K]) =>
    setFilters((f) => ({ ...f, [k]: v }))

  // Facets come from ALL slots (not the filtered set) so the dropdown options stay stable while you
  // narrow — an option never vanishes out from under the cursor.
  const facets = checkpointFacets(slots, envs)
  const groups = organizeCheckpoints(slots, envs, locale, filters)
  const total = groups.reduce((n, g) => n + g.items.length, 0)

  const groupHeader = (key: string): string => {
    if (filters.group === 'category') return categoryLabel(key)[locale]
    if (filters.group === 'game') return envs.find((e) => e.id === key)?.display_name[locale] ?? key
    if (filters.group === 'algo') return algoLabel(t, key as Algo)
    return ''
  }

  return (
    <>
      {/* Toolbar — search + category + algo filters, then sort + group. Wraps on a narrow modal. */}
      <div style={{
        display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center',
        paddingBottom: 8, marginBottom: 2, borderBottom: '1px solid var(--border-default)',
      }}>
        <input
          type="search"
          value={filters.search}
          onChange={(e) => set('search', e.target.value)}
          placeholder={t('saveload.filter_search')}
          aria-label={t('saveload.filter_search')}
          style={{ ...browserCtrl, flex: '1 1 130px', minWidth: 110, cursor: 'text' }}
        />
        <select
          value={filters.category}
          onChange={(e) => set('category', e.target.value)}
          aria-label={t('saveload.filter_category')}
          style={browserCtrl}
        >
          <option value="">{t('saveload.filter_all_categories')}</option>
          {facets.categories.map((id) => (
            <option key={id} value={id}>{categoryLabel(id)[locale]}</option>
          ))}
        </select>
        <select
          value={filters.algo}
          onChange={(e) => set('algo', e.target.value)}
          aria-label={t('saveload.filter_algo')}
          style={browserCtrl}
        >
          <option value="">{t('saveload.filter_all_algos')}</option>
          {facets.algos.map((a) => (
            <option key={a} value={a}>{algoLabel(t, a)}</option>
          ))}
        </select>
        <select
          value={filters.sort}
          onChange={(e) => set('sort', e.target.value as CkptSort)}
          aria-label={t('saveload.sort_label')}
          style={browserCtrl}
        >
          <option value="newest">{t('saveload.sort_newest')}</option>
          <option value="oldest">{t('saveload.sort_oldest')}</option>
          <option value="best">{t('saveload.sort_best')}</option>
          <option value="game">{t('saveload.sort_game')}</option>
        </select>
        <select
          value={filters.group}
          onChange={(e) => set('group', e.target.value as CkptGroup)}
          aria-label={t('saveload.group_label')}
          style={browserCtrl}
        >
          <option value="none">{t('saveload.group_none')}</option>
          <option value="category">{t('saveload.group_category')}</option>
          <option value="game">{t('saveload.group_game')}</option>
          <option value="algo">{t('saveload.group_algo')}</option>
        </select>
      </div>

      {total === 0 ? (
        <div style={{ padding: '16px 4px', textAlign: 'center', color: 'var(--text-muted)', fontSize: 'var(--fs-label)' }}>
          {slots.length === 0 ? t('saveload.empty') : t('saveload.no_match')}
        </div>
      ) : (
        groups.map((g) => (
          <div key={g.key || '_all'}>
            {filters.group !== 'none' && (
              <div style={{
                display: 'flex', alignItems: 'baseline', gap: 6, marginTop: 12, marginBottom: 2,
              }}>
                <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-strong)', textTransform: 'uppercase', letterSpacing: 0.3 }}>
                  {groupHeader(g.key)}
                </span>
                <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  {g.items.length}
                </span>
              </div>
            )}
            {g.items.map((s) => (
              <SlotCard key={s.id} slot={s} manage={manage} onLoad={onLoad} onDelete={onDelete} />
            ))}
          </div>
        ))
      )}
    </>
  )
}

export default function SaveLoadControls() {
  const { t } = useTranslation()
  const trainState = useAppStore((s) => s.trainState)
  const bumpCheckpoints = useAppStore((s) => s.bumpCheckpoints)

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
      bumpCheckpoints()  // so the AI-play picker shows the new save without a reload
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
      // Send the current sidebar settings: when they target the same game + algorithm as the slot, the
      // resumed run adopts them so the user can extend/retune (e.g. raise AlphaZero iterations to train
      // for hours); a mismatch is ignored server-side and the saved config continues unchanged.
      const status = await loadCheckpoint(slot.id, buildTrainConfig())
      // Mirror the resumed run into the sidebar so the controls match what's actually training.
      const st = useAppStore.getState()
      st.clearMetrics()
      st.setSelectedEnvId(slot.env_id)
      st.setAlgo(slot.algo)
      if (status.config) {
        st.setSeed(status.config.seed)
        st.setTotalTimesteps(status.config.total_timesteps)
        st.setHyperparams(status.config.hyperparams)
        if (status.config.evolution) st.setEvolutionParams(status.config.evolution)
        if (status.config.q_learning) st.setQLearningParams(status.config.q_learning)
        if (status.config.self_play) st.setSelfPlayParams(status.config.self_play)
        if (status.config.alphazero) st.setAlphaZeroParams(status.config.alphazero)
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
      bumpCheckpoints()  // keep the AI-play picker in sync after a delete
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
          position: 'fixed', left: '50%', top: '50%', transform: 'translate(-50%, -50%)', zIndex: 1100,
          display: 'flex', alignItems: 'center', gap: 10, pointerEvents: 'none',
          background: 'var(--surface-1)', border: '1px solid var(--border-default)',
          borderRadius: 'var(--radius-lg)', boxShadow: 'var(--shadow-popover)',
          padding: '14px 26px', fontSize: 'var(--fs-body)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)',
        }}>
          <span style={{ color: 'var(--success)', fontSize: 18 }}>✓</span> {toast}
        </div>
      )}

      {modal && (
        <Modal
          title={modal === 'load' ? t('saveload.load_title') : t('saveload.manage_title')}
          hint={modal === 'load' ? t('saveload.pick_hint') : undefined}
          onClose={() => setModal(null)}
        >
          <CheckpointBrowser
            slots={slots}
            manage={modal === 'manage'}
            onLoad={handleLoad}
            onDelete={handleDelete}
          />
        </Modal>
      )}
    </div>
  )
}
