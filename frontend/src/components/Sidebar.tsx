import { useMemo } from 'react'
import type { CSSProperties, ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { useRunControls } from '../api/trainingControls'
import type { Algo } from '../api/types'
import { formatCount } from '../format'
import ParamInfo from './ParamInfo'
import SaveLoadControls from './SaveLoadControls'
import EnvSelector from './EnvSelector'
import LabSelect from './LabSelect'

// ── Shared style helpers ─────────────────────────────────────────────────────

const fieldLabel: CSSProperties = {
  fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)', color: 'var(--text-muted)',
  display: 'inline-flex', alignItems: 'center', gap: 4,
}
const sectionEyebrow: CSSProperties = {
  fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)',
  letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase', color: 'var(--text-faint)',
}
// Trigger layout for the custom LabSelect dropdowns (visuals come from `.lab-trigger`).
const selectStyle: CSSProperties = {
  width: '100%', height: 'var(--control-md)', fontSize: 'var(--fs-sm)',
}
// Quiet mono count pill beside the algorithm label — mirrors EnvSelector's games "Total: N" pill
// exactly (same format, for the algorithm catalogue count).
const countStyle: CSSProperties = {
  fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
  fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-faint)',
  background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-pill)', padding: '0 7px', lineHeight: '16px',
}

const PlayGlyph  = <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden><path d="M7 5v14l12-7z" /></svg>
const PauseGlyph = <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor" aria-hidden><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
const StopGlyph  = <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" aria-hidden><rect x="6" y="6" width="12" height="12" rx="2" /></svg>

// ── Segmented (algorithm / activation switch) ────────────────────────────────

function Segmented<T extends string>({ options, value, disabled, onChange }: {
  options: { id: T; label: ReactNode }[]
  value: T
  disabled?: boolean
  onChange: (v: T) => void
}) {
  return (
    <div role="tablist" style={{
      display: 'flex', padding: 3, gap: 3,
      background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
      borderRadius: 'var(--radius-md)', opacity: disabled ? 0.6 : 1,
    }}>
      {options.map((opt) => {
        const active = value === opt.id
        return (
          <button
            key={opt.id}
            role="tab"
            aria-selected={active}
            onClick={() => !disabled && onChange(opt.id)}
            disabled={disabled}
            style={{
              flex: 1, height: 30, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
              background: active ? 'var(--surface-2)' : 'transparent',
              color: active ? 'var(--text-strong)' : 'var(--text-muted)',
              borderWidth: 1, borderStyle: 'solid',
              borderColor: active ? 'var(--border-default)' : 'transparent',
              boxShadow: active ? 'var(--shadow-xs)' : 'none',
              borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-sans)',
              fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-medium)',
              cursor: disabled ? 'default' : 'pointer', whiteSpace: 'nowrap',
              transition: 'var(--t-colors)',
            }}
          >
            {opt.label}
          </button>
        )
      })}
    </div>
  )
}

// ── Param-level helpers ─────────────────────────────────────────────────────

const LOG_SCALE = new Set<string>(['learning_rate'])

function formatValue(id: string, v: number | string): string {
  if (typeof v === 'string') return v
  switch (id) {
    case 'learning_rate':
    case 'az_learning_rate': {  // AlphaZero's lr slider — same scientific format (else it rounds 5e-4 → "0")
      const exp = Math.floor(Math.log10(v))
      return `${(v / Math.pow(10, exp)).toFixed(2)}e${exp}`
    }
    case 'gamma':          return v.toFixed(4)
    case 'clip_range':     return v.toFixed(2)
    case 'ent_coef':       return v.toFixed(3)
    case 'sac_tau':        return v.toFixed(3)  // small soft-update coefficient (≈0.005)
    case 'sac_buffer_size': return formatCount(v)  // 1000000 → "1M" (the shared count formatter)
    case 'td3_tau':        return v.toFixed(3)  // same small soft-update coefficient as SAC
    case 'td3_buffer_size': return formatCount(v)  // 1000000 → "1M"
    case 'td3_train_noise': return v.toFixed(2)  // exploration-noise std (≈0.10)
    case 'dqn_buffer_size': return formatCount(v)  // 100000 → "100k" (the shared count formatter)
    case 'dqn_exploration_fraction':
    case 'dqn_exploration_final_eps': return v.toFixed(2)  // ε-greedy fractions (≈0.16 / 0.04)
    case 'a2c_gae_lambda': return v.toFixed(2)  // GAE λ (≈1.00 for classic A2C)
    case 'q_learning_rate':
    case 'epsilon_start':
    case 'epsilon_end':
    case 'epsilon_decay':
    case 'mutation_rate':
    case 'crossover_rate': return v.toFixed(2)
    default:               return String(Math.round(v))
  }
}

function atRecommended(val: number | string, rec: number | string): boolean {
  if (typeof val === 'string' || typeof rec === 'string') return val === rec
  if (rec === 0) return Math.abs(val) < 1e-10
  return Math.abs(val - rec) / Math.abs(rec) < 0.001
}

// ── ParamSlider ─────────────────────────────────────────────────────────────

interface SliderProps {
  id: string
  label: string
  value: number
  min: number
  max: number
  step: number
  recommended: number
  disabled?: boolean
  onChange: (v: number) => void
}

function ParamSlider({ id, label, value, min, max, step, recommended, disabled, onChange }: SliderProps) {
  const log = LOG_SCALE.has(id)
  const isRec = atRecommended(value, recommended)

  const sVal = log ? Math.log10(value) : value
  const sMin = log ? Math.log10(min) : min
  const sMax = log ? Math.log10(max) : max
  const sRec = log ? Math.log10(recommended) : recommended

  // Fractional position [0,1] of the recommended value along the track.
  // The browser thumb is ~14 px wide; correct for the half-thumb inset so
  // the tick aligns with the visual travel range of the thumb.
  const recFrac = Math.max(0, Math.min(1, (sRec - sMin) / (sMax - sMin)))
  const THUMB_PX = 14
  const recLeft = `calc(${recFrac * 100}% + ${(0.5 - recFrac) * THUMB_PX}px)`

  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6, alignItems: 'baseline' }}>
        <span style={fieldLabel}>
          {label}
          <ParamInfo paramId={id} label={label} />
        </span>
        <span style={{
          fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
          fontSize: 'var(--fs-label)', letterSpacing: 'var(--ls-tight)',
          color: disabled ? 'var(--text-faint)' : isRec ? 'var(--success)' : 'var(--text-strong)',
        }}>
          {formatValue(id, value)}
        </span>
      </div>
      {/* Slider + recommended tick — dims while a run locks the config (opacity keeps the ⓘ popup legible) */}
      <div style={{ position: 'relative', opacity: disabled ? 0.45 : 1, transition: 'var(--t-colors)' }}>
        <input
          type="range"
          aria-label={label}
          min={sMin} max={sMax} step={log ? 0.01 : step}
          value={sVal}
          disabled={disabled}
          onChange={(e) => {
            const raw = parseFloat(e.target.value)
            let v = log ? Math.pow(10, raw) : raw
            // Snap to the recommended value when the thumb lands within half a step of it, so the green
            // ★ tick is always exactly selectable even if the recommendation isn't on the step grid
            // (otherwise the thumb only stops just left or right of the tick — never on it).
            if (!log && recommended >= min && recommended <= max && Math.abs(v - recommended) <= step / 2) {
              v = recommended
            }
            onChange(v)
          }}
          style={{ width: '100%', cursor: disabled ? 'default' : 'pointer', accentColor: 'var(--accent)', display: 'block' }}
        />
        {/* Thin green tick at the recommended position */}
        <div
          aria-hidden
          style={{
            position: 'absolute',
            left: recLeft,
            top: '50%',
            transform: 'translate(-50%, -50%)',
            width: 2,
            height: 11,
            background: 'var(--success)',
            borderRadius: 1,
            pointerEvents: 'none',
            opacity: 0.9,
          }}
        />
      </div>
    </div>
  )
}

// ── ActivationToggle ────────────────────────────────────────────────────────

function ActivationToggle({ value, label, disabled, onChange }: {
  value: 'tanh' | 'relu'
  label: string
  disabled?: boolean
  onChange: (v: 'tanh' | 'relu') => void
}) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ marginBottom: 6 }}>
        <span style={fieldLabel}>
          {label}
          <ParamInfo paramId="activation" label={label} />
        </span>
      </div>
      <Segmented
        value={value}
        disabled={disabled}
        onChange={onChange}
        options={[{ id: 'tanh', label: '★ tanh' }, { id: 'relu', label: 'relu' }]}
      />
    </div>
  )
}

// ── AlgoSwitch ───────────────────────────────────────────────────────────────

// Algorithm picker — a dropdown (not a 2-button switch) because the catalogue of algorithms will
// grow (Q-learning, DQN, SAC…). The options are the *selected env's* supported_algos, so a game can
// opt out of one (e.g. an image env may be PPO-only); the store snaps to a valid algo on env switch.
function ALGO_LABEL(t: (k: string) => string, id: string): string {
  switch (id) {
    case 'ppo':            return t('sidebar.algo_ppo')
    case 'neuroevolution': return t('sidebar.algo_evo')
    case 'q_learning':     return t('sidebar.algo_q')
    case 'alphazero':      return t('sidebar.algo_az')
    case 'sac':            return t('sidebar.algo_sac')
    case 'td3':            return t('sidebar.algo_td3')
    case 'dqn':            return t('sidebar.algo_dqn')
    case 'a2c':            return t('sidebar.algo_a2c')
    case 'qrdqn':          return t('sidebar.algo_qrdqn')
    default:               return id
  }
}

function AlgoSwitch({ value, options, recommended, algoCount, disabled, onChange }: {
  value: Algo
  options: string[]
  recommended?: string | null
  algoCount: number
  disabled?: boolean
  onChange: (a: Algo) => void
}) {
  const { t } = useTranslation()
  // The ★ recommended algo is the best fit for THIS game (often not PPO). We only MARK it — a ★ on its
  // dropdown option (the same convention as the activation/ent-coef pickers) plus an always-visible hint
  // line below, since with mark-only the closed picker often shows a different (e.g. PPO) algo. Picking
  // it stays a deliberate click; switching env never auto-selects it.
  const onRecommended = !!recommended && value === recommended
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <label style={{ ...fieldLabel, justifyContent: 'space-between', width: '100%' }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          {t('sidebar.algorithm')}
          <ParamInfo paramId="algorithm" label={t('sidebar.algorithm')} />
        </span>
        {algoCount > 0 && (
          <span
            style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}
            title={t('sidebar.algo_count')}
          >
            <span>{t('sidebar.total')}:</span>
            <span style={countStyle} aria-label={t('sidebar.algo_count')}>{algoCount}</span>
          </span>
        )}
      </label>
      <LabSelect
        ariaLabel={t('sidebar.algorithm')}
        value={value}
        disabled={disabled || options.length === 0}
        onChange={(v) => onChange(v as Algo)}
        style={selectStyle}
        options={options.map((id) => ({
          value: id,
          label: id === recommended ? `★ ${ALGO_LABEL(t, id)}` : ALGO_LABEL(t, id),
        }))}
      />
      {recommended && (
        onRecommended ? (
          <span style={recHint('var(--success)')}>
            <span aria-hidden style={recStar}>★</span>
            {t('sidebar.algo_recommended_here')}
          </span>
        ) : (
          <button
            type="button"
            disabled={disabled}
            onClick={() => onChange(recommended as Algo)}
            title={t('sidebar.algo_use_recommended')}
            style={{ ...recHint('var(--text-muted)'), background: 'none', border: 'none', padding: 0,
              cursor: disabled ? 'default' : 'pointer', textAlign: 'left' }}
          >
            <span aria-hidden style={recStar}>★</span>
            <span>{t('sidebar.algo_recommended')}: <strong style={{ color: 'var(--text-strong)' }}>{ALGO_LABEL(t, recommended)}</strong></span>
          </button>
        )
      )}
    </div>
  )
}

// Gold ★ + a quiet caption — the always-visible recommendation marker under the algorithm picker.
const recStar: CSSProperties = { color: 'var(--goal)', fontSize: 'var(--fs-meta)' }
function recHint(color: string): CSSProperties {
  return {
    display: 'inline-flex', alignItems: 'center', gap: 5,
    fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-meta)', color,
    transition: 'var(--t-colors)',
  }
}

// ── Run controls ─────────────────────────────────────────────────────────────

type BtnKind = 'primary' | 'pause' | 'resume' | 'stop' | 'disabled'

function runBtn(kind: BtnKind, lg = false): CSSProperties {
  const base: CSSProperties = {
    flex: 1, display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 7,
    height: lg ? 'var(--control-lg)' : 'var(--control-md)',
    borderRadius: 'var(--radius-md)', fontFamily: 'var(--font-sans)',
    fontSize: lg ? 'var(--fs-body)' : 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)',
    borderWidth: 1, borderStyle: 'solid', borderColor: 'transparent',
    cursor: 'pointer', whiteSpace: 'nowrap', transition: 'var(--t-colors)',
  }
  switch (kind) {
    case 'primary':  return { ...base, background: 'var(--accent-grad)', color: 'var(--accent-contrast)', boxShadow: 'var(--shadow-cta)' }
    case 'resume':   return { ...base, background: 'var(--success-surface)', color: 'var(--success)', boxShadow: 'var(--ring-inset)' }
    case 'pause':    return { ...base, background: 'var(--warning-surface)', color: 'var(--warning)', boxShadow: 'var(--ring-inset)' }
    case 'stop':     return { ...base, background: 'var(--danger-surface)', color: 'var(--danger)', boxShadow: 'var(--ring-inset)' }
    case 'disabled': return { ...base, background: 'var(--surface-2)', color: 'var(--text-muted)', borderColor: 'var(--border-default)', cursor: 'not-allowed' }
  }
}

// The step-budget dropdown is built per-env from the registry's default_total_timesteps: a
// ladder of ×0.2 … ×8 around the recommended value, with ★ on the recommendation. So CartPole
// (50k → 10k…400k) and LunarLander (500k → 100k…4M) each get an appropriate range. The ×8 top rung
// (= 2× the previous ×4 max) gives headroom for envs that keep improving past the default budget —
// learning is rarely linear, so a hard game may need well beyond the recommended steps.
const STEP_FACTORS = [0.2, 0.5, 1, 2, 4, 8]

function stepsLadder(defaultSteps: number): number[] {
  return STEP_FACTORS.map((f) => Math.round((defaultSteps * f) / 1000) * 1000)
}

const formatSteps = formatCount  // shared "use M past 1000k" formatter

// ── Sidebar ──────────────────────────────────────────────────────────────────

export default function Sidebar() {
  const { t } = useTranslation()

  const envs            = useAppStore((s) => s.envs)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)

  const algo            = useAppStore((s) => s.algo)
  const setAlgo         = useAppStore((s) => s.setAlgo)
  const hyperparams     = useAppStore((s) => s.hyperparams)
  const setHyperparams  = useAppStore((s) => s.setHyperparams)
  const evolutionParams = useAppStore((s) => s.evolutionParams)
  const setEvolutionParams = useAppStore((s) => s.setEvolutionParams)
  const qLearningParams = useAppStore((s) => s.qLearningParams)
  const setQLearningParams = useAppStore((s) => s.setQLearningParams)
  const selfPlayParams  = useAppStore((s) => s.selfPlayParams)
  const setSelfPlayParams = useAppStore((s) => s.setSelfPlayParams)
  const alphaZeroParams = useAppStore((s) => s.alphaZeroParams)
  const setAlphaZeroParams = useAppStore((s) => s.setAlphaZeroParams)
  const sacParams       = useAppStore((s) => s.sacParams)
  const setSacParams    = useAppStore((s) => s.setSacParams)
  const td3Params       = useAppStore((s) => s.td3Params)
  const setTd3Params    = useAppStore((s) => s.setTd3Params)
  const dqnParams       = useAppStore((s) => s.dqnParams)
  const setDqnParams    = useAppStore((s) => s.setDqnParams)
  const a2cParams       = useAppStore((s) => s.a2cParams)
  const setA2cParams    = useAppStore((s) => s.setA2cParams)
  const qrdqnParams     = useAppStore((s) => s.qrdqnParams)
  const setQrdqnParams  = useAppStore((s) => s.setQrdqnParams)
  const seed            = useAppStore((s) => s.seed)
  const setSeed         = useAppStore((s) => s.setSeed)
  const totalTimesteps  = useAppStore((s) => s.totalTimesteps)
  const setTotalTimesteps = useAppStore((s) => s.setTotalTimesteps)
  const sweepCount      = useAppStore((s) => s.sweepCount)
  const setSweepCount   = useAppStore((s) => s.setSweepCount)
  const sweep           = useAppStore((s) => s.sweep)

  const { handleRun, handleRunSweep, handlePause, handleResume, handleStop, isRunning, isPaused, isStopping, isActive, canRun, trainGated, trainGatedReason } =
    useRunControls()

  const selectedEnv = envs.find((e) => e.id === selectedEnvId)
  // Total distinct learning algorithms across the whole catalogue (a "Total: N" pill beside the picker,
  // mirroring the games count) — data-driven from the union of every env's supported_algos, so it tracks
  // automatically as algorithms are added.
  const algoCount = useMemo(() => new Set(envs.flatMap((e) => e.supported_algos)).size, [envs])
  const ppoDefs = selectedEnv?.hyperparams?.['ppo'] ?? {}
  const evoDefs = selectedEnv?.hyperparams?.['neuroevolution'] ?? {}
  const qlDefs  = selectedEnv?.hyperparams?.['q_learning'] ?? {}
  const azDefs  = selectedEnv?.hyperparams?.['alphazero'] ?? {}
  const sacDefs = selectedEnv?.hyperparams?.['sac'] ?? {}
  const sacEntChoices = sacDefs.ent_coef?.choices ?? null  // SAC entropy: ["auto", "0.1", "0.2"]
  const td3Defs = selectedEnv?.hyperparams?.['td3'] ?? {}
  const dqnDefs = selectedEnv?.hyperparams?.['dqn'] ?? {}
  const a2cDefs = selectedEnv?.hyperparams?.['a2c'] ?? {}
  const qrdqnDefs = selectedEnv?.hyperparams?.['qrdqn'] ?? {}
  const isEvo = algo === 'neuroevolution'
  const isQ   = algo === 'q_learning'
  const isAz  = algo === 'alphazero'
  const isSac = algo === 'sac'
  const isTd3 = algo === 'td3'
  const isDqn = algo === 'dqn'
  const isA2c = algo === 'a2c'
  const isQrdqn = algo === 'qrdqn'

  // Per-env step ladder + the ★ recommended budget; always include the current value so the
  // <select> can render it even after a reload with a value off the ladder. The off-policy algos (SAC +
  // TD3 + DQN) carry their own ★ budget (offpolicy_total_timesteps) — the ladder + ★ then reflect that
  // real budget (~500k for BipedalWalker-SAC, 100k for CartPole-DQN), not the PPO default.
  const defaultSteps =
    ((isSac || isTd3 || isDqn) && selectedEnv?.offpolicy_total_timesteps) || selectedEnv?.default_total_timesteps || 50_000
  const stepsOptions = Array.from(new Set([...stepsLadder(defaultSteps), totalTimesteps])).sort((a, b) => a - b)

  return (
    <aside style={{
      width: 'var(--sidebar-w)', flexShrink: 0,
      background: 'var(--surface-1)', borderRight: '2px solid var(--border-default)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        height: 'var(--panel-head-h)', flexShrink: 0,
        padding: '0 var(--space-5)', borderBottom: '1px solid var(--border-default)',
        display: 'flex', alignItems: 'center', gap: 8,
        fontWeight: 'var(--fw-semibold)', fontSize: 'var(--fs-meta)',
        letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase', color: 'var(--text-muted)',
      }}>
        <span aria-hidden style={{
          width: 3, height: 14, borderRadius: 2, background: 'var(--accent)',
          boxShadow: 'var(--accent-glow)', flexShrink: 0,
        }} />
        {t('sidebar.title')}
      </div>

      {/* Game selector + algorithm switch */}
      <div style={{
        padding: 'var(--space-4) var(--space-5)', borderBottom: '1px solid var(--border-default)',
        flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 'var(--space-4)',
      }}>
        <EnvSelector disabled={isActive} />

        <AlgoSwitch
          value={algo}
          options={selectedEnv?.supported_algos ?? ['ppo', 'neuroevolution']}
          recommended={selectedEnv?.recommended_algo}
          algoCount={algoCount}
          disabled={isActive}
          onChange={setAlgo}
        />
      </div>

      {/* Scrollable params */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-5)' }}>
        <div style={{ ...sectionEyebrow, marginBottom: 'var(--space-4)' }}>
          {isEvo ? t('sidebar.evo_params_title') : isQ ? t('sidebar.q_params_title') : isAz ? t('sidebar.az_params_title') : isSac ? t('sidebar.sac_params_title') : isTd3 ? t('sidebar.td3_params_title') : isDqn ? t('sidebar.dqn_params_title') : isA2c ? t('sidebar.a2c_params_title') : isQrdqn ? t('sidebar.qrdqn_params_title') : t('sidebar.params_title')}
        </div>

        {/* While a run holds the config (train/pause/stopping), the params below lock like the env/algo/
            seed/steps already do — a small notice so a mid-run edit doesn't look like it took effect. */}
        {isActive && (
          <div role="note" style={{
            display: 'flex', alignItems: 'center', gap: 6, marginBottom: 'var(--space-4)',
            padding: '6px 10px', borderRadius: 'var(--radius-md)',
            background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
            fontSize: 'var(--fs-meta)', color: 'var(--text-muted)',
          }}>
            <span aria-hidden>🔒</span>
            <span>{t('sidebar.locked_while_running')}</span>
          </div>
        )}

        {algo === 'ppo' && (
          <>
            {/* Competitive multi-agent self-play (simple_tag, G7b-2): the round schedule — how many
                times the two species alternate (predators learn vs. frozen prey, then prey vs. frozen
                predators, …). Only simple_tag defines `rounds`, so it shows just for those envs. */}
            {ppoDefs.rounds && (
              <ParamSlider disabled={isActive}
                id="rounds" label={t('sidebar.rounds')}
                value={selfPlayParams.rounds}
                min={ppoDefs.rounds.min!} max={ppoDefs.rounds.max!} step={ppoDefs.rounds.step!}
                recommended={ppoDefs.rounds.recommended as number}
                onChange={(v) => setSelfPlayParams({ rounds: Math.round(v) })}
              />
            )}

            {ppoDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={hyperparams.learning_rate}
                min={ppoDefs.learning_rate.min!} max={ppoDefs.learning_rate.max!} step={0.01}
                recommended={ppoDefs.learning_rate.recommended as number}
                onChange={(v) => setHyperparams({ learning_rate: v })}
              />
            )}

            {ppoDefs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={hyperparams.gamma}
                min={ppoDefs.gamma.min!} max={ppoDefs.gamma.max!} step={ppoDefs.gamma.step!}
                recommended={ppoDefs.gamma.recommended as number}
                onChange={(v) => setHyperparams({ gamma: v })}
              />
            )}

            {ppoDefs.clip_range && (
              <ParamSlider disabled={isActive}
                id="clip_range" label={t('sidebar.clip_range')}
                value={hyperparams.clip_range}
                min={ppoDefs.clip_range.min!} max={ppoDefs.clip_range.max!} step={ppoDefs.clip_range.step!}
                recommended={ppoDefs.clip_range.recommended as number}
                onChange={(v) => setHyperparams({ clip_range: v })}
              />
            )}

            {ppoDefs.ent_coef && (
              <ParamSlider disabled={isActive}
                id="ent_coef" label={t('sidebar.ent_coef')}
                value={hyperparams.ent_coef}
                min={ppoDefs.ent_coef.min!} max={ppoDefs.ent_coef.max!} step={ppoDefs.ent_coef.step!}
                recommended={ppoDefs.ent_coef.recommended as number}
                onChange={(v) => setHyperparams({ ent_coef: v })}
              />
            )}

            {ppoDefs.n_hidden_layers && (
              <ParamSlider disabled={isActive}
                id="n_hidden_layers" label={t('sidebar.n_hidden_layers')}
                value={hyperparams.n_hidden_layers}
                min={ppoDefs.n_hidden_layers.min!} max={ppoDefs.n_hidden_layers.max!} step={ppoDefs.n_hidden_layers.step!}
                recommended={ppoDefs.n_hidden_layers.recommended as number}
                onChange={(v) => setHyperparams({ n_hidden_layers: Math.round(v) })}
              />
            )}

            {ppoDefs.neurons_per_layer && (
              <ParamSlider disabled={isActive}
                id="neurons_per_layer" label={t('sidebar.neurons_per_layer')}
                value={hyperparams.neurons_per_layer}
                min={ppoDefs.neurons_per_layer.min!} max={ppoDefs.neurons_per_layer.max!} step={ppoDefs.neurons_per_layer.step!}
                recommended={ppoDefs.neurons_per_layer.recommended as number}
                onChange={(v) => setHyperparams({ neurons_per_layer: Math.round(v) })}
              />
            )}

            {ppoDefs.activation && (
              <ActivationToggle disabled={isActive}
                value={hyperparams.activation}
                label={t('sidebar.activation')}
                onChange={(v) => setHyperparams({ activation: v })}
              />
            )}
          </>
        )}

        {isEvo && (
          <>
            {evoDefs.population_size && (
              <ParamSlider disabled={isActive}
                id="population_size" label={t('sidebar.population_size')}
                value={evolutionParams.population_size}
                min={evoDefs.population_size.min!} max={evoDefs.population_size.max!} step={evoDefs.population_size.step!}
                recommended={evoDefs.population_size.recommended as number}
                onChange={(v) => setEvolutionParams({ population_size: Math.round(v) })}
              />
            )}

            {evoDefs.top_k_parents && (
              <ParamSlider disabled={isActive}
                id="top_k_parents" label={t('sidebar.top_k_parents')}
                value={evolutionParams.top_k_parents}
                min={evoDefs.top_k_parents.min!} max={evoDefs.top_k_parents.max!} step={evoDefs.top_k_parents.step!}
                recommended={evoDefs.top_k_parents.recommended as number}
                onChange={(v) => setEvolutionParams({ top_k_parents: Math.round(v) })}
              />
            )}

            {evoDefs.mutation_rate && (
              <ParamSlider disabled={isActive}
                id="mutation_rate" label={t('sidebar.mutation_rate')}
                value={evolutionParams.mutation_rate}
                min={evoDefs.mutation_rate.min!} max={evoDefs.mutation_rate.max!} step={evoDefs.mutation_rate.step!}
                recommended={evoDefs.mutation_rate.recommended as number}
                onChange={(v) => setEvolutionParams({ mutation_rate: v })}
              />
            )}

            {evoDefs.crossover_rate && (
              <ParamSlider disabled={isActive}
                id="crossover_rate" label={t('sidebar.crossover_rate')}
                value={evolutionParams.crossover_rate}
                min={evoDefs.crossover_rate.min!} max={evoDefs.crossover_rate.max!} step={evoDefs.crossover_rate.step!}
                recommended={evoDefs.crossover_rate.recommended as number}
                onChange={(v) => setEvolutionParams({ crossover_rate: v })}
              />
            )}

            {evoDefs.generations && (
              <ParamSlider disabled={isActive}
                id="generations" label={t('sidebar.generations')}
                value={evolutionParams.generations}
                min={evoDefs.generations.min!} max={evoDefs.generations.max!} step={evoDefs.generations.step!}
                recommended={evoDefs.generations.recommended as number}
                onChange={(v) => setEvolutionParams({ generations: Math.round(v) })}
              />
            )}
          </>
        )}

        {isQ && (
          <>
            {qlDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="q_learning_rate" label={t('sidebar.q_learning_rate')}
                value={qLearningParams.learning_rate}
                min={qlDefs.learning_rate.min!} max={qlDefs.learning_rate.max!} step={qlDefs.learning_rate.step!}
                recommended={qlDefs.learning_rate.recommended as number}
                onChange={(v) => setQLearningParams({ learning_rate: v })}
              />
            )}

            {qlDefs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={qLearningParams.gamma}
                min={qlDefs.gamma.min!} max={qlDefs.gamma.max!} step={qlDefs.gamma.step!}
                recommended={qlDefs.gamma.recommended as number}
                onChange={(v) => setQLearningParams({ gamma: v })}
              />
            )}

            {qlDefs.epsilon_start && (
              <ParamSlider disabled={isActive}
                id="epsilon_start" label={t('sidebar.epsilon_start')}
                value={qLearningParams.epsilon_start}
                min={qlDefs.epsilon_start.min!} max={qlDefs.epsilon_start.max!} step={qlDefs.epsilon_start.step!}
                recommended={qlDefs.epsilon_start.recommended as number}
                onChange={(v) => setQLearningParams({ epsilon_start: v })}
              />
            )}

            {qlDefs.epsilon_end && (
              <ParamSlider disabled={isActive}
                id="epsilon_end" label={t('sidebar.epsilon_end')}
                value={qLearningParams.epsilon_end}
                min={qlDefs.epsilon_end.min!} max={qlDefs.epsilon_end.max!} step={qlDefs.epsilon_end.step!}
                recommended={qlDefs.epsilon_end.recommended as number}
                onChange={(v) => setQLearningParams({ epsilon_end: v })}
              />
            )}

            {qlDefs.epsilon_decay && (
              <ParamSlider disabled={isActive}
                id="epsilon_decay" label={t('sidebar.epsilon_decay')}
                value={qLearningParams.epsilon_decay}
                min={qlDefs.epsilon_decay.min!} max={qlDefs.epsilon_decay.max!} step={qlDefs.epsilon_decay.step!}
                recommended={qlDefs.epsilon_decay.recommended as number}
                onChange={(v) => setQLearningParams({ epsilon_decay: v })}
              />
            )}

            {qlDefs.episodes && (
              <ParamSlider disabled={isActive}
                id="episodes" label={t('sidebar.episodes')}
                value={qLearningParams.episodes}
                min={qlDefs.episodes.min!} max={qlDefs.episodes.max!} step={qlDefs.episodes.step!}
                recommended={qlDefs.episodes.recommended as number}
                onChange={(v) => setQLearningParams({ episodes: Math.round(v) })}
              />
            )}
          </>
        )}

        {/* AlphaZero (board games, G6f/G6h): the self-play budget (iterations × games_per_iter) + the
            Gumbel search depth + considered-move breadth + the net's learning rate. `iterations` is this
            algorithm's budget, so there's no separate "Total Steps" control (it's PPO-only below). */}
        {isAz && (
          <>
            {azDefs.iterations && (
              <ParamSlider disabled={isActive}
                id="iterations" label={t('sidebar.iterations')}
                value={alphaZeroParams.iterations}
                min={azDefs.iterations.min!} max={azDefs.iterations.max!} step={azDefs.iterations.step!}
                recommended={azDefs.iterations.recommended as number}
                onChange={(v) => setAlphaZeroParams({ iterations: Math.round(v) })}
              />
            )}

            {azDefs.gumbel_sims && (
              <ParamSlider disabled={isActive}
                id="gumbel_sims" label={t('sidebar.gumbel_sims')}
                value={alphaZeroParams.gumbel_sims}
                min={azDefs.gumbel_sims.min!} max={azDefs.gumbel_sims.max!} step={azDefs.gumbel_sims.step!}
                recommended={azDefs.gumbel_sims.recommended as number}
                onChange={(v) => setAlphaZeroParams({ gumbel_sims: Math.round(v) })}
              />
            )}

            {azDefs.gumbel_considered && (
              <ParamSlider disabled={isActive}
                id="gumbel_considered" label={t('sidebar.gumbel_considered')}
                value={alphaZeroParams.gumbel_considered}
                min={azDefs.gumbel_considered.min!} max={azDefs.gumbel_considered.max!} step={azDefs.gumbel_considered.step!}
                recommended={azDefs.gumbel_considered.recommended as number}
                onChange={(v) => setAlphaZeroParams({ gumbel_considered: Math.round(v) })}
              />
            )}

            {azDefs.games_per_iter && (
              <ParamSlider disabled={isActive}
                id="games_per_iter" label={t('sidebar.games_per_iter')}
                value={alphaZeroParams.games_per_iter}
                min={azDefs.games_per_iter.min!} max={azDefs.games_per_iter.max!} step={azDefs.games_per_iter.step!}
                recommended={azDefs.games_per_iter.recommended as number}
                onChange={(v) => setAlphaZeroParams({ games_per_iter: Math.round(v) })}
              />
            )}

            {azDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="az_learning_rate" label={t('sidebar.learning_rate')}
                value={alphaZeroParams.learning_rate}
                min={azDefs.learning_rate.min!} max={azDefs.learning_rate.max!} step={0.0001}
                recommended={azDefs.learning_rate.recommended as number}
                onChange={(v) => setAlphaZeroParams({ learning_rate: v })}
              />
            )}

            {azDefs.actor_processes && (
              <ParamSlider disabled={isActive}
                id="actor_processes" label={t('sidebar.actor_processes')}
                value={alphaZeroParams.actor_processes}
                min={azDefs.actor_processes.min!} max={azDefs.actor_processes.max!} step={azDefs.actor_processes.step!}
                recommended={azDefs.actor_processes.recommended as number}
                onChange={(v) => setAlphaZeroParams({ actor_processes: Math.round(v) })}
              />
            )}
          </>
        )}

        {/* SAC (S5a — off-policy continuous control): the learning rate + discount, the target soft-update
            (tau), the replay-buffer size, how often it updates (train_freq), and the entropy temperature
            (auto-tuned by default, or pinned). batch_size / learning_starts / gradient_steps are fixed
            backend defaults. SAC reuses the PPO "Total Steps" budget below (it runs that many env steps). */}
        {isSac && (
          <>
            {sacDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={sacParams.learning_rate}
                min={sacDefs.learning_rate.min!} max={sacDefs.learning_rate.max!} step={0.01}
                recommended={sacDefs.learning_rate.recommended as number}
                onChange={(v) => setSacParams({ learning_rate: v })}
              />
            )}

            {sacDefs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={sacParams.gamma}
                min={sacDefs.gamma.min!} max={sacDefs.gamma.max!} step={sacDefs.gamma.step!}
                recommended={sacDefs.gamma.recommended as number}
                onChange={(v) => setSacParams({ gamma: v })}
              />
            )}

            {sacDefs.tau && (
              <ParamSlider disabled={isActive}
                id="sac_tau" label={t('sidebar.sac_tau')}
                value={sacParams.tau}
                min={sacDefs.tau.min!} max={sacDefs.tau.max!} step={sacDefs.tau.step!}
                recommended={sacDefs.tau.recommended as number}
                onChange={(v) => setSacParams({ tau: v })}
              />
            )}

            {sacDefs.buffer_size && (
              <ParamSlider disabled={isActive}
                id="sac_buffer_size" label={t('sidebar.sac_buffer_size')}
                value={sacParams.buffer_size}
                min={sacDefs.buffer_size.min!} max={sacDefs.buffer_size.max!} step={sacDefs.buffer_size.step!}
                recommended={sacDefs.buffer_size.recommended as number}
                onChange={(v) => setSacParams({ buffer_size: Math.round(v) })}
              />
            )}

            {sacDefs.train_freq && (
              <ParamSlider disabled={isActive}
                id="sac_train_freq" label={t('sidebar.sac_train_freq')}
                value={sacParams.train_freq}
                min={sacDefs.train_freq.min!} max={sacDefs.train_freq.max!} step={sacDefs.train_freq.step!}
                recommended={sacDefs.train_freq.recommended as number}
                onChange={(v) => setSacParams({ train_freq: Math.round(v) })}
              />
            )}

            {sacEntChoices && (
              <div style={{ marginBottom: 14 }}>
                <div style={{ marginBottom: 6 }}>
                  <span style={fieldLabel}>
                    {t('sidebar.sac_ent_coef')}
                    <ParamInfo paramId="sac_ent_coef" label={t('sidebar.sac_ent_coef')} />
                  </span>
                </div>
                <Segmented
                  value={sacParams.ent_coef}
                  disabled={isActive}
                  onChange={(v) => setSacParams({ ent_coef: v })}
                  options={sacEntChoices.map((c) => ({
                    id: c,
                    label: c === 'auto' ? `★ ${t('sidebar.sac_ent_auto')}` : c,
                  }))}
                />
              </div>
            )}
          </>
        )}

        {/* TD3 (S5b — off-policy continuous control, SAC's deterministic sibling): the learning rate +
            discount, the target soft-update (tau), the replay-buffer size, how often it updates
            (train_freq), and the exploration-noise std (train_noise — a deterministic policy must inject
            noise to explore; TD3 has no entropy term). policy_delay / target-smoothing / batch_size /
            learning_starts are fixed backend defaults. TD3 reuses the PPO "Total Steps" budget below. */}
        {isTd3 && (
          <>
            {td3Defs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={td3Params.learning_rate}
                min={td3Defs.learning_rate.min!} max={td3Defs.learning_rate.max!} step={0.01}
                recommended={td3Defs.learning_rate.recommended as number}
                onChange={(v) => setTd3Params({ learning_rate: v })}
              />
            )}

            {td3Defs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={td3Params.gamma}
                min={td3Defs.gamma.min!} max={td3Defs.gamma.max!} step={td3Defs.gamma.step!}
                recommended={td3Defs.gamma.recommended as number}
                onChange={(v) => setTd3Params({ gamma: v })}
              />
            )}

            {td3Defs.tau && (
              <ParamSlider disabled={isActive}
                id="td3_tau" label={t('sidebar.td3_tau')}
                value={td3Params.tau}
                min={td3Defs.tau.min!} max={td3Defs.tau.max!} step={td3Defs.tau.step!}
                recommended={td3Defs.tau.recommended as number}
                onChange={(v) => setTd3Params({ tau: v })}
              />
            )}

            {td3Defs.buffer_size && (
              <ParamSlider disabled={isActive}
                id="td3_buffer_size" label={t('sidebar.td3_buffer_size')}
                value={td3Params.buffer_size}
                min={td3Defs.buffer_size.min!} max={td3Defs.buffer_size.max!} step={td3Defs.buffer_size.step!}
                recommended={td3Defs.buffer_size.recommended as number}
                onChange={(v) => setTd3Params({ buffer_size: Math.round(v) })}
              />
            )}

            {td3Defs.train_freq && (
              <ParamSlider disabled={isActive}
                id="td3_train_freq" label={t('sidebar.td3_train_freq')}
                value={td3Params.train_freq}
                min={td3Defs.train_freq.min!} max={td3Defs.train_freq.max!} step={td3Defs.train_freq.step!}
                recommended={td3Defs.train_freq.recommended as number}
                onChange={(v) => setTd3Params({ train_freq: Math.round(v) })}
              />
            )}

            {td3Defs.train_noise && (
              <ParamSlider disabled={isActive}
                id="td3_train_noise" label={t('sidebar.td3_train_noise')}
                value={td3Params.train_noise}
                min={td3Defs.train_noise.min!} max={td3Defs.train_noise.max!} step={td3Defs.train_noise.step!}
                recommended={td3Defs.train_noise.recommended as number}
                onChange={(v) => setTd3Params({ train_noise: v })}
              />
            )}
          </>
        )}

        {/* DQN (S5c — off-policy value-based, discrete actions; PPO's counterpart): the learning rate +
            discount, the replay-buffer size, how often it updates (train_freq), how often the slow target
            net is synced (target_update_interval), and the two ε-greedy exploration knobs — the fraction
            of the run to anneal ε over and the final ε held after. batch_size / learning_starts /
            gradient_steps are fixed/derived backend defaults. DQN reuses the PPO "Total Steps" budget. */}
        {isDqn && (
          <>
            {dqnDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={dqnParams.learning_rate}
                min={dqnDefs.learning_rate.min!} max={dqnDefs.learning_rate.max!} step={0.01}
                recommended={dqnDefs.learning_rate.recommended as number}
                onChange={(v) => setDqnParams({ learning_rate: v })}
              />
            )}

            {dqnDefs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={dqnParams.gamma}
                min={dqnDefs.gamma.min!} max={dqnDefs.gamma.max!} step={dqnDefs.gamma.step!}
                recommended={dqnDefs.gamma.recommended as number}
                onChange={(v) => setDqnParams({ gamma: v })}
              />
            )}

            {dqnDefs.buffer_size && (
              <ParamSlider disabled={isActive}
                id="dqn_buffer_size" label={t('sidebar.dqn_buffer_size')}
                value={dqnParams.buffer_size}
                min={dqnDefs.buffer_size.min!} max={dqnDefs.buffer_size.max!} step={dqnDefs.buffer_size.step!}
                recommended={dqnDefs.buffer_size.recommended as number}
                onChange={(v) => setDqnParams({ buffer_size: Math.round(v) })}
              />
            )}

            {dqnDefs.train_freq && (
              <ParamSlider disabled={isActive}
                id="dqn_train_freq" label={t('sidebar.dqn_train_freq')}
                value={dqnParams.train_freq}
                min={dqnDefs.train_freq.min!} max={dqnDefs.train_freq.max!} step={dqnDefs.train_freq.step!}
                recommended={dqnDefs.train_freq.recommended as number}
                onChange={(v) => setDqnParams({ train_freq: Math.round(v) })}
              />
            )}

            {dqnDefs.target_update_interval && (
              <ParamSlider disabled={isActive}
                id="dqn_target_update" label={t('sidebar.dqn_target_update')}
                value={dqnParams.target_update_interval}
                min={dqnDefs.target_update_interval.min!} max={dqnDefs.target_update_interval.max!} step={dqnDefs.target_update_interval.step!}
                recommended={dqnDefs.target_update_interval.recommended as number}
                onChange={(v) => setDqnParams({ target_update_interval: Math.round(v) })}
              />
            )}

            {dqnDefs.exploration_fraction && (
              <ParamSlider disabled={isActive}
                id="dqn_exploration_fraction" label={t('sidebar.dqn_exploration_fraction')}
                value={dqnParams.exploration_fraction}
                min={dqnDefs.exploration_fraction.min!} max={dqnDefs.exploration_fraction.max!} step={dqnDefs.exploration_fraction.step!}
                recommended={dqnDefs.exploration_fraction.recommended as number}
                onChange={(v) => setDqnParams({ exploration_fraction: v })}
              />
            )}

            {dqnDefs.exploration_final_eps && (
              <ParamSlider disabled={isActive}
                id="dqn_exploration_final_eps" label={t('sidebar.dqn_exploration_final_eps')}
                value={dqnParams.exploration_final_eps}
                min={dqnDefs.exploration_final_eps.min!} max={dqnDefs.exploration_final_eps.max!} step={dqnDefs.exploration_final_eps.step!}
                recommended={dqnDefs.exploration_final_eps.recommended as number}
                onChange={(v) => setDqnParams({ exploration_final_eps: v })}
              />
            )}
          </>
        )}

        {/* A2C (S5d — on-policy actor-critic, PPO's simpler predecessor): the learning rate + discount,
            n_steps (A2C's signature short rollout), the GAE λ (1.0 = full Monte-Carlo returns), the
            entropy bonus, and the same network-size + activation knobs as PPO. No clip/batch/epochs —
            A2C does one plain policy-gradient update per rollout. */}
        {isA2c && (
          <>
            {a2cDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={a2cParams.learning_rate}
                min={a2cDefs.learning_rate.min!} max={a2cDefs.learning_rate.max!} step={0.01}
                recommended={a2cDefs.learning_rate.recommended as number}
                onChange={(v) => setA2cParams({ learning_rate: v })}
              />
            )}

            {a2cDefs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={a2cParams.gamma}
                min={a2cDefs.gamma.min!} max={a2cDefs.gamma.max!} step={a2cDefs.gamma.step!}
                recommended={a2cDefs.gamma.recommended as number}
                onChange={(v) => setA2cParams({ gamma: v })}
              />
            )}

            {a2cDefs.n_steps && (
              <ParamSlider disabled={isActive}
                id="a2c_n_steps" label={t('sidebar.a2c_n_steps')}
                value={a2cParams.n_steps}
                min={a2cDefs.n_steps.min!} max={a2cDefs.n_steps.max!} step={a2cDefs.n_steps.step!}
                recommended={a2cDefs.n_steps.recommended as number}
                onChange={(v) => setA2cParams({ n_steps: Math.round(v) })}
              />
            )}

            {a2cDefs.gae_lambda && (
              <ParamSlider disabled={isActive}
                id="a2c_gae_lambda" label={t('sidebar.a2c_gae_lambda')}
                value={a2cParams.gae_lambda}
                min={a2cDefs.gae_lambda.min!} max={a2cDefs.gae_lambda.max!} step={a2cDefs.gae_lambda.step!}
                recommended={a2cDefs.gae_lambda.recommended as number}
                onChange={(v) => setA2cParams({ gae_lambda: v })}
              />
            )}

            {a2cDefs.ent_coef && (
              <ParamSlider disabled={isActive}
                id="ent_coef" label={t('sidebar.ent_coef')}
                value={a2cParams.ent_coef}
                min={a2cDefs.ent_coef.min!} max={a2cDefs.ent_coef.max!} step={a2cDefs.ent_coef.step!}
                recommended={a2cDefs.ent_coef.recommended as number}
                onChange={(v) => setA2cParams({ ent_coef: v })}
              />
            )}

            {a2cDefs.n_hidden_layers && (
              <ParamSlider disabled={isActive}
                id="n_hidden_layers" label={t('sidebar.n_hidden_layers')}
                value={a2cParams.n_hidden_layers}
                min={a2cDefs.n_hidden_layers.min!} max={a2cDefs.n_hidden_layers.max!} step={a2cDefs.n_hidden_layers.step!}
                recommended={a2cDefs.n_hidden_layers.recommended as number}
                onChange={(v) => setA2cParams({ n_hidden_layers: Math.round(v) })}
              />
            )}

            {a2cDefs.neurons_per_layer && (
              <ParamSlider disabled={isActive}
                id="neurons_per_layer" label={t('sidebar.neurons_per_layer')}
                value={a2cParams.neurons_per_layer}
                min={a2cDefs.neurons_per_layer.min!} max={a2cDefs.neurons_per_layer.max!} step={a2cDefs.neurons_per_layer.step!}
                recommended={a2cDefs.neurons_per_layer.recommended as number}
                onChange={(v) => setA2cParams({ neurons_per_layer: Math.round(v) })}
              />
            )}

            {a2cDefs.activation && (
              <ActivationToggle disabled={isActive}
                value={a2cParams.activation}
                label={t('sidebar.activation')}
                onChange={(v) => setA2cParams({ activation: v })}
              />
            )}
          </>
        )}

        {/* QR-DQN (S5e — distributional DQN): the same off-policy knobs as DQN (learning rate, discount,
            replay buffer, update frequency, target sync, and the two ε-greedy exploration knobs — sharing
            DQN's slider ids so they read identically), plus the ONE knob DQN doesn't have: n_quantiles,
            how many quantiles represent each action's return distribution. */}
        {isQrdqn && (
          <>
            {qrdqnDefs.learning_rate && (
              <ParamSlider disabled={isActive}
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={qrdqnParams.learning_rate}
                min={qrdqnDefs.learning_rate.min!} max={qrdqnDefs.learning_rate.max!} step={0.01}
                recommended={qrdqnDefs.learning_rate.recommended as number}
                onChange={(v) => setQrdqnParams({ learning_rate: v })}
              />
            )}

            {qrdqnDefs.gamma && (
              <ParamSlider disabled={isActive}
                id="gamma" label={t('sidebar.gamma')}
                value={qrdqnParams.gamma}
                min={qrdqnDefs.gamma.min!} max={qrdqnDefs.gamma.max!} step={qrdqnDefs.gamma.step!}
                recommended={qrdqnDefs.gamma.recommended as number}
                onChange={(v) => setQrdqnParams({ gamma: v })}
              />
            )}

            {qrdqnDefs.n_quantiles && (
              <ParamSlider disabled={isActive}
                id="qrdqn_n_quantiles" label={t('sidebar.qrdqn_n_quantiles')}
                value={qrdqnParams.n_quantiles}
                min={qrdqnDefs.n_quantiles.min!} max={qrdqnDefs.n_quantiles.max!} step={qrdqnDefs.n_quantiles.step!}
                recommended={qrdqnDefs.n_quantiles.recommended as number}
                onChange={(v) => setQrdqnParams({ n_quantiles: Math.round(v) })}
              />
            )}

            {qrdqnDefs.buffer_size && (
              <ParamSlider disabled={isActive}
                id="dqn_buffer_size" label={t('sidebar.dqn_buffer_size')}
                value={qrdqnParams.buffer_size}
                min={qrdqnDefs.buffer_size.min!} max={qrdqnDefs.buffer_size.max!} step={qrdqnDefs.buffer_size.step!}
                recommended={qrdqnDefs.buffer_size.recommended as number}
                onChange={(v) => setQrdqnParams({ buffer_size: Math.round(v) })}
              />
            )}

            {qrdqnDefs.train_freq && (
              <ParamSlider disabled={isActive}
                id="dqn_train_freq" label={t('sidebar.dqn_train_freq')}
                value={qrdqnParams.train_freq}
                min={qrdqnDefs.train_freq.min!} max={qrdqnDefs.train_freq.max!} step={qrdqnDefs.train_freq.step!}
                recommended={qrdqnDefs.train_freq.recommended as number}
                onChange={(v) => setQrdqnParams({ train_freq: Math.round(v) })}
              />
            )}

            {qrdqnDefs.target_update_interval && (
              <ParamSlider disabled={isActive}
                id="dqn_target_update" label={t('sidebar.dqn_target_update')}
                value={qrdqnParams.target_update_interval}
                min={qrdqnDefs.target_update_interval.min!} max={qrdqnDefs.target_update_interval.max!} step={qrdqnDefs.target_update_interval.step!}
                recommended={qrdqnDefs.target_update_interval.recommended as number}
                onChange={(v) => setQrdqnParams({ target_update_interval: Math.round(v) })}
              />
            )}

            {qrdqnDefs.exploration_fraction && (
              <ParamSlider disabled={isActive}
                id="dqn_exploration_fraction" label={t('sidebar.dqn_exploration_fraction')}
                value={qrdqnParams.exploration_fraction}
                min={qrdqnDefs.exploration_fraction.min!} max={qrdqnDefs.exploration_fraction.max!} step={qrdqnDefs.exploration_fraction.step!}
                recommended={qrdqnDefs.exploration_fraction.recommended as number}
                onChange={(v) => setQrdqnParams({ exploration_fraction: v })}
              />
            )}

            {qrdqnDefs.exploration_final_eps && (
              <ParamSlider disabled={isActive}
                id="dqn_exploration_final_eps" label={t('sidebar.dqn_exploration_final_eps')}
                value={qrdqnParams.exploration_final_eps}
                min={qrdqnDefs.exploration_final_eps.min!} max={qrdqnDefs.exploration_final_eps.max!} step={qrdqnDefs.exploration_final_eps.step!}
                recommended={qrdqnDefs.exploration_final_eps.recommended as number}
                onChange={(v) => setQrdqnParams({ exploration_final_eps: v })}
              />
            )}
          </>
        )}

        {/* Divider */}
        <div style={{ height: 1, background: 'var(--border-default)', margin: 'var(--space-3) 0 var(--space-4)' }} />

        {/* Seed */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--space-3)' }}>
          <label style={fieldLabel}>
            {t('sidebar.seed')}
            <ParamInfo paramId="seed" label={t('sidebar.seed')} />
          </label>
          <div style={{
            display: 'flex', alignItems: 'center', gap: 8, height: 'var(--control-sm)',
            padding: '0 10px', background: 'var(--surface-inset)',
            border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
            opacity: isActive ? 0.5 : 1,
          }}>
            <input
              type="number"
              aria-label={t('sidebar.seed')}
              min={0} max={999999}
              value={seed}
              disabled={isActive}
              onChange={(e) => setSeed(Math.max(0, parseInt(e.target.value, 10) || 0))}
              style={{
                width: 56, textAlign: 'right', background: 'transparent', border: 'none', outline: 'none',
                color: 'var(--text-strong)', fontFamily: 'var(--font-mono)',
                fontFeatureSettings: 'var(--ff-tabular)', fontSize: 'var(--fs-label)',
              }}
            />
            <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-faint)', fontFamily: 'var(--font-mono)' }}>int</span>
          </div>
        </div>

        {/* Total Steps — PPO + SAC + TD3 + DQN + A2C + QR-DQN (all run an env-step budget; evolution uses
            Generations, Q-learning Episodes, AlphaZero Iterations). */}
        {(algo === 'ppo' || algo === 'sac' || algo === 'td3' || algo === 'dqn' || algo === 'a2c' || algo === 'qrdqn') && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={fieldLabel}>
              {t('sidebar.total_steps')}
              <ParamInfo paramId="total_steps" label={t('sidebar.total_steps')} />
            </label>
            <LabSelect
              ariaLabel={t('sidebar.total_steps')}
              value={String(totalTimesteps)}
              disabled={isActive}
              onChange={(v) => setTotalTimesteps(parseInt(v, 10))}
              style={{ fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)' }}
              options={stepsOptions.map((n) => ({
                value: String(n),
                label: n === defaultSteps ? `${formatSteps(n)} ★` : formatSteps(n),
              }))}
            />
          </div>
        )}
      </div>

      {/* Train-gated note: the game can't be trained here yet. Reasons — the optional ale-py package
          missing so the Atari family is unusable (R1/ADR-101), a missing GPU (the vector heavies, still
          human-playable), an unbuilt image trainer (Atari/CarRacing, still playable), or a watch-only
          multi-agent env whose per-species trainer isn't built yet (simple_tag, G7b). */}
      {trainGated && !isActive && (
        <div style={{
          padding: '0 var(--space-5)', flexShrink: 0,
          fontSize: 'var(--fs-meta)', lineHeight: 1.45, color: 'var(--text-muted)',
        }}>
          {t(
            trainGatedReason === 'no_atari' ? 'sidebar.atari_needs_ale'
            : trainGatedReason === 'not_implemented_ma' ? 'sidebar.train_not_implemented_ma'
            : trainGatedReason === 'not_implemented_board' ? 'sidebar.train_not_implemented_board'
            : trainGatedReason === 'not_implemented' ? 'sidebar.train_not_implemented'
            : 'sidebar.train_needs_gpu',
          )}
        </div>
      )}

      {/* Seed-sweep progress (X3): while a sweep drains its queue, show which seed of N is live so the
          user knows a batch is running (Stop below cancels the whole sweep). */}
      {sweep && isActive && (
        <div style={{
          padding: '0 var(--space-5)', flexShrink: 0,
          fontSize: 'var(--fs-meta)', lineHeight: 1.45, color: 'var(--text-muted)',
          display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span aria-hidden style={{ color: 'var(--accent)' }}>⚂</span>
          <span>{t('sidebar.sweep_progress', { index: sweep.index, total: sweep.total, seed: sweep.running_seed })}</span>
        </div>
      )}

      {/* Run controls — outer div is always the flex row so flex:1 works in every branch */}
      <div style={{
        padding: 'var(--space-4) var(--space-5)', borderTop: '1px solid var(--border-default)',
        flexShrink: 0, display: 'flex', gap: 'var(--space-2)',
      }}>
        {isStopping ? (
          <button disabled style={runBtn('disabled', true)}>{t('sidebar.stopping')}</button>
        ) : isRunning ? (
          <>
            <button onClick={handlePause} className="btn-press" style={runBtn('pause')}>{PauseGlyph} {t('sidebar.pause')}</button>
            <button onClick={handleStop} className="btn-press" style={runBtn('stop')}>{StopGlyph} {t(sweep ? 'sidebar.cancel_sweep' : 'sidebar.stop')}</button>
          </>
        ) : isPaused ? (
          <>
            <button onClick={handleResume} className="btn-press" style={runBtn('resume')}>{PlayGlyph} {t('sidebar.resume')}</button>
            <button onClick={handleStop} className="btn-press" style={runBtn('stop')}>{StopGlyph} {t(sweep ? 'sidebar.cancel_sweep' : 'sidebar.stop')}</button>
          </>
        ) : (
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            <button onClick={handleRun} disabled={!canRun} className={canRun ? 'btn-cta' : undefined} style={canRun ? runBtn('primary', true) : runBtn('disabled', true)}>
              {PlayGlyph} {t('sidebar.run')}
            </button>
            {/* Seed sweep (X3): run the current config across N seeds (seed … seed+N−1), queued
                sequentially, for multi-seed analysis (X4). The seed input above is the first seed. */}
            <div style={{ display: 'flex', gap: 'var(--space-2)', alignItems: 'center' }}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6, height: 'var(--control-md)',
                padding: '0 10px', background: 'var(--surface-inset)',
                border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
                opacity: canRun ? 1 : 0.5,
              }}>
                <span style={{ ...fieldLabel, gap: 4 }}>
                  {t('sidebar.sweep_seeds')}
                  <ParamInfo paramId="sweep_count" label={t('sidebar.sweep_seeds')} />
                </span>
                <input
                  type="number"
                  aria-label={t('sidebar.sweep_seeds')}
                  min={1} max={20}
                  value={sweepCount}
                  disabled={!canRun}
                  onChange={(e) => setSweepCount(Math.min(20, Math.max(1, parseInt(e.target.value, 10) || 1)))}
                  style={{
                    width: 34, textAlign: 'right', background: 'transparent', border: 'none', outline: 'none',
                    color: 'var(--text-strong)', fontFamily: 'var(--font-mono)',
                    fontFeatureSettings: 'var(--ff-tabular)', fontSize: 'var(--fs-label)',
                  }}
                />
              </div>
              <button
                onClick={handleRunSweep}
                disabled={!canRun}
                title={t('sidebar.run_sweep_hint')}
                className={canRun ? 'btn-cta' : undefined}
                style={canRun ? runBtn('primary') : runBtn('disabled')}
              >
                {t('sidebar.run_sweep', { count: sweepCount })}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Save / Load / Manage (D1) — the checkpoint slots live in modals here, not in the dashboard. */}
      <div style={{ padding: '0 var(--space-5) var(--space-4)', flexShrink: 0 }}>
        <SaveLoadControls />
      </div>
    </aside>
  )
}
