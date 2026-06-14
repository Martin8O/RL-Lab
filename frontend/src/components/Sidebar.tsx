import type { CSSProperties, ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { useRunControls } from '../api/trainingControls'
import type { Algo } from '../api/types'
import ParamInfo from './ParamInfo'
import SaveLoadControls from './SaveLoadControls'
import EnvSelector from './EnvSelector'

// ── Shared style helpers ─────────────────────────────────────────────────────

const fieldLabel: CSSProperties = {
  fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)', color: 'var(--text-muted)',
  display: 'inline-flex', alignItems: 'center', gap: 4,
}
const sectionEyebrow: CSSProperties = {
  fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)',
  letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase', color: 'var(--text-faint)',
}
const selectStyle: CSSProperties = {
  width: '100%', height: 'var(--control-md)', padding: '0 12px',
  background: 'var(--surface-2)', color: 'var(--text-strong)',
  border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
  fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-sm)', cursor: 'pointer',
  transition: 'var(--t-colors)',
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
    case 'learning_rate': {
      const exp = Math.floor(Math.log10(v))
      return `${(v / Math.pow(10, exp)).toFixed(2)}e${exp}`
    }
    case 'gamma':          return v.toFixed(4)
    case 'clip_range':     return v.toFixed(2)
    case 'ent_coef':       return v.toFixed(3)
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
          color: isRec ? 'var(--success)' : 'var(--text-strong)',
        }}>
          {formatValue(id, value)}
        </span>
      </div>
      {/* Slider + recommended tick */}
      <div style={{ position: 'relative' }}>
        <input
          type="range"
          aria-label={label}
          min={sMin} max={sMax} step={log ? 0.01 : step}
          value={sVal}
          disabled={disabled}
          onChange={(e) => {
            const raw = parseFloat(e.target.value)
            onChange(log ? Math.pow(10, raw) : raw)
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
    default:               return id
  }
}

function AlgoSwitch({ value, options, disabled, onChange }: {
  value: Algo
  options: string[]
  disabled?: boolean
  onChange: (a: Algo) => void
}) {
  const { t } = useTranslation()
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <label style={fieldLabel}>
        {t('sidebar.algorithm')}
        <ParamInfo paramId="algorithm" label={t('sidebar.algorithm')} />
      </label>
      <select
        aria-label={t('sidebar.algorithm')}
        value={value}
        disabled={disabled || options.length === 0}
        onChange={(e) => onChange(e.target.value as Algo)}
        style={{ ...selectStyle, cursor: disabled || options.length === 0 ? 'default' : 'pointer' }}
      >
        {options.map((id) => (
          <option key={id} value={id}>{ALGO_LABEL(t, id)}</option>
        ))}
      </select>
    </div>
  )
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
    case 'primary':  return { ...base, background: 'var(--accent)', color: 'var(--accent-contrast)', boxShadow: 'var(--shadow-xs)' }
    case 'resume':   return { ...base, background: 'var(--success-surface)', color: 'var(--success)' }
    case 'pause':    return { ...base, background: 'var(--warning-surface)', color: 'var(--warning)' }
    case 'stop':     return { ...base, background: 'var(--danger-surface)', color: 'var(--danger)' }
    case 'disabled': return { ...base, background: 'var(--surface-2)', color: 'var(--text-muted)', borderColor: 'var(--border-default)', cursor: 'not-allowed' }
  }
}

// The step-budget dropdown is built per-env from the registry's default_total_timesteps: a
// ladder of ×0.2 … ×4 around the recommended value, with ★ on the recommendation. So CartPole
// (50k → 10k…200k) and LunarLander (500k → 100k…2M) each get an appropriate range.
const STEP_FACTORS = [0.2, 0.5, 1, 2, 4]

function stepsLadder(defaultSteps: number): number[] {
  return STEP_FACTORS.map((f) => Math.round((defaultSteps * f) / 1000) * 1000)
}

function formatSteps(n: number): string {
  return n >= 1_000_000 ? `${n / 1_000_000}M` : `${Math.round(n / 1000)}k`
}

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
  const seed            = useAppStore((s) => s.seed)
  const setSeed         = useAppStore((s) => s.setSeed)
  const totalTimesteps  = useAppStore((s) => s.totalTimesteps)
  const setTotalTimesteps = useAppStore((s) => s.setTotalTimesteps)

  const { handleRun, handlePause, handleResume, handleStop, isRunning, isPaused, isStopping, isActive, canRun } =
    useRunControls()

  const selectedEnv = envs.find((e) => e.id === selectedEnvId)
  const ppoDefs = selectedEnv?.hyperparams?.['ppo'] ?? {}
  const evoDefs = selectedEnv?.hyperparams?.['neuroevolution'] ?? {}
  const isEvo = algo === 'neuroevolution'

  // Per-env step ladder + the ★ recommended budget; always include the current value so the
  // <select> can render it even after a reload with a value off the ladder.
  const defaultSteps = selectedEnv?.default_total_timesteps ?? 50_000
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
        display: 'flex', alignItems: 'center',
        fontWeight: 'var(--fw-semibold)', fontSize: 'var(--fs-sm)', color: 'var(--text-strong)',
      }}>
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
          disabled={isActive}
          onChange={setAlgo}
        />
      </div>

      {/* Scrollable params */}
      <div style={{ flex: 1, overflowY: 'auto', padding: 'var(--space-5)' }}>
        <div style={{ ...sectionEyebrow, marginBottom: 'var(--space-4)' }}>
          {isEvo ? t('sidebar.evo_params_title') : t('sidebar.params_title')}
        </div>

        {!isEvo && (
          <>
            {ppoDefs.learning_rate && (
              <ParamSlider
                id="learning_rate" label={t('sidebar.learning_rate')}
                value={hyperparams.learning_rate}
                min={ppoDefs.learning_rate.min!} max={ppoDefs.learning_rate.max!} step={0.01}
                recommended={ppoDefs.learning_rate.recommended as number}
                onChange={(v) => setHyperparams({ learning_rate: v })}
              />
            )}

            {ppoDefs.gamma && (
              <ParamSlider
                id="gamma" label={t('sidebar.gamma')}
                value={hyperparams.gamma}
                min={ppoDefs.gamma.min!} max={ppoDefs.gamma.max!} step={ppoDefs.gamma.step!}
                recommended={ppoDefs.gamma.recommended as number}
                onChange={(v) => setHyperparams({ gamma: v })}
              />
            )}

            {ppoDefs.clip_range && (
              <ParamSlider
                id="clip_range" label={t('sidebar.clip_range')}
                value={hyperparams.clip_range}
                min={ppoDefs.clip_range.min!} max={ppoDefs.clip_range.max!} step={ppoDefs.clip_range.step!}
                recommended={ppoDefs.clip_range.recommended as number}
                onChange={(v) => setHyperparams({ clip_range: v })}
              />
            )}

            {ppoDefs.ent_coef && (
              <ParamSlider
                id="ent_coef" label={t('sidebar.ent_coef')}
                value={hyperparams.ent_coef}
                min={ppoDefs.ent_coef.min!} max={ppoDefs.ent_coef.max!} step={ppoDefs.ent_coef.step!}
                recommended={ppoDefs.ent_coef.recommended as number}
                onChange={(v) => setHyperparams({ ent_coef: v })}
              />
            )}

            {ppoDefs.n_hidden_layers && (
              <ParamSlider
                id="n_hidden_layers" label={t('sidebar.n_hidden_layers')}
                value={hyperparams.n_hidden_layers}
                min={ppoDefs.n_hidden_layers.min!} max={ppoDefs.n_hidden_layers.max!} step={ppoDefs.n_hidden_layers.step!}
                recommended={ppoDefs.n_hidden_layers.recommended as number}
                onChange={(v) => setHyperparams({ n_hidden_layers: Math.round(v) })}
              />
            )}

            {ppoDefs.neurons_per_layer && (
              <ParamSlider
                id="neurons_per_layer" label={t('sidebar.neurons_per_layer')}
                value={hyperparams.neurons_per_layer}
                min={ppoDefs.neurons_per_layer.min!} max={ppoDefs.neurons_per_layer.max!} step={ppoDefs.neurons_per_layer.step!}
                recommended={ppoDefs.neurons_per_layer.recommended as number}
                onChange={(v) => setHyperparams({ neurons_per_layer: Math.round(v) })}
              />
            )}

            {ppoDefs.activation && (
              <ActivationToggle
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
              <ParamSlider
                id="population_size" label={t('sidebar.population_size')}
                value={evolutionParams.population_size}
                min={evoDefs.population_size.min!} max={evoDefs.population_size.max!} step={evoDefs.population_size.step!}
                recommended={evoDefs.population_size.recommended as number}
                onChange={(v) => setEvolutionParams({ population_size: Math.round(v) })}
              />
            )}

            {evoDefs.top_k_parents && (
              <ParamSlider
                id="top_k_parents" label={t('sidebar.top_k_parents')}
                value={evolutionParams.top_k_parents}
                min={evoDefs.top_k_parents.min!} max={evoDefs.top_k_parents.max!} step={evoDefs.top_k_parents.step!}
                recommended={evoDefs.top_k_parents.recommended as number}
                onChange={(v) => setEvolutionParams({ top_k_parents: Math.round(v) })}
              />
            )}

            {evoDefs.mutation_rate && (
              <ParamSlider
                id="mutation_rate" label={t('sidebar.mutation_rate')}
                value={evolutionParams.mutation_rate}
                min={evoDefs.mutation_rate.min!} max={evoDefs.mutation_rate.max!} step={evoDefs.mutation_rate.step!}
                recommended={evoDefs.mutation_rate.recommended as number}
                onChange={(v) => setEvolutionParams({ mutation_rate: v })}
              />
            )}

            {evoDefs.crossover_rate && (
              <ParamSlider
                id="crossover_rate" label={t('sidebar.crossover_rate')}
                value={evolutionParams.crossover_rate}
                min={evoDefs.crossover_rate.min!} max={evoDefs.crossover_rate.max!} step={evoDefs.crossover_rate.step!}
                recommended={evoDefs.crossover_rate.recommended as number}
                onChange={(v) => setEvolutionParams({ crossover_rate: v })}
              />
            )}

            {evoDefs.generations && (
              <ParamSlider
                id="generations" label={t('sidebar.generations')}
                value={evolutionParams.generations}
                min={evoDefs.generations.min!} max={evoDefs.generations.max!} step={evoDefs.generations.step!}
                recommended={evoDefs.generations.recommended as number}
                onChange={(v) => setEvolutionParams({ generations: Math.round(v) })}
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

        {/* Total Steps — PPO only (evolution is bounded by Generations, not a step budget) */}
        {!isEvo && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={fieldLabel}>
              {t('sidebar.total_steps')}
              <ParamInfo paramId="total_steps" label={t('sidebar.total_steps')} />
            </label>
            <select
              aria-label={t('sidebar.total_steps')}
              value={totalTimesteps}
              disabled={isActive}
              onChange={(e) => setTotalTimesteps(parseInt(e.target.value, 10))}
              style={{
                width: 'auto', height: 'var(--control-sm)', padding: '0 10px',
                background: 'var(--surface-2)', color: 'var(--text-strong)',
                border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
                fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
                fontSize: 'var(--fs-label)', cursor: isActive ? 'default' : 'pointer',
              }}
            >
              {stepsOptions.map((n) => (
                <option key={n} value={n}>
                  {n === defaultSteps ? `${formatSteps(n)} ★` : formatSteps(n)}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Run controls — outer div is always the flex row so flex:1 works in every branch */}
      <div style={{
        padding: 'var(--space-4) var(--space-5)', borderTop: '1px solid var(--border-default)',
        flexShrink: 0, display: 'flex', gap: 'var(--space-2)',
      }}>
        {isStopping ? (
          <button disabled style={runBtn('disabled', true)}>{t('sidebar.stopping')}</button>
        ) : isRunning ? (
          <>
            <button onClick={handlePause} style={runBtn('pause')}>{PauseGlyph} {t('sidebar.pause')}</button>
            <button onClick={handleStop} style={runBtn('stop')}>{StopGlyph} {t('sidebar.stop')}</button>
          </>
        ) : isPaused ? (
          <>
            <button onClick={handleResume} style={runBtn('resume')}>{PlayGlyph} {t('sidebar.resume')}</button>
            <button onClick={handleStop} style={runBtn('stop')}>{StopGlyph} {t('sidebar.stop')}</button>
          </>
        ) : (
          <button onClick={handleRun} disabled={!canRun} style={canRun ? runBtn('primary', true) : runBtn('disabled', true)}>
            {PlayGlyph} {t('sidebar.run')}
          </button>
        )}
      </div>

      {/* Save / Load / Manage (D1) — the checkpoint slots live in modals here, not in the dashboard. */}
      <div style={{ padding: '0 var(--space-5) var(--space-4)', flexShrink: 0 }}>
        <SaveLoadControls />
      </div>
    </aside>
  )
}
