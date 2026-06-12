import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { useRunControls } from '../api/trainingControls'
import type { Algo } from '../api/types'
import ParamInfo from './ParamInfo'

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
    <div style={{ marginBottom: 9 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2, alignItems: 'baseline' }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          {label}
          <ParamInfo paramId={id} label={label} />
        </span>
        <span style={{
          fontSize: 11, fontFamily: 'monospace',
          color: isRec ? 'var(--ok)' : 'var(--text)',
        }}>
          {formatValue(id, value)}
        </span>
      </div>
      {/* Slider + recommended tick */}
      <div style={{ position: 'relative' }}>
        <input
          type="range"
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
            height: 10,
            background: 'var(--ok)',
            borderRadius: 1,
            pointerEvents: 'none',
            opacity: 0.85,
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
    <div style={{ marginBottom: 9 }}>
      <div style={{ marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
          {label}
          <ParamInfo paramId="activation" label={label} />
        </span>
      </div>
      <div style={{ display: 'flex', gap: 5 }}>
        {(['tanh', 'relu'] as const).map((opt) => {
          const active = value === opt
          return (
            <button
              key={opt}
              onClick={() => !disabled && onChange(opt)}
              style={{
                flex: 1, padding: '3px 0',
                background: active ? 'var(--accent)' : 'var(--surface-2)',
                color: active ? '#fff' : 'var(--text-muted)',
                border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius: 4, cursor: disabled ? 'default' : 'pointer',
                fontSize: 11, fontWeight: active ? 600 : 400,
              }}
            >
              {opt === 'tanh' ? '★ tanh' : 'relu'}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── AlgoSwitch ───────────────────────────────────────────────────────────────

function AlgoSwitch({ value, disabled, onChange }: {
  value: Algo
  disabled?: boolean
  onChange: (a: Algo) => void
}) {
  const { t } = useTranslation()
  const opts: { id: Algo; label: string }[] = [
    { id: 'ppo', label: t('sidebar.algo_ppo') },
    { id: 'neuroevolution', label: t('sidebar.algo_evo') },
  ]
  return (
    <div>
      <label style={{
        display: 'block', marginBottom: 4,
        fontSize: 11, color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.05em',
      }}>
        {t('sidebar.algorithm')}
      </label>
      <div style={{ display: 'flex', gap: 5 }}>
        {opts.map((opt) => {
          const active = value === opt.id
          return (
            <button
              key={opt.id}
              onClick={() => !disabled && onChange(opt.id)}
              disabled={disabled}
              style={{
                flex: 1, padding: '5px 0',
                background: active ? 'var(--accent)' : 'var(--surface-2)',
                color: active ? '#fff' : 'var(--text-muted)',
                border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius: 4, cursor: disabled ? 'default' : 'pointer',
                fontSize: 11, fontWeight: active ? 600 : 400,
              }}
            >
              {opt.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}

// ── Run controls ─────────────────────────────────────────────────────────────

function btnStyle(bg: string, disabled = false): CSSProperties {
  return {
    flex: 1, padding: '11px 0',
    background: disabled ? 'var(--surface-2)' : bg,
    color: disabled ? 'var(--text-muted)' : '#fff',
    border: 'none', borderRadius: 6,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontSize: 13, fontWeight: 700,
    letterSpacing: '0.02em',
  }
}

const STEPS_OPTIONS = [10_000, 25_000, 50_000, 100_000, 200_000]

// ── Sidebar ──────────────────────────────────────────────────────────────────

export default function Sidebar() {
  const { t } = useTranslation()

  const envs            = useAppStore((s) => s.envs)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const locale          = useAppStore((s) => s.locale)
  const setSelectedEnvId = useAppStore((s) => s.setSelectedEnvId)

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

  return (
    <aside style={{
      width: 264, flexShrink: 0,
      background: 'var(--surface)', borderRight: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 14px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 13, color: 'var(--text-h)', flexShrink: 0,
      }}>
        {t('sidebar.title')}
      </div>

      {/* Game selector + algorithm switch */}
      <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 10 }}>
        <div>
          <label style={{
            display: 'block', marginBottom: 4,
            fontSize: 11, color: 'var(--text-muted)',
            textTransform: 'uppercase', letterSpacing: '0.05em',
          }}>
            {t('sidebar.game_selector')}
          </label>
          <select
            value={selectedEnvId ?? ''}
            onChange={(e) => setSelectedEnvId(e.target.value || null)}
            disabled={envs.length === 0 || isActive}
            style={{
              width: '100%', padding: '5px 8px',
              background: 'var(--surface-2)', color: 'var(--text)',
              border: '1px solid var(--border)', borderRadius: 4,
              fontSize: 13, cursor: envs.length === 0 || isActive ? 'default' : 'pointer',
            }}
          >
            {envs.length === 0
              ? <option value="">{t('sidebar.loading_envs')}</option>
              : envs.map((env) => (
                  <option key={env.id} value={env.id}>
                    {env.display_name[locale]}
                  </option>
                ))
            }
          </select>
        </div>

        <AlgoSwitch value={algo} disabled={isActive} onChange={setAlgo} />
      </div>

      {/* Scrollable params */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '10px 14px' }}>
        <div style={{
          fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.05em',
          textTransform: 'uppercase', marginBottom: 10,
        }}>
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
        <div style={{ borderTop: '1px solid var(--border)', margin: '10px 0' }} />

        {/* Seed */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
            {t('sidebar.seed')}
            <ParamInfo paramId="seed" label={t('sidebar.seed')} />
          </label>
          <input
            type="number"
            min={0} max={999999}
            value={seed}
            disabled={isActive}
            onChange={(e) => setSeed(Math.max(0, parseInt(e.target.value, 10) || 0))}
            style={{
              width: 72, padding: '3px 6px', textAlign: 'right',
              background: 'var(--surface-2)', color: 'var(--text)',
              border: '1px solid var(--border)', borderRadius: 4, fontSize: 12,
            }}
          />
        </div>

        {/* Total Steps — PPO only (evolution is bounded by Generations, not a step budget) */}
        {!isEvo && (
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              {t('sidebar.total_steps')}
              <ParamInfo paramId="total_steps" label={t('sidebar.total_steps')} />
            </label>
            <select
              value={totalTimesteps}
              disabled={isActive}
              onChange={(e) => setTotalTimesteps(parseInt(e.target.value, 10))}
              style={{
                padding: '3px 6px',
                background: 'var(--surface-2)', color: 'var(--text)',
                border: '1px solid var(--border)', borderRadius: 4,
                fontSize: 12, cursor: isActive ? 'default' : 'pointer',
              }}
            >
              {STEPS_OPTIONS.map((n) => (
                <option key={n} value={n}>
                  {n === 50_000 ? `${n / 1000}k ★` : `${n / 1000}k`}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Run controls — outer div is always the flex row so flex:1 works in every branch */}
      <div style={{
        padding: '10px 14px', borderTop: '1px solid var(--border)',
        flexShrink: 0, display: 'flex', gap: 6,
      }}>
        {isStopping ? (
          <button disabled style={btnStyle('', true)}>{t('sidebar.stopping')}</button>
        ) : isRunning ? (
          <>
            <button onClick={handlePause} style={btnStyle('var(--warn)')}>
              ⏸ {t('sidebar.pause')}
            </button>
            <button onClick={handleStop} style={btnStyle('var(--err)')}>
              ⏹ {t('sidebar.stop')}
            </button>
          </>
        ) : isPaused ? (
          <>
            <button onClick={handleResume} style={btnStyle('var(--ok)')}>
              ▶ {t('sidebar.resume')}
            </button>
            <button onClick={handleStop} style={btnStyle('var(--err)')}>
              ⏹ {t('sidebar.stop')}
            </button>
          </>
        ) : (
          <button onClick={handleRun} disabled={!canRun} style={btnStyle('var(--accent)', !canRun)}>
            ▶ {t('sidebar.run')}
          </button>
        )}
      </div>
    </aside>
  )
}
