import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { currentBand, scaleForPlay, scaleFromEnvSkill, scaleFromEnvSpec, skillScaleFor } from '../content/skill'

// Fixed low→high skill gradient (red = beginner, green = superhuman). Theme-agnostic.
const GRADIENT =
  'linear-gradient(to right, #e2453c 0%, #f0883e 28%, #e3c000 50%, #86c440 75%, #3fae4f 100%)'

const MARKER_COLORS = { human: '#4aa3ff', ai: '#c07cf0' } as const

/** The dashboard's single, context-aware skill gauge (one full-width row under both top panels).
 *
 *  Training and play-vs-AI never need a skill readout at the same time, so instead of two identical
 *  meters this one **relabels by whichever context is live**:
 *   - a play session (active or just finished) wins → shows the play rating: *Your skill* when the
 *     human plays, *AI skill* when watching the AI, plus starred best-human / best-AI record marks;
 *   - otherwise → the training agent's live skill (*AI skill*).
 *
 *  It reads everything from the store itself (no props) so App can mount it once, full-width. */
export default function SkillMeter({ slot, overlay = false }: {
  slot: 'play' | 'train'
  /** When true, float as an absolute HUD chip inside the (position:relative) stage instead of a
   *  footer row — reclaims the vertical space below the visualization for the stats panels. */
  overlay?: boolean
}) {
  const { t } = useTranslation()
  const envId    = useAppStore((s) => s.selectedEnvId)
  const envs     = useAppStore((s) => s.envs)
  const envSkill = useAppStore((s) => s.envSkill)

  const playState      = useAppStore((s) => s.playState)
  const playMode       = useAppStore((s) => s.playMode)
  const playScore      = useAppStore((s) => s.playScore)
  const playScores     = useAppStore((s) => s.playScores)
  const trainState     = useAppStore((s) => s.trainState)
  const algo           = useAppStore((s) => s.algo)
  const lastProgress   = useAppStore((s) => s.lastProgress)
  const metricsHistory = useAppStore((s) => s.metricsHistory)
  const lastEvolution  = useAppStore((s) => s.lastEvolution)

  // One meter, relabelled by what's relevant now: you actively playing always wins (you're at the
  // keyboard); otherwise a training run that has started this session owns the readout — its live
  // value, and its final value after it stops, so it won't snap back to an old play score. A
  // finished play result only lingers as feedback when no training has run this session (e.g. a
  // reconciled prior session on a fresh load).
  const playActive = playState === 'playing'
  const trainStarted = trainState !== 'idle'
  const playVisible = playActive || (!trainStarted && playState !== 'idle')

  // One meter, but it lives in BOTH visualization panels and shows in whichever matches the live
  // context: the env/cart panel while playing, the chart panel while training (or idle). The other
  // instance renders nothing — so there's never a duplicate, and no separate full-width row.
  if (slot === 'play' && !playVisible) return null
  if (slot === 'train' && playVisible) return null

  let score: number | null
  let titleKey: string
  let markerHuman: number | null = null
  let markerAi: number | null = null
  if (playVisible) {
    score = playScore
    titleKey = playMode === 'human' ? 'skill.you' : 'skill.title'
    markerHuman = playScores?.human[0]?.score ?? null
    markerAi = playScores?.ai[0]?.score ?? null
  } else {
    const lastMetrics = metricsHistory.at(-1)
    score =
      algo === 'neuroevolution'
        ? lastEvolution?.best_fitness ?? null
        : lastProgress?.ep_rew_mean ?? lastMetrics?.ep_rew_mean ?? null
    titleKey = 'skill.title'
  }

  // Prefer the backend's per-env thresholds (single source of truth with the play rating), but
  // only when they're for the *currently selected* env — a stale fetch from a previous env must
  // never scale this one. Otherwise derive the scale from the selected env's own spec (so e.g.
  // CartPole's 0–500 can't leak onto LunarLander), and only then the local table.
  const baseScale =
    envSkill && envSkill.env_id === envId
      ? scaleFromEnvSkill(envSkill)
      : scaleFromEnvSpec(envs.find((e) => e.id === envId)) ?? skillScaleFor(envId)
  // While playing, widen the floor for envs whose play episode runs longer than training
  // (play_step_scale) so the meter span matches the longer episode (mirrors the backend rating).
  const playStepScale = envs.find((e) => e.id === envId)?.play_step_scale ?? 1
  const scale = playVisible ? scaleForPlay(baseScale, playStepScale) : baseScale

  const hasScore = score !== null
  const value = score ?? 0
  // A valid skill READING needs a meaningful score. For an env whose reward climbs from zero
  // (CartPole, scale.min >= 0) the running/partial score is a valid lower bound, so it reads live.
  // For shaped/penalty envs (scale.min < 0 — MountainCar, Acrobot, LunarLander) the running
  // cumulative score starts ABOVE the "solved" mark and only falls as steps/fuel are spent, so it
  // is NOT a reading until the episode ends — showing it would make the meter start full and drain
  // leftward (or, on an early stop, freeze on a bogus high band). So for those envs only a
  // *finished* episode produces a band; while playing we show "measuring…", and an aborted session
  // shows nothing. Training (not playVisible) reads ep_rew_mean, already a final-episode metric.
  const partialNotAReading = playVisible && scale.min < 0 && playState !== 'finished'
  const measuring = partialNotAReading && playState === 'playing'
  const showReading = hasScore && !partialNotAReading
  // Fill + tick positions measured across [scale.min, scale.max], so a shaped env that starts
  // negative (LunarLander, min -200) shows real progress through the red instead of a flat 0%.
  const span = scale.max - scale.min || 1
  const posOf = (v: number) => Math.max(0, Math.min(1, (v - scale.min) / span))
  const frac = posOf(value)
  const band = currentBand(value, scale)
  const ticks = scale.bands.slice(1).map((b) => posOf(b.min)) // band boundaries

  const recordMarks = [
    { key: 'human' as const, value: markerHuman },
    { key: 'ai' as const, value: markerAi },
  ]
    .filter((m): m is { key: 'human' | 'ai'; value: number } => typeof m.value === 'number' && m.value > 0)
    .map((m) => ({
      ...m,
      frac: posOf(m.value),
      color: MARKER_COLORS[m.key],
      label: t(m.key === 'human' ? 'playscore.best_human' : 'playscore.best_ai', { score: Math.round(m.value) }),
    }))

  const containerStyle: CSSProperties = overlay
    ? {
        position: 'absolute', left: 10, right: 10, bottom: 10, zIndex: 3,
        background: 'var(--surface-1)', border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-md)',
        padding: '5px 11px', minHeight: 30,
        display: 'flex', alignItems: 'center', gap: 10,
      }
    : {
        flexShrink: 0, borderTop: '1px solid var(--border-default)',
        background: 'var(--surface)', padding: '6px 12px', minHeight: 34,
        display: 'flex', alignItems: 'center', gap: 10,
      }

  return (
    <div style={containerStyle}>
      <span style={{
        fontSize: 11, fontWeight: 600, letterSpacing: '0.02em',
        color: 'var(--text-muted)', whiteSpace: 'nowrap',
      }}>
        {t(titleKey)}
      </span>

      {/* Gradient bar with band-boundary ticks + a needle at the current skill */}
      <div style={{ position: 'relative', flex: 1, height: 14 }}>
        <div style={{
          position: 'absolute', inset: 0, borderRadius: 7,
          background: GRADIENT, opacity: showReading ? 1 : 0.8,
        }} />
        {ticks.map((tk, i) => (
          <div key={i} style={{
            position: 'absolute', left: `${tk * 100}%`, top: 0, bottom: 0,
            width: 1, background: 'rgba(255,255,255,0.5)',
          }} />
        ))}
        {recordMarks.map((m) => (
          <div key={m.key} title={m.label} style={{
            position: 'absolute', left: `${m.frac * 100}%`, top: -3, bottom: -3,
            width: 2, marginLeft: -1, background: m.color, borderRadius: 1,
          }}>
            <span style={{
              position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)',
              fontSize: 9, lineHeight: 1, color: m.color, textShadow: '0 0 2px rgba(0,0,0,0.6)',
            }}>
              ★
            </span>
          </div>
        ))}
        {showReading && (
          <div style={{
            position: 'absolute', left: `${frac * 100}%`, top: -3, bottom: -3,
            width: 2, marginLeft: -1, background: 'var(--text-h)',
            borderRadius: 1, boxShadow: '0 0 3px rgba(0,0,0,0.55)',
          }}>
            <div style={{
              position: 'absolute', left: '50%', top: -4, transform: 'translateX(-50%)',
              width: 8, height: 8, borderRadius: '50%',
              background: 'var(--text-h)', border: '1.5px solid var(--surface)',
            }} />
          </div>
        )}
      </div>

      <span style={{
        fontSize: 11, fontWeight: 600, whiteSpace: 'nowrap',
        color: showReading ? 'var(--text-h)' : 'var(--text-muted)',
        minWidth: 120, textAlign: 'right',
      }}>
        {measuring
          ? t('skill.measuring')
          : showReading
            ? `${t(`skill.${band.key}`)} · ${Math.round(value)}`
            : t('skill.no_data')}
      </span>
    </div>
  )
}
