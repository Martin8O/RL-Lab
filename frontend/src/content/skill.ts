// Per-environment skill thresholds: map a run's Score → a skill band.
// CartPole's Score is mean episode length (max 500). As of E2 the live thresholds come from
// the backend (`GET /api/skill/{env}` → `scaleFromEnvSkill`, single source of truth); the table
// below is only a fallback used until that fetch lands (and a default for unknown envs).

import type { EnvSkill, EnvSpec, SkillBandId } from '../api/types'

export type SkillKey = 'child' | 'below' | 'average' | 'above' | 'superhuman'

// Backend band ids (schemas/skill.py) → the shorter frontend keys the i18n + UI already use.
const BAND_ID_TO_KEY: Record<SkillBandId, SkillKey> = {
  child:         'child',
  below_average: 'below',
  average:       'average',
  above_average: 'above',
  superhuman:    'superhuman',
}

export interface SkillBand {
  key: SkillKey
  /** inclusive lower bound on Score for this band */
  min: number
}

export interface SkillScale {
  /** bottom of the bar (0% fill) — 0 for CartPole, negative for shaped envs like LunarLander */
  min: number
  max: number
  bands: SkillBand[] // ascending by min; first band's min == scale.min
}

export const SKILL_SCALES: Record<string, SkillScale> = {
  cartpole: {
    min: 0,
    max: 500,
    bands: [
      { key: 'child', min: 0 },
      { key: 'below', min: 50 },
      { key: 'average', min: 150 },
      { key: 'above', min: 300 },
      { key: 'superhuman', min: 450 },
    ],
  },
}

export const DEFAULT_SKILL_SCALE: SkillScale = SKILL_SCALES.cartpole

export function skillScaleFor(envId: string | null): SkillScale {
  return (envId !== null && SKILL_SCALES[envId]) || DEFAULT_SKILL_SCALE
}

// Band lower bounds as a fraction of the env's [min_score, solved_score] range — must match the
// backend's _BAND_FRACTIONS (services/skill.py).
const BAND_FRACTIONS: { key: SkillKey; frac: number }[] = [
  { key: 'child', frac: 0 },
  { key: 'below', frac: 0.1 },
  { key: 'average', frac: 0.3 },
  { key: 'above', frac: 0.6 },
  { key: 'superhuman', frac: 0.95 },
]

/** Build the skill scale from an env's own registry spec (solved_score + min_score). The robust
 *  fallback when the backend thresholds haven't loaded yet — so the meter always uses the SELECTED
 *  env's range and never another env's (e.g. CartPole's 0–500) leaks onto LunarLander. */
export function scaleFromEnvSpec(env: EnvSpec | undefined): SkillScale | null {
  if (!env) return null
  const min = env.min_score ?? 0
  const max = env.solved_score
  const span = max - min
  return { min, max, bands: BAND_FRACTIONS.map((b) => ({ key: b.key, min: min + b.frac * span })) }
}

/** Widen a scale's 0% floor for longer **play** episodes (EnvSpec.play_step_scale). A
 *  step-penalty env played at 3× the steps has a ~3× deeper failure floor, while a *success*
 *  score is unchanged — so multiply min by the factor and recompute the bands over the new range
 *  (must mirror the backend's `env_skill(min_scale=...)`). Only meaningful for min < 0; a no-op at
 *  factor 1. */
export function scaleForPlay(scale: SkillScale, factor: number): SkillScale {
  if (factor === 1 || scale.min >= 0) return scale
  const min = scale.min * factor
  const span = scale.max - min
  return { min, max: scale.max, bands: BAND_FRACTIONS.map((b) => ({ key: b.key, min: min + b.frac * span })) }
}

/** Convert the backend's per-env thresholds into the local SkillScale shape (the meter's
 *  bands + the rating now share one source: the env's solved_score). */
export function scaleFromEnvSkill(envSkill: EnvSkill): SkillScale {
  return {
    min: envSkill.min_score ?? 0,  // default guards an older backend that predates min_score
    max: envSkill.max_score,
    bands: envSkill.bands.map((b) => ({ key: BAND_ID_TO_KEY[b.id], min: b.min_score })),
  }
}

/** The highest band whose threshold the score has reached. */
export function currentBand(score: number, scale: SkillScale): SkillBand {
  let band = scale.bands[0]
  for (const b of scale.bands) if (score >= b.min) band = b
  return band
}
