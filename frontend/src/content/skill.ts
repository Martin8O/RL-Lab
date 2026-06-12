// Per-environment skill thresholds: map a run's Score → a skill band.
// CartPole's Score is mean episode length (max 500). As of E2 the live thresholds come from
// the backend (`GET /api/skill/{env}` → `scaleFromEnvSkill`, single source of truth); the table
// below is only a fallback used until that fetch lands (and a default for unknown envs).

import type { EnvSkill, SkillBandId } from '../api/types'

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
  max: number
  bands: SkillBand[] // ascending by min; first must be min:0
}

export const SKILL_SCALES: Record<string, SkillScale> = {
  cartpole: {
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

/** Convert the backend's per-env thresholds into the local SkillScale shape (the meter's
 *  bands + the rating now share one source: the env's solved_score). */
export function scaleFromEnvSkill(envSkill: EnvSkill): SkillScale {
  return {
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
