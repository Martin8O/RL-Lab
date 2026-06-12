// Per-environment skill thresholds: map a run's Score → a skill band.
// CartPole's Score is mean episode length (max 500), so the bands below are illustrative
// and tunable. Phase E will move these to a backend evaluator served per environment.

export type SkillKey = 'child' | 'below' | 'average' | 'above' | 'superhuman'

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

/** The highest band whose threshold the score has reached. */
export function currentBand(score: number, scale: SkillScale): SkillBand {
  let band = scale.bands[0]
  for (const b of scale.bands) if (score >= b.min) band = b
  return band
}
