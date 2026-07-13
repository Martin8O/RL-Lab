// #2b — expectation-setting for training time (Simple mode). The worst newcomer experience is
// starting a run expecting "done in two minutes" and then staring at a flat chart for 20 (e.g. Pong
// needs millions of steps — that's not a bug, some games just learn slowly). So Simple mode shows a
// qualitative "how long will this take" cue instead of a raw step count, and — for the slow ones — an
// explicit "you won't see much for a while, that's normal" note.
//
// Honest by design: real wall-clock depends on hardware + algorithm + env throughput, so we DON'T
// promise exact minutes. We derive a coarse tier from the registry `difficulty` (already curated per
// env) plus the chosen training length, and let the copy stay qualitative. All strings are i18n keys.
import type { Difficulty } from '../api/types'

// The three friendly training-length choices Simple mode offers instead of the ×0.2…×8 step ladder.
// Each maps to a real multiple of the env's ★ budget, so the honest step count still drives training.
export type TrainLength = 'short' | 'normal' | 'long'
export const TRAIN_LENGTHS: TrainLength[] = ['short', 'normal', 'long']
export const LENGTH_FACTOR: Record<TrainLength, number> = { short: 0.5, normal: 1, long: 4 }

// The qualitative time tier: 🟢 fast (a few minutes) / 🟡 medium (takes a while) / 🔴 slow (a long
// run — let it run in the background). Drives the colour dot + the expectation copy.
export type TrainTier = 'fast' | 'medium' | 'slow'
const TIERS: TrainTier[] = ['fast', 'medium', 'slow']
const DIFFICULTY_BASE: Record<Difficulty, number> = { beginner: 0, intermediate: 1, advanced: 2 }
export const TIER_DOT: Record<TrainTier, string> = { fast: '🟢', medium: '🟡', slow: '🔴' }

// Tier from the env difficulty, bumped by the chosen length (Long escalates a step, Short eases one),
// clamped to [fast, slow]. So a beginner game on Normal reads 🟢, but the same game on Long reads 🟡;
// an advanced game already reads 🔴 on Normal.
export function trainTier(difficulty: Difficulty, length: TrainLength): TrainTier {
  const bump = length === 'long' ? 1 : length === 'short' ? -1 : 0
  const idx = Math.max(0, Math.min(TIERS.length - 1, (DIFFICULTY_BASE[difficulty] ?? 1) + bump))
  return TIERS[idx]
}

// The slow tier gets the explicit "this learns slowly — that's normal, let it run" sentence.
export function needsPatience(tier: TrainTier): boolean {
  return tier === 'slow'
}
