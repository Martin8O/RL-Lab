// #2b — capability badges for the game picker. A quick, scannable "what can I do with this game?"
// read *before* clicking, so a newcomer isn't surprised. Three mutually-exclusive axes derived from
// the registry's EnvSpec flags (Martin's refinement: icon + text, since a bare joystick/robot glyph
// is ambiguous):
//   🕹 solo   — you play the game yourself        (human_playable, single-player)
//   🤖 vs AI  — you play against the trained AI    (competitive board games)
//   👀 watch  — you can only watch it learn/play   (multi-agent swarms, ecosystems)
//
// The three are exclusive so each game shows exactly one badge (the headline capability). Shown in
// Simple mode's picker; harmless to reuse elsewhere. Labels are i18n keys under `mode.cap_*`.
import type { EnvSpec } from '../api/types'

export type Capability = 'solo' | 'vs_ai' | 'watch'

export interface CapabilityBadge {
  cap: Capability
  icon: string      // emoji glyph
  labelKey: string  // i18n key → short text next to the glyph
}

const BADGES: Record<Capability, CapabilityBadge> = {
  solo:  { cap: 'solo',  icon: '🕹', labelKey: 'mode.cap_solo' },
  vs_ai: { cap: 'vs_ai', icon: '🤖', labelKey: 'mode.cap_vs_ai' },
  watch: { cap: 'watch', icon: '👀', labelKey: 'mode.cap_watch' },
}

// The single headline capability for an env. `competitive && human_playable` = a board game you play
// against the net (🤖 vs AI); any other human-playable game = 🕹 solo; everything you can't control
// (competitive-but-not-playable ecosystems like simple_tag, or the cooperative swarms) = 👀 watch.
export function capabilityFor(env: EnvSpec): CapabilityBadge {
  if (env.competitive && env.human_playable) return BADGES.vs_ai
  if (env.human_playable) return BADGES.solo
  return BADGES.watch
}
