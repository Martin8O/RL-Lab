// Display labels + order for env *categories* — these are the registry's `family` values, surfaced
// in the game picker (EnvSelector) so dozens of future games stay browseable by family. Adding a
// family = one entry here; an unknown family falls back to showing its raw id. The order mirrors the
// Phase-G roadmap (cheapest/CPU families first → GPU families later).

import type { Bilingual } from '../api/types'

export const ENV_CATEGORIES: { id: string; label: Bilingual }[] = [
  { id: 'classic_control', label: { en: 'Classic Control', cz: 'Klasické řízení' } },
  { id: 'toy_text',        label: { en: 'Toy Text (tabular)', cz: 'Toy Text (tabulkové)' } },
  { id: 'box2d',           label: { en: 'Box2D (physics)', cz: 'Box2D (fyzika)' } },
  { id: 'atari',           label: { en: 'Atari', cz: 'Atari' } },
  { id: 'mujoco',          label: { en: 'MuJoCo (robotics)', cz: 'MuJoCo (robotika)' } },
  { id: 'board',           label: { en: 'Board games', cz: 'Deskové hry' } },
  { id: 'petting_zoo',     label: { en: 'Multi-agent', cz: 'Více agentů' } },
]

const ORDER = ENV_CATEGORIES.map((c) => c.id)

export function categoryLabel(id: string): Bilingual {
  return ENV_CATEGORIES.find((c) => c.id === id)?.label ?? { en: id, cz: id }
}

/** Stable sort index for a family id (known families in roadmap order, unknown ones last). */
export function categoryOrder(id: string): number {
  const i = ORDER.indexOf(id)
  return i === -1 ? ORDER.length : i
}
