// Default static boards for the Toy Text grid-worlds, shown when an env is *selected but idle*
// (before any frame streams). During play / training the backend streams the real `grid` layout
// each frame (client_render.grid_layout) and that takes over; these defaults just mirror it so a
// picked grid renders immediately, the way the physics envs show a resting cart/pendulum. Keep the
// maps in sync with Gymnasium's FrozenLake maps + the fixed CliffWalking/Taxi boards.

import type { GridLayout } from '../api/types'

const FL_4X4 = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
const FL_8X8 = [
  'SFFFFFFF', 'FFFFFFFF', 'FFFHFFFF', 'FFFFFHFF',
  'FFFHFFFF', 'FHHFFFHF', 'FHFFHFHF', 'FFFHFFFG',
]
const CHAR_TAG: Record<string, string> = { S: 'start', F: 'normal', H: 'hole', G: 'goal' }

function frozenlake(map: string[]): GridLayout {
  const rows = map.length
  const cols = map[0].length
  const cells = map.flatMap((row) => [...row].map((ch) => CHAR_TAG[ch] ?? 'normal'))
  return { kind: 'frozenlake', rows, cols, cells }
}

function cliffwalking(): GridLayout {
  const rows = 4
  const cols = 12
  const cells = Array<string>(rows * cols).fill('normal')
  const bottom = (rows - 1) * cols
  cells[bottom] = 'start'
  cells[bottom + cols - 1] = 'goal'
  for (let c = 1; c < cols - 1; c++) cells[bottom + c] = 'cliff'
  return { kind: 'cliffwalking', rows, cols, cells }
}

function taxi(): GridLayout {
  const rows = 5
  const cols = 5
  const cells = Array<string>(rows * cols).fill('normal')
  for (const [r, c] of [[0, 0], [0, 4], [4, 0], [4, 3]]) cells[r * cols + c] = 'stop'  // R, G, Y, B
  return { kind: 'taxi', rows, cols, cells }
}

export const DEFAULT_GRIDS: Record<string, GridLayout> = {
  frozenlake: frozenlake(FL_4X4),
  frozenlake_noslip: frozenlake(FL_4X4),
  frozenlake8x8: frozenlake(FL_8X8),
  taxi: taxi(),
  cliffwalking: cliffwalking(),
}

// Resting position before the first frame: the agent at its start cell. Taxi shows a representative
// idle pose ([row, col, passenger_loc, destination]) — passenger waiting at R, destination G.
export const DEFAULT_AGENT: Record<string, number[]> = {
  frozenlake: [0, 0],
  frozenlake_noslip: [0, 0],
  frozenlake8x8: [0, 0],
  cliffwalking: [3, 0],
  taxi: [2, 2, 0, 1],
}

const TOY_TEXT_IDS = new Set(Object.keys(DEFAULT_GRIDS))

export function isGridEnv(envId: string | null): boolean {
  return envId !== null && TOY_TEXT_IDS.has(envId)
}
