// Action-column labels for the Q-table heatmap (G2b). Language-neutral glyphs (arrows / letters),
// so no i18n is needed — they mirror the verified action order in content/playKeymaps.ts. Adding a
// game = one row here. Unknown envs fall back to "a0…a{n}".

export const Q_TABLE_ACTIONS: Record<string, string[]> = {
  // FrozenLake (all variants): 0=Left, 1=Down, 2=Right, 3=Up.
  frozenlake: ['←', '↓', '→', '↑'],
  frozenlake_noslip: ['←', '↓', '→', '↑'],
  frozenlake8x8: ['←', '↓', '→', '↑'],
  // CliffWalking: 0=Up, 1=Right, 2=Down, 3=Left.
  cliffwalking: ['↑', '→', '↓', '←'],
  // Taxi: 0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Drop-off.
  taxi: ['↓', '↑', '→', '←', 'P', 'D'],
}

export function qActionsFor(envId: string | null, n: number): string[] {
  const labels = (envId !== null && Q_TABLE_ACTIONS[envId]) || null
  if (labels && labels.length === n) return labels
  return Array.from({ length: n }, (_, i) => `a${i}`)
}
