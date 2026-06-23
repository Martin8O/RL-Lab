import type { CSSProperties } from 'react'

// Global "this panel isn't relevant to the current activity" treatment. A panel with no applicable
// data — Evolution Stats during a PPO run, High Scores with no records, anything that depends on an
// algorithm/mode the user isn't in — recedes instead of sitting blank and reading as broken. One
// shared affordance (not a per-panel hack) so every panel dims the same way.
//
// The dim itself lives in index.css (.panel-dim) so it can be theme-aware: in dark mode low opacity
// alone just fades into the near-black background too subtly, so dark dims harder and also drops
// brightness. The transition stays here (always applied) so both dimming AND un-dimming animate —
// if it lived only in the class, removing the class would snap back instantly.
export const PANEL_DIM_BASE: CSSProperties = {
  transition: 'opacity 0.25s ease, filter 0.25s ease',
}

export function panelDimClass(dimmed: boolean): string | undefined {
  return dimmed ? 'panel-dim' : undefined
}
