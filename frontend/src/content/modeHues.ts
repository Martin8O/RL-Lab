// #2b — per-mode hue palette for the audience-mode marks (shared by ModeChooser + ModeToggle). Kept in
// a plain .ts module (not the .tsx marks file) so that file can export components only — required for
// React Fast Refresh (react-refresh/only-export-components).
import type { AudienceMode } from '../store/useAppStore'

export interface Hue {
  main: string      // CSS var for text/border/tick (theme-aware)
  tint: string      // translucent wash over the card surface
  ring: string      // translucent hover ring/glow
  tileFrom: string  // icon-tile gradient (hex — the tile is dark in both themes)
  tileTo: string
}

// Simple = warm amber (arcade), Advanced = violet (technical) — both on-brand (the logo pairs purple
// with amber).
export const HUES: Record<AudienceMode, Hue> = {
  simple:   { main: 'var(--goal)',  tint: 'rgba(240, 184, 74, 0.20)',  ring: 'rgba(240, 184, 74, 0.55)',  tileFrom: '#F4C25A', tileTo: '#C67512' },
  advanced: { main: 'var(--viz-5)', tint: 'rgba(176, 140, 255, 0.20)', ring: 'rgba(176, 140, 255, 0.55)', tileFrom: '#a175f5', tileTo: '#5a2fb0' },
}
