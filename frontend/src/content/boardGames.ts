// Board-game render metadata (G6a) — the one game-specific bit of the OpenSpiel board subsystem,
// kept in content so adding Connect Four / chess / go later is a data + renderer change, not engine
// code (the backend board_engine + play session + contract are fully game-agnostic). The backend
// streams a game-agnostic BoardState (row-major glyph chars + legal moves + whose turn + winner);
// this maps each glyph char to a drawn piece, and provides an idle board to show before the first
// frame (mirrors content/gridMaps.ts for the Toy Text grids).

import type { BoardState } from '../api/types'

/** How to draw one player's piece: a glyph + a theme-token colour (player 0 = accent, 1 = danger). */
export interface BoardPiece {
  /** The player index this glyph belongs to (0 = first to move). */
  player: number
  /** Display glyph drawn in the cell. */
  glyph: string
  /** Theme token for the glyph colour. */
  color: string
}

export interface BoardGameMeta {
  /** Map a (lower-cased) board glyph char → the piece to draw. Empty cells ("." / " ") are omitted. */
  pieces: Record<string, BoardPiece>
  /** The idle board shown when the game is selected but no session is running. */
  idle: BoardState
}

function emptyBoard(rows: number, cols: number): BoardState {
  const n = rows * cols
  return {
    cells: Array<string>(n).fill('.'),
    rows,
    cols,
    legal_actions: Array.from({ length: n }, (_, i) => i),
    current_player: 0,
    last_action: null,
    is_terminal: false,
    winner: null,
  }
}

export const BOARD_GAMES: Record<string, BoardGameMeta> = {
  // Tic-Tac-Toe: glyph 'x' = player 0 (moves first, accent), 'o' = player 1 (danger). The action
  // index equals the cell index (0–8), so a legal cell is directly clickable.
  tictactoe: {
    pieces: {
      x: { player: 0, glyph: '✕', color: 'var(--accent)' },
      o: { player: 1, glyph: '◯', color: 'var(--danger)' },
    },
    idle: emptyBoard(3, 3),
  },
}

export function isBoardGameEnv(envId: string | null): boolean {
  return envId !== null && envId in BOARD_GAMES
}

/** The render metadata for an env id (Tic-Tac-Toe today), or null if it isn't a known board game. */
export function boardMetaFor(envId: string | null): BoardGameMeta | null {
  return envId !== null ? BOARD_GAMES[envId] ?? null : null
}
