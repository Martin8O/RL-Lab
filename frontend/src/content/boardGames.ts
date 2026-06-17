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
  /**
   * How a clicked cell maps to a game action (G6c). `'cell'` (default) — the action index *equals* the
   * cell index, so a legal cell is directly clickable (Tic-Tac-Toe: 9 cells = 9 actions). `'column'` —
   * the action is the cell's **column** and the piece drops to the lowest empty row (Connect Four: 7
   * column-actions over a 6×7 board). The backend `BoardState` is identical either way; only this flag +
   * `BoardStage`'s click/highlight maths differ. Everything else stays game-agnostic.
   */
  actionMode?: 'cell' | 'column'
  /** The idle board shown when the game is selected but no session is running. */
  idle: BoardState
}

/** An empty idle board. `legalCount` is the number of legal actions to advertise (cells for `'cell'`
 *  games, columns for `'column'` games) — only used to seed the idle render before the first frame. */
function emptyBoard(rows: number, cols: number, legalCount: number = rows * cols): BoardState {
  const n = rows * cols
  return {
    cells: Array<string>(n).fill('.'),
    rows,
    cols,
    legal_actions: Array.from({ length: legalCount }, (_, i) => i),
    current_player: 0,
    last_action: null,
    is_terminal: false,
    winner: null,
  }
}

/** Othello's standard opening: four discs in the centre of the 8×8 board, Black ('x') to move, with
 *  the four legal opening placements advertised. Shown before a session so the empty grid isn't. */
function othelloIdleBoard(): BoardState {
  const cells = Array<string>(64).fill('.')
  cells[27] = 'o' // d4
  cells[28] = 'x' // e4
  cells[35] = 'x' // d5
  cells[36] = 'o' // e5
  return {
    cells,
    rows: 8,
    cols: 8,
    legal_actions: [19, 26, 37, 44], // d3, c4, f5, e6
    current_player: 0,
    last_action: null,
    is_terminal: false,
    winner: null,
    pass_action: null,
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
  // Connect Four: glyph 'x' = player 0 (drops first), 'o' = player 1. Filled discs read as a Connect
  // Four board. Both players use one glyph, so colour is the ONLY thing telling them apart — so we use
  // two strongly-saturated, high-contrast hues (red vs cyan), NOT the pale --accent periwinkle, which
  // at small disc size read as plain white. The action is the COLUMN (0–6), not the cell — actionMode
  // 'column' tells BoardStage to map a clicked cell to its column and drop to the lowest empty row.
  connect_four: {
    pieces: {
      x: { player: 0, glyph: '●', color: 'var(--danger)' },
      o: { player: 1, glyph: '●', color: 'var(--viz-6)' },
    },
    actionMode: 'column',
    idle: emptyBoard(6, 7, 7), // seven column-actions
  },
  // Othello/Reversi (G6d): an 8×8 disc game. Player 0 ('x') = the filled disc ●, player 1 ('o') = the
  // open disc ◯ — same high-contrast --text-strong token, distinguished by FILL so both read clearly in
  // dark AND light (true black/white discs can't: no theme token is always-dark or always-light, and a
  // pale fill reads as white — the Connect Four lesson). A move is a single CELL placement (the default
  // actionMode), flips arrive pre-applied in the streamed board (no flip logic here), and a forced pass
  // surfaces as BoardState.pass_action → a Pass button handled generically in BoardStage.
  othello: {
    pieces: {
      x: { player: 0, glyph: '●', color: 'var(--text-strong)' },
      o: { player: 1, glyph: '◯', color: 'var(--text-strong)' },
    },
    idle: othelloIdleBoard(),
  },
}

export function isBoardGameEnv(envId: string | null): boolean {
  return envId !== null && envId in BOARD_GAMES
}

/** The render metadata for an env id (Tic-Tac-Toe today), or null if it isn't a known board game. */
export function boardMetaFor(envId: string | null): BoardGameMeta | null {
  return envId !== null ? BOARD_GAMES[envId] ?? null : null
}
