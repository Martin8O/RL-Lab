// Client-side rendered env "stages" (SVG) — drawn from the raw physics state the backend streams
// (app/services/client_render.py → client_state). These are *presentational*: EnvPreview owns the
// refs and updates them imperatively from each frame (no React re-render per frame), exactly like
// the CartPole cart/pole. The geometry both files need lives in ./envGeometry. Keep the env set +
// state layout in sync with the backend's client_state.

import type { CSSProperties, RefObject } from 'react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { BoardMove, BoardState, GridLayout } from '../api/types'
import type { BoardGameMeta } from '../content/boardGames'
import {
  MC_GOAL, MC_GROUND_PATH, MC_SURFACE_PATH, mcX, mcY,
  PEND_CX, PEND_CY, PEND_L, ACRO_CX, ACRO_CY, ACRO_L, ACRO_JOINT_Y,
  LL_PAD_HALF, LL_PAD_OBS_Y, llX, llY, llTerrainPaths, LL_DEFAULT_TERRAIN,
} from './envGeometry'

const SVG_STYLE = { width: '100%', maxWidth: 820, maxHeight: '100%' } as const

// ── CartPole ────────────────────────────────────────────────────────────────────────────────
export function CartPoleStage({ envName, cartRef, poleRef }: {
  envName: string
  cartRef: RefObject<SVGGElement | null>
  poleRef: RefObject<SVGGElement | null>
}) {
  return (
    <svg viewBox="0 0 600 260" preserveAspectRatio="xMidYMid meet" style={SVG_STYLE} role="img" aria-label={envName}>
      <line x1="30" y1="210" x2="570" y2="210" stroke="var(--border-strong)" strokeWidth="2.5" />
      <g ref={cartRef}>
        <g ref={poleRef}>
          <line x1="300" y1="190" x2="300" y2="70" stroke="var(--accent)" strokeWidth="7" strokeLinecap="round" />
          <circle cx="300" cy="66" r="9" fill="var(--accent)" />
        </g>
        <rect x="268" y="184" width="64" height="26" rx="5" fill="var(--surface-3)" stroke="var(--border-strong)" strokeWidth="2.5" />
        <circle cx="282" cy="214" r="6" fill="var(--text-faint)" />
        <circle cx="318" cy="214" r="6" fill="var(--text-faint)" />
      </g>
    </svg>
  )
}

// ── MountainCar (mountaincar + mountaincarcontinuous) ─────────────────────────────────────────
export function MountainCarStage({ envName, carRef }: {
  envName: string; carRef: RefObject<SVGGElement | null>
}) {
  return (
    <svg viewBox="0 0 600 340" preserveAspectRatio="xMidYMid meet" style={SVG_STYLE} role="img" aria-label={envName}>
      <path d={MC_GROUND_PATH} fill="var(--surface-3)" />
      <path d={MC_SURFACE_PATH} fill="none" stroke="var(--border-strong)" strokeWidth="2.5" strokeLinejoin="round" />
      {/* goal flag on the right hill */}
      <line x1={mcX(MC_GOAL)} y1={mcY(MC_GOAL)} x2={mcX(MC_GOAL)} y2={mcY(MC_GOAL) - 40} stroke="var(--border-strong)" strokeWidth="2.5" />
      <path d={`M ${mcX(MC_GOAL)} ${mcY(MC_GOAL) - 40} l 24 8 l -24 8 z`} fill="var(--accent)" />
      {/* car — slid + tilted along the hill imperatively from each frame's position */}
      <g ref={carRef}>
        <rect x="-16" y="-17" width="32" height="12" rx="4" fill="var(--accent)" />
        <circle cx="-9" cy="-4" r="4.5" fill="var(--text-strong)" />
        <circle cx="9" cy="-4" r="4.5" fill="var(--text-strong)" />
      </g>
    </svg>
  )
}

// ── Pendulum ──────────────────────────────────────────────────────────────────────────────────
export function PendulumStage({ envName, rodRef }: {
  envName: string; rodRef: RefObject<SVGGElement | null>
}) {
  return (
    <svg viewBox="0 0 600 300" preserveAspectRatio="xMidYMid meet" style={SVG_STYLE} role="img" aria-label={envName}>
      {/* faint marker for the upright target (θ = 0) */}
      <line x1={PEND_CX} y1={PEND_CY - PEND_L - 18} x2={PEND_CX} y2={PEND_CY - PEND_L - 5}
        stroke="var(--text-faint)" strokeWidth="2" strokeDasharray="3 3" />
      <g ref={rodRef}>
        <line x1={PEND_CX} y1={PEND_CY} x2={PEND_CX} y2={PEND_CY - PEND_L} stroke="var(--accent)" strokeWidth="9" strokeLinecap="round" />
        <circle cx={PEND_CX} cy={PEND_CY - PEND_L} r="13" fill="var(--accent)" />
      </g>
      <circle cx={PEND_CX} cy={PEND_CY} r="6" fill="var(--text-faint)" />
    </svg>
  )
}

// ── Acrobot ─────────────────────────────────────────────────────────────────────────────────
// Swing the tip above the dashed target line (one link length above the pivot).
export function AcrobotStage({ envName, link1Ref, link2Ref }: {
  envName: string; link1Ref: RefObject<SVGGElement | null>; link2Ref: RefObject<SVGGElement | null>
}) {
  return (
    <svg viewBox="0 0 600 300" preserveAspectRatio="xMidYMid meet" style={SVG_STYLE} role="img" aria-label={envName}>
      <line x1="110" y1={ACRO_CY - ACRO_L} x2="490" y2={ACRO_CY - ACRO_L}
        stroke="var(--text-faint)" strokeWidth="2" strokeDasharray="5 4" />
      <g ref={link1Ref}>
        <line x1={ACRO_CX} y1={ACRO_CY} x2={ACRO_CX} y2={ACRO_JOINT_Y} stroke="var(--accent)" strokeWidth="8" strokeLinecap="round" />
        <circle cx={ACRO_CX} cy={ACRO_CY} r="6" fill="var(--text-faint)" />
        <g ref={link2Ref}>
          <line x1={ACRO_CX} y1={ACRO_JOINT_Y} x2={ACRO_CX} y2={ACRO_JOINT_Y + ACRO_L} stroke="var(--viz-2)" strokeWidth="8" strokeLinecap="round" />
          <circle cx={ACRO_CX} cy={ACRO_JOINT_Y} r="5" fill="var(--text-strong)" />
          <circle cx={ACRO_CX} cy={ACRO_JOINT_Y + ACRO_L} r="7" fill="var(--viz-2)" />
        </g>
      </g>
    </svg>
  )
}

// ── LunarLander ───────────────────────────────────────────────────────────────────────────────
// The real moon surface (streamed per episode → groundRef/surfaceRef, flat until it arrives) + the
// fixed central pad + flags, with the lander placed/rotated from the obs. Thruster flames live inside
// the lander group (so they rotate with it) and are toggled by EnvPreview per the firing engine: main
// = big downward plume, side engines = small side puffs.
const FLAME_OUTER = '#f0883e', FLAME_INNER = '#ffd23f'
export function LunarLanderStage({ envName, landerRef, mainPlumeRef, leftPlumeRef, rightPlumeRef, groundRef, surfaceRef }: {
  envName: string
  landerRef: RefObject<SVGGElement | null>
  mainPlumeRef: RefObject<SVGGElement | null>
  leftPlumeRef: RefObject<SVGGElement | null>
  rightPlumeRef: RefObject<SVGGElement | null>
  groundRef: RefObject<SVGPathElement | null>
  surfaceRef: RefObject<SVGPathElement | null>
}) {
  const padL = llX(-LL_PAD_HALF), padR = llX(LL_PAD_HALF), padY = llY(LL_PAD_OBS_Y)
  const init = llTerrainPaths(LL_DEFAULT_TERRAIN)
  return (
    <svg viewBox="0 0 600 400" preserveAspectRatio="xMidYMid meet" style={SVG_STYLE} role="img" aria-label={envName}>
      {/* real moon terrain — updated imperatively from each episode's streamed surface points */}
      <path ref={groundRef} d={init.ground} fill="var(--surface-3)" />
      <path ref={surfaceRef} d={init.surface} fill="none" stroke="var(--border-strong)" strokeWidth="2.5" strokeLinejoin="round" />
      {/* landing pad + flags (the fixed centre region between the flags) */}
      <rect x={padL} y={padY - 3} width={padR - padL} height="5" rx="2" fill="var(--accent)" />
      {[padL, padR].map((fx, i) => (
        <g key={i}>
          <line x1={fx} y1={padY - 2} x2={fx} y2={padY - 26} stroke="var(--border-strong)" strokeWidth="2" />
          <path d={`M ${fx} ${padY - 26} l 14 5 l -14 5 z`} fill="var(--accent)" />
        </g>
      ))}
      {/* lander (slid + rotated imperatively from each frame's obs) */}
      <g ref={landerRef}>
        {/* thruster flames — behind the hull, toggled by EnvPreview; main is large, sides small */}
        <g ref={mainPlumeRef} opacity="0">
          <polygon points="-9,11 0,46 9,11" fill={FLAME_OUTER} />
          <polygon points="-4.5,11 0,33 4.5,11" fill={FLAME_INNER} />
        </g>
        <g ref={leftPlumeRef} opacity="0">
          <polygon points="-13,3 -30,12 -13,15" fill={FLAME_OUTER} />
        </g>
        <g ref={rightPlumeRef} opacity="0">
          <polygon points="13,3 30,12 13,15" fill={FLAME_OUTER} />
        </g>
        {/* hull + window + legs */}
        <polygon points="-10,-13 10,-13 15,-3 15,7 10,12 -10,12 -15,7 -15,-3"
          fill="var(--surface-2)" stroke="var(--border-strong)" strokeWidth="2" strokeLinejoin="round" />
        <circle cx="0" cy="-3" r="4.5" fill="var(--accent)" />
        <line x1="-12" y1="10" x2="-20" y2="21" stroke="var(--border-strong)" strokeWidth="3" strokeLinecap="round" />
        <line x1="12" y1="10" x2="20" y2="21" stroke="var(--border-strong)" strokeWidth="3" strokeLinecap="round" />
        <line x1="-23" y1="21" x2="-17" y2="21" stroke="var(--border-strong)" strokeWidth="3" strokeLinecap="round" />
        <line x1="17" y1="21" x2="23" y2="21" stroke="var(--border-strong)" strokeWidth="3" strokeLinecap="round" />
      </g>
    </svg>
  )
}

// ── Toy Text grid-worlds (FrozenLake / CliffWalking / Taxi) ──────────────────────────────────
// One renderer for all three: the backend streams the static board (grid: kind/rows/cols/cells)
// plus the dynamic agent position (state), and this draws the board + agent declaratively. Grid
// moves are infrequent (turn-based human / paced AI), so a React render per move is fine — no need
// for the imperative per-frame transforms the physics stages use. Keep in sync with the backend's
// client_render.grid_layout cell tags and the default boards in content/gridMaps.ts.
const GRID_CELL = 56
const TAXI_STOPS: ReadonlyArray<readonly [number, number]> = [[0, 0], [0, 4], [4, 0], [4, 3]]  // R G Y B
const TAXI_STOP_COLOR = ['var(--viz-3)', 'var(--viz-2)', 'var(--viz-5)', 'var(--viz-1)']
const TAXI_STOP_LABEL = ['R', 'G', 'Y', 'B']
// Taxi's fixed internal walls: an impassable edge on the right of cell [row, col].
const TAXI_WALLS: ReadonlyArray<readonly [number, number]> = [[0, 1], [1, 1], [3, 0], [3, 2], [4, 0], [4, 2]]

function GridFlag({ x, y, color }: { x: number; y: number; color: string }) {
  return (
    <g>
      <line x1={x} y1={y + 13} x2={x} y2={y - 15} stroke="var(--border-strong)" strokeWidth="2.5" />
      <path d={`M ${x} ${y - 15} l 16 6 l -16 6 z`} fill={color} />
    </g>
  )
}

function GridPerson({ x, y, color }: { x: number; y: number; color: string }) {
  return (
    <g>
      <circle cx={x} cy={y - 8} r="5" fill={color} />
      <path d={`M ${x - 7} ${y + 10} q 7 -16 14 0 z`} fill={color} />
    </g>
  )
}

function TaxiPieces({ agent, cx, cy }: {
  agent: number[]; cx: (c: number) => number; cy: (r: number) => number
}) {
  const taxiRow = agent[0] ?? 0
  const taxiCol = agent[1] ?? 0
  const passLoc = agent[2] ?? 0   // 0..3 = waiting at R/G/Y/B; 4 = riding in the taxi
  const dest = agent[3] ?? 0      // 0..3 = drop-off at R/G/Y/B
  const [dr, dc] = TAXI_STOPS[Math.min(3, Math.max(0, dest))]
  return (
    <g>
      {TAXI_WALLS.map(([r, c], i) => (
        <line key={`w${i}`} x1={(c + 1) * GRID_CELL} y1={r * GRID_CELL} x2={(c + 1) * GRID_CELL} y2={(r + 1) * GRID_CELL}
          stroke="var(--border-strong)" strokeWidth="4" strokeLinecap="round" />
      ))}
      {TAXI_STOPS.map(([r, c], i) => (
        <text key={`s${i}`} x={c * GRID_CELL + 7} y={r * GRID_CELL + 18}
          fontSize="13" fontWeight={700} fontFamily="var(--font-mono)" fill={TAXI_STOP_COLOR[i]}>
          {TAXI_STOP_LABEL[i]}
        </text>
      ))}
      <GridFlag x={cx(dc)} y={cy(dr)} color="var(--accent)" />
      {passLoc < 4 && (
        <GridPerson x={cx(TAXI_STOPS[passLoc][1])} y={cy(TAXI_STOPS[passLoc][0])} color="var(--viz-1)" />
      )}
      <g transform={`translate(${cx(taxiCol)} ${cy(taxiRow)})`}>
        <rect x="-16" y="-9" width="32" height="18" rx="4" fill="var(--accent)" stroke="var(--surface-1)" strokeWidth="2" />
        <circle cx="-9" cy="10" r="3.5" fill="var(--text-strong)" />
        <circle cx="9" cy="10" r="3.5" fill="var(--text-strong)" />
        {passLoc === 4 && <circle cx="0" cy="-2" r="4.5" fill="var(--viz-1)" />}
      </g>
    </g>
  )
}

// ── Multi-agent "swarm" (PettingZoo / MPE) ───────────────────────────────────────────────────
// Unlike the SVG physics stages, the swarm is drawn to a <canvas> (ADR-029's precedent for live,
// many-element scenes — and it scales to a desktop GPU swarm of many agents). EnvPreview owns the
// canvas ref and redraws it imperatively from each frame's per-agent + landmark world positions
// (the `agents`/`world` frame fields). This component is just the canvas + a colour legend.
export const SWARM_CANVAS_PX = 680  // square logical resolution; CSS scales it to fit the stage

// One swarm-legend entry: a colour swatch matching the actual canvas render + its label. `ring` draws
// an open marker (simple_spread coverage targets); otherwise a filled dot (agents, predators, prey,
// obstacles). Colours MUST match drawSwarm so the legend and the canvas read together (visual-labels rule).
export interface SwarmLegendItem {
  color: string
  label: string
  ring?: boolean
}

export function SwarmStage({ envName, canvasRef, legend }: {
  envName: string
  canvasRef: RefObject<HTMLCanvasElement | null>
  legend: SwarmLegendItem[]
}) {
  const dot = (color: string): CSSProperties => ({
    display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: color,
  })
  const ringStyle = (color: string): CSSProperties => ({
    display: 'inline-block', width: 10, height: 10, borderRadius: '50%',
    border: `2px solid ${color}`, boxSizing: 'border-box',
  })
  return (
    <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <canvas
        ref={canvasRef}
        width={SWARM_CANVAS_PX}
        height={SWARM_CANVAS_PX}
        role="img"
        aria-label={envName}
        style={{ width: '100%', maxWidth: 480, maxHeight: '100%', display: 'block' }}
      />
      <div style={{
        display: 'flex', gap: 18, fontSize: 'var(--fs-label)', color: 'var(--text-muted)',
        alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center',
      }}>
        {legend.map((it) => (
          <span key={it.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={it.ring ? ringStyle(it.color) : dot(it.color)} /> {it.label}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Board games (OpenSpiel — Tic-Tac-Toe now, Connect Four / chess / go later) ───────────────
// G6a. Unlike the physics/grid stages this is an HTML grid of cells (not SVG), because a board is
// fundamentally clickable: legal empty cells on the human's turn are real <button>s (native focus +
// keyboard + accessible names), occupied cells draw the piece glyph from the game's render metadata
// (content/boardGames.ts), and the last move gets a ring. The board payload (cells/legal/turn/winner)
// is fully game-agnostic; only `meta` (glyph → piece) is game-specific. The status line shows whose
// turn it is; the result banner shows the honest win/draw/loss outcome (no continuous skill %).
export function BoardStage({ envName, board, meta, humanTurn, humanSide, onCellClick, statusText, banner, maxBoardPx = 0 }: {
  envName: string
  board: BoardState
  meta: BoardGameMeta
  humanTurn: boolean
  // The board player the human controls (for orientation), or null in watch/training (default view).
  humanSide: number | null
  onCellClick: (action: number) => void
  statusText: string
  // `mark` colours the winner's piece in the banner — for games whose two players share one glyph
  // (Connect Four: both '●', only the colour differs), "● wins" would otherwise be unreadable.
  banner: { text: string; kind: 'win' | 'draw' | 'loss'; mark?: { glyph: string; color: string } } | null
  // The largest square (px) the board may occupy (the measured stage box, grows in fullscreen). 0 = fall
  // back to the old fixed sizing. Lets the board fill the panel instead of leaving big empty margins.
  maxBoardPx?: number
}) {
  const { t } = useTranslation()
  const { rows, cols, cells, legal_actions, last_action, is_terminal, pass_action } = board
  const legal = new Set(legal_actions)
  // Size the squares to fill the measured stage (clamped so a 3×3 board doesn't become enormous and an
  // 8×8 stays readable); fall back to the fixed sizing when the stage hasn't been measured yet.
  const cellPx = maxBoardPx > 0
    ? Math.max(40, Math.min(180, Math.floor((maxBoardPx - 12) / Math.max(rows, cols, 1))))
    : Math.max(48, Math.min(110, Math.floor(360 / Math.max(rows, cols, 1))))
  const bannerColor =
    banner?.kind === 'win' ? 'var(--success)' : banner?.kind === 'loss' ? 'var(--danger)' : 'var(--text-strong)'

  // Three interaction modes (content/boardGames.ts actionMode). Cell games (Tic-Tac-Toe): action ==
  // cell index. Column games (Connect Four, G6c): a cell's action is its column; the disc drops to the
  // lowest empty row. Move games (Breakthrough, G6e): a move is (from-square → to-square) — click your
  // piece, then a highlighted destination. The backend BoardState is identical for cell/column; move
  // games additionally carry `moves` (per legal action's from/to cells). No HOVER highlighting at all —
  // a clickable cell is just a button (cursor only); the kept highlights are the last move + (move mode)
  // the gold selection/destination markers.
  const columnMode = meta.actionMode === 'column'
  const moveMode = meta.actionMode === 'move'
  const actionOf = (i: number) => (columnMode ? i % cols : i)

  // Board orientation (G6e): for a directional game, flip the board 180° while the human plays the side
  // that isn't `bottomPlayer`, so their pieces sit at the bottom and advance upward. A pure CSS rotation
  // — the triangle glyphs flip to point the right way for free, and clicks/highlights ride the transform
  // (the cell handlers use backend indices, which the rotation doesn't touch). No flip in watch/training.
  const flip = !!meta.orient && humanSide != null && humanSide !== meta.orient.bottomPlayer

  // Move-mode selection — which of the human's pieces is picked, and (chess) a pending promotion choice.
  // Reset whenever the turn passes, the board advances (a move was played) or the game ends, via React's
  // canonical "reset state on prop change during render" pattern (no effect — setState-in-effect cascades).
  const [selectedFrom, setSelectedFrom] = useState<number | null>(null)
  const [promoAt, setPromoAt] = useState<{ cell: number; moves: BoardMove[] } | null>(null)
  const turnKey = `${humanTurn}:${last_action}:${is_terminal}`
  const [prevTurnKey, setPrevTurnKey] = useState(turnKey)
  if (turnKey !== prevTurnKey) {
    setPrevTurnKey(turnKey)
    setSelectedFrom(null)
    setPromoAt(null)
  }

  const moves = board.moves ?? []
  // The cells the human can move FROM, and (once a piece is picked) that piece's destinations. A chess
  // promotion has SEVERAL actions landing on the same square (=Q/=R/=B/=N), so a destination maps to a
  // *list* of moves: one → submit directly, many → open the promotion picker (selectedDest below).
  const fromCells = moveMode ? new Set(moves.map((m) => m.from_cell)) : null
  const destForSelected = new Map<number, BoardMove[]>()
  if (moveMode && selectedFrom != null) {
    for (const m of moves) {
      if (m.from_cell !== selectedFrom) continue
      const at = destForSelected.get(m.to_cell)
      if (at) at.push(m)
      else destForSelected.set(m.to_cell, [m])
    }
  }
  const moveInteractive = moveMode && humanTurn && !is_terminal

  const isFilled = (i: number) => {
    const v = cells[i]?.trim() ?? ''
    return v !== '' && v !== '.'
  }
  // The just-played cell(s) to ring. Cell mode: the last_action cell. Column mode: the top-most filled
  // cell of the last_action column. Move mode: the last move's from + to cells (streamed as last_from/to,
  // since a move int doesn't map to a single cell). A subtle "what just happened" marker.
  const lastCells = new Set<number>()
  if (moveMode) {
    if (board.last_from != null) lastCells.add(board.last_from)
    if (board.last_to != null) lastCells.add(board.last_to)
  } else if (last_action !== null) {
    if (columnMode) {
      for (let r = 0; r < rows; r++) {
        const idx = r * cols + last_action
        if (isFilled(idx)) { lastCells.add(idx); break }
      }
    } else {
      lastCells.add(last_action)
    }
  }

  // A real chessboard look (chess): flush light/dark squares in the classic lichess brown, vs the small
  // games' flat single-surface cells. Theme-independent on purpose — a chessboard is always brown/cream
  // (like the Atari CRT skin is game-specific), and the cburnett pieces are drawn for exactly this board.
  const checkered = !!meta.checkered
  const LIGHT_SQ = '#f0d9b5'
  const DARK_SQ = '#b58863'
  const pad = checkered ? 8 : 6
  const sqIsLight = (r: number, c: number) => (r + c) % 2 === 0
  const cellBox = (r: number, c: number): CSSProperties => ({
    width: cellPx, height: cellPx, display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: checkered ? ((r + c) % 2 === 0 ? LIGHT_SQ : DARK_SQ) : 'var(--surface-2)',
    borderRadius: checkered ? 0 : 'var(--radius-sm)',
    border: checkered ? 'none' : '1px solid var(--border-default)',
    padding: 0, margin: 0,
  })

  // While the human is choosing a move, a mode-specific hint replaces the generic status (no banner up).
  const moveHint = moveInteractive && !banner
    ? (selectedFrom == null ? t('board.select_piece') : t('board.select_dest'))
    : null

  // Chess promotion: show the picker glyphs in the human's own colour. The backend streams the promo
  // letter lower-cased; case it to whatever case the human's pieces use in the glyph map (chess: white
  // pieces are UPPER-cased keys, black lower) — derived, not a hardcoded side, since white isn't player 0.
  const humanUsesUpperGlyphs = Object.entries(meta.pieces).some(
    ([key, pc]) => pc.player === humanSide && key === key.toUpperCase() && key !== key.toLowerCase(),
  )
  const promoCase = (letter: string) => (humanUsesUpperGlyphs ? letter.toUpperCase() : letter.toLowerCase())
  const PROMO_ORDER: Record<string, number> = { q: 0, r: 1, b: 2, n: 3 }

  return (
    <div role="group" aria-label={envName} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14, position: 'relative' }}>
      {/* Whose turn / the final result — leads the board so play always reads as labelled. */}
      <div style={{
        minHeight: 24, fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)',
        color: banner ? bannerColor : 'var(--text-muted)', textAlign: 'center',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
      }}>
        {banner?.mark && (
          <span aria-hidden style={{ color: banner.mark.color, fontWeight: 800 }}>{banner.mark.glyph}</span>
        )}
        {banner ? banner.text : (moveHint ?? statusText)}
      </div>

      {/* Wrap so a coordinate overlay (chess files/ranks) can sit over the board without riding the
          board's orientation flip — it computes visual positions itself. */}
      <div style={{ position: 'relative' }}>
      <div style={{
        display: 'grid', gap: checkered ? 0 : 5,
        gridTemplateColumns: `repeat(${cols}, ${cellPx}px)`,
        gridTemplateRows: `repeat(${rows}, ${cellPx}px)`,
        background: 'var(--surface-3)', padding: pad, borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-md)', position: 'relative',
        transform: flip ? 'rotate(180deg)' : undefined,
      }}>
        {cells.map((ch, i) => {
          const r = Math.floor(i / cols)
          const c = i % cols
          const mark = ch.trim().toUpperCase()
          // Case-sensitive first (chess tells white 'P' from black 'p' by case), then the lower-cased
          // fallback the single-case games (x/o, b/w) rely on. Backward-compatible with all of them.
          const piece = meta.pieces[ch.trim()] ?? meta.pieces[ch.trim().toLowerCase()]
          // Counter-rotate the piece when the board is flipped for orientation, so chess's upright pieces
          // (Unicode glyphs OR the lichess SVGs) stay readable; triangle games let their glyphs ride the flip.
          const uprightFix = flip && meta.uprightGlyphs ? 'rotate(180deg)' : undefined
          const glyph = piece && (meta.pieceImageBase && piece.image
            ? (
              <img src={`${meta.pieceImageBase}/${piece.image}.svg`} alt="" draggable={false}
                style={{ width: cellPx * 0.84, height: cellPx * 0.84, pointerEvents: 'none', transform: uprightFix }} />
            )
            : (
              <span aria-hidden style={{
                fontSize: cellPx * 0.62, lineHeight: 1, color: piece.color, fontWeight: 800, transform: uprightFix,
              }}>
                {piece.glyph}
              </span>
            ))
          // The last-move trail ring — a chess-style yellow on the checkered board, else muted in move
          // mode (so it never competes with the gold selection markers) / piece-coloured for placement.
          const trailRing = lastCells.has(i)
            ? `inset 0 0 0 ${checkered ? 5 : 3}px ${
                checkered ? 'rgba(255,205,0,0.8)' : moveMode ? 'var(--text-muted)' : (piece?.color ?? 'var(--accent)')
              }`
            : undefined

          // ── Move mode (Breakthrough): pick a piece, then click one of its gold-marked destinations ──
          if (moveMode) {
            if (moveInteractive && selectedFrom === i) {
              // The picked piece — a gold ring; click again to deselect.
              return (
                <button key={i} onClick={() => setSelectedFrom(null)}
                  aria-label={t('board.cell_selected', { row: r + 1, col: c + 1 })}
                  style={{ ...cellBox(r, c), cursor: 'pointer', boxShadow: 'inset 0 0 0 3px var(--goal)' }}>
                  {glyph}
                </button>
              )
            }
            const destMoves = moveInteractive ? destForSelected.get(i) : undefined
            if (destMoves && destMoves.length > 0) {
              // A legal destination: an empty target shows a centred gold dot, a capture shows the gold
              // ring around the enemy glyph. One action → submit it; several (a chess promotion landing
              // on this square) → open the piece picker so the player chooses ♕/♖/♗/♘.
              const onDest = () => {
                if (destMoves.length === 1) { onCellClick(destMoves[0].action); setSelectedFrom(null) }
                else setPromoAt({ cell: i, moves: destMoves })
              }
              return (
                <button key={i} onClick={onDest}
                  aria-label={t('board.cell_move_to', { row: r + 1, col: c + 1 })}
                  style={{ ...cellBox(r, c), cursor: 'pointer', boxShadow: piece ? 'inset 0 0 0 3px var(--goal)' : undefined }}>
                  {glyph ?? <span aria-hidden style={{ display: 'block', width: cellPx * 0.3, height: cellPx * 0.3, borderRadius: '50%', background: 'var(--goal)' }} />}
                </button>
              )
            }
            if (moveInteractive && fromCells?.has(i)) {
              // One of the human's movable pieces — selectable.
              return (
                <button key={i} onClick={() => setSelectedFrom(i)}
                  aria-label={t('board.cell_select', { row: r + 1, col: c + 1 })}
                  style={{ ...cellBox(r, c), cursor: 'pointer', boxShadow: trailRing }}>
                  {glyph}
                </button>
              )
            }
            return (
              <div key={i} role="img"
                aria-label={piece
                  ? t('board.cell_taken', { mark, row: r + 1, col: c + 1 })
                  : t('board.cell_empty', { row: r + 1, col: c + 1 })}
                style={{ ...cellBox(r, c), boxShadow: trailRing }}>
                {glyph}
              </div>
            )
          }

          // ── Cell / column mode (Tic-Tac-Toe, Connect Four) ──
          const action = actionOf(i)
          const playable = !is_terminal && humanTurn && legal.has(action)
          // In column mode the landing spot is an empty cell, so the empty cells of a legal column are
          // the buttons; in cell mode the legal (empty) cell itself is the button — as before.
          const isButton = playable && (columnMode ? !piece : true)
          if (isButton) {
            return (
              <button
                key={i}
                onClick={() => onCellClick(action)}
                aria-label={columnMode
                  ? t('board.cell_drop', { col: c + 1 })
                  : t('board.cell_play', { row: r + 1, col: c + 1 })}
                style={{ ...cellBox(r, c), cursor: 'pointer', boxShadow: trailRing }}
              >
                {glyph}
              </button>
            )
          }
          return (
            <div
              key={i}
              role="img"
              aria-label={piece
                ? t('board.cell_taken', { mark, row: r + 1, col: c + 1 })
                : t('board.cell_empty', { row: r + 1, col: c + 1 })}
              style={{ ...cellBox(r, c), boxShadow: trailRing }}
            >
              {glyph}
            </div>
          )
        })}
        {/* Pass move (Othello, G6d): a forced pass when no placement is legal. Absolutely positioned to
            the LEFT of the grid and vertically centred, so it toggling on/off never reflows the board (a
            button *below* the grid shoved the whole board up, then back down on the pass). Game-agnostic
            — driven by the backend-detected `pass_action`; the AI/net pass automatically. */}
        {humanTurn && !is_terminal && pass_action != null && (
          <button
            onClick={() => onCellClick(pass_action)}
            style={{
              position: 'absolute', right: '100%', top: '50%', transform: 'translateY(-50%)',
              marginRight: 14, whiteSpace: 'nowrap', padding: '8px 22px', cursor: 'pointer',
              fontWeight: 'var(--fw-semibold)', fontSize: 'var(--fs-sm)', color: 'var(--text-strong)',
              background: 'var(--surface-2)', border: '1px solid var(--border-strong)',
              borderRadius: 'var(--radius-sm)',
            }}
          >
            {t('board.pass')}
          </button>
        )}
      </div>
      {/* Coordinate labels (chess): files a–h along the player's bottom edge, ranks 1–8 up the right
          edge — lichess-style, each tinted to contrast its square. Computed from the orientation (and the
          flip) in VISUAL coordinates, so they stay correct + upright whether you play white or black. */}
      {checkered && (
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }} aria-hidden>
          {Array.from({ length: cols }, (_, vc) => {
            const gc = flip ? cols - 1 - vc : vc // grid file under this visual column
            const gr = flip ? 0 : rows - 1 // the visual bottom row's grid row
            return (
              <span key={`file-${vc}`} style={{
                position: 'absolute', left: pad + vc * cellPx + 3, bottom: pad + 1,
                fontSize: Math.max(8, Math.round(cellPx * 0.2)), fontWeight: 700, lineHeight: 1,
                color: sqIsLight(gr, gc) ? DARK_SQ : LIGHT_SQ,
              }}>
                {String.fromCharCode(97 + gc)}
              </span>
            )
          })}
          {Array.from({ length: rows }, (_, vr) => {
            const gr = flip ? rows - 1 - vr : vr // grid rank under this visual row
            const gc = flip ? 0 : cols - 1 // the visual right column's grid col
            return (
              <span key={`rank-${vr}`} style={{
                position: 'absolute', right: pad + 3, top: pad + vr * cellPx + 2,
                fontSize: Math.max(8, Math.round(cellPx * 0.2)), fontWeight: 700, lineHeight: 1,
                color: sqIsLight(gr, gc) ? DARK_SQ : LIGHT_SQ,
              }}>
                {rows - gr}
              </span>
            )
          })}
        </div>
      )}
      </div>

      {/* Chess promotion picker (G6g): when a pawn reaches the back rank the (from,to) carries four
          actions (=Q/=R/=B/=N); this overlay lets the player choose which piece. Rendered OUTSIDE the
          (possibly flipped) grid so it stays upright, and as a centred panel + backdrop so it never
          depends on per-cell coordinates. Clicking a piece submits its action; the backdrop cancels. */}
      {promoAt && (
        <>
          <button
            aria-label={t('board.promote_cancel')}
            onClick={() => setPromoAt(null)}
            style={{
              position: 'absolute', inset: 0, background: 'color-mix(in srgb, var(--surface-1) 55%, transparent)',
              border: 'none', cursor: 'pointer', borderRadius: 'var(--radius-md)',
            }}
          />
          <div role="group" aria-label={t('board.promote_prompt')} style={{
            position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, padding: '14px 18px',
            background: 'var(--surface-2)', border: '2px solid var(--border-strong)',
            borderRadius: 'var(--radius-md)', boxShadow: 'var(--shadow-md)',
          }}>
            <span style={{ fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)' }}>
              {t('board.promote_prompt')}
            </span>
            <div style={{ display: 'flex', gap: 8 }}>
              {[...promoAt.moves]
                .sort((a, b) => (PROMO_ORDER[a.promotion ?? ''] ?? 9) - (PROMO_ORDER[b.promotion ?? ''] ?? 9))
                .map((m) => {
                  const letter = m.promotion ?? 'q'
                  const promoPiece = meta.pieces[promoCase(letter)]
                  return (
                    <button key={m.action}
                      onClick={() => { onCellClick(m.action); setSelectedFrom(null); setPromoAt(null) }}
                      aria-label={t('board.promote_to', { piece: t(`board.piece_${letter}`) })}
                      style={{
                        width: 52, height: 52, display: 'flex', alignItems: 'center', justifyContent: 'center',
                        background: 'var(--surface-1)', border: '1px solid var(--border-default)',
                        borderRadius: 'var(--radius-sm)', cursor: 'pointer',
                      }}>
                      <span aria-hidden style={{ fontSize: 30, lineHeight: 1, color: promoPiece?.color ?? 'var(--text-strong)', fontWeight: 800 }}>
                        {promoPiece?.glyph ?? letter.toUpperCase()}
                      </span>
                    </button>
                  )
                })}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export function GridStage({ envName, grid, agent }: {
  envName: string; grid: GridLayout; agent: number[]
}) {
  const { rows, cols, cells, kind } = grid
  const w = cols * GRID_CELL
  const h = rows * GRID_CELL
  const cx = (c: number) => c * GRID_CELL + GRID_CELL / 2
  const cy = (r: number) => r * GRID_CELL + GRID_CELL / 2
  const maxW = kind === 'cliffwalking' ? 780 : 460
  return (
    <svg viewBox={`-6 -6 ${w + 12} ${h + 12}`} preserveAspectRatio="xMidYMid meet"
      style={{ width: '100%', maxWidth: maxW, maxHeight: '100%' }} role="img" aria-label={envName}>
      {cells.map((tag, i) => {
        const r = Math.floor(i / cols)
        const c = i % cols
        const base = tag === 'goal' ? 'var(--accent-surface)'
          : tag === 'start' || tag === 'stop' ? 'var(--surface-3)'
          : 'var(--surface-2)'
        return (
          <g key={i}>
            <rect x={c * GRID_CELL} y={r * GRID_CELL} width={GRID_CELL} height={GRID_CELL} rx="5"
              fill={base} stroke="var(--border-default)" strokeWidth="1.5" />
            {tag === 'cliff' && (
              <rect x={c * GRID_CELL} y={r * GRID_CELL} width={GRID_CELL} height={GRID_CELL} rx="5"
                fill="var(--danger)" fillOpacity="0.32" />
            )}
            {tag === 'hole' && (
              <circle cx={cx(c)} cy={cy(r)} r={GRID_CELL * 0.3}
                fill="var(--surface-1)" stroke="var(--danger)" strokeWidth="2" />
            )}
            {tag === 'goal' && kind !== 'taxi' && <GridFlag x={cx(c)} y={cy(r)} color="var(--accent)" />}
          </g>
        )
      })}
      {kind === 'taxi' ? (
        <TaxiPieces agent={agent} cx={cx} cy={cy} />
      ) : (
        <circle cx={cx(agent[1] ?? 0)} cy={cy(agent[0] ?? 0)} r={GRID_CELL * 0.27}
          fill="var(--accent)" stroke="var(--surface-1)" strokeWidth="2.5" />
      )}
    </svg>
  )
}
