// Client-side rendered env "stages" (SVG) — drawn from the raw physics state the backend streams
// (app/services/client_render.py → client_state). These are *presentational*: EnvPreview owns the
// refs and updates them imperatively from each frame (no React re-render per frame), exactly like
// the CartPole cart/pole. The geometry both files need lives in ./envGeometry. Keep the env set +
// state layout in sync with the backend's client_state.

import type { CSSProperties, RefObject } from 'react'
import { useTranslation } from 'react-i18next'
import type { BoardState, GridLayout } from '../api/types'
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
export function BoardStage({ envName, board, meta, humanTurn, onCellClick, statusText, banner }: {
  envName: string
  board: BoardState
  meta: BoardGameMeta
  humanTurn: boolean
  onCellClick: (action: number) => void
  statusText: string
  // `mark` colours the winner's piece in the banner — for games whose two players share one glyph
  // (Connect Four: both '●', only the colour differs), "● wins" would otherwise be unreadable.
  banner: { text: string; kind: 'win' | 'draw' | 'loss'; mark?: { glyph: string; color: string } } | null
}) {
  const { t } = useTranslation()
  const { rows, cols, cells, legal_actions, last_action, is_terminal } = board
  const legal = new Set(legal_actions)
  const cellPx = Math.max(48, Math.min(110, Math.floor(360 / Math.max(rows, cols, 1))))
  const bannerColor =
    banner?.kind === 'win' ? 'var(--success)' : banner?.kind === 'loss' ? 'var(--danger)' : 'var(--text-strong)'

  // Column games (Connect Four, G6c): a move is a column drop, so a cell's action is its column. Cell
  // games (Tic-Tac-Toe): action == cell index. No HOVER highlighting at all — by request the board is
  // plain; a legal cell is simply a clickable button (cursor only). The one highlight kept is a ring on
  // the *last move played* (a useful, single-cell marker the user asked to keep).
  const columnMode = meta.actionMode === 'column'
  const actionOf = (i: number) => (columnMode ? i % cols : i)

  // The just-played disc to ring: in cell mode it is the last_action cell; in column mode it is the
  // top-most filled cell of the last_action column (pieces stack from the bottom, so the most recent
  // one sits at the smallest occupied row index).
  const isFilled = (i: number) => {
    const v = cells[i]?.trim() ?? ''
    return v !== '' && v !== '.'
  }
  let lastCell: number | null = null
  if (last_action !== null) {
    if (columnMode) {
      for (let r = 0; r < rows; r++) {
        const idx = r * cols + last_action
        if (isFilled(idx)) { lastCell = idx; break }
      }
    } else {
      lastCell = last_action
    }
  }

  const cellBox: CSSProperties = {
    width: cellPx, height: cellPx, display: 'flex', alignItems: 'center', justifyContent: 'center',
    background: 'var(--surface-2)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-default)',
    padding: 0, margin: 0,
  }

  return (
    <div role="group" aria-label={envName} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 14 }}>
      {/* Whose turn / the final result — leads the board so play always reads as labelled. */}
      <div style={{
        minHeight: 24, fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)',
        color: banner ? bannerColor : 'var(--text-muted)', textAlign: 'center',
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 6,
      }}>
        {banner?.mark && (
          <span aria-hidden style={{ color: banner.mark.color, fontWeight: 800 }}>{banner.mark.glyph}</span>
        )}
        {banner ? banner.text : statusText}
      </div>

      <div style={{
        display: 'grid', gap: 5,
        gridTemplateColumns: `repeat(${cols}, ${cellPx}px)`,
        gridTemplateRows: `repeat(${rows}, ${cellPx}px)`,
        background: 'var(--surface-3)', padding: 6, borderRadius: 'var(--radius-md)',
        boxShadow: 'var(--shadow-md)',
      }}>
        {cells.map((ch, i) => {
          const r = Math.floor(i / cols)
          const c = i % cols
          const action = actionOf(i)
          const mark = ch.trim().toUpperCase()
          const piece = meta.pieces[ch.trim().toLowerCase()]
          const ring = lastCell === i ? `inset 0 0 0 3px ${piece?.color ?? 'var(--accent)'}` : undefined
          const glyph = piece && (
            <span aria-hidden style={{ fontSize: cellPx * 0.55, lineHeight: 1, color: piece.color, fontWeight: 800 }}>
              {piece.glyph}
            </span>
          )
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
                style={{ ...cellBox, cursor: 'pointer', boxShadow: ring }}
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
              style={{ ...cellBox, boxShadow: ring }}
            >
              {glyph}
            </div>
          )
        })}
      </div>
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
