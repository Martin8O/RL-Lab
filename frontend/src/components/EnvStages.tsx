// Client-side rendered env "stages" (SVG) — drawn from the raw physics state the backend streams
// (app/services/client_render.py → client_state). These are *presentational*: EnvPreview owns the
// refs and updates them imperatively from each frame (no React re-render per frame), exactly like
// the CartPole cart/pole. The geometry both files need lives in ./envGeometry. Keep the env set +
// state layout in sync with the backend's client_state.

import type { RefObject } from 'react'
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
