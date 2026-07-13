// #2b — shared brand marks for the two audience modes, used by both the first-launch ModeChooser and
// the TopBar ModeToggle so the gamepad (Simple) and neural-network (Advanced) icons read identically
// everywhere. The per-mode hues live in ../content/modeHues (a plain .ts module) so this file exports
// components only — required for React Fast Refresh.
import type { AudienceMode } from '../store/useAppStore'
import { HUES, type Hue } from '../content/modeHues'

// Unique gradient ids per rendered size, so the ModeChooser (large) and ModeToggle (small) instances
// on the same page never collide on a shared <linearGradient id>.
function gid(prefix: string, size: number) { return `${prefix}-${Math.round(size)}` }

// Simple = a detailed dual-stick game controller. Filled dark silhouette with cream d-pad, a 4-button
// diamond, twin analog sticks and shoulder bumps — reads as a real modern controller, not a toy.
export function SimpleMark({ hue, size = 54 }: { hue: Hue; size?: number }) {
  const id = gid('mc-simple', size)
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden style={{ display: 'block' }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={hue.tileFrom} /><stop offset="1" stopColor={hue.tileTo} />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill={`url(#${id})`} />
      {/* whole controller nudged up so its mass sits in the tile centre */}
      <g transform="translate(0 -3)">
        <path d="M22 23.5 q-4-2.5-7 .5 M42 23.5 q4-2.5 7 .5" stroke="#2a1a06" strokeWidth="2.4" strokeLinecap="round" fill="none" />
        <path d="M23 25 H41 a12.5 12.5 0 0 1 12.4 14.2 l-1.1 7.4 a7 7 0 0 1-13.1 2.2 l-1.5-2.8 a3.4 3.4 0 0 0-3-1.8 H31.3 a3.4 3.4 0 0 0-3 1.8 l-1.5 2.8 a7 7 0 0 1-13.1-2.2 l-1.1-7.4 A12.5 12.5 0 0 1 23 25 Z"
          fill="#2a1a06" />
        <path d="M18.6 31 h4.8 M21 28.6 v4.8" stroke="#FFF1D4" strokeWidth="2.3" strokeLinecap="round" />
        <circle cx="43" cy="28.4" r="1.7" fill="#FFF1D4" />
        <circle cx="43" cy="35" r="1.7" fill="#FFF1D4" />
        <circle cx="39.7" cy="31.7" r="1.7" fill="#FFF1D4" />
        <circle cx="46.3" cy="31.7" r="1.7" fill="#FFF1D4" />
        <circle cx="26.5" cy="39.5" r="3.4" fill="#3a2a12" stroke="#FFF1D4" strokeWidth="1.8" />
        <circle cx="37.5" cy="39.5" r="3.4" fill="#3a2a12" stroke="#FFF1D4" strokeWidth="1.8" />
      </g>
    </svg>
  )
}

// Advanced = a neural network (nodes + connections) — the "grown-up" scientific mark for tuning models
// & algorithms. Three fully-connected layers with faint edges; the output node is amber (brand accent).
export function AdvancedMark({ hue, size = 54 }: { hue: Hue; size?: number }) {
  const id = gid('mc-advanced', size)
  const L1 = [[19, 24], [19, 40]]
  const L2 = [[32, 19], [32, 32], [32, 45]]
  const L3 = [[45, 27], [45, 37]]
  const edges: [number[], number[]][] = []
  for (const a of L1) for (const b of L2) edges.push([a, b])
  for (const a of L2) for (const b of L3) edges.push([a, b])
  const node = (p: number[], i: number, amber = false) => (
    <circle key={`${amber ? 'o' : 'n'}${i}`} cx={p[0]} cy={p[1]} r={3}
      fill={amber ? '#F0A93A' : '#fff'} stroke="#1c103a" strokeWidth="1.3" />
  )
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden style={{ display: 'block' }}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={hue.tileFrom} /><stop offset="1" stopColor={hue.tileTo} />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill={`url(#${id})`} />
      {/* enlarge the network ~1.3× about the tile centre so it fills the icon */}
      <g transform="translate(-9.6 -9.6) scale(1.3)">
        {edges.map(([a, b], i) => (
          <line key={i} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="#fff" strokeOpacity="0.5" strokeWidth="0.9" />
        ))}
        {L1.map((p, i) => node(p, i))}
        {L2.map((p, i) => node(p, i + 2))}
        {L3.map((p, i) => node(p, i + 5, i === 0))}
      </g>
    </svg>
  )
}

export function MarkFor({ mode, size }: { mode: AudienceMode; size?: number }) {
  return mode === 'simple'
    ? <SimpleMark hue={HUES.simple} size={size} />
    : <AdvancedMark hue={HUES.advanced} size={size} />
}
