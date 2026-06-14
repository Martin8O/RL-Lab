// Geometry for the client-side env stages (EnvStages.tsx), shared with EnvPreview, which uses it to
// position the moving parts imperatively each frame (no React re-render). Pure module (no JSX) so
// both can import it without tripping the react-refresh "only export components" rule.

// ── CartPole ──────────────────────────────────────────────────────────────────────────────────
export const CART_X_LIMIT = 2.4   // CartPole fails at |x| ≈ 2.4
export const CART_X_SCALE = 250   // px of horizontal travel from ±x-limit (track 30..570, centre 300)

// ── MountainCar (mountaincar + mountaincarcontinuous) ─────────────────────────────────────────
// The car can only ever reach position −1.2 (a hard wall in the env), so we draw −1.2 as the LEFT
// END / top of a big hill: the car climbs to the top-left and stops there, never mid-slope (the old
// drawing extended past −1.2, so the car stopped in the middle with speed left — looked buggy).
// Two smooth cosine humps: a tall left hill + a right hill to the goal flag, valley in the middle.
export const MC_POS_MIN = -1.2, MC_POS_MAX = 0.6, MC_VALLEY = -0.5, MC_GOAL = 0.45, MC_START = -0.5
const MC_LEFT_H = 1.18    // normalized height of the big left hilltop (at −1.2)
const MC_RIGHT_H = 0.82   // normalized height of the right (goal) side
const MC_X0 = 60, MC_X1 = 540, MC_BASE_Y = 298, MC_AMP = 150, MC_FLOOR = 340
const MC_TILT_CAP = 38    // clamp the car's tilt so it stays readable on the steep big hill
// Normalized terrain height (0 at the valley); a cosine hump each side, flat at the very ends.
const mcHeight = (p: number) => {
  if (p <= MC_VALLEY) {
    const t = (MC_VALLEY - p) / (MC_VALLEY - MC_POS_MIN)   // 0 at valley → 1 at −1.2
    return (MC_LEFT_H * (1 - Math.cos(Math.PI * t))) / 2
  }
  const t = (p - MC_VALLEY) / (MC_POS_MAX - MC_VALLEY)     // 0 at valley → 1 at 0.6
  return (MC_RIGHT_H * (1 - Math.cos(Math.PI * t))) / 2
}
const mcHeightSlope = (p: number) => {  // d(height)/d(pos), for the car's tilt
  if (p <= MC_VALLEY) {
    const span = MC_VALLEY - MC_POS_MIN
    return (MC_LEFT_H * (Math.PI / 2) * Math.sin((Math.PI * (MC_VALLEY - p)) / span)) * (-1 / span)
  }
  const span = MC_POS_MAX - MC_VALLEY
  return (MC_RIGHT_H * (Math.PI / 2) * Math.sin((Math.PI * (p - MC_VALLEY)) / span)) * (1 / span)
}
export const mcX = (p: number) => MC_X0 + ((p - MC_POS_MIN) / (MC_POS_MAX - MC_POS_MIN)) * (MC_X1 - MC_X0)
export const mcY = (p: number) => MC_BASE_Y - mcHeight(p) * MC_AMP
/** Car pose at position p: sit on the surface, tilted to the local slope (clamped, screen dY/dX). */
export const mcCarTransform = (p: number) => {
  const dX = (MC_X1 - MC_X0) / (MC_POS_MAX - MC_POS_MIN)
  const dY = -mcHeightSlope(p) * MC_AMP
  const raw = (Math.atan2(dY, dX) * 180) / Math.PI
  const deg = Math.max(-MC_TILT_CAP, Math.min(MC_TILT_CAP, raw))
  return `translate(${mcX(p).toFixed(1)} ${mcY(p).toFixed(1)}) rotate(${deg.toFixed(1)})`
}
const MC_SURFACE = (() => {
  const pts: string[] = []
  for (let p = MC_POS_MIN; p <= MC_POS_MAX + 1e-9; p += 0.02) pts.push(`${mcX(p).toFixed(1)} ${mcY(p).toFixed(1)}`)
  return pts.join(' L ')
})()
export const MC_SURFACE_PATH = `M ${MC_SURFACE}`
export const MC_GROUND_PATH = `M ${MC_SURFACE} L ${mcX(MC_POS_MAX).toFixed(1)} ${MC_FLOOR} L ${mcX(MC_POS_MIN).toFixed(1)} ${MC_FLOOR} Z`

// ── Pendulum ──────────────────────────────────────────────────────────────────────────────────
// Rod pivots at centre; θ = 0 points straight up (the goal). EnvPreview rotates the rod group by θ.
export const PEND_CX = 300, PEND_CY = 158, PEND_L = 116

// ── Acrobot ─────────────────────────────────────────────────────────────────────────────────
// Two links from a top pivot (θ1 = 0 points down, θ2 relative to link 1). EnvPreview rotates link1
// by θ1 (around the pivot) and the nested link2 by θ2 (around the joint).
export const ACRO_CX = 300, ACRO_CY = 112, ACRO_L = 72
export const ACRO_JOINT_Y = ACRO_CY + ACRO_L   // link1's end in the rest (hanging) pose

// ── LunarLander ───────────────────────────────────────────────────────────────────────────────
// Drawn from the 8-number obs [x, y, vx, vy, angle, ang_vel, leg1, leg2] PLUS the per-episode moon
// surface the backend streams (client_render.terrain) — the real terrain is randomly generated and is
// NOT in the obs, so a flat guess made the lander look like it landed above/below ground off-pad.
// We use the env's own viewport→screen mapping (SCALE 30, 600×400) and the SAME obs normalization for
// the lander and the terrain, so they share one coordinate space exactly and nothing clips: obs-x is
// centred on the viewport (±1 at the edges), obs-y = 0 is the lander's pad-touchdown height (the
// surface sits a touch below it), angle 0 is upright.
export const LL_CX = 300, LL_Y0 = 282, LL_HALF_W = 300, LL_H_SCALE = 200
export const LL_PAD_HALF = 0.2             // landing-pad half-width in obs-x (always the fixed centre chunks)
export const LL_PAD_OBS_Y = -0.09          // pad surface height in obs-y (helipad height is fixed every episode)
export const LL_START = { x: 0, y: 1.36 }  // idle pose: near the top, centred
export const llX = (x: number) => LL_CX + x * LL_HALF_W
export const llY = (y: number) => LL_Y0 - y * LL_H_SCALE
/** Lander pose from the obs: translate the body centre to (x, y), then rotate around it (screen-y is
 *  down → −angle). The drawn legs reach the surface; the streamed terrain is at its true height, so
 *  the feet meet the ground at the pad AND off-pad. */
export const llLanderTransform = (x: number, y: number, angle: number) =>
  `translate(${llX(x).toFixed(1)} ${llY(y).toFixed(1)}) rotate(${((-angle * 180) / Math.PI).toFixed(1)})`
/** Build the moon paths from streamed obs-space surface points: a filled ground polygon (down to the
 *  viewBox floor at y=400) plus the crisp surface stroke. */
export function llTerrainPaths(
  points: ReadonlyArray<readonly number[]>,
): { ground: string; surface: string } {
  if (!points.length) return { ground: '', surface: '' }
  const surface = `M ${points.map((p) => `${llX(p[0]).toFixed(1)} ${llY(p[1]).toFixed(1)}`).join(' L ')}`
  const x0 = llX(points[0][0]).toFixed(1)
  const x1 = llX(points[points.length - 1][0]).toFixed(1)
  return { ground: `${surface} L ${x1} 400 L ${x0} 400 Z`, surface }
}
/** Representative jagged moon (11 points like the real env, flat pad in the middle), shown before a
 *  real per-episode terrain streams in — at idle / on env select. A flat line read as "wrong env";
 *  this looks like the moon while staying an honest placeholder until the real surface arrives. */
export const LL_DEFAULT_TERRAIN: ReadonlyArray<readonly [number, number]> = [
  [-1.0, -0.32], [-0.8, -0.19], [-0.6, -0.05], [-0.4, -0.13],
  [-0.2, LL_PAD_OBS_Y], [0.0, LL_PAD_OBS_Y], [0.2, LL_PAD_OBS_Y],
  [0.4, -0.16], [0.6, 0.0], [0.8, -0.12], [1.0, -0.30],
]
