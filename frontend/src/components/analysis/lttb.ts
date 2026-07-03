// Largest-Triangle-Three-Buckets downsampling (Sveinn Steinarsson, 2013) — the standard curve
// simplification that keeps a line's *visual shape* (peaks, troughs, slopes) while cutting the point
// count. DataLab overlays many full-resolution run curves at once; drawing every raw point (a long
// PPO run can log thousands) would bloat the SVG for no visible gain, so each series is thinned to a
// budget of points for display before it's drawn. CSV/analysis stay full-resolution server-side — this
// is a *display-only* thinning, exactly like the XLSX LTTB pass on the backend (X5).
//
// Pure + side-effect-free (unit-tested): given the raw points and a target count, it returns a subset
// (never invents points) that always keeps the first and last sample so the endpoints stay anchored.

export interface Pt {
  x: number
  y: number
}

/** Downsample `points` to at most `threshold` points via LTTB, preserving the first/last and the
 *  shape-defining extrema. Returns the input unchanged when it's already within budget (or the budget
 *  is < 3 — too small to run the three-bucket rule, so we can't honestly thin it). */
export function lttb(points: Pt[], threshold: number): Pt[] {
  const n = points.length
  if (threshold >= n || threshold < 3) return points

  const sampled: Pt[] = [points[0]] // always keep the first point
  // Each of the `threshold - 2` middle output points comes from one bucket of the input; the first and
  // last inputs are fixed, so the interior is divided into that many equal buckets.
  const bucketSize = (n - 2) / (threshold - 2)
  let a = 0 // index of the previously-selected point (the triangle's left vertex)

  for (let i = 0; i < threshold - 2; i++) {
    // The *next* bucket's average point — the triangle's right vertex (a look-ahead anchor).
    const nextStart = Math.floor((i + 1) * bucketSize) + 1
    const nextEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, n)
    let avgX = 0
    let avgY = 0
    const nextCount = nextEnd - nextStart
    for (let j = nextStart; j < nextEnd; j++) {
      avgX += points[j].x
      avgY += points[j].y
    }
    if (nextCount > 0) {
      avgX /= nextCount
      avgY /= nextCount
    }

    // This bucket's range — pick the point forming the largest triangle with (a, next-average).
    const rangeStart = Math.floor(i * bucketSize) + 1
    const rangeEnd = Math.floor((i + 1) * bucketSize) + 1
    const pa = points[a]
    let maxArea = -1
    let chosen = rangeStart
    for (let j = rangeStart; j < rangeEnd; j++) {
      const area = Math.abs(
        (pa.x - avgX) * (points[j].y - pa.y) - (pa.x - points[j].x) * (avgY - pa.y),
      )
      if (area > maxArea) {
        maxArea = area
        chosen = j
      }
    }
    sampled.push(points[chosen])
    a = chosen
  }

  sampled.push(points[n - 1]) // always keep the last point
  return sampled
}
