import { describe, expect, it } from 'vitest'
import { lttb, type Pt } from './lttb'

function ramp(n: number): Pt[] {
  return Array.from({ length: n }, (_, i) => ({ x: i, y: i }))
}

describe('lttb', () => {
  it('passes the input through unchanged when already within budget', () => {
    const pts = ramp(5)
    expect(lttb(pts, 5)).toBe(pts)
    expect(lttb(pts, 10)).toBe(pts)
  })

  it('passes through for a budget below the 3-point minimum', () => {
    const pts = ramp(100)
    expect(lttb(pts, 2)).toBe(pts)
  })

  it('downsamples to exactly the threshold, keeping the first and last points', () => {
    const pts = ramp(100)
    const out = lttb(pts, 10)
    expect(out).toHaveLength(10)
    expect(out[0]).toEqual(pts[0])
    expect(out[out.length - 1]).toEqual(pts[99])
  })

  it('preserves a sharp spike (the shape-defining extremum survives thinning)', () => {
    // A flat line with one tall spike in the middle — LTTB should keep the spike.
    const pts: Pt[] = ramp(100).map((p) => ({ x: p.x, y: 0 }))
    pts[50] = { x: 50, y: 999 }
    const out = lttb(pts, 12)
    expect(Math.max(...out.map((p) => p.y))).toBe(999)
  })

  it('never invents points — every output is a member of the input', () => {
    const pts = ramp(50)
    const set = new Set(pts)
    for (const p of lttb(pts, 8)) expect(set.has(p)).toBe(true)
  })
})
