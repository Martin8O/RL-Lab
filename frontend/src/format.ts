// Shared numeric formatting — the project-wide rule: once a count crosses 1000k it reads in
// **millions** ("1.2M", "2M"), never "1200k". Used everywhere a step/episode count is shown (the
// chart axis + stats, the sidebar budget ladder, the save cards) so the unit is consistent.

export function formatCount(n: number): string {
  const a = Math.abs(n)
  if (a >= 1_000_000) {
    const s = (n / 1_000_000).toFixed(1)
    return `${s.endsWith('.0') ? s.slice(0, -2) : s}M` // 1.0M → "1M", 1.2M → "1.2M"
  }
  if (a >= 1000) return `${Math.round(n / 1000)}k`
  return String(Math.round(n))
}
