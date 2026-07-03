// Pure sorting for the DataLab summary table (Zone 4, X6b). i18n-free / side-effect-free (unit-tested);
// the table component owns all rendering + formatting. The one subtlety worth isolating: a summary
// scalar is `| null` wherever a short / never-solved run can't produce it (a run that never solved has
// no `solved_env_steps`), and those blanks must always sort to the *bottom* regardless of direction —
// otherwise "rank by time-to-solve, ascending" would float the never-solved runs to the top as if they
// were the fastest. Sorting is stable within the null group (the input order — i.e. selection order).

import type { RunSummary } from '../../api/types'

// The numeric RunSummary fields the table can rank on (plus the non-numeric 'label' handled separately).
export type SortKey =
  | 'label'
  | 'final_skill_pct'
  | 'auc_normalized'
  | 'solved_env_steps'
  | 'peak_skill_pct'
  | 'collapse_pct'
  | 'mean_steps_per_sec'

export type SortDir = 'asc' | 'desc'

/** One selected run paired with its display label — what the table sorts and renders. */
export interface SummaryRow {
  summary: RunSummary
  label: string
}

/** Compare two nullable numbers with nulls (and NaN) forced last in *both* directions. */
export function compareNullable(
  a: number | null | undefined,
  b: number | null | undefined,
  dir: SortDir,
): number {
  const aNull = a == null || Number.isNaN(a)
  const bNull = b == null || Number.isNaN(b)
  if (aNull && bNull) return 0
  if (aNull) return 1 // a sinks
  if (bNull) return -1 // b sinks
  return dir === 'asc' ? a - b : b - a
}

/** Sort a copy of `rows` by `key`/`dir`. 'label' sorts case-insensitively by the display label; every
 *  other key sorts the numeric summary field with nulls last. Stable within ties (native sort). */
export function sortRows(rows: SummaryRow[], key: SortKey, dir: SortDir): SummaryRow[] {
  const copy = rows.slice()
  copy.sort((ra, rb) => {
    if (key === 'label') {
      const c = ra.label.localeCompare(rb.label, undefined, { sensitivity: 'base', numeric: true })
      return dir === 'asc' ? c : -c
    }
    return compareNullable(ra.summary[key], rb.summary[key], dir)
  })
  return copy
}
