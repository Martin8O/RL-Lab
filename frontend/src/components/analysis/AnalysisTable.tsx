// The DataLab summary table (Zone 4, X6b) — one row per selected run of the X2 summary statistics, the
// scalars a paper reports (final skill %, AUC, time-to-solve, peak, collapse, throughput). Click a header
// to rank by that column (the "rank by AUC / final % / time-to-solve" the DoD asks for); nulls always
// sink to the bottom so a never-solved run can't masquerade as the fastest. When the selection is
// collapsed into a multi-seed band, a highlighted aggregate row carries the mean ± std across seeds (the
// AggregatedSummary ± columns, X4) above the per-seed rows. Theme tokens, tabular-mono numerics, sortable
// column headers with an aria-sort state. Pure presentation over data the parent fetched.

import { useMemo, useState } from 'react'
import { useTranslation } from 'react-i18next'
import type { AggregatedSummary, RunSummary } from '../../api/types'
import { formatCount } from '../../format'
import { sortRows, type SortDir, type SortKey, type SummaryRow } from './summarySort'

const NA = '—'
const pct0 = (v: number) => `${Math.round(v)}%`
const auc = (v: number) => v.toFixed(2)
const stepsPerSec = (v: number) => `${formatCount(v)}/s`

interface Col {
  key: Exclude<SortKey, 'label'> // the RunSummary numeric field this column ranks + reads
  labelKey: string
  fmt: (v: number) => string
}

// The RunSummary field names double as the sort keys and the AggregatedSummary.metrics keys.
const COLS: Col[] = [
  { key: 'final_skill_pct', labelKey: 'analysis.col_final', fmt: pct0 },
  { key: 'auc_normalized', labelKey: 'analysis.col_auc', fmt: auc },
  { key: 'solved_env_steps', labelKey: 'analysis.col_solved', fmt: formatCount },
  { key: 'peak_skill_pct', labelKey: 'analysis.col_peak', fmt: pct0 },
  { key: 'collapse_pct', labelKey: 'analysis.col_collapse', fmt: pct0 },
  { key: 'mean_steps_per_sec', labelKey: 'analysis.col_speed', fmt: stepsPerSec },
]

// Numeric columns are centred under their header label (X7) — the header and the value share one axis, so
// the number reads directly beneath its name. The first "Run" column overrides this back to left.
const thBase: React.CSSProperties = {
  textAlign: 'center', padding: '5px 10px', fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)',
  color: 'var(--text-muted)', whiteSpace: 'nowrap', userSelect: 'none', cursor: 'pointer',
  borderBottom: '1px solid var(--border-default)', position: 'sticky', top: 0, background: 'var(--surface-1)',
}
const tdBase: React.CSSProperties = {
  textAlign: 'center', padding: '4px 10px', fontSize: 'var(--fs-label)',
  fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
  color: 'var(--text-default)', whiteSpace: 'nowrap',
}

export default function AnalysisTable({
  summaries,
  labelOf,
  colorOf,
  aggregate,
  seedCount,
}: {
  summaries: RunSummary[]
  labelOf: (runId: string) => string
  colorOf: (runId: string) => string | null
  aggregate?: AggregatedSummary | null // when a multi-seed band is active → the ± aggregate row
  seedCount?: number
}) {
  const { t } = useTranslation()
  // Default rank = AUC descending (the "how fast AND how high" ranking key the X2 doc names).
  const [sortKey, setSortKey] = useState<SortKey>('auc_normalized')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const rows: SummaryRow[] = useMemo(
    () => summaries.map((s) => ({ summary: s, label: labelOf(s.run_id) })),
    [summaries, labelOf],
  )
  const sorted = useMemo(() => sortRows(rows, sortKey, sortDir), [rows, sortKey, sortDir])

  const onSort = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir(key === 'label' ? 'asc' : 'desc') // numerics read best high-first
    }
  }

  const ariaSort = (key: SortKey): React.AriaAttributes['aria-sort'] =>
    key === sortKey ? (sortDir === 'asc' ? 'ascending' : 'descending') : 'none'
  const arrow = (key: SortKey) => (key === sortKey ? (sortDir === 'asc' ? ' ▲' : ' ▼') : '')

  // The aggregate row's per-column "mean ± std" — only for columns X4 actually produced a stat for.
  const aggCell = (col: Col): string => {
    const st = aggregate?.metrics[col.key]
    if (!st) return NA
    const mean = col.fmt(st.mean)
    return st.std != null ? `${mean} ± ${col.fmt(st.std)}` : mean
  }

  if (summaries.length === 0) {
    return (
      <div style={{ padding: '10px 4px', fontSize: 'var(--fs-meta)', color: 'var(--text-faint)' }}>
        {t('analysis.table_loading')}
      </div>
    )
  }

  return (
    <div style={{ overflow: 'auto', maxHeight: '100%', border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%', minWidth: 520 }}>
        <thead>
          <tr>
            <th
              scope="col" aria-sort={ariaSort('label')} onClick={() => onSort('label')}
              style={{ ...thBase, textAlign: 'left' }}
            >
              {t('analysis.col_run')}{arrow('label')}
            </th>
            {COLS.map((c) => (
              <th
                key={c.key} scope="col" aria-sort={ariaSort(c.key)} onClick={() => onSort(c.key)}
                title={t('analysis.sort_by', { col: t(c.labelKey) })} style={thBase}
              >
                {t(c.labelKey)}{arrow(c.key)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {aggregate && (seedCount ?? aggregate.n_seeds) >= 2 && (
            <tr style={{ background: 'var(--surface-2)' }}>
              <td style={{ ...tdBase, textAlign: 'left', color: 'var(--text-strong)', fontFamily: 'var(--font-sans)', fontWeight: 'var(--fw-semibold)' }}>
                {t('analysis.agg_row', { n: seedCount ?? aggregate.n_seeds })}
              </td>
              {COLS.map((c) => (
                <td key={c.key} style={{ ...tdBase, color: 'var(--text-strong)' }}>{aggCell(c)}</td>
              ))}
            </tr>
          )}
          {sorted.map(({ summary: s, label }) => (
            <tr key={s.run_id} className="dl-row" style={{ borderTop: '1px solid var(--border-subtle)' }}>
              <td style={{ ...tdBase, textAlign: 'left', fontFamily: 'var(--font-sans)', color: 'var(--text-default)' }}>
                <span style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
                  <span style={{ width: 9, height: 9, borderRadius: 2, flexShrink: 0, background: colorOf(s.run_id) ?? 'var(--text-faint)' }} />
                  <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 220 }}>{label}</span>
                </span>
              </td>
              {COLS.map((c) => {
                const v = s[c.key]
                return <td key={c.key} style={tdBase}>{v == null ? NA : c.fmt(v)}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
