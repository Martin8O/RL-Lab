// The DataLab export zone (Zone 5, X6b) — download the current run selection as a citable dataset. Each
// button is a plain <a download> to an X5 export endpoint (server-side over the full on-disk history, so
// the export is full-resolution, not the capped live store); the pivot-aware formats (CSV, XLSX, the SVG
// figure) honour the active pivot (per-game raw reward / per-algorithm normalized skill-%). X7 promoted
// the two former "coming" formats — a standalone vector figure (SVG) and a TensorBoard event-file zip —
// to live downloads, so all six formats ship. Each button's description ends with its file extension,
// highlighted, so the format-vs-file-type reads uniformly across all six. XLSX + SVG localize their
// descriptive text to the app language (`locale`). Theme tokens, aria-labels.

import { useTranslation } from 'react-i18next'
import { analysisExportUrl, type ExportFmt } from '../../api/client'
import { useAppStore } from '../../store/useAppStore'

interface Fmt {
  fmt: ExportFmt
  labelKey: string
  descKey: string
  ext: string // the file extension the download saves as (shown highlighted in the description)
}

const FORMATS: Fmt[] = [
  { fmt: 'csv', labelKey: 'analysis.export_csv', descKey: 'analysis.export_csv_desc', ext: '.csv' },
  { fmt: 'xlsx', labelKey: 'analysis.export_xlsx', descKey: 'analysis.export_xlsx_desc', ext: '.xlsx' },
  { fmt: 'repro', labelKey: 'analysis.export_repro', descKey: 'analysis.export_repro_desc', ext: '.md' },
  { fmt: 'tex', labelKey: 'analysis.export_latex', descKey: 'analysis.export_latex_desc', ext: '.tex' },
  { fmt: 'svg', labelKey: 'analysis.export_figure', descKey: 'analysis.export_figure_desc', ext: '.svg' },
  { fmt: 'zip', labelKey: 'analysis.export_tensorboard', descKey: 'analysis.export_tensorboard_desc', ext: '.zip' },
]

// The pivot-aware formats (raw reward vs normalized skill-%) — the rest ignore the compare mode.
const PIVOT_AWARE = new Set<ExportFmt>(['csv', 'xlsx', 'svg'])

const btnStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 1,
  padding: '6px 10px', textDecoration: 'none', cursor: 'pointer',
  border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
  background: 'var(--surface-2)', color: 'var(--text-strong)', textAlign: 'left',
}

// The description line: what the format is, then its file extension highlighted (mono + accent) so every
// button states its file type the same way.
function Desc({ text, ext }: { text: string; ext: string }) {
  return (
    <span style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-faint)' }}>
      {text}{' · '}
      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 'var(--fw-semibold)', color: 'var(--accent)' }}>
        {ext}
      </span>
    </span>
  )
}

export default function ExportZone({
  runIds,
  pivot,
}: {
  runIds: string[]
  pivot: 'game' | 'algo'
}) {
  const { t } = useTranslation()
  const locale = useAppStore((s) => s.locale)
  const disabled = runIds.length === 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        {FORMATS.map((f) => {
          const takesPivot = PIVOT_AWARE.has(f.fmt)
          const aria = takesPivot
            ? t('analysis.export_aria_pivot', { fmt: t(f.labelKey), pivot: t(pivot === 'game' ? 'analysis.mode_game' : 'analysis.mode_algo') })
            : t('analysis.export_aria', { fmt: t(f.labelKey) })
          return disabled ? (
            <span key={f.fmt} aria-disabled="true" style={{ ...btnStyle, opacity: 0.45, cursor: 'not-allowed' }}>
              <span style={{ fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)' }}>{t(f.labelKey)}</span>
              <Desc text={t(f.descKey)} ext={f.ext} />
            </span>
          ) : (
            <a
              key={f.fmt}
              href={analysisExportUrl(f.fmt, runIds, pivot, locale)}
              download
              aria-label={aria}
              title={aria}
              style={btnStyle}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent)')}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-default)')}
            >
              <span style={{ fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)' }}>{t(f.labelKey)}</span>
              <Desc text={t(f.descKey)} ext={f.ext} />
            </a>
          )
        })}
      </div>

      <p style={{ margin: 0, fontSize: 'var(--fs-micro)', color: 'var(--text-muted)', lineHeight: 1.5 }}>
        {disabled ? t('analysis.export_empty') : t('analysis.export_note')}
      </p>
    </div>
  )
}
