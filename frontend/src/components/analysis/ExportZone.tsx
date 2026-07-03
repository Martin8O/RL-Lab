// The DataLab export zone (Zone 5, X6b) — download the current run selection as a citable dataset. Each
// button is a plain <a download> to an X5 export endpoint (server-side over the full on-disk history, so
// the export is full-resolution, not the capped live store); the CSV + XLSX honour the active pivot
// (per-game raw reward / per-algorithm normalized skill-%). The Wave-3 formats (TensorBoard event files,
// a vector figure) are shown disabled with a "coming" tag rather than hidden, so the surface advertises
// what's on the way (the DoD asks for them to be visibly marked, not missing). Theme tokens, aria-labels.

import { useTranslation } from 'react-i18next'
import { analysisExportUrl } from '../../api/client'

interface Fmt {
  fmt: 'csv' | 'xlsx' | 'repro' | 'tex'
  labelKey: string
  descKey: string
}

const FORMATS: Fmt[] = [
  { fmt: 'csv', labelKey: 'analysis.export_csv', descKey: 'analysis.export_csv_desc' },
  { fmt: 'xlsx', labelKey: 'analysis.export_xlsx', descKey: 'analysis.export_xlsx_desc' },
  { fmt: 'repro', labelKey: 'analysis.export_repro', descKey: 'analysis.export_repro_desc' },
  { fmt: 'tex', labelKey: 'analysis.export_latex', descKey: 'analysis.export_latex_desc' },
]

const WAVE3 = ['analysis.export_tensorboard', 'analysis.export_figure']

const btnStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 1,
  padding: '6px 10px', textDecoration: 'none', cursor: 'pointer',
  border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)',
  background: 'var(--surface-2)', color: 'var(--text-strong)', textAlign: 'left',
}

export default function ExportZone({
  runIds,
  pivot,
}: {
  runIds: string[]
  pivot: 'game' | 'algo'
}) {
  const { t } = useTranslation()
  const disabled = runIds.length === 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
        {FORMATS.map((f) => {
          const takesPivot = f.fmt === 'csv' || f.fmt === 'xlsx'
          const aria = takesPivot
            ? t('analysis.export_aria_pivot', { fmt: t(f.labelKey), pivot: t(pivot === 'game' ? 'analysis.mode_game' : 'analysis.mode_algo') })
            : t('analysis.export_aria', { fmt: t(f.labelKey) })
          return disabled ? (
            <span key={f.fmt} aria-disabled="true" style={{ ...btnStyle, opacity: 0.45, cursor: 'not-allowed' }}>
              <span style={{ fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)' }}>{t(f.labelKey)}</span>
              <span style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-faint)' }}>{t(f.descKey)}</span>
            </span>
          ) : (
            <a
              key={f.fmt}
              href={analysisExportUrl(f.fmt, runIds, pivot)}
              download
              aria-label={aria}
              title={aria}
              style={btnStyle}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = 'var(--accent)')}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = 'var(--border-default)')}
            >
              <span style={{ fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)' }}>{t(f.labelKey)}</span>
              <span style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-faint)' }}>{t(f.descKey)}</span>
            </a>
          )
        })}
      </div>

      {/* Wave-3 formats — advertised, not yet live. */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
        {WAVE3.map((k) => (
          <span key={k} style={{
            display: 'inline-flex', alignItems: 'center', gap: 5, padding: '3px 8px',
            border: '1px dashed var(--border-default)', borderRadius: 'var(--radius-pill)',
            fontSize: 'var(--fs-micro)', color: 'var(--text-faint)',
          }}>
            {t(k)}
            <span style={{ fontSize: 'var(--fs-micro)', color: 'var(--text-faint)', opacity: 0.8 }}>· {t('analysis.export_wave3')}</span>
          </span>
        ))}
      </div>

      <p style={{ margin: 0, fontSize: 'var(--fs-micro)', color: 'var(--text-faint)', lineHeight: 1.5 }}>
        {disabled ? t('analysis.export_empty') : t('analysis.export_note')}
      </p>
    </div>
  )
}
