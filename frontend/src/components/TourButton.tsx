// #1 — a tour launcher. Re-opens a guided tour on demand (the tours also auto-open once). A quiet
// text button ("Tutorial") matching the About / lang-theme chrome. `flow` picks the dashboard tour
// (TopBar) or the Data Lab tour (Data Lab header); the accessible name stays flow-specific.

import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { TourFlow } from '../content/tourSteps'

export default function TourButton({ flow = 'dashboard' }: { flow?: TourFlow }) {
  const { t } = useTranslation()
  const startTour = useAppStore((s) => s.startTour)
  const aria = t(flow === 'datalab' ? 'tour.datalab_launch_aria' : 'tour.launch_aria')
  return (
    <button
      type="button"
      onClick={() => startTour(flow)}
      aria-label={aria}
      title={aria}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        height: 34, padding: '0 10px',
        background: 'transparent', border: '1px solid transparent',
        borderRadius: 'var(--radius-md)', color: 'var(--text-muted)', cursor: 'pointer',
        fontFamily: 'var(--font-sans)', fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)',
        whiteSpace: 'nowrap', transition: 'var(--t-colors)',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--surface-2)'; e.currentTarget.style.color = 'var(--text-default)' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)' }}
    >
      {t('tour.launch_label')}
    </button>
  )
}
