import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { BackendStatus } from '../store/useAppStore'
import ParamInfo from './ParamInfo'

const DOT_COLOR: Record<BackendStatus, string> = {
  online:     'var(--success)',
  connecting: 'var(--warning)',
  offline:    'var(--danger)',
}

function StatusDot() {
  const { t } = useTranslation()
  const status = useAppStore((s) => s.backendStatus)
  const color = DOT_COLOR[status]
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8 }}>
      <span style={{ position: 'relative', width: 8, height: 8, display: 'inline-flex' }}>
        <span
          style={{
            position: 'relative', width: 8, height: 8, borderRadius: '50%',
            background: color,
            boxShadow: status === 'online' ? `0 0 0 3px var(--success-surface)` : 'none',
            animation: status === 'connecting' ? 'lab-pulse 1.4s var(--ease-in-out) infinite' : undefined,
          }}
        />
      </span>
      <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-label)' }}>{t(`status.${status}`)}</span>
    </div>
  )
}

function Chip({ label, value, title, accent, infoId, infoLabel }: {
  label: string; value: string; title?: string; accent?: boolean; infoId?: string; infoLabel?: string
}) {
  return (
    <div
      title={title}
      style={{
        display: 'inline-flex', alignItems: 'center', gap: 7,
        height: 28, padding: '0 11px',
        background: 'var(--surface-2)', border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-pill)', fontSize: 'var(--fs-label)',
        whiteSpace: 'nowrap',
        cursor: title ? 'help' : undefined,
      }}
    >
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{
        fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)',
        fontWeight: 'var(--fw-medium)', letterSpacing: 'var(--ls-tight)',
        color: accent ? 'var(--success)' : 'var(--text-strong)',
      }}>
        {value}
      </span>
      {infoId && <ParamInfo paramId={infoId} label={infoLabel ?? label} />}
    </div>
  )
}

function IconBtn({ onClick, label, children, text }: {
  onClick: () => void; label: string; children?: React.ReactNode; text?: string
}) {
  return (
    <button
      onClick={onClick}
      aria-label={label}
      title={label}
      style={{
        display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
        height: 34, minWidth: 34, padding: text ? '0 11px' : 0,
        background: 'transparent', border: '1px solid transparent',
        borderRadius: 'var(--radius-md)', color: 'var(--text-muted)',
        fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)',
        letterSpacing: 'var(--ls-wide)', cursor: 'pointer',
        transition: 'var(--t-colors)',
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--surface-2)'; e.currentTarget.style.color = 'var(--text-default)' }}
      onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-muted)' }}
    >
      {text ?? children}
    </button>
  )
}

const SunIcon = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
    <circle cx="12" cy="12" r="4.2" stroke="currentColor" strokeWidth="2" />
    <path d="M12 2.5v2.2M12 19.3v2.2M21.5 12h-2.2M4.7 12H2.5M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6M18.4 18.4l-1.6-1.6M7.2 7.2L5.6 5.6"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
  </svg>
)
const MoonIcon = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
  </svg>
)

export default function TopBar() {
  const { t }  = useTranslation()
  const locale  = useAppStore((s) => s.locale)
  const theme   = useAppStore((s) => s.theme)
  const setLocale = useAppStore((s) => s.setLocale)
  const setTheme  = useAppStore((s) => s.setTheme)

  const algo            = useAppStore((s) => s.algo)
  const metricsHistory  = useAppStore((s) => s.metricsHistory)
  const lastEvolution   = useAppStore((s) => s.lastEvolution)
  const population       = useAppStore((s) => s.evolutionParams.population_size)
  const bestReward      = useAppStore((s) => s.bestReward)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const highScores      = useAppStore((s) => s.highScores)
  const envs            = useAppStore((s) => s.envs)

  const isEvo = algo === 'neuroevolution'

  // Gen / Iter / Pop are algorithm-specific: evolution counts generations + genome evals,
  // PPO counts rollout iterations.
  const genValue  = isEvo && lastEvolution ? `${lastEvolution.generation}/${lastEvolution.total_generations}` : '—'
  const lastIter  = metricsHistory.at(-1)?.iteration
  const iterValue = isEvo
    ? (lastEvolution ? String(lastEvolution.generation * population) : '—')
    : (lastIter !== undefined ? String(lastIter) : '—')
  const popValue  = isEvo ? String(population) : '—'

  // The Best chip shows the *all-time* best for the selected env (persisted, survives
  // restarts) — visibly distinct (★ + green) from the live session high, which is named in
  // the tooltip alongside it.
  const allTime = selectedEnvId ? highScores[selectedEnvId]?.score : undefined
  const envName = envs.find((e) => e.id === selectedEnvId)?.display_name[locale] ?? selectedEnvId ?? ''
  const bestTitle =
    `${t('topbar.best_alltime', { env: envName })}` +
    (bestReward !== null ? ` · ${t('topbar.best_session')}: ${bestReward.toFixed(1)}` : '')

  return (
    <header style={{
      height: 'var(--topbar-h)', flexShrink: 0,
      display: 'flex', alignItems: 'center', gap: 'var(--space-4)', padding: '0 var(--space-5)',
      background: 'var(--surface-1)', borderBottom: '2px solid var(--border-default)',
    }}>
      {/* Brand */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 11 }}>
        <div style={{
          width: 30, height: 30, borderRadius: 'var(--radius-md)', background: 'var(--accent)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: 'var(--shadow-sm)', flexShrink: 0,
        }}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden>
            <path d="M4 18 L9 11 L13 14 L20 5" stroke="#fff" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            <circle cx="20" cy="5" r="2.3" fill="#fff" />
          </svg>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
          <span style={{ fontSize: 15, fontWeight: 'var(--fw-semibold)', color: 'var(--text-strong)', letterSpacing: 'var(--ls-tight)' }}>
            {t('app.title')}
          </span>
          <span style={{
            fontSize: 'var(--fs-micro)', fontWeight: 'var(--fw-semibold)',
            letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase', color: 'var(--text-faint)',
          }}>
            {t('app.subtitle')}
          </span>
        </div>
      </div>

      <div style={{ width: 1, height: 26, background: 'var(--border-default)' }} />

      <StatusDot />

      <div style={{ flex: 1 }} />

      <div style={{ display: 'flex', gap: 'var(--space-2)' }}>
        <Chip label={t('topbar.chips.gen')}  value={genValue}  infoId="topbar_gen" />
        <Chip label={t('topbar.chips.iter')} value={iterValue} infoId="topbar_iter" />
        <Chip
          label={`★ ${t('topbar.chips.best')}`}
          value={allTime !== undefined ? allTime.toFixed(1) : '—'}
          title={bestTitle}
          accent={allTime !== undefined}
          infoId="topbar_best"
          infoLabel={t('topbar.chips.best')}
        />
        <Chip label={t('topbar.chips.pop')}  value={popValue} infoId="topbar_pop" />
      </div>

      <div style={{ width: 1, height: 26, background: 'var(--border-default)' }} />

      <IconBtn onClick={() => setLocale(locale === 'en' ? 'cz' : 'en')} label={t('topbar.toggle_language')} text={locale === 'en' ? 'CZ' : 'EN'} />
      <IconBtn onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} label={t('topbar.toggle_theme')}>
        {theme === 'dark' ? MoonIcon : SunIcon}
      </IconBtn>
    </header>
  )
}
