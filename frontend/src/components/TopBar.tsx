import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { BackendStatus } from '../store/useAppStore'
import ParamInfo from './ParamInfo'

const DOT_COLOR: Record<BackendStatus, string> = {
  online:     'var(--ok)',
  connecting: 'var(--warn)',
  offline:    'var(--err)',
}

function StatusDot() {
  const { t } = useTranslation()
  const status = useAppStore((s) => s.backendStatus)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <span
        style={{
          width: 8, height: 8, borderRadius: '50%',
          background: DOT_COLOR[status],
          boxShadow: `0 0 6px ${DOT_COLOR[status]}`,
          display: 'inline-block',
          animation: status === 'connecting' ? 'pulse 1.4s ease-in-out infinite' : undefined,
        }}
      />
      <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{t(`status.${status}`)}</span>
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
        display: 'flex', alignItems: 'center', gap: 4,
        background: 'var(--surface-2)', border: '1px solid var(--border)',
        borderRadius: 6, padding: '2px 8px', fontSize: 12,
        cursor: title ? 'help' : undefined,
      }}
    >
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span style={{ color: accent ? 'var(--ok)' : 'var(--text-h)', fontVariantNumeric: 'tabular-nums', fontWeight: accent ? 600 : 400 }}>
        {value}
      </span>
      {infoId && <ParamInfo paramId={infoId} label={infoLabel ?? label} />}
    </div>
  )
}

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
      height: 48, flexShrink: 0,
      display: 'flex', alignItems: 'center', gap: 12, padding: '0 16px',
      background: 'var(--surface)', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontWeight: 600, fontSize: 15, color: 'var(--text-h)', letterSpacing: '-0.2px', marginRight: 4 }}>
        {t('app.title')}
      </span>

      <StatusDot />

      <div style={{ flex: 1 }} />

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

      <div style={{ width: 1, height: 20, background: 'var(--border)', margin: '0 4px' }} />

      <button
        onClick={() => setLocale(locale === 'en' ? 'cz' : 'en')}
        style={{
          background: 'var(--surface-2)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '3px 10px', cursor: 'pointer',
          color: 'var(--text-h)', fontSize: 12, fontWeight: 500,
        }}
        aria-label="Toggle language"
      >
        {locale === 'en' ? 'CZ' : 'EN'}
      </button>

      <button
        onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
        style={{
          background: 'var(--surface-2)', border: '1px solid var(--border)',
          borderRadius: 6, padding: '3px 8px', cursor: 'pointer',
          color: 'var(--text-h)', fontSize: 14, lineHeight: 1,
        }}
        aria-label="Toggle theme"
      >
        {theme === 'dark' ? '☀' : '🌙'}
      </button>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.35; }
        }
      `}</style>
    </header>
  )
}
