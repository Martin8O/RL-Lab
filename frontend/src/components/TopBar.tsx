import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import type { BackendStatus } from '../store/useAppStore'
import ParamInfo from './ParamInfo'
import ModeSwitch from './ModeSwitch'
import LangThemeToggle from './LangThemeToggle'
import AboutButton from './AboutButton'

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
            // Online = a slow sonar ring (the instrument is live); connecting keeps the pulse.
            ...(status === 'online' ? { ['--ring-c' as string]: 'rgba(54, 211, 153, 0.45)' } : null),
            animation: status === 'online'
              ? 'lab-sonar 2.8s var(--ease-out) infinite'
              : status === 'connecting' ? 'lab-pulse 1.4s var(--ease-in-out) infinite' : undefined,
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
        boxShadow: 'var(--ring-inset)',
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

export default function TopBar() {
  const { t }  = useTranslation()
  const locale  = useAppStore((s) => s.locale)

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
      background: 'var(--header-grad)', borderBottom: '2px solid var(--border-default)',
    }}>
      {/* View switcher: RL Lab (dashboard) ⇆ DataLab (analysis surface). */}
      <ModeSwitch />

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

      <AboutButton />
      <LangThemeToggle />
    </header>
  )
}
