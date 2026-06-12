import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { skillScaleFor } from '../content/skill'
import ParamInfo from './ParamInfo'
import type { EvolutionChild, MutationDist } from '../api/types'

// ── Shell ────────────────────────────────────────────────────────────────────

function PanelShell({ title, right, borderRight = true, children }: {
  title: string
  right?: React.ReactNode
  borderRight?: boolean
  children: React.ReactNode
}) {
  return (
    <div style={{
      flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
      background: 'var(--surface)',
      borderRight: borderRight ? '1px solid var(--border)' : undefined,
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '6px 12px', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        fontWeight: 600, fontSize: 12, color: 'var(--text-h)', flexShrink: 0, minHeight: 30,
      }}>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>{title}{right}</span>
      </div>
      {children}
    </div>
  )
}

function Empty({ text }: { text: string }) {
  return (
    <div style={{
      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: 'var(--text-muted)', fontSize: 12, textAlign: 'center', padding: 8,
    }}>
      {text}
    </div>
  )
}

// ── Leaderboard (Top-5 children, read-only) ───────────────────────────────────

const GRID = '34px 1fr 1fr 1fr 1fr'

function Leaderboard() {
  const { t } = useTranslation()
  const algo          = useAppStore((s) => s.algo)
  const lastEvolution = useAppStore((s) => s.lastEvolution)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)

  const solveMax = skillScaleFor(selectedEnvId).max
  const children = lastEvolution?.children ?? []

  let body: React.ReactNode
  if (algo !== 'neuroevolution') {
    body = <Empty text={t('leaderboard.placeholder')} />
  } else if (children.length === 0) {
    body = <Empty text={t('leaderboard.evo_hint')} />
  } else {
    body = (
      <div style={{ flex: 1, overflowY: 'auto', padding: '2px 8px 6px' }}>
        {/* Column header */}
        <div style={{
          display: 'grid', gridTemplateColumns: GRID, gap: 4,
          fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase',
          letterSpacing: '0.04em', padding: '2px 4px',
        }}>
          <span>{t('leaderboard.col_id')}</span>
          <span style={{ textAlign: 'right' }}>{t('leaderboard.col_total')}</span>
          <span style={{ textAlign: 'right' }}>{t('leaderboard.col_avg')}</span>
          <span style={{ textAlign: 'right' }}>{t('leaderboard.col_steps')}</span>
          <span style={{ textAlign: 'right' }}>{t('leaderboard.col_seed')}</span>
        </div>
        {children.map((c, i) => (
          <Row key={c.id} child={c} solveMax={solveMax} isBest={i === 0} />
        ))}
      </div>
    )
  }

  return <PanelShell title={t('leaderboard.title')}>{body}</PanelShell>
}

// The list is sorted best-first, so row 0 is this generation's champion — the genome the
// backend keeps by elitism and renders in the live preview. It gets a static ★ + highlight
// (read-only: the backend always breeds from the best, no manual parent picking).
function Row({ child, solveMax, isBest }: {
  child: EvolutionChild
  solveMax: number
  isBest: boolean
}) {
  const { t } = useTranslation()
  const frac = solveMax > 0 ? Math.max(0, Math.min(1, child.avg_reward / solveMax)) : 0
  const num: CSSProperties = { textAlign: 'right', fontFamily: 'monospace', fontSize: 11, color: 'var(--text)' }

  return (
    <div
      style={{
        position: 'relative',
        borderLeft: `2px solid ${isBest ? 'var(--accent)' : 'transparent'}`,
        background: isBest ? 'var(--accent-soft)' : 'transparent',
        borderRadius: 4, marginTop: 2,
      }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: GRID, gap: 4, alignItems: 'center', padding: '3px 4px' }}>
        <span
          title={isBest ? t('leaderboard.best_row') : undefined}
          style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-h)', display: 'inline-flex', alignItems: 'center', gap: 2 }}
        >
          {isBest && <span style={{ color: 'var(--accent)' }}>★</span>}#{child.id}
        </span>
        <span style={num}>{child.total_reward.toFixed(0)}</span>
        <span style={{ ...num, color: 'var(--ok)' }}>{child.avg_reward.toFixed(1)}</span>
        <span style={num}>{child.steps}</span>
        <span style={{ ...num, color: 'var(--text-muted)' }}>{child.seed}</span>
      </div>
      {/* Per-row fitness bar (avg reward toward the env's solved score) */}
      <div style={{ height: 2, background: 'var(--border)', borderRadius: 1, margin: '0 4px 1px' }}>
        <div style={{ width: `${frac * 100}%`, height: '100%', background: 'var(--ok)', borderRadius: 1 }} />
      </div>
    </div>
  )
}

// ── Evolution Stats ───────────────────────────────────────────────────────────

function StatCell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      <span style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>{label}</span>
      <span style={{ fontSize: 13, fontWeight: 600, color: color ?? 'var(--text-h)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  )
}

function MutationBars({ dist }: { dist: MutationDist }) {
  const { t } = useTranslation()
  const max = Math.max(1, ...dist.counts)
  return (
    <div>
      <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 2 }}>
        {t('evolution.mutation_dist')}
      </div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 1, height: 26 }}>
        {dist.counts.map((c, i) => (
          <div
            key={i}
            title={String(c)}
            style={{
              flex: 1, height: `${(c / max) * 100}%`, minHeight: c > 0 ? 1 : 0,
              background: 'var(--accent)', opacity: 0.8, borderRadius: '1px 1px 0 0',
            }}
          />
        ))}
      </div>
    </div>
  )
}

function EvolutionStats() {
  const { t } = useTranslation()
  const algo          = useAppStore((s) => s.algo)
  const lastEvolution = useAppStore((s) => s.lastEvolution)
  const population     = useAppStore((s) => s.evolutionParams.population_size)

  const isEvo = algo === 'neuroevolution'

  let body: React.ReactNode
  if (!isEvo) {
    body = <Empty text={t('evolution.placeholder')} />
  } else if (!lastEvolution) {
    body = <Empty text={t('evolution.evo_hint')} />
  } else {
    const e = lastEvolution
    body = (
      <div style={{ flex: 1, overflowY: 'auto', padding: '6px 12px', display: 'flex', flexDirection: 'column', gap: 7 }}>
        {/* Fixed 3-column grid keeps the five stats to two tidy rows so the mutation bars
            below always stay in view inside the short panel. */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '5px 8px' }}>
          <StatCell label={t('evolution.generation')} value={`${e.generation}/${e.total_generations}`} />
          <StatCell label={t('evolution.total_iters')} value={String(e.generation * population)} />
          <StatCell label={t('evolution.best')}  value={e.best_fitness.toFixed(1)}  color="var(--ok)" />
          <StatCell label={t('evolution.avg')}   value={e.avg_fitness.toFixed(1)} />
          <StatCell label={t('evolution.worst')} value={e.worst_fitness.toFixed(1)} color="var(--text-muted)" />
        </div>
        <MutationBars dist={e.mutation_dist} />
      </div>
    )
  }

  return (
    <PanelShell
      title={t('evolution.title')}
      right={<ParamInfo paramId="evolution_stats" label={t('evolution.title')} />}
    >
      {body}
    </PanelShell>
  )
}

// ── Save / Load (placeholder until Phase D) ───────────────────────────────────

function SaveLoad() {
  const { t } = useTranslation()
  return (
    <PanelShell title={t('saveload.title')} borderRight={false}>
      <Empty text={t('saveload.placeholder')} />
    </PanelShell>
  )
}

// ── BottomPanels ──────────────────────────────────────────────────────────────

export default function BottomPanels() {
  return (
    <div style={{
      height: 168, flexShrink: 0, display: 'flex',
      borderTop: '1px solid var(--border)',
    }}>
      <Leaderboard />
      <EvolutionStats />
      <SaveLoad />
    </div>
  )
}
