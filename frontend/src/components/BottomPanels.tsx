import { type CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { skillScaleFor } from '../content/skill'
import { useShowEvoLeaderboard } from '../hooks/useShowEvoLeaderboard'
import ParamInfo from './ParamInfo'
import PlayLeaderboards from './PlayLeaderboards'
import type { EvolutionChild, MutationDist } from '../api/types'

// ── Shell ────────────────────────────────────────────────────────────────────

function PanelShell({ title, right, borderRight = true, center = false, children }: {
  title: string
  right?: React.ReactNode
  borderRight?: boolean
  center?: boolean
  children: React.ReactNode
}) {
  return (
    <div style={{
      flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
      background: 'var(--surface)',
      borderRight: borderRight ? '2px solid var(--border)' : undefined,
      overflow: 'hidden',
    }}>
      <div style={{
        padding: '6px 12px', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: center ? 'center' : 'space-between',
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

  return <PanelShell title={t('leaderboard.title')} center>{body}</PanelShell>
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
  const num: CSSProperties = { textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text)' }

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
    <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0, gap: 2 }}>
      <span style={{ fontSize: 'var(--fs-meta)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: 'var(--ls-eyebrow)', fontWeight: 'var(--fw-semibold)' }}>{label}</span>
      <span style={{ fontFamily: 'var(--font-mono)', fontFeatureSettings: 'var(--ff-tabular)', fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-medium)', letterSpacing: 'var(--ls-tight)', color: color ?? 'var(--text-strong)' }}>{value}</span>
    </div>
  )
}

function MutationBars({ dist }: { dist: MutationDist }) {
  const { t } = useTranslation()
  const max = Math.max(1, ...dist.counts)
  return (
    <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 3 }}>
        {t('evolution.mutation_dist')}
      </div>
      <div style={{ flex: 1, minHeight: 24, display: 'flex', alignItems: 'flex-end', gap: 1 }}>
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
  const evolutionHistory = useAppStore((s) => s.evolutionHistory)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const population     = useAppStore((s) => s.evolutionParams.population_size)

  const isEvo = algo === 'neuroevolution'

  // Run-level "steps-to-solve" equivalent: the first generation whose best genome reached
  // the solved score. Shown here (a run-level panel) rather than as a per-child leaderboard
  // column, where it has no per-genome meaning.
  const solveMax = skillScaleFor(selectedEnvId).max
  let solvedGen: number | null = null
  for (const e of evolutionHistory) {
    if (e.best_fitness >= solveMax) { solvedGen = e.generation; break }
  }

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
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, max-content)', justifyContent: 'space-between', gap: '6px 8px' }}>
          <StatCell label={t('evolution.generation')} value={`${e.generation}/${e.total_generations}`} />
          <StatCell label={t('evolution.total_iters')} value={String(e.generation * population)} />
          <StatCell label={t('evolution.best')}  value={e.best_fitness.toFixed(1)}  color="var(--ok)" />
          <StatCell label={t('evolution.avg')}   value={e.avg_fitness.toFixed(1)} />
          <StatCell label={t('evolution.worst')} value={e.worst_fitness.toFixed(1)} color="var(--text-muted)" />
          <StatCell
            label={t('evolution.solved')}
            value={solvedGen != null ? `gen ${solvedGen}` : '—'}
            color={solvedGen != null ? 'var(--accent-h)' : undefined}
          />
        </div>
        <MutationBars dist={e.mutation_dist} />
      </div>
    )
  }

  return (
    <PanelShell
      title={t('evolution.title')}
      right={<ParamInfo paramId="evolution_stats" label={t('evolution.title')} />}
      center
    >
      {body}
    </PanelShell>
  )
}

// ── Blank panel ───────────────────────────────────────────────────────────────
// Reserved, intentionally empty space (growth potential) — the old Save / Load panel (moved to the
// sidebar) and the slot freed by narrowing the high-score board, both kept blank for future games.

function BlankPanel({ flex = 1, borderRight = true }: { flex?: number; borderRight?: boolean }) {
  return (
    <div style={{
      flex, minWidth: 0, background: 'var(--surface)',
      borderRight: borderRight ? '2px solid var(--border)' : undefined,
    }} />
  )
}

// ── BottomPanels ──────────────────────────────────────────────────────────────

export default function BottomPanels() {
  const showEvoLeaderboard = useShowEvoLeaderboard()

  return (
    <div style={{
      height: 200, flexShrink: 0, display: 'flex',
      borderTop: '2px solid var(--border-default)',
    }}>
      {/* Left group spans the cart-window width (55%) so its right edge lands exactly on the
          cart/chart divider: high scores (narrowed ~⅓) + evolution stats, side by side. */}
      <div style={{ flex: '0 0 55%', minWidth: 0, display: 'flex', overflow: 'hidden' }}>
        <div style={{ flex: 2, minWidth: 0, display: 'flex', overflow: 'hidden' }}>
          {showEvoLeaderboard ? <Leaderboard /> : <PlayLeaderboards />}
        </div>
        <div style={{ flex: 3, minWidth: 0, display: 'flex', overflow: 'hidden' }}>
          <EvolutionStats />
        </div>
      </div>
      {/* The single empty panel — bottom-right, the width of the chart window (45%) — growth potential. */}
      <BlankPanel flex={1} borderRight={false} />
    </div>
  )
}
