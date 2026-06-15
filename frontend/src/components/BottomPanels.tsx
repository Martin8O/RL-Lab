import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { skillScaleFor } from '../content/skill'
import { qActionsFor } from '../content/qTableActions'
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
  const envs          = useAppStore((s) => s.envs)

  // The env's real solved score (CartPole 500, LunarLander 200) — never skillScaleFor's CartPole
  // fallback, which would mark LunarLander "solved" only at 500.
  const solveMax = envs.find((e) => e.id === selectedEnvId)?.solved_score ?? skillScaleFor(selectedEnvId).max
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
  const envs          = useAppStore((s) => s.envs)
  const population     = useAppStore((s) => s.evolutionParams.population_size)

  const isEvo = algo === 'neuroevolution'

  // Run-level "steps-to-solve" equivalent: the first generation whose best genome reached
  // the env's real solved score (200 for LunarLander, not skillScaleFor's CartPole 500).
  const solveMax = envs.find((e) => e.id === selectedEnvId)?.solved_score ?? skillScaleFor(selectedEnvId).max
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

// ── Q-table heatmap (G2b) ───────────────────────────────────────────────────────
// Tabular Q-learning's "watch the table fill in" view: a [states × actions] heatmap where each
// cell is colour-coded by its Q-value (blank = never learned, green = good, red = bad) and the
// greedy action per state is outlined. Shown ONLY for the q_learning algorithm — it takes over the
// Evolution Stats slot AND the bottom-right blank slot (the user's ask) so a big table (Taxi's 500
// states) has room.
//
// Rendered to a single <canvas>, NOT to per-cell DOM nodes: a 500×6 table is 3 000 cells streamed
// several times a second, and reconciling that many React elements froze the UI on a CPU laptop
// (which also blocked the Pause/Stop clicks). One canvas redraw is cheap, so the dashboard stays
// responsive and the whole table is always visible (cells auto-size to fit — no scrollbar).

// Diverging Q-value colour on a symmetric [-maxAbs, +maxAbs] scale: positive → green, negative →
// red, ~0 → null (unlearned cells stay blank so you see the table light up). Canvas fill strings.
function qCellColor(v: number, maxAbs: number): string | null {
  if (maxAbs <= 0) return null
  const t = Math.max(-1, Math.min(1, v / maxAbs))
  if (Math.abs(t) < 0.001) return null
  return t > 0 ? `rgba(63, 174, 79, ${t.toFixed(3)})` : `rgba(226, 69, 60, ${(-t).toFixed(3)})`
}

// Draw the whole table into the canvas, auto-sizing square cells so every state fits the box (no
// scroll). States flow top-to-bottom within a column-block, blocks wrap left-to-right; the block
// grid is centred. The greedy action per state gets a bright outline.
function drawQTable(
  canvas: HTMLCanvasElement, table: { n_states: number; n_actions: number; values: number[][] },
  actions: string[], w: number, h: number, gridColor: string, greedyColor: string,
): void {
  const { n_states: S, n_actions: A, values } = table
  const dpr = window.devicePixelRatio || 1
  canvas.width = Math.max(1, Math.round(w * dpr))
  canvas.height = Math.max(1, Math.round(h * dpr))
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, w, h)

  let maxAbs = 0
  const best = new Array<number>(S)
  const learned = new Array<boolean>(S)  // any non-zero cell ⇒ this state has been visited
  for (let s = 0; s < S; s++) {
    const row = values[s]
    let bi = 0
    let any = false
    for (let a = 0; a < A; a++) {
      const ab = Math.abs(row[a])
      if (ab > maxAbs) maxAbs = ab
      if (ab !== 0) any = true
      if (row[a] > row[bi]) bi = a
    }
    best[s] = bi
    learned[s] = any
  }

  const gap = 8
  const showGlyphs = S <= 64
  // Largest square cell for which all column-blocks fit the box (search big→small).
  let cell = 0, rowsPerCol = S, nCols = 1, headerH = 0
  for (let c = 20; c >= 2; c--) {
    const hh = showGlyphs && c >= 11 ? c : 0
    const rpc = Math.max(1, Math.floor((h - hh) / c))
    const nc = Math.ceil(S / rpc)
    const totalW = nc * A * c + (nc - 1) * gap
    if (totalW <= w) { cell = c; rowsPerCol = rpc; nCols = nc; headerH = hh; break }
  }
  if (cell === 0) { cell = 2; rowsPerCol = Math.max(1, Math.floor(h / cell)); nCols = Math.ceil(S / rowsPerCol) }

  const blockW = A * cell
  const totalW = nCols * blockW + (nCols - 1) * gap
  const offsetX = Math.max(0, (w - totalW) / 2)
  ctx.font = `${Math.min(cell - 1, 11)}px ui-monospace, monospace`
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'

  for (let s = 0; s < S; s++) {
    const col = Math.floor(s / rowsPerCol)
    const rip = s % rowsPerCol
    const bx = offsetX + col * (blockW + gap)
    const by = rip * cell + headerH
    if (headerH > 0 && rip === 0) {
      ctx.fillStyle = 'rgba(150,150,160,0.85)'
      for (let a = 0; a < A; a++) ctx.fillText(actions[a] ?? '', bx + a * cell + cell / 2, headerH / 2)
    }
    const row = values[s]
    for (let a = 0; a < A; a++) {
      const x = bx + a * cell
      const fill = qCellColor(row[a], maxAbs)
      if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, by, cell - 1, cell - 1) }
      // Outline the greedy action for any *visited* state (learned[s]) — even if that action's own
      // Q stayed 0 (optimistic-init pick among negatives), so the heatmap shows the real policy.
      const isBest = a === best[s] && learned[s]
      ctx.strokeStyle = isBest ? greedyColor : gridColor
      ctx.lineWidth = isBest ? 1.4 : 0.5
      ctx.strokeRect(x + 0.5, by + 0.5, cell - 2, cell - 2)
    }
  }
}

function QTablePanel() {
  const { t } = useTranslation()
  const qtable = useAppStore((s) => s.lastQTable)
  const ql     = useAppStore((s) => s.lastQLearning)
  const envId  = useAppStore((s) => s.selectedEnvId)

  const wrapRef   = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [size, setSize] = useState({ w: 0, h: 0 })

  const hasTable = qtable !== null
  // Attach the observer once the canvas wrapper exists (it only mounts once a table arrives).
  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight })
    measure()
    const obs = new ResizeObserver(measure)
    obs.observe(el)
    return () => obs.disconnect()
  }, [hasTable])

  // Redraw whenever the table or the box size changes. Reads two theme colours so it tracks dark/light.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !qtable || size.w < 4 || size.h < 4) return
    const css = getComputedStyle(document.documentElement)
    const grid = css.getPropertyValue('--border-default').trim() || 'rgba(128,128,128,0.3)'
    const greedy = css.getPropertyValue('--text-strong').trim() || '#fff'
    drawQTable(canvas, qtable.table, qActionsFor(envId, qtable.table.n_actions), size.w, size.h, grid, greedy)
  }, [qtable, size, envId])

  // Derived stats for the strip.
  let fillPct = 0
  if (qtable) {
    let filled = 0
    for (const r of qtable.table.values) for (const v of r) if (v !== 0) filled++
    const tot = qtable.table.n_states * qtable.table.n_actions
    fillPct = tot > 0 ? (filled / tot) * 100 : 0
  }

  return (
    <PanelShell
      title={t('qtable.title')}
      right={<ParamInfo paramId="qtable" label={t('qtable.title')} />}
      borderRight={false}
      center
    >
      {!qtable ? (
        <Empty text={t('qtable.hint')} />
      ) : (
        <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column', padding: '6px 12px', gap: 6 }}>
          {/* Stats + legends — centred above the table. */}
          <div style={{ display: 'flex', gap: 16, flexShrink: 0, justifyContent: 'center', alignItems: 'center', flexWrap: 'wrap' }}>
            <StatCell label={t('qtable.episode')} value={ql ? `${ql.episode}/${ql.total_episodes}` : `—/${qtable.total_episodes}`} />
            <StatCell label={t('qtable.epsilon')} value={ql ? ql.epsilon.toFixed(2) : '—'} />
            <StatCell label={t('qtable.filled')} value={`${Math.round(fillPct)}%`} color="var(--accent-h)" />
            <StatCell label={t('qtable.score')} value={ql?.ep_rew_mean != null ? ql.ep_rew_mean.toFixed(2) : '—'} color="var(--ok)" />
            {/* Value colour legend */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 9, color: 'var(--text-muted)' }}>
              <span>{t('qtable.legend_low')}</span>
              <span style={{ width: 54, height: 8, borderRadius: 2, background: 'linear-gradient(to right, rgba(226,69,60,0.9), var(--surface-2), rgba(63,174,79,0.9))' }} />
              <span>{t('qtable.legend_high')}</span>
            </div>
            {/* Greedy-outline legend (what the white box means) */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 9, color: 'var(--text-muted)' }}>
              <span style={{ width: 10, height: 10, border: '1.4px solid var(--text-strong)', borderRadius: 1, display: 'inline-block' }} />
              {t('qtable.legend_best')}
            </div>
            {/* Action-column key — the heatmap's columns left→right (so Taxi's tiny cells are legible). */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 9, color: 'var(--text-muted)' }}>
              <span>{t('qtable.actions')}:</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-default)', letterSpacing: '0.12em' }}>
                {qActionsFor(envId, qtable.table.n_actions).join(' ')}
              </span>
            </div>
          </div>
          {/* Canvas heatmap — fills the box; the whole table is always visible (no scroll). */}
          <div ref={wrapRef} style={{ flex: 1, minHeight: 0, position: 'relative' }}>
            <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
          </div>
        </div>
      )}
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
  const isQ = useAppStore((s) => s.algo === 'q_learning')

  // Q-learning mode: the Q-table heatmap takes over the Evolution Stats slot AND the bottom-right
  // blank slot (the user's ask), so a big table has room. The leaderboard slot keeps its width.
  if (isQ) {
    return (
      <div style={{
        height: 200, flexShrink: 0, display: 'flex',
        borderTop: '2px solid var(--border-default)',
      }}>
        {/* Leaderboard slot keeps the same ~22% width (2/5 of the old 55% left group). */}
        <div style={{ flex: '0 0 22%', minWidth: 0, display: 'flex', overflow: 'hidden', borderRight: '2px solid var(--border-default)' }}>
          <PlayLeaderboards />
        </div>
        {/* Q-table spans the old Evolution-Stats + blank area (the remaining 78%). */}
        <div style={{ flex: 1, minWidth: 0, display: 'flex', overflow: 'hidden' }}>
          <QTablePanel />
        </div>
      </div>
    )
  }

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
