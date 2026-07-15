import { useTranslation } from 'react-i18next'
import type { CSSProperties } from 'react'
import { useAppStore } from '../store/useAppStore'
import { PANEL_DIM_BASE, panelDimClass } from './panelDim'
import { PLAY_SCORE_TOP_N } from '../api/types'
import type { PlayScoreEntry } from '../api/types'

// The persistent high-score slot (E2): one shared "High Scores" header over two boards — a
// named Human hall of fame and an AI one (keyed by model). Same footprint as the bottom row,
// sized so the top-N fits without scrolling.
export default function PlayLeaderboards({ standalone = false }: { standalone?: boolean }) {
  const { t } = useTranslation()
  const playScores = useAppStore((s) => s.playScores)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const envs = useAppStore((s) => s.envs)
  const human = playScores?.human ?? []
  const ai = playScores?.ai ?? []
  // Board games (G6a) report win/draw/loss, not a high score, so a score ladder doesn't fit — show a
  // "deferred" note instead of empty boards (it returns once self-play training lands in G6b).
  const isBoard = envs.find((e) => e.id === selectedEnvId)?.family === 'board'
  // No records yet (or board games, which don't keep a high-score ladder) → recede instead of
  // sitting blank and reading as broken (panelDim). Fades back in the moment a first score lands.
  const dimmed = isBoard || (human.length === 0 && ai.length === 0)

  return (
    <div className={panelDimClass(dimmed)} style={{
      flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
      background: 'var(--surface)',
      // Standalone (Simple mode, full-width): drop the right divider — it's not sitting left of another
      // panel there, so the border would draw a stray line down the middle of the centered board.
      borderRight: standalone ? undefined : '2px solid var(--border)',
      overflow: 'hidden',
      ...PANEL_DIM_BASE,
    }}>
      {/* Shared title — makes it clear both columns below are high-score boards */}
      <div style={{
        padding: '6px 12px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 12, color: 'var(--text-h)', flexShrink: 0, minHeight: 30,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
      }}>
        🏆 {t('playscore.title')}
      </div>
      {isBoard ? (
        <div style={{
          flex: 1, minHeight: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--text-muted)', fontSize: 11, textAlign: 'center', padding: 16, lineHeight: 1.5,
        }}>
          {t('playscore.board_deferred')}
        </div>
      ) : (
        <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
          <Board title={`🧑 ${t('playscore.human_title')}`} entries={human} divider />
          <Board title={`🤖 ${t('playscore.ai_title')}`} entries={ai} />
        </div>
      )}
    </div>
  )
}

function Board({ title, entries, divider = false }: {
  title: string
  entries: PlayScoreEntry[]
  divider?: boolean
}) {
  const { t } = useTranslation()
  return (
    <div style={{
      flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden',
      borderRight: divider ? '1px solid var(--border)' : undefined,
    }}>
      {/* Sub-label per column */}
      <div style={{
        padding: '3px 8px', fontSize: 10, fontWeight: 600, color: 'var(--text-muted)',
        textTransform: 'uppercase', letterSpacing: '0.04em', flexShrink: 0, textAlign: 'center',
        whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
      }}>
        {title}
      </div>
      {entries.length === 0 ? (
        <div style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: 'var(--text-muted)', fontSize: 11, textAlign: 'center', padding: 8,
        }}>
          {t('playscore.empty')}
        </div>
      ) : (
        <div style={{ flex: 1, overflowY: 'auto', padding: '0 4px 4px' }}>
          {/* Display count == backend TOP_N, so "made the board" matches exactly what's shown. */}
          {entries.slice(0, PLAY_SCORE_TOP_N).map((e, i) => <Row key={`${e.name}-${e.achieved_at}`} entry={e} rank={i + 1} />)}
        </div>
      )}
    </div>
  )
}

// Top-3 get a medal; the rest a plain rank number.
const MEDALS = ['🥇', '🥈', '🥉']

function Row({ entry, rank }: { entry: PlayScoreEntry; rank: number }) {
  const score: CSSProperties = {
    fontFamily: 'var(--font-mono)', fontSize: 11, fontWeight: 600, color: 'var(--ok)', flexShrink: 0,
  }
  return (
    <div
      title={`${entry.name} · ${entry.score.toFixed(1)}`}
      style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '2px 4px' }}
    >
      <span style={{ width: 16, fontSize: 11, textAlign: 'center', flexShrink: 0, color: 'var(--text-muted)' }}>
        {MEDALS[rank - 1] ?? rank}
      </span>
      <span style={{
        flex: 1, minWidth: 0, fontSize: 11, color: 'var(--text-h)',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>
        {entry.name}
      </span>
      <span style={score}>{Math.round(entry.score)}</span>
    </div>
  )
}
