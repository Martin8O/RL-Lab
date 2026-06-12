import { useTranslation } from 'react-i18next'
import type { CSSProperties } from 'react'
import { useAppStore } from '../store/useAppStore'
import type { PlayScoreEntry } from '../api/types'

// The persistent high-score slot (E2): one shared "High Scores" header over two boards — a
// named Human hall of fame and an AI one (keyed by model). Same footprint as the bottom row,
// sized so the top-7 fits without scrolling.
export default function PlayLeaderboards() {
  const { t } = useTranslation()
  const playScores = useAppStore((s) => s.playScores)
  const human = playScores?.human ?? []
  const ai = playScores?.ai ?? []

  return (
    <div style={{
      flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column',
      background: 'var(--surface)', borderRight: '1px solid var(--border)', overflow: 'hidden',
    }}>
      {/* Shared title — makes it clear both columns below are high-score boards */}
      <div style={{
        padding: '6px 12px', borderBottom: '1px solid var(--border)',
        fontWeight: 600, fontSize: 12, color: 'var(--text-h)', flexShrink: 0, minHeight: 30,
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
      }}>
        🏆 {t('playscore.title')}
      </div>
      <div style={{ flex: 1, minHeight: 0, display: 'flex' }}>
        <Board title={`🧑 ${t('playscore.human_title')}`} entries={human} divider />
        <Board title={`🤖 ${t('playscore.ai_title')}`} entries={ai} />
      </div>
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
          {entries.map((e, i) => <Row key={`${e.name}-${e.achieved_at}`} entry={e} rank={i + 1} />)}
        </div>
      )}
    </div>
  )
}

// Top-3 get a medal; the rest a plain rank number.
const MEDALS = ['🥇', '🥈', '🥉']

function Row({ entry, rank }: { entry: PlayScoreEntry; rank: number }) {
  const score: CSSProperties = {
    fontFamily: 'monospace', fontSize: 11, fontWeight: 600, color: 'var(--ok)', flexShrink: 0,
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
