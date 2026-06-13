import { useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { submitPlayScore } from '../api/client'
import { PLAY_SCORE_TOP_N } from '../api/types'

// Invisible coordinator (E2): watches the finished play result and places it on a board.
// - AI sessions auto-submit under the model's label (de-duped server-side).
// - Human sessions that crack the top-N pop a name-entry modal before submitting.
// Only a session we actually saw *start* (passed through 'playing') is acted on, so a reload
// that reconciles a previously-finished session from the backend can't re-prompt or re-submit.
// The "armed" flag also dedupes the result being set twice (play_result frame + terminal status).
export default function PlayScoreGate() {
  const playState  = useAppStore((s) => s.playState)
  const playResult = useAppStore((s) => s.playResult)

  const armedRef = useRef(false)
  const [pending, setPending] = useState<{ score: number; steps: number } | null>(null)

  useEffect(() => {
    if (playState === 'playing') {
      armedRef.current = true
      setPending(null) // a new game supersedes any unsubmitted prompt
      return
    }
    if (playState !== 'finished' || !playResult || !armedRef.current) return
    armedRef.current = false

    const { score, steps, mode } = playResult
    const st = useAppStore.getState()

    if (mode === 'ai') {
      // Never record an anonymous AI row: without a model id the board can't de-dup it, so a
      // missing checkpoint just isn't scored (belt-and-braces with the armed-ref guard above).
      const modelId = st.playCheckpointId
      if (!modelId) return
      void submitPlayScore(st.selectedEnvId ?? 'cartpole', {
        category: 'ai',
        name: st.playCheckpointLabel ?? 'AI',
        score, steps,
        model_id: modelId,
      }).then((r) => st.setPlayScores(r.scores)).catch(() => {})
      return
    }

    // Human: only prompt if the score would actually make the board.
    const board = st.playScores?.human ?? []
    const qualifies = board.length < PLAY_SCORE_TOP_N || score > Math.min(...board.map((e) => e.score))
    if (qualifies) setPending({ score, steps })
  }, [playState, playResult])

  if (!pending) return null
  return <NameModal score={pending.score} steps={pending.steps} onDone={() => setPending(null)} />
}

function NameModal({ score, steps, onDone }: { score: number; steps: number; onDone: () => void }) {
  const { t } = useTranslation()
  const storedName  = useAppStore((s) => s.playerName)
  const setPlayerName = useAppStore((s) => s.setPlayerName)
  const setPlayScores = useAppStore((s) => s.setPlayScores)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)

  const [name, setName] = useState(storedName)
  const [saving, setSaving] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    inputRef.current?.focus()
    inputRef.current?.select()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onDone() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onDone])

  async function save() {
    const clean = name.trim()
    if (!clean || saving) return
    setSaving(true)
    setPlayerName(clean)
    try {
      const r = await submitPlayScore(selectedEnvId ?? 'cartpole', {
        category: 'human', name: clean, score, steps,
      })
      setPlayScores(r.scores)
    } catch { /* keep UX flowing even if the write fails */ }
    onDone()
  }

  return (
    <div
      onClick={onDone}
      style={{
        position: 'fixed', inset: 0, zIndex: 1100, background: 'rgba(0,0,0,0.5)',
        display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20,
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={t('playscore.new_record')}
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%', maxWidth: 360,
          background: 'var(--surface-1)', color: 'var(--text-default)',
          border: '1px solid var(--border-default)', borderRadius: 'var(--radius-xl)',
          boxShadow: 'var(--shadow-popover)', padding: 'var(--space-5)',
          display: 'flex', flexDirection: 'column', gap: 'var(--space-3)',
        }}
      >
        <div style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-h)' }}>
          🏆 {t('playscore.new_record')}
        </div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          {t('playscore.you_scored', { score: Math.round(score) })}
        </div>
        <input
          ref={inputRef}
          value={name}
          maxLength={24}
          placeholder={t('playscore.your_name')}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') void save() }}
          style={{
            padding: '9px 11px', borderRadius: 'var(--radius-md)', fontSize: 13,
            background: 'var(--surface-inset)', color: 'var(--text-strong)',
            border: '1px solid var(--border-default)', outline: 'none',
          }}
        />
        <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button
            onClick={onDone}
            style={{
              padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 600,
              background: 'var(--surface-2)', color: 'var(--text-muted)',
              border: '1px solid var(--border)', cursor: 'pointer',
            }}
          >
            {t('playscore.skip')}
          </button>
          <button
            onClick={() => void save()}
            disabled={!name.trim() || saving}
            style={{
              padding: '6px 14px', borderRadius: 6, fontSize: 12, fontWeight: 700,
              background: 'var(--accent)', color: '#fff', border: 'none',
              cursor: name.trim() && !saving ? 'pointer' : 'not-allowed',
              opacity: name.trim() && !saving ? 1 : 0.5,
            }}
          >
            {t('playscore.save')}
          </button>
        </div>
      </div>
    </div>
  )
}
