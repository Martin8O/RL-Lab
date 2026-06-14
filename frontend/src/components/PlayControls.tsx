import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { fetchCheckpoints, startPlay, stopPlay } from '../api/client'
import type { CheckpointMeta } from '../api/types'
import PlayInstructions from './PlayInstructions'

const PLAY_SPEEDS = [0.1, 0.15, 0.25, 0.5, 1, 2, 4]

// Compact label for the checkpoint picker: drop the leading "env · " (the env is already chosen)
// and cap the length so a long name doesn't run into the <select> arrow on the right.
function optionLabel(label: string): string {
  const i = label.indexOf('·')
  const s = (i >= 0 ? label.slice(i + 1) : label).trim()
  return s.length > 20 ? `${s.slice(0, 19)}…` : s
}

// Play-vs-AI controls (E2): start/stop one interactive episode, pick who plays (human at the
// keyboard ↔ AI watch from a checkpoint) and the pacing, and open the how-to-play guide.
// The canvas + keyboard live in EnvPreview; this row owns the lifecycle + config.
export default function PlayControls() {
  const { t } = useTranslation()

  const selectedEnvId  = useAppStore((s) => s.selectedEnvId)
  const envs           = useAppStore((s) => s.envs)
  const seed           = useAppStore((s) => s.seed)
  const backendStatus  = useAppStore((s) => s.backendStatus)
  const trainState     = useAppStore((s) => s.trainState)
  const playState      = useAppStore((s) => s.playState)
  const playSpeed      = useAppStore((s) => s.playSpeed)
  const playCheckpointId = useAppStore((s) => s.playCheckpointId)
  const setPlayMode      = useAppStore((s) => s.setPlayMode)
  const setPlaySpeed     = useAppStore((s) => s.setPlaySpeed)
  const setPlayCheckpointId = useAppStore((s) => s.setPlayCheckpointId)
  const setPlayCheckpointLabel = useAppStore((s) => s.setPlayCheckpointLabel)
  const applyPlayStatus  = useAppStore((s) => s.applyPlayStatus)

  const [checkpoints, setCheckpoints] = useState<CheckpointMeta[]>([])
  const [error, setError] = useState<string | null>(null)

  const env          = envs.find((e) => e.id === selectedEnvId)
  const humanPlayable = env?.human_playable ?? false
  const playing      = playState === 'playing'
  const trainLive    = trainState === 'running' || trainState === 'paused' || trainState === 'stopping'
  // Checkpoints that can actually be played here (same env; any algo works via the AI policy).
  const envCheckpoints = checkpoints.filter((c) => c.env_id === selectedEnvId)

  // Load checkpoints when the backend comes online and whenever a play session ends (a new
  // checkpoint may have been saved meanwhile). Cheap, read-only.
  useEffect(() => {
    if (backendStatus !== 'online') return
    void fetchCheckpoints().then(setCheckpoints).catch(() => {})
  }, [backendStatus, playState])

  // Keep the AI checkpoint selection valid: default to the newest matching checkpoint, and drop a
  // stale pick if the env changed out from under it (the "AI plays" button needs a valid model).
  useEffect(() => {
    const stillValid = playCheckpointId && envCheckpoints.some((c) => c.id === playCheckpointId)
    if (!stillValid) setPlayCheckpointId(envCheckpoints[0]?.id ?? null)
  }, [selectedEnvId, checkpoints]) // eslint-disable-line react-hooks/exhaustive-deps

  const ready     = backendStatus === 'online' && !trainLive
  const canHuman  = ready && humanPlayable
  const aiReady   = !!playCheckpointId && envCheckpoints.length > 0
  const canAi     = ready && aiReady

  // Two explicit actions instead of a Play button + a who-plays dropdown: you Play, or the AI
  // plays by itself. The store's playMode (skill-meter label + keyboard wiring) is set per action.
  async function handlePlay(mode: 'human' | 'ai') {
    setError(null)
    setPlayMode(mode)
    // Remember which model the AI plays so its leaderboard identity is known on finish
    // (the checkpoint label already encodes algo + size, so use it verbatim).
    const ckpt = mode === 'ai'
      ? envCheckpoints.find((c) => c.id === playCheckpointId) ?? null
      : null
    setPlayCheckpointLabel(ckpt ? ckpt.label : null)
    try {
      const status = await startPlay({
        env_id: selectedEnvId ?? 'cartpole',
        mode,
        checkpoint_id: mode === 'ai' ? playCheckpointId : null,
        seed,
        speed: playSpeed,
      })
      applyPlayStatus(status)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleStop() {
    try {
      applyPlayStatus(await stopPlay())
    } catch { /* status will reconcile via WS */ }
  }

  return (
    <div style={{
      flexShrink: 0, borderTop: '1px solid var(--border-default)',
      background: 'var(--surface-1)', padding: '0 var(--space-3)', minHeight: 52,
      display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap',
    }}>
      {playing ? (
        /* Stop the active session */
        <button
          onClick={handleStop}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            height: 'var(--control-sm)', padding: '0 12px', borderRadius: 'var(--radius-md)',
            cursor: 'pointer', fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)',
            background: 'var(--danger-surface)', color: 'var(--danger)',
            border: '1px solid transparent', transition: 'var(--t-colors)',
          }}
        >
          <span aria-hidden>■</span> {t('play.stop')}
        </button>
      ) : (
        <>
          {/* You play */}
          <button
            onClick={() => handlePlay('human')}
            disabled={!canHuman}
            title={!humanPlayable ? t('play.not_playable') : trainLive ? t('play.busy_training') : undefined}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              height: 'var(--control-sm)', padding: '0 12px', borderRadius: 'var(--radius-md)',
              cursor: canHuman ? 'pointer' : 'not-allowed',
              fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)',
              background: 'var(--accent)', color: 'var(--accent-contrast)',
              border: '1px solid transparent', boxShadow: 'var(--shadow-xs)',
              opacity: canHuman ? 1 : 0.5, transition: 'var(--t-colors)',
            }}
          >
            <span aria-hidden>▶</span> {t('play.play')}
          </button>

          {/* AI plays by itself */}
          <button
            onClick={() => handlePlay('ai')}
            disabled={!canAi}
            title={trainLive ? t('play.busy_training') : !aiReady ? t('play.no_checkpoints') : undefined}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              height: 'var(--control-sm)', padding: '0 12px', borderRadius: 'var(--radius-md)',
              cursor: canAi ? 'pointer' : 'not-allowed',
              fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-semibold)',
              background: 'var(--surface-2)', color: 'var(--text-strong)',
              border: '1px solid var(--border-default)',
              opacity: canAi ? 1 : 0.5, transition: 'var(--t-colors)',
            }}
          >
            <span aria-hidden>🤖</span> {t('play.ai_play')}
          </button>

          {/* Model picker for the AI button (shown when checkpoints exist for this env) */}
          {envCheckpoints.length > 0 ? (
            <label style={labelStyle}>
              {t('play.checkpoint')}
              <select
                value={playCheckpointId ?? ''}
                onChange={(e) => setPlayCheckpointId(e.target.value)}
                style={{ ...selectStyle, maxWidth: 160 }}
              >
                {envCheckpoints.map((c) => (
                  <option key={c.id} value={c.id} title={c.label}>
                    {optionLabel(c.label)}
                  </option>
                ))}
              </select>
            </label>
          ) : (
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t('play.no_checkpoints')}</span>
          )}
        </>
      )}

      {/* Pacing — play allows slow-mo so a beginner can react */}
      <label style={labelStyle}>
        {t('play.speed')}
        <select
          value={playSpeed}
          onChange={(e) => setPlaySpeed(parseFloat(e.target.value))}
          style={selectStyle}
        >
          {PLAY_SPEEDS.map((s) => (
            <option key={s} value={s}>{s}×</option>
          ))}
        </select>
      </label>

      <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
        {error && (
          <span style={{ fontSize: 11, color: 'var(--danger, #e2453c)', maxWidth: 200 }}>{error}</span>
        )}
        <PlayInstructions />
      </div>
    </div>
  )
}

const labelStyle: CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 5,
  fontSize: 11, color: 'var(--text-muted)',
}

const selectStyle: CSSProperties = {
  height: 'var(--control-sm)', padding: '0 10px', borderRadius: 'var(--radius-md)',
  fontSize: 'var(--fs-label)', fontFamily: 'var(--font-sans)',
  background: 'var(--surface-2)', color: 'var(--text-strong)',
  border: '1px solid var(--border-default)', cursor: 'pointer', transition: 'var(--t-colors)',
}
