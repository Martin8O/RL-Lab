import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { fetchCheckpoints, startPlay, stopPlay } from '../api/client'
import type { CheckpointMeta } from '../api/types'
import PlayInstructions from './PlayInstructions'

const PLAY_SPEEDS = [0.1, 0.15, 0.25, 0.5, 1, 2, 4]

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
  const playMode       = useAppStore((s) => s.playMode)
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

  // Keep the AI-mode selection valid: default to the newest matching checkpoint, and drop a
  // stale pick if the env changed out from under it.
  useEffect(() => {
    if (playMode !== 'ai') return
    const stillValid = playCheckpointId && envCheckpoints.some((c) => c.id === playCheckpointId)
    if (!stillValid) setPlayCheckpointId(envCheckpoints[0]?.id ?? null)
  }, [playMode, selectedEnvId, checkpoints]) // eslint-disable-line react-hooks/exhaustive-deps

  const aiReady   = playMode !== 'ai' || (!!playCheckpointId && envCheckpoints.length > 0)
  const canPlay   = backendStatus === 'online' && humanPlayable && !trainLive && aiReady

  async function handlePlay() {
    setError(null)
    // Remember which model the AI plays so its leaderboard identity is known on finish
    // (the checkpoint label already encodes algo + size, so use it verbatim).
    const ckpt = playMode === 'ai'
      ? envCheckpoints.find((c) => c.id === playCheckpointId) ?? null
      : null
    setPlayCheckpointLabel(ckpt ? ckpt.label : null)
    try {
      const status = await startPlay({
        env_id: selectedEnvId ?? 'cartpole',
        mode: playMode,
        checkpoint_id: playMode === 'ai' ? playCheckpointId : null,
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
      flexShrink: 0, borderTop: '1px solid var(--border)',
      background: 'var(--surface)', padding: '6px 12px', minHeight: 46,
      display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap',
    }}>
      {/* Play / Stop */}
      <button
        onClick={playing ? handleStop : handlePlay}
        disabled={!playing && !canPlay}
        title={!humanPlayable ? t('play.not_playable') : trainLive ? t('play.busy_training') : undefined}
        style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '5px 12px', borderRadius: 6,
          cursor: (!playing && !canPlay) ? 'not-allowed' : 'pointer',
          fontSize: 11, fontWeight: 700,
          background: playing ? 'var(--danger, #e2453c)' : 'var(--accent)',
          color: '#fff', border: 'none',
          opacity: (!playing && !canPlay) ? 0.5 : 1,
        }}
      >
        <span aria-hidden>{playing ? '■' : '▶'}</span>
        {playing ? t('play.stop') : t('play.start')}
      </button>

      {/* Who plays */}
      <label style={labelStyle}>
        {t('play.mode')}
        <select
          value={playMode}
          disabled={playing}
          onChange={(e) => setPlayMode(e.target.value as 'human' | 'ai')}
          style={selectStyle}
        >
          <option value="human">{t('play.human')}</option>
          <option value="ai">{t('play.ai')}</option>
        </select>
      </label>

      {/* Checkpoint picker (AI watch only) */}
      {playMode === 'ai' && (
        envCheckpoints.length > 0 ? (
          <label style={labelStyle}>
            {t('play.checkpoint')}
            <select
              value={playCheckpointId ?? ''}
              disabled={playing}
              onChange={(e) => setPlayCheckpointId(e.target.value)}
              style={{ ...selectStyle, maxWidth: 160 }}
            >
              {envCheckpoints.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.label} · {c.algo === 'ppo' ? 'PPO' : 'EVO'}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{t('play.no_checkpoints')}</span>
        )
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
  padding: '3px 6px', borderRadius: 5, fontSize: 11,
  background: 'var(--surface-2)', color: 'var(--text)',
  border: '1px solid var(--border)', cursor: 'pointer',
}
