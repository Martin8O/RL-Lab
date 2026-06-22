import { useEffect, useState } from 'react'
import type { CSSProperties } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { fetchCheckpoints, startPlay, stopPlay, updatePlaySpeed } from '../api/client'
import type { CheckpointMeta } from '../api/types'
import { keymapFor } from '../content/playKeymaps'
import { boardMetaFor } from '../content/boardGames'
import PlayInstructions from './PlayInstructions'
import ParamInfo from './ParamInfo'
import type { BoardStrength } from '../api/types'

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
  const boardSide        = useAppStore((s) => s.boardSide)
  const boardStrength    = useAppStore((s) => s.boardStrength)
  const setBoardSide     = useAppStore((s) => s.setBoardSide)
  const setBoardStrength = useAppStore((s) => s.setBoardStrength)
  const checkpointsNonce = useAppStore((s) => s.checkpointsNonce)

  const [checkpoints, setCheckpoints] = useState<CheckpointMeta[]>([])
  const [error, setError] = useState<string | null>(null)
  // Board games (G6b): which opponent the human (or the AI-vs-AI watch) faces — '' = the built-in MCTS
  // (G6a, paced by the difficulty selector), or a saved checkpoint id = your trained net. Local +
  // unpersisted; defaults to the MCTS so the G6a behaviour is the default and the net is opt-in.
  const [boardOpponent, setBoardOpponent] = useState<string>('')

  const env          = envs.find((e) => e.id === selectedEnvId)
  const humanPlayable = env?.human_playable ?? false
  // Board games (G6a): the built-in "AI" side is an MCTS (no checkpoint), so "AI plays" is an AI-vs-AI
  // watch and the controls swap the checkpoint picker for side + difficulty; G6b adds an opponent picker
  // so you can instead face your **trained net** (a checkpoint).
  const isBoard      = env?.family === 'board'
  // The piece to show for each side in the side picker (G6e). For a **directional** game (Breakthrough)
  // the human always plays from the bottom advancing up, so both sides show that "up" glyph (the
  // `bottomPlayer`'s), told apart by colour. **Glyph-distinguished** games (chess, `uprightGlyphs`) and
  // placement games show each side's own lead glyph instead (chess: ♔ white vs ♚ black — same colour,
  // told apart by fill, so showing one glyph for both would be indistinguishable).
  const boardMeta    = boardMetaFor(selectedEnvId)
  const sideMark = (player: number): { glyph: string; color: string } => {
    const pieces = Object.values(boardMeta?.pieces ?? {})
    const ownOf = (p: number) => pieces.find((pc) => pc.player === p && pc.lead) ?? pieces.find((pc) => pc.player === p)
    const own = ownOf(player)
    const up = boardMeta?.orient && !boardMeta?.uprightGlyphs ? ownOf(boardMeta.orient.bottomPlayer) : undefined
    return { glyph: (up ?? own)?.glyph ?? (player === 0 ? '①' : '②'), color: own?.color ?? 'var(--text-strong)' }
  }
  const playing      = playState === 'playing'
  const trainLive    = trainState === 'running' || trainState === 'paused' || trainState === 'stopping'
  // Checkpoints that can actually be played here (same env; any algo works via the AI policy).
  const envCheckpoints = checkpoints.filter((c) => c.env_id === selectedEnvId)
  // Board: face the trained net when a valid checkpoint is picked, else the built-in MCTS.
  const useBoardNet  = isBoard && boardOpponent !== '' && envCheckpoints.some((c) => c.id === boardOpponent)

  // Load checkpoints when the backend comes online, whenever a play session ends, and whenever a
  // checkpoint is saved/deleted elsewhere (checkpointsNonce) — so a fresh Save shows up in the AI-play
  // picker without a page reload. Cheap, read-only.
  useEffect(() => {
    if (backendStatus !== 'online') return
    void fetchCheckpoints().then(setCheckpoints).catch(() => {})
  }, [backendStatus, playState, checkpointsNonce])

  // Keep the AI checkpoint selection valid: default to the newest matching checkpoint, and drop a
  // stale pick if the env changed out from under it (the "AI plays" button needs a valid model).
  useEffect(() => {
    const stillValid = playCheckpointId && envCheckpoints.some((c) => c.id === playCheckpointId)
    if (!stillValid) setPlayCheckpointId(envCheckpoints[0]?.id ?? null)
  }, [selectedEnvId, checkpoints]) // eslint-disable-line react-hooks/exhaustive-deps

  const ready     = backendStatus === 'online' && !trainLive
  const canHuman  = ready && humanPlayable
  const aiReady   = !!playCheckpointId && envCheckpoints.length > 0
  // Board "AI plays" = watch the built-in MCTS play itself → no checkpoint needed; every other env
  // needs a loaded checkpoint for AI play.
  const canAi     = ready && (isBoard || aiReady)

  // Two explicit actions instead of a Play button + a who-plays dropdown: you Play, or the AI
  // plays by itself. The store's playMode (skill-meter label + keyboard wiring) is set per action.
  async function handlePlay(mode: 'human' | 'ai') {
    setError(null)
    setPlayMode(mode)
    // Remember which model the AI plays so its leaderboard identity is known on finish
    // (the checkpoint label already encodes algo + size, so use it verbatim). Board games are W/D/L
    // (no score leaderboard), so they carry no label.
    const ckpt = mode === 'ai' && !isBoard
      ? envCheckpoints.find((c) => c.id === playCheckpointId) ?? null
      : null
    setPlayCheckpointLabel(ckpt ? ckpt.label : null)
    // Board opponent: the picked trained net (a checkpoint) or null = the built-in MCTS (G6b). For a
    // non-board env, only AI play carries a checkpoint.
    const boardCheckpoint = useBoardNet ? boardOpponent : null
    try {
      const status = await startPlay({
        env_id: selectedEnvId ?? 'cartpole',
        mode,
        // Board: the trained net (or null = built-in MCTS). Other envs: the AI policy for an AI watch.
        checkpoint_id: isBoard ? boardCheckpoint : (mode === 'ai' ? playCheckpointId : null),
        // Human play: a fresh random seed each game, so envs with a randomized scene (LunarLander's
        // moon terrain) vary like the training preview does — a fixed seed made every game identical.
        // AI play keeps the configured seed so a checkpoint demo stays reproducible. For board games
        // this also varies the MCTS opponent each human game while keeping the AI-vs-AI watch a
        // reproducible demo (the play convention).
        seed: mode === 'human' ? null : seed,
        speed: playSpeed,
        // Hold the env's "do nothing" until a key is pressed (so MountainCar/Acrobot don't get
        // shoved by the default action 0). null for CartPole, which has no idle; 0 (NOOP) for Atari.
        idle_action: keymapFor(selectedEnvId, env?.family).idleAction,
        // Board games (G6a): which side the human takes + the MCTS opponent strength. Ignored elsewhere.
        side: boardSide,
        ai_strength: boardStrength,
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

  // "How to play" sits right beside the primary action (Play / Stop), nudged slightly up so it
  // reads as a help badge attached to that button rather than a peer control.
  const howToPlay = (
    <span style={{ display: 'inline-flex', transform: 'translateY(-8px)' }}>
      <PlayInstructions />
    </span>
  )

  return (
    <div style={{
      flexShrink: 0, borderTop: '1px solid var(--border-default)',
      // Board games carry the most controls (Play + Watch + side picker + opponent + difficulty), which
      // wrap to TWO rows once a saved checkpoint adds the opponent picker — while the *playing* bar is one
      // row (Stop + speed). That height swing grew/shrank the flex:1 stage above and re-centred the board
      // (the "board jumps bigger on Play" report). Reserve the 2-row height for board games so idle and
      // playing render at the SAME height → stable stage → the board neither resizes nor shifts. (28px
      // control × 2 + gap ≈ 63; 76 leaves headroom. Non-board bars keep the compact single-row floor.)
      background: 'var(--surface-1)', padding: '0 var(--space-3)', minHeight: isBoard ? 76 : 52,
      display: 'flex', alignItems: 'center', gap: 7, flexWrap: 'wrap', alignContent: 'center',
    }}>
      {playing ? (
        /* Stop the active session */
        <>
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
          {howToPlay}
        </>
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

          {howToPlay}

          {/* AI plays by itself (board: an AI-vs-AI watch, no checkpoint needed) */}
          <button
            onClick={() => handlePlay('ai')}
            disabled={!canAi}
            title={trainLive ? t('play.busy_training') : (!isBoard && !aiReady) ? t('play.no_checkpoints') : undefined}
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
            <span aria-hidden>🤖</span> {isBoard ? t('play.watch_ai') : t('play.ai_play')}
          </button>

          {isBoard ? (
            <>
              {/* Board games (G6a): pick your side, shown as the game's actual coloured pieces (G6e) — a
                  native <select> can't colour its options, so this is a small segmented radio control.
                  For a directional game (Breakthrough) the human always ends up at the bottom advancing
                  up, so both options show that "up" glyph distinguished only by colour. */}
              <label style={labelStyle}>
                {t('play.board_side')}
                <div role="radiogroup" aria-label={t('play.board_side')} style={sideGroupStyle}>
                  {/* Offer the sides in MOVE ORDER: the first mover is "Go first". For most games that's
                      player 0, but OpenSpiel chess makes white = player 1 the first mover (boardMeta.
                      firstPlayer), so picking "Go first" there correctly gives you white + the opening move. */}
                  {(() => { const first = boardMeta?.firstPlayer ?? 0; return [first, 1 - first] })().map((p, idx) => {
                    const mark = sideMark(p)
                    const active = boardSide === p
                    const label = t(idx === 0 ? 'play.board_side_first' : 'play.board_side_second')
                    return (
                      <button
                        key={p}
                        type="button"
                        role="radio"
                        aria-checked={active}
                        aria-label={label}
                        onClick={() => setBoardSide(p)}
                        style={{
                          ...sideBtnStyle,
                          background: active ? 'var(--surface-2)' : 'transparent',
                          color: active ? 'var(--text-strong)' : 'var(--text-muted)',
                          borderColor: active ? 'var(--border-default)' : 'transparent',
                          boxShadow: active ? 'var(--shadow-xs)' : 'none',
                        }}
                      >
                        <span aria-hidden style={{ color: mark.color, fontWeight: 800 }}>{mark.glyph}</span>
                        {label}
                      </button>
                    )
                  })}
                </div>
              </label>
              {/* Opponent (G6b): the built-in search AI (MCTS, with a difficulty) or your trained net.
                  Only shown once a checkpoint exists for this game; otherwise it's always the MCTS. */}
              {envCheckpoints.length > 0 && (
                <label style={labelStyle}>
                  {t('play.board_opponent')}
                  <select
                    /* Derived so a stale pick (e.g. after an env switch) falls back to the built-in AI
                       without a setState-in-effect — useBoardNet already encodes the id's validity. */
                    value={useBoardNet ? boardOpponent : ''}
                    onChange={(e) => setBoardOpponent(e.target.value)}
                    style={{ ...selectStyle, maxWidth: 160 }}
                  >
                    <option value="">{t('play.board_opponent_ai')}</option>
                    {envCheckpoints.map((c) => (
                      <option key={c.id} value={c.id} title={c.label}>
                        {optionLabel(c.label)}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {/* Difficulty only applies to the built-in MCTS; a trained net has no sims knob. */}
              {!useBoardNet && (
                <>
                  <label style={labelStyle}>
                    {t('play.board_difficulty')}
                    <select
                      value={boardStrength}
                      onChange={(e) => setBoardStrength(e.target.value as BoardStrength)}
                      style={selectStyle}
                    >
                      <option value="easy">{t('play.diff_easy')}</option>
                      <option value="medium">{t('play.diff_medium')}</option>
                      <option value="hard">{t('play.diff_hard')}</option>
                    </select>
                  </label>
                  <ParamInfo paramId="board_difficulty" label={t('play.board_difficulty')} />
                </>
              )}
            </>
          ) : envCheckpoints.length > 0 ? (
            /* Model picker for the AI button (shown when checkpoints exist for this env) */
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
          onChange={(e) => {
            const v = parseFloat(e.target.value)
            setPlaySpeed(v)
            // Apply to the running session immediately (not just the next start).
            if (playing) void updatePlaySpeed(v).catch(() => {})
          }}
          style={selectStyle}
        >
          {PLAY_SPEEDS.map((s) => (
            <option key={s} value={s}>{s}×</option>
          ))}
        </select>
      </label>

      {error && (
        <span style={{
          marginLeft: 'auto', fontSize: 11, color: 'var(--danger, #e2453c)', maxWidth: 200,
        }}>
          {error}
        </span>
      )}
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

// Board side picker (G6e) — a small segmented radio control so each side can show its coloured piece
// glyph (a native <select> can't colour individual options). Mirrors the sidebar's Segmented look.
const sideGroupStyle: CSSProperties = {
  display: 'flex', gap: 3, padding: 3,
  background: 'var(--surface-inset)', border: '1px solid var(--border-default)',
  borderRadius: 'var(--radius-md)',
}

const sideBtnStyle: CSSProperties = {
  display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 5,
  height: 'calc(var(--control-sm) - 6px)', padding: '0 9px', borderRadius: 'var(--radius-sm)',
  borderWidth: 1, borderStyle: 'solid',
  fontSize: 'var(--fs-label)', fontFamily: 'var(--font-sans)', fontWeight: 'var(--fw-medium)',
  whiteSpace: 'nowrap', cursor: 'pointer', transition: 'var(--t-colors)',
}
