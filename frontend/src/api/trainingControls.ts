// Shared training run-controls (C2). Both the Sidebar and the Evolution Stats panel expose
// Run/Stop, so the start/stop logic lives here once. `handleRun` reads the current algorithm
// from the store and sends PPO hyperparams or the evolution block accordingly.

import { useAppStore } from '../store/useAppStore'
import { pauseTraining, resumeTraining, startSweep, startTraining, stopTraining } from './client'
import type { TrainConfig } from './types'

/** Build the TrainConfig from the store's current settings (the sidebar). Shared by Start and by
 *  Load-to-continue (so resuming a run can extend/retune it with the current params). Reads
 *  `getState()` so it works outside React too. Returns null when no env is selected. */
export function buildTrainConfig(): TrainConfig | null {
  const s = useAppStore.getState()
  if (!s.selectedEnvId) return null
  const env = s.envs.find((e) => e.id === s.selectedEnvId)
  const isAz = s.algo === 'alphazero'
  const isSelfPlay = !!env && env.family === 'petting_zoo' && env.competitive === true
  return {
    env_id: s.selectedEnvId,
    algo: s.algo,
    seed: s.seed,
    // AlphaZero's budget is iterations × games_per_iter self-play games; total_timesteps mirrors that
    // so the progress bar / status agree with the trainer (which reports games as timesteps).
    total_timesteps: isAz ? s.alphaZeroParams.iterations * s.alphaZeroParams.games_per_iter : s.totalTimesteps,
    hyperparams: s.hyperparams,
    evolution: s.algo === 'neuroevolution' ? s.evolutionParams : null,
    q_learning: s.algo === 'q_learning' ? s.qLearningParams : null,
    self_play: isSelfPlay ? s.selfPlayParams : null,
    alphazero: isAz ? s.alphaZeroParams : null,
    sac: s.algo === 'sac' ? s.sacParams : null,
    td3: s.algo === 'td3' ? s.td3Params : null,
    dqn: s.algo === 'dqn' ? s.dqnParams : null,
    a2c: s.algo === 'a2c' ? s.a2cParams : null,
  }
}

export interface RunControls {
  handleRun: () => Promise<void>
  /** X3: launch a seed-sweep — the current config across `sweepCount` seeds, queued back-to-back. */
  handleRunSweep: () => Promise<void>
  handlePause: () => Promise<void>
  handleResume: () => Promise<void>
  handleStop: () => Promise<void>
  isRunning: boolean
  isPaused: boolean
  isStopping: boolean
  isActive: boolean
  canRun: boolean
  /** True when training the selected env is disabled while the game is still human-playable (G4a). Two
   *  reasons (see `trainGatedReason`): the env needs a GPU that isn't present, OR its trainer isn't
   *  built yet (image envs). The Sidebar shows the matching explanatory note. */
  trainGated: boolean
  /** Why training is gated, so the Sidebar picks the right note. `'no_gpu'` = needs a CUDA device (the
   *  vector heavies un-gate on a GPU); `'not_implemented'` = image-obs CnnPolicy trainer not built yet
   *  (stays gated even on a GPU, G4b/G3c-train); `'not_implemented_ma'` = a watch-only multi-agent env
   *  whose per-species trainer isn't built yet (simple_tag, G7b) — not pixel-based and not hand-playable,
   *  so it needs its own note; `'not_implemented_board'` = a board game whose self-play trainer isn't
   *  built yet (G6b) — it IS hand-playable (vs the built-in AI), so its note points to the Play button;
   *  `null` = not gated. */
  trainGatedReason: 'no_gpu' | 'not_implemented' | 'not_implemented_ma' | 'not_implemented_board' | null
}

export function useRunControls(): RunControls {
  // The run-config values (algo, seed, hyperparams, …) are read at click time via buildTrainConfig()
  // (getState()), so the hook itself only needs what drives the gating + control state below.
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const envs            = useAppStore((s) => s.envs)
  const trainState      = useAppStore((s) => s.trainState)
  const gpuAvailable    = useAppStore((s) => s.gpuAvailable)
  const clearMetrics    = useAppStore((s) => s.clearMetrics)
  const setTrainState   = useAppStore((s) => s.setTrainState)

  const isRunning  = trainState === 'running'
  const isPaused   = trainState === 'paused'
  const isStopping = trainState === 'stopping'
  const isActive   = isRunning || isPaused || isStopping
  // Gate Run while the game stays human-playable. Two reasons, in priority order:
  //  1. the image-obs envs (Atari, CarRacing) have no trainer yet (CnnPolicy/GPU seam, G4b/G3c-train) →
  //     gated even on a GPU machine until that lands;
  //  2. the GPU-gated *vector* heavies (BipedalWalker, MuJoCo) train with the existing MlpPolicy but
  //     need a CUDA device — gated only while none is present (a GPU machine un-gates them).
  const selectedEnv     = envs.find((e) => e.id === selectedEnvId)
  const notImplemented  = !!selectedEnv && selectedEnv.train_implemented === false
  // A watch-only multi-agent env (simple_tag): its trainer isn't built either, but the image-trainer
  // note ("pixel-based games … use the Play button") is doubly wrong here — it's vector, and there's
  // no Play button (a swarm has no single human driver) — so it gets its own note.
  const watchOnlyMa     = !!selectedEnv && selectedEnv.family === 'petting_zoo' && selectedEnv.human_playable === false
  // A board game (G6a): its neural self-play trainer isn't built yet (G6b), but unlike the MA/image
  // cases it IS hand-playable now (vs the built-in MCTS AI), so its gate note points to Play.
  const isBoard         = !!selectedEnv && selectedEnv.family === 'board'
  const needsAbsentGpu  = !!selectedEnv && selectedEnv.hw_requirement === 'gpu' && !gpuAvailable
  const trainGated      = notImplemented || needsAbsentGpu
  const trainGatedReason: 'no_gpu' | 'not_implemented' | 'not_implemented_ma' | 'not_implemented_board' | null =
    notImplemented
      ? (watchOnlyMa ? 'not_implemented_ma' : isBoard ? 'not_implemented_board' : 'not_implemented')
      : needsAbsentGpu ? 'no_gpu' : null
  const canRun          = !!selectedEnvId && envs.length > 0 && !trainGated

  // Apply the REST response's state to the store immediately, so the controls flip (Run → Pause/Stop,
  // etc.) without waiting for the WS `status` frame. A heavy first-time env import — notably the
  // multi-agent stack (supersuit+mpe2, ~8 s cold) running on the trainer thread — can delay that WS
  // frame, which otherwise left a started run looking unstarted and unstoppable (the controls never
  // showed Pause/Stop). The authoritative WS frames still reconcile any later transitions.
  async function handleRun() {
    if (!canRun) return
    clearMetrics()
    try {
      const config = buildTrainConfig()
      if (!config) return
      const status = await startTraining(config)
      setTrainState(status.state)
    } catch (err) {
      console.error('Failed to start training:', err)
    }
  }

  // Seed sweep (X3): same config as handleRun, but POST /sweep with the store's seed count. The backend
  // queues N runs (seeds config.seed … +N−1) and runs them back-to-back; the WS status carries which
  // seed of N is live. Reuses handleRun's clear + config build so a sweep starts from a clean chart.
  async function handleRunSweep() {
    if (!canRun) return
    clearMetrics()
    try {
      const config = buildTrainConfig()
      if (!config) return
      const count = Math.max(1, Math.round(useAppStore.getState().sweepCount))
      const status = await startSweep(config, count)
      setTrainState(status.state)
    } catch (err) {
      console.error('Failed to start seed sweep:', err)
    }
  }

  async function handlePause()  { try { setTrainState((await pauseTraining()).state)  } catch { /* ignore */ } }
  async function handleResume() { try { setTrainState((await resumeTraining()).state) } catch { /* ignore */ } }
  async function handleStop()   { try { setTrainState((await stopTraining()).state)   } catch { /* ignore */ } }

  return { handleRun, handleRunSweep, handlePause, handleResume, handleStop, isRunning, isPaused, isStopping, isActive, canRun, trainGated, trainGatedReason }
}
