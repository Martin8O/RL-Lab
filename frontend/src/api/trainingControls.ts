// Shared training run-controls (C2). Both the Sidebar and the Evolution Stats panel expose
// Run/Stop, so the start/stop logic lives here once. `handleRun` reads the current algorithm
// from the store and sends PPO hyperparams or the evolution block accordingly.

import { useAppStore } from '../store/useAppStore'
import { pauseTraining, resumeTraining, startTraining, stopTraining } from './client'

export interface RunControls {
  handleRun: () => Promise<void>
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
   *  (stays gated even on a GPU, G4b/G3c-train); `null` = not gated. */
  trainGatedReason: 'no_gpu' | 'not_implemented' | null
}

export function useRunControls(): RunControls {
  const algo            = useAppStore((s) => s.algo)
  const selectedEnvId   = useAppStore((s) => s.selectedEnvId)
  const envs            = useAppStore((s) => s.envs)
  const seed            = useAppStore((s) => s.seed)
  const totalTimesteps  = useAppStore((s) => s.totalTimesteps)
  const hyperparams     = useAppStore((s) => s.hyperparams)
  const evolutionParams = useAppStore((s) => s.evolutionParams)
  const qLearningParams = useAppStore((s) => s.qLearningParams)
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
  const needsAbsentGpu  = !!selectedEnv && selectedEnv.hw_requirement === 'gpu' && !gpuAvailable
  const trainGated      = notImplemented || needsAbsentGpu
  const trainGatedReason: 'no_gpu' | 'not_implemented' | null =
    notImplemented ? 'not_implemented' : needsAbsentGpu ? 'no_gpu' : null
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
      const status = await startTraining({
        env_id: selectedEnvId!,
        algo,
        seed,
        total_timesteps: totalTimesteps,
        hyperparams,
        // Each block is sent only for its own algorithm; null keeps the recorded config clean.
        evolution: algo === 'neuroevolution' ? evolutionParams : null,
        q_learning: algo === 'q_learning' ? qLearningParams : null,
      })
      setTrainState(status.state)
    } catch (err) {
      console.error('Failed to start training:', err)
    }
  }

  async function handlePause()  { try { setTrainState((await pauseTraining()).state)  } catch { /* ignore */ } }
  async function handleResume() { try { setTrainState((await resumeTraining()).state) } catch { /* ignore */ } }
  async function handleStop()   { try { setTrainState((await stopTraining()).state)   } catch { /* ignore */ } }

  return { handleRun, handlePause, handleResume, handleStop, isRunning, isPaused, isStopping, isActive, canRun, trainGated, trainGatedReason }
}
