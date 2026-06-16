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
  /** True when the selected env needs a GPU that isn't present (G4a) — training is disabled but the
   *  game is still human-playable. The Sidebar shows an explanatory note in this case. */
  trainGated: boolean
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
  // GPU-only envs (Atari + other image-obs games) can't train without a CUDA device — gate Run, but
  // keep them human-playable. On a GPU machine gpuAvailable is true, so nothing is gated (G4a).
  const selectedEnv = envs.find((e) => e.id === selectedEnvId)
  const trainGated  = !!selectedEnv && selectedEnv.hw_requirement === 'gpu' && !gpuAvailable
  const canRun      = !!selectedEnvId && envs.length > 0 && !trainGated

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

  return { handleRun, handlePause, handleResume, handleStop, isRunning, isPaused, isStopping, isActive, canRun, trainGated }
}
