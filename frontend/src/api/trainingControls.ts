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

  const isRunning  = trainState === 'running'
  const isPaused   = trainState === 'paused'
  const isStopping = trainState === 'stopping'
  const isActive   = isRunning || isPaused || isStopping
  // GPU-only envs (Atari + other image-obs games) can't train without a CUDA device — gate Run, but
  // keep them human-playable. On a GPU machine gpuAvailable is true, so nothing is gated (G4a).
  const selectedEnv = envs.find((e) => e.id === selectedEnvId)
  const trainGated  = !!selectedEnv && selectedEnv.hw_requirement === 'gpu' && !gpuAvailable
  const canRun      = !!selectedEnvId && envs.length > 0 && !trainGated

  async function handleRun() {
    if (!canRun) return
    clearMetrics()
    try {
      await startTraining({
        env_id: selectedEnvId!,
        algo,
        seed,
        total_timesteps: totalTimesteps,
        hyperparams,
        // Each block is sent only for its own algorithm; null keeps the recorded config clean.
        evolution: algo === 'neuroevolution' ? evolutionParams : null,
        q_learning: algo === 'q_learning' ? qLearningParams : null,
      })
    } catch (err) {
      console.error('Failed to start training:', err)
    }
  }

  async function handlePause()  { try { await pauseTraining()  } catch { /* ignore */ } }
  async function handleResume() { try { await resumeTraining() } catch { /* ignore */ } }
  async function handleStop()   { try { await stopTraining()   } catch { /* ignore */ } }

  return { handleRun, handlePause, handleResume, handleStop, isRunning, isPaused, isStopping, isActive, canRun, trainGated }
}
