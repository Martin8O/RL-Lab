import type { Algo, EnvSpec } from './types'

/** Whether a training run uses the GPU — a function of BOTH the env and the algorithm, not of static
 *  env metadata alone (the trap the first device badge fell into). Image-obs envs (Atari) train a
 *  CnnPolicy on CUDA (G4b). Board games are the algorithm-dependent case: **AlphaZero** runs the batched
 *  self-play engine on the GPU (G6g), while **MaskablePPO** (`ppo`) trains its small net on the CPU — so
 *  the same board game is GPU or CPU depending on the picked algorithm. Everything else trains an
 *  MlpPolicy on the CPU (faster there than shuttling tiny batches to a GPU — ADR-056). Board AlphaZero
 *  falls back to the CPU when no GPU is present (it is ungated), so it is GPU only when one is available.
 *  **SAC** (S5a) is also an MlpPolicy: its per-step gradient updates are tiny batch-256 forwards that the
 *  CPU runs faster than a latency-bound GPU shuttle (measured — same ADR-056 result), so SAC trains on the
 *  CPU and reads CPU here. (MuJoCo/BipedalWalker stay GPU-*gated* by step count, but the device is CPU.) */
export function trainsOnGpu(
  env: EnvSpec | undefined,
  algo: Algo,
  gpuAvailable: boolean,
): boolean {
  if (!env) return false
  if (env.obs_type === 'image') return true // Atari CnnPolicy (gated → always GPU when it runs)
  if (env.family === 'board' && algo === 'alphazero') return gpuAvailable // batched AZ engine (G6g)
  return false  // SAC + every other MlpPolicy trains on CPU (ADR-056)
}
