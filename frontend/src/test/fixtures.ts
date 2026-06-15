import type { EnvSpec, HyperparamDef, PlayScores } from '../api/types'

// A minimal, fully-typed CartPole spec for component tests. Mirrors the shape the backend
// registry serves (algo → param → definition) closely enough to drive the Sidebar sliders.
function num(
  d: Partial<HyperparamDef> & { default: number; recommended: number; min: number; max: number; step: number },
): HyperparamDef {
  return { type: 'float', choices: null, ...d }
}

export const cartpoleEnv: EnvSpec = {
  id: 'cartpole',
  gym_id: 'CartPole-v1',
  display_name: { en: 'CartPole', cz: 'CartPole' },
  description: { en: 'Balance a pole on a cart.', cz: 'Vyvaž tyč na vozíku.' },
  family: 'classic_control',
  obs_type: 'vector',
  action_space: 'discrete',
  supported_algos: ['ppo', 'neuroevolution'],
  hyperparams: {
    ppo: {
      learning_rate: num({ type: 'float', default: 3e-4, recommended: 3e-4, min: 1e-5, max: 1e-2, step: 0.01 }),
      gamma:         num({ default: 0.99, recommended: 0.99, min: 0.8, max: 0.999, step: 0.001 }),
      clip_range:    num({ default: 0.2, recommended: 0.2, min: 0.1, max: 0.4, step: 0.01 }),
      ent_coef:      num({ default: 0, recommended: 0, min: 0, max: 0.1, step: 0.001 }),
      n_hidden_layers:   num({ type: 'int', default: 2, recommended: 2, min: 1, max: 4, step: 1 }),
      neurons_per_layer: num({ type: 'int', default: 64, recommended: 64, min: 16, max: 256, step: 16 }),
      activation: { type: 'categorical', default: 'tanh', recommended: 'tanh', min: null, max: null, step: null, choices: ['tanh', 'relu'] },
    },
    neuroevolution: {
      population_size: num({ type: 'int', default: 50, recommended: 50, min: 10, max: 200, step: 10 }),
      top_k_parents:   num({ type: 'int', default: 10, recommended: 10, min: 2, max: 50, step: 1 }),
      mutation_rate:   num({ default: 0.1, recommended: 0.1, min: 0.01, max: 1, step: 0.01 }),
      crossover_rate:  num({ default: 0.5, recommended: 0.5, min: 0, max: 1, step: 0.01 }),
      generations:     num({ type: 'int', default: 30, recommended: 30, min: 5, max: 200, step: 5 }),
    },
  },
  default_total_timesteps: 50_000,
  play_step_scale: 1,
  floor_scales_with_steps: true,
  sparse_reward: false,
  turn_based: false,
  human_playable: true,
  competitive: false,
  difficulty: 'beginner',
  hw_requirement: 'cpu',
}

export const samplePlayScores: PlayScores = {
  env_id: 'cartpole',
  human: [
    { name: 'Ada', score: 480.0, steps: 480, achieved_at: '2026-06-13T10:00:00Z' },
    { name: 'Grace', score: 310.5, steps: 310, achieved_at: '2026-06-13T09:00:00Z' },
  ],
  ai: [
    { name: 'PPO 50k', score: 500.0, steps: 500, achieved_at: '2026-06-13T08:00:00Z', model_id: 'ckpt-1', algo: 'ppo' },
  ],
}
