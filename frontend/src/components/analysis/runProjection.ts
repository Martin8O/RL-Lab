// Project a finished run's recorded frames onto a DataLab compare curve. This mirrors the backend's
// analysis engine EXACTLY so a client-drawn overlay reads the same numbers as the server summary /
// aggregate:
//   • x  = the canonical X1 axis — cumulative `env_steps` (sample efficiency) or `wall_clock` seconds.
//   • y  = `score_of_frame` (best_fitness for neuroevolution, else ep_rew_mean) as raw *reward*, or,
//          for the per-algorithm pivot, that reward normalized to the 0–100 % skill scale over the
//          env's [min_score, solved_score] — the same clamp the live meter + `solvedPct` use.
// (Backend refs: services/analysis/stats.py `score_of_frame` / `skill_pct`, api/analysis.py `_seed_curve`.)

import type { Algo, EnvSpec, RunDetail } from '../../api/types'
import type { Pt } from './lttb'
import { solvedPct } from '../checkpointBrowser'

export type Axis = 'env_steps' | 'wall_clock'
export type Metric = 'reward' | 'skill_pct'

// A run frame carries the X1 axes on every member of the union; the reward field differs by trainer.
type RunFrame = RunDetail['metrics'][number]

/** The per-frame headline score on the env's reward scale — `best_fitness` for neuroevolution (its
 *  curve is fitness), `ep_rew_mean` for every other trainer. Mirrors backend `score_of_frame`. */
function frameReward(frame: RunFrame, algo: Algo): number | null {
  if (algo === 'neuroevolution') return 'best_fitness' in frame ? frame.best_fitness : null
  return 'ep_rew_mean' in frame ? frame.ep_rew_mean : null
}

/** The recorded curve of one run as (x, y) points on the chosen axis + metric. Frames whose reward is
 *  null (or unnormalizable for skill_pct) are dropped, so a sparse run yields a shorter clean line. */
export function runToPoints(
  detail: RunDetail,
  axis: Axis,
  metric: Metric,
  env: EnvSpec | undefined,
): Pt[] {
  const algo = detail.config.algo
  const min = env?.min_score ?? 0
  const solved = env?.solved_score ?? 0
  const out: Pt[] = []
  for (const f of detail.metrics) {
    const reward = frameReward(f, algo)
    if (reward === null) continue
    const y = metric === 'reward' ? reward : solvedPct(reward, min, solved)
    if (y === null) continue
    out.push({ x: axis === 'env_steps' ? f.env_steps : f.wall_clock, y })
  }
  // Guard against any out-of-order frames so the line + LTTB read left-to-right.
  out.sort((a, b) => a.x - b.x)
  return out
}
