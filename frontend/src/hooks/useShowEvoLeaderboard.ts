import { useEffect, useState } from 'react'
import { useAppStore } from '../store/useAppStore'

// Keep the Top-5 leaderboard up this long after a neuroevolution run ends, so the final
// generation can be studied before the slot reverts to the persistent high-score boards.
export const EVO_GRACE_MS = 15_000

/** When to show the evolution Top-5 (vs. the high-score boards): only while a neuroevolution
 *  run is active, plus a short grace window after it ends. Selecting neuroevolution without
 *  pressing Run keeps the high scores; switching back to PPO drops the leaderboard at once.
 *
 *  The grace window is a boolean flipped *on* at the run's falling edge using the render-time
 *  "adjust state when inputs change" pattern (https://react.dev/learn/you-might-not-need-an-effect)
 *  and flipped *off* by the async timer below. So there is no synchronous setState in an effect
 *  body (no cascading renders) and no impure clock read during render — and no flicker, because
 *  React re-renders from the render-time setState before painting. */
export function useShowEvoLeaderboard(): boolean {
  const algo       = useAppStore((s) => s.algo)
  const trainState = useAppStore((s) => s.trainState)
  const isEvo = algo === 'neuroevolution'
  const evoRunActive =
    isEvo && (trainState === 'running' || trainState === 'paused' || trainState === 'stopping')

  const [prevActive, setPrevActive] = useState(false)
  const [grace, setGrace] = useState(false)

  if (evoRunActive !== prevActive) {
    setPrevActive(evoRunActive)
    // Falling edge (a run just ended) → open the grace window; rising edge → clear any stale grace
    // (the live `evoRunActive` keeps the board up on its own).
    setGrace(!evoRunActive)
  }
  // Leaving evo mode (switch back to PPO) drops the board at once — no grace is honored.
  if (!isEvo && grace) setGrace(false)

  // Close the grace window after EVO_GRACE_MS. The setState here is asynchronous (a timer callback),
  // which the cascading-render rule allows.
  useEffect(() => {
    if (!grace) return
    const timer = setTimeout(() => setGrace(false), EVO_GRACE_MS)
    return () => clearTimeout(timer)
  }, [grace])

  return evoRunActive || grace
}
