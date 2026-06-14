// Per-environment keyboard → action bindings for human play (the "2→4 actions" seam).
// Data-driven and game-agnostic, mirroring content/playGuides.ts (which holds the *displayed*
// controls): adding a game's keyboard is a content-only change here. The backend holds the last
// received action between WS frames (latency-tolerant), so we send on keydown and, for envs with
// an explicit idle action, send `idleAction` on release.
//
// `keys` are matched against `KeyboardEvent.key` verbatim, so list every variant (e.g. both 'a'
// and 'A', the arrow name). Keep the bindings in sync with the human-readable controls in
// content/playGuides.ts.

export interface KeyBinding {
  /** KeyboardEvent.key values that trigger this action (e.g. ['ArrowLeft', 'a', 'A']). */
  keys: string[]
  /** The discrete action id to send for the env (e.g. CartPole 0=left, LunarLander 2=main engine). */
  action: number
}

export interface PlayKeymap {
  bindings: KeyBinding[]
  /**
   * Action to send when all bound keys are released. `null` = hold the last action (CartPole has
   * no "do nothing", so a release would be ambiguous — keep pushing the last way). A number = the
   * env's idle action (LunarLander 0 = no thrust), so letting go cuts the engines.
   */
  idleAction: number | null
}

export const PLAY_KEYMAPS: Record<string, PlayKeymap> = {
  // CartPole: 0 = push left, 1 = push right. No idle action — the cart always moves.
  cartpole: {
    bindings: [
      { keys: ['ArrowLeft', 'a', 'A'], action: 0 },
      { keys: ['ArrowRight', 'd', 'D'], action: 1 },
    ],
    idleAction: null,
  },
  // LunarLander: 0 = do nothing, 1 = left engine (drifts left), 2 = main engine (thrust up),
  // 3 = right engine (drifts right). Releasing all keys coasts (idle = 0).
  lunarlander: {
    bindings: [
      { keys: ['ArrowUp', 'w', 'W'], action: 2 },
      { keys: ['ArrowLeft', 'a', 'A'], action: 1 },
      { keys: ['ArrowRight', 'd', 'D'], action: 3 },
    ],
    idleAction: 0,
  },
}

export const DEFAULT_KEYMAP: PlayKeymap = PLAY_KEYMAPS.cartpole

export function keymapFor(envId: string | null): PlayKeymap {
  return (envId !== null && PLAY_KEYMAPS[envId]) || DEFAULT_KEYMAP
}
