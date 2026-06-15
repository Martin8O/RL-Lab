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
  /**
   * The value to send for the env. For a *discrete* env this is the action id (CartPole 0=left,
   * LunarLander 2=main engine). For a *scalar continuous* (box) env it is the analog command — a
   * real number the backend wraps into the whole action vector (Pendulum ±2 = full torque each
   * way, MountainCarContinuous ±1 = full throttle each way). For a *multi-joint continuous* env
   * (BipedalWalker, Box(4)) it is a per-joint VECTOR contribution (e.g. ← = [-1,0,0,0]); the
   * EnvPreview handler sums the held keys element-wise into one action vector that the backend
   * reshapes + clips. A keyboard can only do "full one way / full the other / nothing", which is
   * plenty to play these by hand.
   */
  action: number | number[]
}

export interface PlayKeymap {
  bindings: KeyBinding[]
  /**
   * Action to send when all bound keys are released. `null` = hold the last action (CartPole has
   * no "do nothing", so a release would be ambiguous — keep pushing the last way). A number = the
   * env's idle action (LunarLander 0 = no thrust), so letting go cuts the engines.
   */
  idleAction: number | null
  /**
   * Grid-worlds (Toy Text): play is turn-based — send exactly one action per key press, with no
   * auto-repeat or idle (the backend steps one cell per received action). Mirrors EnvSpec.turn_based.
   */
  turnBased?: boolean
}

// Atari (ALE) — ONE shared keymap for the whole family (G4a). Every Atari env is registered with
// full_action_space=True, so the 18 ALE actions sit at fixed indices in every game:
//   0 NOOP · 1 FIRE · 2 UP · 3 RIGHT · 4 LEFT · 5 DOWN (+ 6–17 = diagonals / fire combos).
// We bind the four directions + FIRE; releasing all keys sends NOOP (0). The combo actions
// (UPFIRE…) aren't bound directly — holding e.g. ↑ and Space makes the EnvPreview handler alternate
// UP and FIRE, which approximates them well enough to play every game by hand. Real-time, not
// turn-based; the play speed slider (down to 0.1×) paces these fast arcade games for a human.
const ATARI_KEYMAP: PlayKeymap = {
  bindings: [
    { keys: [' ', 'Spacebar'], action: 1 }, // FIRE
    { keys: ['ArrowUp', 'w', 'W'], action: 2 }, // UP
    { keys: ['ArrowRight', 'd', 'D'], action: 3 }, // RIGHT
    { keys: ['ArrowLeft', 'a', 'A'], action: 4 }, // LEFT
    { keys: ['ArrowDown', 's', 'S'], action: 5 }, // DOWN
  ],
  idleAction: 0, // NOOP — releasing all keys does nothing (the game keeps running)
}

// FrozenLake (4×4 / 4×4 no-slip / 8×8 share one keymap). Actions verified in the venv:
// 0 = Left, 1 = Down, 2 = Right, 3 = Up. Turn-based — one move per key press.
const FROZENLAKE_KEYMAP: PlayKeymap = {
  bindings: [
    { keys: ['ArrowLeft', 'a', 'A'], action: 0 },
    { keys: ['ArrowDown', 's', 'S'], action: 1 },
    { keys: ['ArrowRight', 'd', 'D'], action: 2 },
    { keys: ['ArrowUp', 'w', 'W'], action: 3 },
  ],
  idleAction: null,
  turnBased: true,
}

// MiniGrid (G2c) — ONE shared keymap for the whole family (mirrors Atari). The action space is
// Discrete(7); we bind navigation + interaction: 0 turn-left, 1 turn-right, 2 move-forward, 3 pickup,
// 4 drop, 5 toggle (open a door). DoorKey/KeyCorridor need pickup (3) + toggle (5). Turn-based — one
// move per key press, no auto-repeat, no idle. There is no "move backward", so ↓ is intentionally unbound.
const MINIGRID_KEYMAP: PlayKeymap = {
  bindings: [
    { keys: ['ArrowLeft', 'a', 'A'], action: 0 }, // turn left
    { keys: ['ArrowRight', 'd', 'D'], action: 1 }, // turn right
    { keys: ['ArrowUp', 'w', 'W'], action: 2 }, // move forward
    { keys: ['p', 'P', 'Enter'], action: 3 }, // pick up (the key / the ball)
    { keys: ['o', 'O'], action: 4 }, // drop
    { keys: [' ', 'Spacebar', 'e', 'E'], action: 5 }, // toggle (open a door)
  ],
  idleAction: null,
  turnBased: true,
}

// BipedalWalker (G3b) — continuous Box(4): the four leg-joint torques [hip1, knee1, hip2, knee2],
// each in [-1, 1]. Unlike Pendulum/MountainCarContinuous (one scalar torque filled across the whole
// action), this needs PER-JOINT control, so each key carries a 4-element VECTOR contribution and the
// EnvPreview handler SUMS the held keys element-wise into one action vector (the backend reshapes +
// clips it — the WS action frame + play_session already accept list[float] from the G1b seam). Arrow
// keys drive leg 1, WASD drives leg 2; releasing all keys sends the scalar idle 0, which the backend
// fills into a zero-torque vector [0,0,0,0]. Real-time (not turn-based) — pace it with the speed slider.
const BIPEDAL_KEYMAP: PlayKeymap = {
  bindings: [
    // Leg 1 — arrow keys: ← / → hip torque each way, ↑ / ↓ knee torque each way
    { keys: ['ArrowLeft'], action: [-1, 0, 0, 0] },
    { keys: ['ArrowRight'], action: [1, 0, 0, 0] },
    { keys: ['ArrowUp'], action: [0, 1, 0, 0] },
    { keys: ['ArrowDown'], action: [0, -1, 0, 0] },
    // Leg 2 — WASD: A / D hip torque each way, W / S knee torque each way
    { keys: ['a', 'A'], action: [0, 0, -1, 0] },
    { keys: ['d', 'D'], action: [0, 0, 1, 0] },
    { keys: ['w', 'W'], action: [0, 0, 0, 1] },
    { keys: ['s', 'S'], action: [0, 0, 0, -1] },
  ],
  idleAction: 0, // all keys released → backend fills a zero-torque vector [0,0,0,0]
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
  // MountainCar: 0 = accelerate left, 1 = don't accelerate, 2 = accelerate right.
  // Releasing all keys cuts the engine (idle = 1, "don't accelerate").
  mountaincar: {
    bindings: [
      { keys: ['ArrowLeft', 'a', 'A'], action: 0 },
      { keys: ['ArrowRight', 'd', 'D'], action: 2 },
    ],
    idleAction: 1,
  },
  // Acrobot: 0 = torque toward "−", 1 = no torque, 2 = torque toward "+". Releasing all
  // keys applies no torque (idle = 1) so the arm coasts.
  acrobot: {
    bindings: [
      { keys: ['ArrowLeft', 'a', 'A'], action: 0 },
      { keys: ['ArrowRight', 'd', 'D'], action: 2 },
    ],
    idleAction: 1,
  },
  // Pendulum (continuous): torque ∈ [-2, 2]. ← / → apply full torque each way; releasing all
  // keys applies zero torque (idle = 0) so the pendulum coasts. Bang-bang keyboard control.
  pendulum: {
    bindings: [
      { keys: ['ArrowLeft', 'a', 'A'], action: -2 },
      { keys: ['ArrowRight', 'd', 'D'], action: 2 },
    ],
    idleAction: 0,
  },
  // MountainCarContinuous (continuous): force ∈ [-1, 1]. ← / → apply full throttle each way;
  // releasing all keys cuts the throttle (idle = 0). Bang-bang keyboard control.
  mountaincarcontinuous: {
    bindings: [
      { keys: ['ArrowLeft', 'a', 'A'], action: -1 },
      { keys: ['ArrowRight', 'd', 'D'], action: 1 },
    ],
    idleAction: 0,
  },
  // BipedalWalker (+ Hardcore) — per-joint vector control (see BIPEDAL_KEYMAP).
  bipedalwalker: BIPEDAL_KEYMAP,
  bipedalwalkerhardcore: BIPEDAL_KEYMAP,
  // Toy Text grid-worlds — turn-based: one move per key press (see FROZENLAKE_KEYMAP).
  frozenlake: FROZENLAKE_KEYMAP,
  frozenlake_noslip: FROZENLAKE_KEYMAP,
  frozenlake8x8: FROZENLAKE_KEYMAP,
  // CliffWalking (verified in venv): 0 = Up, 1 = Right, 2 = Down, 3 = Left.
  cliffwalking: {
    bindings: [
      { keys: ['ArrowUp', 'w', 'W'], action: 0 },
      { keys: ['ArrowRight', 'd', 'D'], action: 1 },
      { keys: ['ArrowDown', 's', 'S'], action: 2 },
      { keys: ['ArrowLeft', 'a', 'A'], action: 3 },
    ],
    idleAction: null,
    turnBased: true,
  },
  // Taxi (verified in venv): 0 = South(↓), 1 = North(↑), 2 = East(→), 3 = West(←), 4 = Pickup,
  // 5 = Drop-off. Move with the arrows/WASD; P (or Space) picks up, O (or Enter) drops off.
  taxi: {
    bindings: [
      { keys: ['ArrowDown', 's', 'S'], action: 0 },
      { keys: ['ArrowUp', 'w', 'W'], action: 1 },
      { keys: ['ArrowRight', 'd', 'D'], action: 2 },
      { keys: ['ArrowLeft', 'a', 'A'], action: 3 },
      { keys: ['p', 'P', ' '], action: 4 },
      { keys: ['o', 'O', 'Enter'], action: 5 },
    ],
    idleAction: null,
    turnBased: true,
  },
}

export const DEFAULT_KEYMAP: PlayKeymap = PLAY_KEYMAPS.cartpole

// Atari is family-driven (one shared keymap for ~60 games), so the lookup takes the env's `family`:
// an explicit per-id entry wins, else the whole Atari family maps to ATARI_KEYMAP, else the default.
export function keymapFor(envId: string | null, family?: string): PlayKeymap {
  if (envId !== null && PLAY_KEYMAPS[envId]) return PLAY_KEYMAPS[envId]
  if (family === 'atari') return ATARI_KEYMAP
  if (family === 'minigrid') return MINIGRID_KEYMAP
  return DEFAULT_KEYMAP
}
