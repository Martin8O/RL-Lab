# Adding an environment

> Living document (seeded in Phase F4). The fastest path is data-only; this guide shows that path and
> the typed seams where a game needs real code. See also [`architecture.md`](architecture.md) and, for a
> new learning method, [`adding-an-algorithm.md`](adding-an-algorithm.md).

## TL;DR

A **vector-observation + discrete-action** Gymnasium env (CartPole, LunarLander, MountainCar, Acrobot …)
is added as **data**: one row in the registry plus bilingual content. A **continuous (`box`) action** env
reuses the now-built continuous seam, and a **discrete (single-integer) observation** env (Toy Text:
FrozenLake / Taxi / CliffWalking) reuses the now-built one-hot seam — both are also data + content.
An **image-observation** env (Atari / ALE — `obs_type="image"`) is **human-playable as data too**: it
reuses the existing server-JPEG render + play loop, so a registry row + `hw_requirement="gpu"` +
`train_implemented=False` + `supported_algos=["ppo"]` + a shared keymap is enough to play it now (G4a). Its
*training* still needs the `CnnPolicy`+CUDA seam, so `train_implemented=False` keeps Run gated off **even on a
GPU** (ADR-043) — distinct from the *vector* GPU heavies (BipedalWalker/MuJoCo, `train_implemented=True`) that
un-gate the moment a CUDA device is present. The two
compose: **CarRacing** (`obs_type="image"` **and** `action_space="box"`, G3c) is human-playable as data by
reusing *both* the server-JPEG path and the continuous-box play path at once (a steer/gas/brake vector keymap),
with training the same GPU-gated `CnnPolicy` case — confirming the seams stack without new engine code. A
**new image *family*** (**VizDoom** — a 3D FPS, `family="vizdoom"`, G8b / ADR-097) is the one image case that needs
a little code once: the Gymnasium VizDoom wrapper emits a `Dict` obs (`{'screen': Box(240,320,3), …}`), so it gets
its own vec builder `image_vec.make_vizdoom` (a screen-extraction wrapper `Dict→screen Box` → WarpFrame 84×84 →
`VecFrameStack`, `SubprocVecEnv`, **not** `AtariWrapper`) plus a `make_image_vec` branch and a `Vizdoom`-id
registration in `factory.make_env` (for human play). After that seam exists, each *scenario* is data + content +
a short calibration probe — PPO/DQN/QR-DQN, AI-play and the server-JPEG render all ride the shared image path
unchanged. One tuning note baked into the family: PPO's Atari default `ent_coef=0` **entropy-collapses** on VizDoom
(one critical action = shoot), so the family recipe adds a small `ent_coef=0.01` (like the board games). A
**dict-observation** env (MiniGrid — a 7×7×3 view + `direction` + a `mission` string) is **data too**:
`make_env` applies `FlatObsWrapper` for `family=="minigrid"`, flattening it to a vector so the same
`MlpPolicy`/genome train (on CPU) with no engine change, while the colourful grid still renders server-side
as a JPEG and play is turn-based (G2c). A **MuJoCo robotics** env (Hopper, Walker2d, HalfCheetah, Ant, Reacher,
Swimmer, Humanoid — vector obs + continuous `Box`, G5a) is **data + content only**: it reuses the continuous-box play path
(a per-joint vector keymap, like BipedalWalker) and the server-JPEG render path (not in `client_render`) at once,
with `hw_requirement="gpu"` gating training (a gait needs millions of steps — like BipedalWalker, *not* a
`CnnPolicy` gate). Two play fields tune the human feel for these fall-fast 125 fps robots: human play is capped at
the frame rate (ADR-041) and `human_play_slowdown` stretches the per-step wall-clock for envs that end on an
unpreventable fall (ADR-042). A **multi-agent** env (PettingZoo — N agents in one shared world) rides the now-built multi-agent seam: data
plus the `ma_env` adapter, trained with parameter-sharing PPO and drawn as a "swarm" (G7a; see below). A
**2-agent / competitive setup or turn-based self-play** needs code at one of the seams (bottom of this page).

## 1. Verify the env's truths in the venv — never guess

```python
import gymnasium as gym
s = gym.spec("YourEnv-v0")
print(s.reward_threshold, s.max_episode_steps)   # solved score + episode length (may be None)
e = gym.make("YourEnv-v0")
print(e.observation_space, e.action_space)        # vector vs image; discrete vs box (+ bounds)
```

Choose the meter's 0% floor (`min_score`) as the score of an agent that **achieves nothing** — *not* the
deepest score a flailing/random agent can reach (ADR-026). For a step-penalty env that baseline is "ran out
the clock doing nothing" ≈ `−1 × max_episode_steps` (e.g. CliffWalking/Taxi = −200), so an idle/stuck agent
reads ~0% and only real progress lifts the meter; worse runs simply clamp to 0%. A too-deep floor (set from
random returns) makes a do-nothing run read as near-mastery — the bug ADR-026 fixes. For a shaped env
(LunarLander) it's where a non-progressing agent sits (≈ −200) — and if such a shaped/terminal env also sets
`play_step_scale>1`, set **`floor_scales_with_steps=False`** so the longer play episode doesn't widen that
floor (a crash ends early; its score doesn't scale with the cap — ADR-027). If `reward_threshold` is `None`,
pick and **document** a sensible "solved" value.

## 2. Register the env (data) — `backend/app/envs/registry.py`

```python
register(EnvSpec(
    id="yourenv",                      # used as the key in all content maps
    gym_id="YourEnv-v0",
    display_name=Bilingual(en="…", cz="…"),
    description=Bilingual(en="…", cz="…"),
    family="classic_control",          # category for the game picker (see content/envCategories.ts)
    obs_type="vector",                 # "vector" | "image" | "discrete" (single int → one-hot, Toy Text)
    action_space="discrete",           # "discrete" | "box"
    supported_algos=["ppo", "neuroevolution"],   # gates the algo dropdown per-env
    hyperparams=_standard_hyperparams(),         # shared PPO + neuroevolution surface
    solved_score=200.0,                # 100% on the meter; the run-archive / "solved @" threshold
    min_score=-200.0,                  # 0% on the meter (see "skill floor" in §1 — the idle/timeout baseline)
    default_total_timesteps=500_000,   # the ★ PPO budget; the sidebar step ladder is ×0.2…×8 of this
    play_step_scale=1,                 # play episodes run this × longer (so a human has time to play)
    # Optional: make_kwargs={...} for variants sharing one gym_id (FrozenLake map_name/is_slippery);
    # episode_step_limit=N for an env with no native TimeLimit (CliffWalking); turn_based=True for a
    # grid-world the human plays one move per key press (see "Discrete observations" below).
    human_playable=True, competitive=False,
    difficulty="intermediate", hw_requirement="cpu",
))
```

### Discrete (single-integer) observations — Toy Text

If `observation_space` is `Discrete(n)` (the state is one int — which grid cell / which Taxi configuration),
set `obs_type="discrete"`. The shared `make_env` factory wraps the env in `OneHotObservation` (int →
length-`n` vector) automatically, so PPO's `MlpPolicy` and the numpy genome train with **no engine change** —
it's still data + content. Notes: a deprecated gym id (e.g. `CliffWalking-v0` → `-v1`) and an env with **no
native `TimeLimit`** both surface here — use the verified id and set `episode_step_limit` so a poor policy
can't loop forever. Variants that differ only by kwargs (FrozenLake's `map_name`/`is_slippery`) are separate
registry rows sharing one `gym_id` via `make_kwargs`.

### Dict observations (image + extras) — MiniGrid

If the obs is a `Dict` bundling a small image with extra fields (MiniGrid: a 7×7×3 partial view + the agent's
`direction` + a `mission` string), keep `obs_type="vector"` and set `family="minigrid"`: `make_env` applies
`minigrid.wrappers.FlatObsWrapper` for that family (flattening the Dict to a length-2835 `Box`), exactly as it
applies the one-hot wrapper for Toy Text — so PPO/neuroevolution train with no engine change, on CPU. Rendering
is **server-side** (the family isn't in `client_render`, so `client_state` returns None → the env's `rgb_array`
→ JPEG, like Atari but no retro skin), and play is `turn_based=True`. These envs have **no native `TimeLimit`**
but self-truncate at their internal `max_steps`, so no `episode_step_limit` is needed (G2c).

### Multi-agent (PettingZoo) — the swarm seam

If the env has **N agents in one shared world** (PettingZoo's *parallel* API — each agent its own
obs/action/reward), it is **not** a single-agent `make_env` row: it rides the multi-agent seam (ADR-038).
Set `family="petting_zoo"`, `obs_type="vector"` (the per-agent obs), `supported_algos=["ppo"]` (no
neuroevolution / Q-learning path), `human_playable=False` (a swarm has no single human driver — watch + train
only), and put the PettingZoo constructor kwargs in `make_kwargs` (`{"N": 3, "max_cycles": 25, ...}`); the
spec's `gym_id` is the scenario module name (e.g. `"simple_spread_v3"`). The adapter `app/services/ma_env.py`
builds the raw parallel env (preview/render) and the SuperSuit **parameter-sharing** vec env (one shared
`MlpPolicy` over all homogeneous agents); `trainer_ppo` and the preview streamer branch on `is_multi_agent`.
Rendering is a **client-side "swarm" canvas** drawn from the additive `agents`/`world` frame fields (per-agent
+ landmark world positions extracted by `ma_env.agent_sprites`/`world_entities`). MA reproducibility is
policy-level (the SuperSuit vec env can't be seeded by SB3; the trainer seeds numpy/torch/python instead).

**Homogeneous agents** (simple_spread) train via parameter sharing. **Heterogeneous species** (simple_tag —
predators vs. prey, different obs sizes + opposite rewards) break parameter sharing and need **per-species
policies** (G7b-2). Such an env can still be *registered watch-only* now (G7b-1 / ADR-047): set
`train_implemented=False` (gates Run with a multi-agent note) and rely on the **training-free watch** —
`POST /api/preview/watch` drives the streamer with no policy (random rollout) so the swarm renders without a
trainer. The render + frame contract already cover heterogeneous worlds (`role="adversary"`, `kind="obstacle"`),
so registering simple_tag was a data row + content only.

The registry is the source of truth: the sidebar, skill bands, step ladder, compare filter, and the algo
dropdown all read from it. `supported_algos` is how a game opts out of an algorithm (e.g. an image env can be
PPO-only); the store snaps to a valid algo when you switch games.

## 3. Bilingual content (frontend)

- **`content/parameters.ts`** — add a `perEnv["yourenv"]` note to **every** parameter block (and the chart
  concepts), CZ+EN, honest about the env's quirks. Keep the shared `general`/`recommended` game-neutral.
- **`content/playGuides.ts`** — goal / controls / tips for the "How to play" popup.
- **`content/playKeymaps.ts`** — key → action (`action: number | number[]`). For discrete, `action` is the
  action id; for a **single-torque** box env it's a scalar analog command the backend fills across the whole
  action (e.g. Pendulum ±2 = full torque); for a **multi-joint** box env (BipedalWalker `Box(4)`) each key
  carries a per-joint **vector** (e.g. `←` = `[-1,0,0,0]`) and `EnvPreview` sums the held keys into one action
  (the backend reshapes + clips it). `idleAction` is the env's true no-op (so the agent isn't shoved before any
  input) — `null` if the env always moves (CartPole); a scalar `0` for a box env (the backend fills a zero vector).
- **`content/envCategories.ts`** — add the `family` label if it's a new category.

i18n parity (`en.json`/`cz.json`) is enforced by `.\tasks.ps1 i18n`; the per-env prose above lives in the
content files, not the i18n JSON.

## 4. Rendering — client-side (preferred) or server image

- **Client render** (lighter, crisper, lets us draw nicer scenes): return the raw state from
  `backend/app/services/client_render.py` `client_state`, add a stage to `frontend/src/components/EnvStages.tsx`
  with its geometry in `envGeometry.ts`, and a branch to `EnvPreview.tsx`'s `clientKind` + draw dispatch
  (update the moving parts imperatively from each frame). Frames may include the just-applied `action` so the
  scene can show the firing effect (e.g. LunarLander thruster plumes). For scene geometry the obs can't
  provide (a randomized terrain), also return it from `client_render.terrain` in the **same obs-normalized
  coordinates** as the agent — draw both in one space, and clamp the agent's lowest point to the surface so
  it rests on the ground, not in it (see LunarLander's moon).
- **Grid-worlds** (Toy Text): `client_state` returns the agent's `[row, col]` (Taxi: the decoded
  `[taxi_row, taxi_col, passenger, destination]`) and `client_render.grid_layout` returns the **static board**
  as a `GridLayout` (`kind`+`rows`/`cols`+row-major `cells`), streamed in the `grid` frame field. Draw it with
  a `GridStage` in `EnvStages.tsx` (declarative — grids change slowly) and add an idle default to
  `content/gridMaps.ts`. Set `turn_based=True` so the human advances one cell per key press (the AI/preview
  still step continuously); the keymap sends one action per `keydown` for these.
- **Server image**: do nothing — envs not in the client-render set are rendered to JPEG automatically.

## 5. Skill bands — automatic

`services/skill.py` derives 5 bands from `[min_score, solved_score]` (fractions 0/10/30/60/95%). No per-env
config needed for monotonic-score envs. A symmetric-score game (e.g. Pong −21…21) will want an optional
per-env band override (see ADR-015) when that family lands.

## 6. Pre-delivery checklist (mandatory)

A green build ≠ a correct UI. Penalty/shaped envs expose CartPole-shaped assumptions. Before calling a new
env done, drive it **end-to-end in the browser** — train (every supported algo) + human play + AI play — and
confirm **every shown number** reflects this env's `[min_score, solved_score]` at the start *and* end of an
episode: skill-meter scale/zones/needle/%, the live chart, idle action (no shove), live speed, the per-env
popups, record-stars, and the start/idle pose. Then `.\tasks.ps1 all` must be green.

## The seams (when it's not data-only)

| Seam | Trigger | Where |
|---|---|---|
| Policy + device | image obs → `CnnPolicy` + CUDA | `trainer_ppo._build_model` |
| Shared frame-stack | Atari (vec-env + `VecFrameStack`) used by trainer **and** both streamers | new shared env path |
| Action space | continuous `box` (done: classic control, multi-joint, **image-box CarRacing human play G3c**, **Atari image AI-play G4c / ADR-046** via `play_session._run_image_ai` on the `make_atari` vec env); only `box` *training* on image obs stays GPU-gated | `play_session`, `policy`, `trainer_*`, preview, keymaps |
| Competitive | 2-agent / `side` selector (Pong) | `play_session` + a 2-agent env |
| Board games | turn-based self-play (OpenSpiel) | a parallel subsystem, not a registry row |
| Multi-agent (**done G7a**) | N agents in one shared world (PettingZoo) | `ma_env` adapter + `trainer_ppo`/preview branches; `agents`/`world` frame |
