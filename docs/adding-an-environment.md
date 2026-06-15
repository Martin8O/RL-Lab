# Adding an environment

> Living document (seeded in Phase F4). The fastest path is data-only; this guide shows that path and
> the five seams where a game needs real code. See also [`architecture.md`](architecture.md) and, for a
> new learning method, [`adding-an-algorithm.md`](adding-an-algorithm.md).

## TL;DR

A **vector-observation + discrete-action** Gymnasium env (CartPole, LunarLander, MountainCar, Acrobot …)
is added as **data**: one row in the registry plus bilingual content. A **continuous (`box`) action** env
reuses the now-built continuous seam (also data + content). Anything with **image observations, a 2-agent /
competitive setup, or turn-based self-play** needs code at one of the five seams (bottom of this page).

## 1. Verify the env's truths in the venv — never guess

```python
import gymnasium as gym
s = gym.spec("YourEnv-v0")
print(s.reward_threshold, s.max_episode_steps)   # solved score + episode length (may be None)
e = gym.make("YourEnv-v0")
print(e.observation_space, e.action_space)        # vector vs image; discrete vs box (+ bounds)
```

Measure a random / do-nothing policy's return to choose the meter's 0% floor (`min_score`). For a
step-penalty env this is roughly `−1 × max_episode_steps`; for a shaped env it's where a flailing agent
bottoms out. If `reward_threshold` is `None`, pick and **document** a sensible "solved" value.

## 2. Register the env (data) — `backend/app/envs/registry.py`

```python
register(EnvSpec(
    id="yourenv",                      # used as the key in all content maps
    gym_id="YourEnv-v0",
    display_name=Bilingual(en="…", cz="…"),
    description=Bilingual(en="…", cz="…"),
    family="classic_control",          # category for the game picker (see content/envCategories.ts)
    obs_type="vector",                 # "vector" | "image"
    action_space="discrete",           # "discrete" | "box"
    supported_algos=["ppo", "neuroevolution"],   # gates the algo dropdown per-env
    hyperparams=_standard_hyperparams(),         # shared PPO + neuroevolution surface
    solved_score=200.0,                # 100% on the meter; the run-archive / "solved @" threshold
    min_score=-200.0,                  # 0% on the meter (negative for shaped/penalty envs)
    default_total_timesteps=500_000,   # the ★ PPO budget; the sidebar step ladder is ×0.2…×4 of this
    play_step_scale=1,                 # play episodes run this × longer (so a human has time to play)
    human_playable=True, competitive=False,
    difficulty="intermediate", hw_requirement="cpu",
))
```

The registry is the source of truth: the sidebar, skill bands, step ladder, compare filter, and the algo
dropdown all read from it. `supported_algos` is how a game opts out of an algorithm (e.g. an image env can be
PPO-only); the store snaps to a valid algo when you switch games.

## 3. Bilingual content (frontend)

- **`content/parameters.ts`** — add a `perEnv["yourenv"]` note to **every** parameter block (and the chart
  concepts), CZ+EN, honest about the env's quirks. Keep the shared `general`/`recommended` game-neutral.
- **`content/playGuides.ts`** — goal / controls / tips for the "How to play" popup.
- **`content/playKeymaps.ts`** — key → action. For discrete, `action` is the action id; for a box env it's
  the analog command (e.g. Pendulum ±2 = full torque). `idleAction` is the env's true no-op (so the agent
  isn't shoved before any input) — `null` if the env always moves (CartPole).
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

## The five seams (when it's not data-only)

| Seam | Trigger | Where |
|---|---|---|
| Policy + device | image obs → `CnnPolicy` + CUDA | `trainer_ppo._build_model` |
| Shared frame-stack | Atari (vec-env + `VecFrameStack`) used by trainer **and** both streamers | new shared env path |
| Action space | continuous `box` (done for classic control); image-box (CarRacing) next | `play_session`, `policy`, `trainer_*`, preview, keymaps |
| Competitive | 2-agent / `side` selector (Pong) | `play_session` + a 2-agent env |
| Board games | turn-based self-play (OpenSpiel) | a parallel subsystem, not a registry row |
