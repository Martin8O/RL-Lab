# Adding an algorithm

> Living document (Phase F4). The dashboard ships **six** learning methods — PPO (gradient, via
> Stable-Baselines3), a custom **neuroevolution** (population/fitness/mutation), tabular **Q-learning**
> (value-based, G2b), **AlphaZero-lite** (CNN policy+value + neural-guided MCTS self-play, board games
> only, G6f/ADR-055), **SAC** (off-policy continuous control, S5a), and **TD3** (off-policy continuous
> control, S5b/ADR-068) — as **peer trainers** behind one manager (ADR-004/028). A seventh (DQN, A2C…)
> plugs into the same seam. See also
> [`architecture.md`](architecture.md) and [`adding-an-environment.md`](adding-an-environment.md).

## The peer-trainer contract

`services/training_manager.py` owns the run lifecycle and routes by `config.algo` in `_run()`:

```python
if config.algo == "neuroevolution":
    terminal = train_evolution(config, gym_id, control,
                               emit_evolution, publish_predict, on_snapshot, resume)
elif config.algo == "q_learning":
    terminal = train_q_learning(config, gym_id, control,
                                emit_q_learning, emit_qtable, publish_predict, on_snapshot, resume)
else:
    terminal = train_ppo(config, gym_id, control,
                         emit_metrics, emit_progress, publish_predict, on_snapshot, resume)
```

A trainer is a **function** that runs the learning loop **on the manager's daemon thread** (never the
event loop — ADR-007) and returns a terminal `TrainState` (`finished` / `stopped` / `error`). It receives:

| Argument | Purpose |
|---|---|
| `config: TrainConfig` | env, seed, budget, and the algo's hyperparameters |
| `gym_id: str` | the resolved Gymnasium id from the registry |
| `control: TrainControl` | cooperative **pause** (park between steps on an `Event`) + **stop** (return early) |
| `emit_* callbacks` | push WS metric frames (see below) — marshalled to the loop by the manager |
| `publish_predict(fn)` | hand the preview a **decoupled** read-only policy (ADR-019) — never the live model |
| `on_snapshot(artifact)` | hand up a `CheckpointArtifact` (bytes) at a quiescent point, so a Save can persist it |
| `resume: bytes \| None` | a previously-saved artifact to continue from |

## Steps to add one

1. **Pick its metric frame.** Reuse `TrainingMetrics`/`TrainingProgress` (gradient methods) or
   `EvolutionMetrics` (population methods), or add a new frame to `schemas/training.py` **and**
   `frontend/src/api/types.ts` (one pydantic model + one TS type — contracts in one place). Add the new
   `type` to the WS union in [`api.md`](api.md).

2. **Write `services/trainer_<name>.py`.** Import torch/numpy **lazily inside the function** (like the two
   existing trainers) so `/health`, `/envs`, and the WS echo stay import-light. Run the loop; between
   steps, honor `control` (pause/stop); periodically call the `emit_*` callbacks.

3. **Publish a decoupled preview policy (load-bearing, not optional).** Call `publish_predict` with a
   *snapshot* forward — for PPO that's `_build_numpy_predict(model)` (a numpy forward over copied weights,
   rebuilt at each rollout boundary); for neuroevolution it's the generation's leader. **Do not** let the
   preview call into the live model — concurrent access measurably perturbs a same-seed run (ADR-019). The
   forward must be **action-space-agnostic** (`int` for `Discrete`, a clipped float vector for `Box`) per
   the ADR-021 contract.

4. **Emit checkpoint snapshots.** At quiescent points, serialize the model to bytes and call
   `on_snapshot(CheckpointArtifact(...))`; implement load-from-bytes for `resume`. This gives you
   Save/Load/Export and run-resume for free.

5. **Register it as data.** Add the algo id to each env's `supported_algos` in `registry.py` where it
   applies (this gates the algorithm dropdown per-env; the store snaps to a valid algo on env switch), and
   add its tunables to the env's `hyperparams` block (with `recommended` ★ values).

6. **Bilingual content + tests.** Add the algorithm's info-popup copy to `content/parameters.ts` (CZ+EN,
   general + per-env), wire any new i18n keys (both `en.json`/`cz.json` — `.\tasks.ps1 i18n` enforces
   parity), and add a smoke test that the trainer reaches a terminal state on CartPole. Then
   `.\tasks.ps1 all` must be green.

## Worked example — tabular Q-learning (G2b, ADR-028)

The third trainer, `services/trainer_q.py`, is the value-based peer and a concrete tour of the steps:

- **New frames (step 1).** Q-learning is *episodic*, not rollout/generation-based, so it added a
  `QLearningMetrics` frame (`type:"q_learning"`, x-axis = `episode`, `ep_rew_mean` is the learning curve)
  plus a separate **`QTableFrame`** (`type:"qtable"`) carrying the `[n_states × n_actions]` table for the
  live heatmap. The table frame is **decoupled and never logged** into the run's metric history (it is
  large — Taxi is 3 000 cells) and only the latest is retained on `TrainStatus.last_qtable` for reconnect.
- **Discrete-obs decode (step 3).** The shared `make_env` factory one-hot-wraps a `Discrete(n)` observation
  (ADR-024), so Q-learning's greedy preview/AI policy `argmax`-decodes the one-hot back to the int state,
  then `argmax`es the table row — the same action-space-agnostic predict→step contract (ADR-021).
- **Checkpoint (step 4).** Snapshots a numpy `qtable.npz`; resume continues the episode counter.
- **Data + gating (step 5).** `q_learning` is in `supported_algos` only on the discrete-obs Toy Text envs;
  its `hyperparams` block carries α / γ / ε-start / ε-end / ε-decay / episodes (per-env ★ episode budget).
- **UI for free + two small UI seams.** Reward chart, skill meter, checkpoints, run history and play-vs-AI
  all lit up from the shared contracts. Two deliberate additions: the **Q-table heatmap** panel (renders to
  one `<canvas>`, not per-cell DOM — ADR-029 — and takes over the Evolution-Stats + blank slots only for
  `q_learning`), and **algorithm-aware chart-tab messaging** (ADR-031): the empty-state hint reads from a
  single `algo → tabs` map (`ALGO_CHART_TABS`), so a tab the algorithm doesn't produce says so ("Q-learning
  doesn't use the Loss chart — see Reward") instead of a misleading "start training". Add a row there for
  any new algorithm and the messaging stays correct.

## Worked example — AlphaZero-lite (G6f, ADR-055)

The fourth trainer, `services/trainer_az.py`, is **board-only** and shows two ways the seam stretches:

- **Routed by algo *within a family*.** Board games (`is_board_game`) route to the board subsystem; the
  manager then sub-routes on `config.algo` — `"alphazero"` → `train_az`, else G6b's `train_board`. So a
  family carries **two trainers** and the user picks (`supported_algos = ["ppo", "alphazero"]`) — the first
  head-to-head algorithm comparison on one game.
- **Reuses an existing frame contract — no new frame.** AlphaZero's honest curve is the same
  eval-vs-reference-MCTS ∈ [−1, 1] as G6b, so it emits the standard `metrics` + `progress` frames and adds
  `alphazero: [reward, loss]` to `ALGO_CHART_TABS`. Its budget is **iterations × games_per_iter self-play
  games** (reported as `timesteps`), so the algo-aware sidebar shows AlphaZero's own sliders and **hides**
  the PPO Total-Steps ladder (same pattern as Q-learning's Episodes / evolution's Generations). A progress
  frame is emitted **per self-play game** so the chart advances smoothly between iterations.
- **PyTorch CNN + OpenSpiel MCTS, decoupled the ADR-019 way.** `services/az_net.py` is a CNN over the
  `observation_tensor` planes; inference (eval + Play) runs **neural-MCTS** (`az_move_fn`) for real strength
  while the live preview keeps a fast raw-policy snapshot — and `board_engine.eval_vs_mcts` was generalised
  to a `(state) -> action` move fn so both board trainers feed it.
- **CPU "lite" — and the scope lesson.** It pins `device="cpu"`: self-play is single-position (batch-of-1)
  forwards, which the CPU runs faster than the GPU (a GPU only pays off with **batched** self-play + a bigger
  net — measured 6–18× at batch 64–256). That batched-GPU engine is the **G6g (chess) foundation**, not part
  of this lite version.

## Worked example — SAC (S5a, ADR-067)

The fifth trainer, `services/trainer_sac.py`, is the first **off-policy** method and shows where the
shape stretches without forking the contract:

- **No rollout boundary ⇒ a step-interval cadence.** PPO emits a metrics frame per rollout; SAC collects
  single transitions into a replay buffer and updates continuously. So the metrics callback emits on a
  **step interval** (`_METRICS_INTERVAL_STEPS`) — snapshot + a fresh decoupled preview ride the same
  interval — and the shared ~1 Hz `_progress_ticker` (reused verbatim from `trainer_ppo`) keeps the live
  stats + reward curve smooth between them. No new frame: SAC emits the **standard `metrics` + `progress`**
  and adds `sac: ['reward', 'loss']` to `ALGO_CHART_TABS` (the Loss tab shows SAC's `train/critic_loss`).
- **Decoupled preview = a CPU save/load copy of the policy.** SAC's actor (squashed-Gaussian MLP) isn't
  the PPO `mlp_extractor.policy_net` + `action_net`, so the numpy forward doesn't apply; instead
  `_build_sac_predict` round-trips `model.policy` through SB3's `save`/`load` into an isolated CPU policy
  and calls `predict(deterministic=True)` (the CnnPolicy preview's trick, ADR-019) — never the live model.
- **Raw obs/rewards — explicitly NOT VecNormalize (the one cross-algorithm coordination).** Unlike the PPO
  MuJoCo path (G5c), SAC does not wrap the env in `VecNormalize`: reward normalization is on-policy-shaped
  (a running return scaling) and would drift against a replay buffer of rewards stamped with old stats, and
  SAC's standard recipe needs neither. So `ep_rew_mean` stays raw and `_sac_predict` (AI-play) applies no
  normalizer — a PPO-vs-SAC comparison on one robot is apples-to-apples on the same raw scale.
- **Data + gating (step 5).** `sac` is in `supported_algos` only on the continuous-`Box` envs (MuJoCo +
  BipedalWalker + Pendulum + MountainCarContinuous); its `hyperparams` block (lr/γ/τ/buffer/train_freq +
  the `auto`/fixed `ent_coef` categorical) rides on every env but is exposed only there — the same "block
  on all, exposed where listed" pattern as the `q_learning` block. SAC reuses the PPO `total_timesteps`
  step budget (so the sidebar shows the Total-Steps ladder for `ppo` **and** `sac`); device is `"cpu"`
  — its per-step gradient updates are tiny batch-256 MLP forwards that the CPU runs faster than a
  latency-bound GPU shuttle (measured: HalfCheetah CPU 163 vs CUDA 120 steps/s, the same ADR-056 result
  PPO's MlpPolicy has), so `api/device.trainsOnGpu` reads CPU for SAC too.

## Worked example — TD3 (S5b, ADR-068)

The sixth trainer, `services/trainer_td3.py`, shows how cheap a *sibling* algorithm is once the seam fits:
TD3 is SAC's off-policy twin, so it is a near-copy of `trainer_sac.py` and reuses everything that file does
(the PPO `metrics`/`progress` frames, the CPU save/load preview, raw obs / no-VecNormalize, the step-interval
cadence, `_ep_means`/`_progress_ticker`). Two things differ:

- **A deterministic policy ⇒ explore by injected noise, not entropy.** TD3 has no `ent_coef`; its actor is
  deterministic and its signature tricks (twin clipped critics, delayed policy updates, target-policy
  smoothing) are fixed SB3 defaults. So the one new tunable is **`train_noise`** — the std of the Gaussian
  `NormalActionNoise` added to collected actions, sized to the env's action dimension in `_td3_kwargs`. It is
  the conceptual analogue of SAC's entropy temperature and sits in the same sidebar slot.
- **It shares SAC's data, not just its shape.** TD3 lists the *same* `supported_algos` envs and re-uses the
  existing `EnvSpec.sac_total_timesteps` off-policy budget (no new field) — `budgetFor` and the sidebar treat
  `sac` and `td3` identically. `device` is `"cpu"` for the same ADR-056 reason as SAC.

**Off-policy live-curve gate (applies to both SAC and TD3).** Their ~1 Hz ticker fires within a few hundred
steps, when the episode buffer holds only one or two high-variance episodes (often a lucky random-warmup one),
which plotted as a misleading "starts high then dips". `_ep_means` gained a `min_episodes` param (default 1 for
PPO + snapshots) and the off-policy trainers pass **5** for their live chart frames, so the curve starts at the
settled baseline and climbs. Snapshots/checkpoints still record any available reward.

## What you get for free

The reward chart, progress ticker, skill meter, checkpoints, run history/compare, and play-vs-AI all read
the shared contracts — a peer trainer that emits the standard frames and publishes a decoupled predict fn
lights all of them up without UI-specific code.
