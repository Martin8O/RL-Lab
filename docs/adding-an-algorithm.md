# Adding an algorithm

> Living document (Phase F4). The dashboard ships **three** learning methods — PPO (gradient, via
> Stable-Baselines3), a custom **neuroevolution** (population/fitness/mutation), and tabular
> **Q-learning** (value-based, G2b) — as **peer trainers** behind one manager (ADR-004/028). A fourth
> (DQN, SAC…) plugs into the same seam. See also [`architecture.md`](architecture.md) and
> [`adding-an-environment.md`](adding-an-environment.md).

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

## What you get for free

The reward chart, progress ticker, skill meter, checkpoints, run history/compare, and play-vs-AI all read
the shared contracts — a peer trainer that emits the standard frames and publishes a decoupled predict fn
lights all of them up without UI-specific code.
