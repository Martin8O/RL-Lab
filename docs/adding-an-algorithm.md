# Adding an algorithm

> Living document (Phase F4). The dashboard ships **two** learning methods — PPO (gradient, via
> Stable-Baselines3) and a custom **neuroevolution** (population/fitness/mutation) — as **peer trainers**
> behind one manager (ADR-004). A third (e.g. tabular Q-learning, DQN) plugs into the same seam. See also
> [`architecture.md`](architecture.md) and [`adding-an-environment.md`](adding-an-environment.md).

## The peer-trainer contract

`services/training_manager.py` owns the run lifecycle and routes by `config.algo` in `_run()`:

```python
if config.algo == "neuroevolution":
    terminal = train_evolution(config, gym_id, control,
                               emit_evolution, publish_predict, on_snapshot, resume)
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

## What you get for free

The reward chart, progress ticker, skill meter, checkpoints, run history/compare, and play-vs-AI all read
the shared contracts — a peer trainer that emits the standard frames and publishes a decoupled predict fn
lights all of them up without UI-specific code.
