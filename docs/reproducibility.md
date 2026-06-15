# Reproducibility

> Living document (Phase F4). Reproducibility is a first-class constraint, not an afterthought: every
> training and evaluation path takes an **explicit seed** and records the **full config** of the run.

## The rule

Every training/eval path takes an explicit `seed`, and the run's complete configuration is recorded. A
finished run can be reproduced from its archived config alone.

## How it holds end-to-end

- **Config is explicit and echoed.** `TrainConfig` (env, algo, seed, budget, every hyperparameter) is the
  start request *and* is echoed back in `TrainStatus` — what you asked for is what ran.
- **Training is headless and off-thread (ADR-007).** SB3 `learn()` runs on a daemon thread with no
  rendering in the loop; the seed flows into the env and the model.
- **The preview can't change the result (ADR-008/019).** The live preview renders from a **decoupled
  policy snapshot** (a numpy forward over copied weights for PPO; the generation leader for
  neuroevolution) — never `model.predict` on the live model, which was proven to perturb a same-seed PPO
  trajectory. Turning the visual on/off does not alter a run.
- **Neuroevolution scores are reproducible per child.** Fitness is the **mean undiscounted return** over
  `episodes` evaluation episodes; each Top-K child surfaces the **deterministic env seed** it was scored
  with (instead of a meaningless γ/α), so any child's run can be re-created exactly (ADR-010).
- **Run archive (ADR-014).** Meaningful finished runs (those that reached ≥10% of the env's
  `solved_score`) are auto-archived under gitignored `data/runs/<id>/` as `meta` + `config` + `metrics`,
  and record **`solved_at`** — the x (timesteps for PPO, generation for neuroevolution) where the run
  first hit 100% of `solved_score`. This is the headline "steps-to-solve" metric and the basis of the
  compare overlay's "solved @" marker.

## The one deliberate exception: human play

Human play passes a **random** seed so that environments with a randomized scene (e.g. LunarLander's
procedurally-generated moon) vary each game — otherwise a person would replay the identical layout every
time. **AI** play keeps its configured seed, so an AI demo stays a reproducible demo. This is the only
place a random seed is intentional; see `CLAUDE.md` → Reproducibility and ADR-022.

## Reproduce a run

1. `GET /api/runs/{id}` → read the archived `config` (it is a full `TrainConfig`).
2. `POST /api/train/start` with that config.
3. Same seed + same config ⇒ the same trajectory (CPU; cross-machine float determinism caveats aside).

For a neuroevolution child specifically, take its `seed` from the leaderboard and evaluate the champion
checkpoint against that seed.
