"""X1 — canonical comparison axes (env_steps + wall_clock) on every persisted metric frame.

Two guarantees the analysis suite (Phase X / DataLab) is built on:
  1. Every *logged* frame model auto-carries ``env_steps`` (cumulative env interactions) + ``wall_clock``
     (elapsed seconds), filled from the frame's ``timesteps`` / ``elapsed`` — unless a trainer whose
     ``timesteps`` is not env steps (AlphaZero) sets ``env_steps`` explicitly, in which case it is kept.
  2. A finished run round-trips through the store with those axes present + monotonic, and a *legacy*
     run (frames recorded before this contract) is backfilled on read (no archive mutation).
"""

from pathlib import Path

from app.schemas.training import (
    EvolutionChild,
    EvolutionMetrics,
    MultiAgentMetrics,
    MutationDist,
    QLearningMetrics,
    SpeciesMetrics,
    TrainConfig,
    TrainingMetrics,
)
from app.services.runs import RunStore, backfill_axes


def _store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "runs")


# -- schema auto-fill -------------------------------------------------------


def test_training_metrics_autofills_axes_from_timesteps_and_elapsed() -> None:
    m = TrainingMetrics(
        iteration=2, timesteps=4096, total_timesteps=50_000, ep_rew_mean=120.0,
        ep_len_mean=200.0, loss=0.3, learning_rate=3e-4, elapsed=12.5,
    )
    # env_steps == timesteps and wall_clock == elapsed for the step-based trainers.
    assert m.env_steps == 4096
    assert m.wall_clock == 12.5
    assert m.model_dump()["env_steps"] == 4096  # rides the serialized frame (→ metrics.json)


def test_training_metrics_keeps_explicit_env_steps_for_alphazero() -> None:
    # AlphaZero's timesteps is self-play GAMES; it passes env_steps = cumulative plies explicitly.
    m = TrainingMetrics(
        iteration=3, timesteps=90, total_timesteps=720, env_steps=1873, ep_rew_mean=0.4,
        ep_len_mean=None, loss=0.1, learning_rate=5e-4, elapsed=40.0,
    )
    assert m.env_steps == 1873 != m.timesteps  # the explicit ply count wins, not the games count
    assert m.wall_clock == 40.0


def test_evolution_metrics_autofills_axes() -> None:
    ev = EvolutionMetrics(
        generation=2, total_generations=3, best_fitness=500.0, avg_fitness=100.0,
        worst_fitness=-10.0, children=[EvolutionChild(id=1, total_reward=1.0, avg_reward=1.0, steps=5, seed=1)],
        mutation_dist=MutationDist(bins=[0.0, 1.0], counts=[3]), timesteps=12_000, elapsed=8.0,
    )
    assert ev.env_steps == 12_000 and ev.wall_clock == 8.0


def test_q_learning_metrics_autofills_axes() -> None:
    q = QLearningMetrics(
        iteration=5, episode=2500, total_episodes=5000, epsilon=0.3, ep_rew_mean=0.6,
        ep_len_mean=12.0, timesteps=41_234, elapsed=3.2,
    )
    assert q.env_steps == 41_234 and q.wall_clock == 3.2


def test_multi_agent_metrics_autofills_axes() -> None:
    ma = MultiAgentMetrics(
        round=2, total_rounds=8, learning_role="adversary",
        species=[SpeciesMetrics(role="adversary", ep_rew_mean=1.0, ep_len_mean=25.0, timesteps=5000)],
        ep_rew_mean=1.0, timesteps=10_000, total_timesteps=80_000, elapsed=15.0,
    )
    assert ma.env_steps == 10_000 and ma.wall_clock == 15.0


# -- loader backfill (pure) -------------------------------------------------


def test_backfill_adds_axes_to_legacy_frames() -> None:
    legacy = [
        {"type": "metrics", "iteration": 1, "timesteps": 2048, "ep_rew_mean": 30.0, "elapsed": 1.0},
        {"type": "metrics", "iteration": 2, "timesteps": 4096, "ep_rew_mean": 90.0, "elapsed": 2.0},
    ]
    out = backfill_axes(legacy)
    assert [f["env_steps"] for f in out] == [2048, 4096]
    assert [f["wall_clock"] for f in out] == [1.0, 2.0]


def test_backfill_is_idempotent_and_preserves_explicit_env_steps() -> None:
    # A post-X1 AZ frame carries env_steps (plies) ≠ timesteps (games); backfill must NOT clobber it.
    frames = [{"type": "metrics", "timesteps": 90, "env_steps": 1873, "elapsed": 40.0, "wall_clock": 40.0}]
    out = backfill_axes(frames)
    assert out[0]["env_steps"] == 1873 and out[0]["wall_clock"] == 40.0


# -- store round-trip: monotonic env_steps ----------------------------------


def test_ppo_run_roundtrips_with_monotonic_env_steps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="ppo", seed=7)
    frames = [
        {"type": "metrics", "iteration": 1, "timesteps": 2048, "ep_rew_mean": 30.0, "elapsed": 1.0},
        {"type": "metrics", "iteration": 2, "timesteps": 4096, "ep_rew_mean": 500.0, "elapsed": 2.0},
        {"type": "metrics", "iteration": 3, "timesteps": 6144, "ep_rew_mean": 500.0, "elapsed": 3.0},
    ]
    meta = store.save(cfg, frames, state="finished", started_at="2026-07-02T10:00:00+00:00", solved_score=500.0)
    detail = store.get(meta.id)
    assert detail is not None
    steps = [f["env_steps"] for f in detail.metrics]
    assert steps == [2048, 4096, 6144]
    assert all(b >= a for a, b in zip(steps, steps[1:], strict=False))  # monotonic non-decreasing
    assert all("wall_clock" in f for f in detail.metrics)


def test_evolution_run_roundtrips_with_env_steps(tmp_path: Path) -> None:
    store = _store(tmp_path)
    cfg = TrainConfig(env_id="cartpole", algo="neuroevolution", seed=3)
    frames = [
        {"type": "evolution", "generation": 1, "total_generations": 2, "best_fitness": 120.0,
         "timesteps": 6000, "elapsed": 4.0},
        {"type": "evolution", "generation": 2, "total_generations": 2, "best_fitness": 500.0,
         "timesteps": 12000, "elapsed": 8.0},
    ]
    meta = store.save(cfg, frames, state="stopped", started_at="2026-07-02T11:00:00+00:00", solved_score=500.0)
    detail = store.get(meta.id)
    assert detail is not None
    assert [f["env_steps"] for f in detail.metrics] == [6000, 12000]
