"""C1 — Neuroevolution trainer: per-generation frames, reproducibility, stop, manager wiring."""

from app.schemas.training import EvolutionHyperparams, EvolutionMetrics, TrainConfig
from app.services.connection_manager import manager
from app.services.train_control import TrainControl
from app.services.trainer_evolution import train_evolution
from app.services.training_manager import TrainingManager


def _tiny_config(generations: int = 3, seed: int = 42) -> TrainConfig:
    """A small, fast CartPole evolution run that finishes in well under a second on CPU."""
    return TrainConfig(
        env_id="cartpole",
        algo="neuroevolution",
        seed=seed,
        evolution=EvolutionHyperparams(
            population_size=8,
            top_k_parents=3,
            mutation_rate=0.1,
            crossover_rate=0.5,
            generations=generations,
            episodes=1,
        ),
    )


def _run(config: TrainConfig, control: TrainControl | None = None) -> list[EvolutionMetrics]:
    seen: list[EvolutionMetrics] = []
    train_evolution(
        config, "CartPole-v1", control or TrainControl(), seen.append, lambda _predict: None
    )
    return seen


# -- trainer ----------------------------------------------------------------


def test_evolution_streams_every_generation() -> None:
    frames = _run(_tiny_config(generations=3))
    assert len(frames) == 3
    for gen, f in enumerate(frames, start=1):
        assert f.generation == gen
        assert f.total_generations == 3
        assert f.best_fitness >= f.avg_fitness >= f.worst_fitness
        assert 1 <= len(f.children) <= 5
        assert sum(f.mutation_dist.counts) > 0
        assert len(f.mutation_dist.bins) == len(f.mutation_dist.counts) + 1
    # Cumulative env steps grow across generations.
    assert frames[0].timesteps < frames[-1].timesteps


def test_evolution_reproducible_with_same_seed() -> None:
    first, second = _run(_tiny_config(seed=7)), _run(_tiny_config(seed=7))
    assert [f.best_fitness for f in first] == [f.best_fitness for f in second]
    assert [c.id for f in first for c in f.children] == [c.id for f in second for c in f.children]
    assert [c.avg_reward for f in first for c in f.children] == [
        c.avg_reward for f in second for c in f.children
    ]


def test_evolution_stop_aborts_after_current_generation() -> None:
    control = TrainControl()
    seen: list[EvolutionMetrics] = []

    def sink(m: EvolutionMetrics) -> None:
        seen.append(m)
        control.request_stop()  # stop right after the first generation

    terminal = train_evolution(
        _tiny_config(generations=5), "CartPole-v1", control, sink, lambda _p: None
    )
    assert terminal == "stopped"
    assert len(seen) == 1


# -- manager ----------------------------------------------------------------


def test_manager_routes_to_evolution_and_stops_clean() -> None:
    mgr = TrainingManager(manager)  # no loop bound → broadcasts are skipped
    status = mgr.start(_tiny_config(generations=1000))  # long enough to still be running
    try:
        assert status.state == "running"
        assert status.algo == "neuroevolution"
    finally:
        mgr.stop()
        mgr.join(timeout=30)
    assert mgr.status().state == "stopped"
