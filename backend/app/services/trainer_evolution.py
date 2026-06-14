"""Neuroevolution trainer for CartPole — numpy MLP genomes on a background thread.

The cookbook's "200 cars" idea as a peer to the PPO trainer: a population of small,
fixed-topology MLPs is scored by playing episodes, the Top-K become parents, and the next
generation is bred by crossover + Gaussian mutation with the single best genome carried
over unchanged (elitism). Pure numpy + gymnasium — no torch/SB3 — and imported lazily by
the training manager so /health, /envs and the WS echo stay fast to boot.

Reproducibility: every random draw (population init, parent picks, crossover masks,
mutation noise) comes from one ``np.random.default_rng(seed)``; every evaluation episode is
seeded deterministically from the run seed. Same seed ⇒ identical generations on CPU.

Fitness is the *undiscounted* mean episode return, so "solved ≈ 500" is literal for
CartPole-v1 — selection uses no γ. (Surfacing a γ per child would only mislead the beginner
audience this tool is for, so the leaderboard reports the reproducible per-child seed.)
"""

import io
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from app.schemas.training import (
    EvolutionChild,
    EvolutionHyperparams,
    EvolutionMetrics,
    MutationDist,
    TrainConfig,
    TrainState,
)
from app.services.checkpoints import CheckpointArtifact
from app.services.train_control import TrainControl

EvolutionSink = Callable[[EvolutionMetrics], None]
# The predict fn returns an int (discrete) or a numpy action vector (continuous box) — Any.
PredictPublisher = Callable[[Callable[[object], Any]], None]
SnapshotSink = Callable[[CheckpointArtifact], None]

_HIDDEN = 16  # single hidden layer — ample for CartPole, fast to evolve
_INIT_SCALE = 0.5  # std of the initial random weights
_MUT_HIST_BINS = 21  # bins in the per-generation mutation-distribution histogram
_TOP_CHILDREN = 5  # leaderboard size emitted each generation


class _Policy:
    """A flat weight vector viewed as a 2-layer tanh MLP (obs → hidden → action head).

    Discrete envs (``act_low``/``act_high`` is ``None``) read the head as action logits and pick
    the arg-max button. Continuous (box) envs squash the head through ``tanh`` and affine-map it
    into ``[act_low, act_high]``, so the genome emits a real-valued action vector that is always
    in range — the continuous-action seam (G1b) for neuroevolution.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden: int,
        act_dim: int,
        flat: np.ndarray,
        act_low: np.ndarray | None = None,
        act_high: np.ndarray | None = None,
    ) -> None:
        s1 = obs_dim * hidden
        s2 = s1 + hidden
        s3 = s2 + hidden * act_dim
        self.w1 = flat[:s1].reshape(obs_dim, hidden)
        self.b1 = flat[s1:s2]
        self.w2 = flat[s2:s3].reshape(hidden, act_dim)
        self.b2 = flat[s3:]
        self.act_low = None if act_low is None else np.asarray(act_low, dtype=np.float64)
        self.act_high = None if act_high is None else np.asarray(act_high, dtype=np.float64)

    def act(self, obs: np.ndarray) -> Any:
        hidden = np.tanh(obs @ self.w1 + self.b1)
        out = hidden @ self.w2 + self.b2
        low, high = self.act_low, self.act_high
        if low is None or high is None:  # discrete: pick the highest-scoring action
            return int(np.argmax(out))
        # continuous: tanh ∈ [-1, 1] → affine into [low, high]; always a valid in-bounds action
        scaled = low + (np.tanh(out) + 1.0) * 0.5 * (high - low)
        return scaled.astype(np.float32)


def _genome_size(obs_dim: int, hidden: int, act_dim: int) -> int:
    return obs_dim * hidden + hidden + hidden * act_dim + act_dim


def _child_seed(seed: int, generation: int, rank: int) -> int:
    """A unique, deterministic env seed per (generation, genome) for reproducible scoring."""
    return seed + generation * 100_000 + rank * 100


def _evaluate(
    flat: np.ndarray,
    env: Any,
    obs_dim: int,
    act_dim: int,
    episodes: int,
    base_seed: int,
    act_low: np.ndarray | None = None,
    act_high: np.ndarray | None = None,
) -> tuple[float, int]:
    """Play ``episodes`` episodes with this genome; return (total_reward, total_steps)."""
    policy = _Policy(obs_dim, _HIDDEN, act_dim, flat, act_low=act_low, act_high=act_high)
    total_reward = 0.0
    total_steps = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy.act(obs))
            total_reward += float(reward)
            total_steps += 1
            done = bool(terminated or truncated)
    return total_reward, total_steps


def _make_predict(net: _Policy) -> Callable[[object], Any]:
    """A standalone predict fn over a snapshot genome — handed to the live preview."""

    def predict(obs: object) -> Any:
        return net.act(np.asarray(obs, dtype=np.float64))

    return predict


def _breed(
    population: np.ndarray,
    order: np.ndarray,
    hp: EvolutionHyperparams,
    rng: np.random.Generator,
    dim: int,
) -> tuple[np.ndarray, MutationDist]:
    """Produce the next generation and a histogram of every mutation perturbation applied.

    Elitism keeps the best genome unchanged (so best-fitness can't regress on a stable
    policy); the rest are bred from the Top-K parents via uniform crossover then Gaussian
    mutation.
    """
    top_k = max(1, min(hp.top_k_parents, hp.population_size))
    parents = population[order[:top_k]]

    next_pop = np.empty_like(population)
    next_pop[0] = population[order[0]]  # elitism: carry the best over unchanged
    noise_acc: list[np.ndarray] = []
    for i in range(1, hp.population_size):
        if top_k >= 2 and rng.random() < hp.crossover_rate:
            a = parents[rng.integers(top_k)]
            b = parents[rng.integers(top_k)]
            child = np.where(rng.random(dim) < 0.5, a, b)
        else:
            child = parents[rng.integers(top_k)].copy()
        noise = rng.normal(0.0, hp.mutation_rate, dim)
        next_pop[i] = child + noise
        noise_acc.append(noise)

    edge = 3.0 * hp.mutation_rate if hp.mutation_rate > 0 else 1.0
    edges = np.linspace(-edge, edge, _MUT_HIST_BINS + 1)
    flat_noise = np.concatenate(noise_acc) if noise_acc else np.zeros(1)
    counts, _ = np.histogram(flat_noise, bins=edges)
    return next_pop, MutationDist(bins=edges.tolist(), counts=counts.tolist())


def _snapshot(
    population: np.ndarray,
    obs_dim: int,
    act_dim: int,
    generation: int,
    total_generations: int,
    best_fitness: float,
    total_steps: int,
    act_low: np.ndarray | None = None,
    act_high: np.ndarray | None = None,
) -> CheckpointArtifact:
    """Serialize the bred population to an in-memory ``population.npz`` for the store.

    The saved ``generation`` is the one that just finished, so resume continues at
    ``generation + 1`` evaluating this already-bred population. ``population[0]`` is the
    elite (champion) carried over by :func:`_breed`. For a continuous (box) env the action
    bounds are stored too, so :func:`app.services.policy._evolution_predict` can rebuild a
    champion that squashes into range; discrete checkpoints omit them (back-compatible).
    """
    buf = io.BytesIO()
    if act_low is not None and act_high is not None:  # continuous: also store the action bounds
        np.savez(
            buf, population=population, obs_dim=obs_dim, act_dim=act_dim,
            hidden=_HIDDEN, generation=generation, act_low=act_low, act_high=act_high,
        )
    else:  # discrete: bounds omitted (back-compatible with pre-G1b checkpoints)
        np.savez(
            buf, population=population, obs_dim=obs_dim, act_dim=act_dim,
            hidden=_HIDDEN, generation=generation,
        )
    return CheckpointArtifact(
        algo="neuroevolution",
        blob=buf.getvalue(),
        artifact_name="population.npz",
        reward=best_fitness,
        timesteps=total_steps,
        total_timesteps=0,
        generation=generation,
        total_generations=total_generations,
    )


def _load_population(resume_blob: bytes, dim: int) -> tuple[np.ndarray, int]:
    """Read a saved ``population.npz``; returns ``(population, start_generation)``.

    Raises ``ValueError`` if the genome width no longer matches the env (e.g. the env's
    observation/action space changed) — the manager surfaces this as a clear error.
    """
    data = np.load(io.BytesIO(resume_blob))
    population = np.asarray(data["population"], dtype=np.float64)
    if population.ndim != 2 or population.shape[1] != dim:
        raise ValueError(
            "Checkpoint genome size does not match this environment — cannot resume."
        )
    return population, int(data["generation"])


def train_evolution(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: EvolutionSink,
    on_policy: PredictPublisher,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Evolve a population to completion (or until stopped). Returns the terminal state.

    Blocks the calling thread; the manager runs this off the event loop. Emits one
    :class:`EvolutionMetrics` frame per generation. ``on_policy`` publishes each
    generation's best genome so the decoupled preview streamer can render the leader.

    ``resume_blob`` continues from a saved ``population.npz`` (generation numbering, child
    ids and seeds pick up where the checkpoint left off; the RNG is re-seeded deterministically
    from the run seed + that generation, so a resumed run is reproducible from the checkpoint
    but not bit-identical to an uninterrupted one). ``on_snapshot`` receives the bred
    population each generation so the checkpoint store can persist it.
    """
    import gymnasium as gym  # lazy: keep gym out of startup

    hp = config.evolution or EvolutionHyperparams()

    env: Any = gym.make(gym_id)
    started_at = time.monotonic()
    total_steps = 0
    try:
        obs_dim = int(env.observation_space.shape[0])
        # Discrete envs expose `.n` (one of N buttons); Box (continuous) envs expose `.low`/`.high`
        # and a `.shape` — the genome then emits `act_dim` real numbers squashed into those bounds.
        action_space = env.action_space
        if getattr(action_space, "n", None) is not None:
            act_dim = int(action_space.n)
            act_low: np.ndarray | None = None
            act_high: np.ndarray | None = None
        else:
            act_dim = int(action_space.shape[0])
            act_low = np.asarray(action_space.low, dtype=np.float64)
            act_high = np.asarray(action_space.high, dtype=np.float64)
        dim = _genome_size(obs_dim, _HIDDEN, act_dim)

        if resume_blob is not None:
            population, start_generation = _load_population(resume_blob, dim)
            rng = np.random.default_rng(config.seed + start_generation)
        else:
            start_generation = 0
            rng = np.random.default_rng(config.seed)
            population = rng.standard_normal((hp.population_size, dim)) * _INIT_SCALE

        total_generations = start_generation + hp.generations

        for local in range(1, hp.generations + 1):
            generation = start_generation + local
            avg_rewards = np.empty(hp.population_size)
            totals = np.empty(hp.population_size)
            steps = np.empty(hp.population_size, dtype=int)
            seeds = np.empty(hp.population_size, dtype=int)
            for idx in range(hp.population_size):
                # Park here while paused; bail out promptly (≈ one genome) on stop.
                control.wait_if_paused()
                if control.stop_requested:
                    return "stopped"
                base_seed = _child_seed(config.seed, generation, idx)
                total_reward, ep_steps = _evaluate(
                    population[idx], env, obs_dim, act_dim, hp.episodes, base_seed,
                    act_low=act_low, act_high=act_high,
                )
                totals[idx] = total_reward
                avg_rewards[idx] = total_reward / hp.episodes
                steps[idx] = ep_steps
                seeds[idx] = base_seed
                total_steps += ep_steps

            order = np.argsort(avg_rewards)[::-1]

            # Publish the generation's best genome (a snapshot copy, so the next
            # generation's mutation can't race the preset live preview reads).
            best_net = _Policy(
                obs_dim, _HIDDEN, act_dim, population[order[0]].copy(),
                act_low=act_low, act_high=act_high,
            )
            on_policy(_make_predict(best_net))

            children = [
                EvolutionChild(
                    id=(generation - 1) * hp.population_size + rank,
                    total_reward=float(totals[gi]),
                    avg_reward=float(avg_rewards[gi]),
                    steps=int(steps[gi]),
                    seed=int(seeds[gi]),
                )
                for rank, gi in enumerate(order[:_TOP_CHILDREN])
            ]

            best_fitness = float(avg_rewards[order[0]])
            population, mutation_dist = _breed(population, order, hp, rng, dim)

            on_metrics(
                EvolutionMetrics(
                    generation=generation,
                    total_generations=total_generations,
                    best_fitness=best_fitness,
                    avg_fitness=float(avg_rewards.mean()),
                    worst_fitness=float(avg_rewards[order[-1]]),
                    children=children,
                    mutation_dist=mutation_dist,
                    timesteps=total_steps,
                    elapsed=time.monotonic() - started_at,
                )
            )
            # Snapshot the bred population so "Save" can persist (and later resume) this run.
            if on_snapshot is not None:
                on_snapshot(
                    _snapshot(
                        population, obs_dim, act_dim, generation,
                        total_generations, best_fitness, total_steps,
                        act_low=act_low, act_high=act_high,
                    )
                )
        return "finished"
    finally:
        env.close()
