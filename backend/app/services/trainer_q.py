"""Tabular Q-learning trainer — the 3rd peer algorithm (G2b), numpy-only on a worker thread.

The native consumer of a **discrete observation** (Toy Text): instead of a neural network it
keeps a plain ``[n_states × n_actions]`` table of action values and improves it with the
Bellman update after every step

    Q[s, a] += α · (r + γ · maxₐ' Q[s', a'] − Q[s, a])

while behaving ε-greedily (mostly the current-best action, sometimes a random one to explore).
The beginner literally *watches the table fill in* — the UI draws it as a heatmap.

The env is built through the shared factory, so its ``Discrete(n)`` observation is one-hot
wrapped exactly as for PPO/neuroevolution; Q-learning simply ``argmax``-decodes the one-hot back
to the integer state (so all three algorithms share one env path — see ``app.envs.factory``).

Pure numpy + gymnasium — no torch/SB3 — and imported lazily by the training manager so /health,
/envs and the WS echo stay fast to boot.

Reproducibility: every random draw (ε-greedy exploration, the per-episode env seed) comes from
one ``np.random.default_rng(seed)`` + a deterministic per-episode seed, so the same seed yields
the same table on CPU. ε anneals from ``epsilon_start`` to ``epsilon_end`` over the first
``epsilon_decay`` *fraction* of the episode budget, then holds — a budget-relative schedule that
behaves the same whether a game wants 3 000 or 20 000 episodes.
"""

import io
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from app.envs.factory import make_env
from app.schemas.training import (
    QLearningHyperparams,
    QLearningMetrics,
    QTable,
    QTableFrame,
    TrainConfig,
    TrainState,
)
from app.services.checkpoints import CheckpointArtifact
from app.services.train_control import TrainControl

MetricsSink = Callable[[QLearningMetrics], None]
QTableSink = Callable[[QTableFrame], None]
# The greedy table policy handed to the decoupled preview — predict(obs) -> int action.
PredictPublisher = Callable[[Callable[[object], int]], None]
SnapshotSink = Callable[[CheckpointArtifact], None]

_TARGET_REPORTS = 300  # aim for ~this many metric/table frames across a run (cadence = budget/this)


def _decode_state(obs: object) -> int:
    """Recover the integer state from the factory's one-hot observation (arg-max of the vector)."""
    return int(np.argmax(np.asarray(obs)))


def _epsilon(episode: int, total_episodes: int, hp: QLearningHyperparams) -> float:
    """ε for a (global) episode index: linear anneal start→end over the first ``epsilon_decay``
    fraction of the *total* episode budget, then hold at ``epsilon_end``. Budget-relative so the
    schedule has the same shape regardless of episode count — and keyed off the global episode (not
    a per-run offset) so a resumed checkpoint continues annealing instead of re-exploring from ε₀."""
    decay_episodes = max(1.0, hp.epsilon_decay * total_episodes)
    frac = min(1.0, episode / decay_episodes)
    return float(hp.epsilon_start + (hp.epsilon_end - hp.epsilon_start) * frac)


def _make_predict(table: np.ndarray) -> Callable[[object], int]:
    """A standalone greedy predict fn over a snapshot of the table — handed to the live preview
    and reused by AI play (see ``app.services.policy._q_learning_predict``)."""

    def predict(obs: object) -> int:
        return int(np.argmax(table[_decode_state(obs)]))

    return predict


def _qtable_frame(table: np.ndarray, episode: int, total_episodes: int) -> QTableFrame:
    """Build the WS heatmap frame from a table snapshot (rounded to keep the payload small)."""
    rows, cols = table.shape
    values = np.round(table, 4).tolist()
    return QTableFrame(
        episode=episode,
        total_episodes=total_episodes,
        table=QTable(n_states=int(rows), n_actions=int(cols), values=values),
    )


def _snapshot(
    table: np.ndarray, episode: int, total_episodes: int, reward: float | None, total_steps: int
) -> CheckpointArtifact:
    """Serialize the table to an in-memory ``qtable.npz`` for the checkpoint store.

    The saved ``episode`` is the one just finished, so resume continues at ``episode + 1`` from
    this table. Kept numpy-only so the manager/checkpoint store stay ML-free.
    """
    buf = io.BytesIO()
    np.savez(buf, qtable=table, episode=episode)
    return CheckpointArtifact(
        algo="q_learning",
        blob=buf.getvalue(),
        artifact_name="qtable.npz",
        reward=reward,
        timesteps=total_steps,
        total_timesteps=0,
        iteration=episode,  # episodes elapsed — the q_learning progress counter
        total_generations=total_episodes,  # reused as the run's episode budget (for the meta)
    )


def _load_table(resume_blob: bytes, n_states: int, n_actions: int) -> tuple[np.ndarray, int]:
    """Read a saved ``qtable.npz``; returns ``(table, start_episode)``.

    Raises ``ValueError`` if the table shape no longer matches the env (its state/action count
    changed) — the manager surfaces this as a clear error.
    """
    data = np.load(io.BytesIO(resume_blob))
    table = np.asarray(data["qtable"], dtype=np.float64)
    if table.shape != (n_states, n_actions):
        raise ValueError(
            "Checkpoint Q-table size does not match this environment — cannot resume."
        )
    return table, int(data["episode"])


def train_q_learning(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: MetricsSink,
    on_qtable: QTableSink,
    on_policy: PredictPublisher,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Learn a Q-table to completion (or until stopped). Returns the terminal state.

    Blocks the calling thread; the manager runs this off the event loop. Emits a
    :class:`QLearningMetrics` frame (the learning curve / stats) **and** a :class:`QTableFrame`
    (the live heatmap) every ``budget / ~300`` episodes plus a final frame, and publishes the
    greedy table policy each report so the decoupled preview renders the current policy.

    ``resume_blob`` continues from a saved ``qtable.npz`` (episode numbering + the RNG pick up
    from the checkpoint, deterministic from the run seed but not bit-identical to an uninterrupted
    run). ``on_snapshot`` receives the table each report so the checkpoint store can persist it.
    """
    hp = config.q_learning or QLearningHyperparams()

    # Shared factory: applies variant kwargs + the discrete-obs one-hot wrapper, so the obs is a
    # length-n one-hot vector we decode back to the integer state with arg-max.
    env: Any = make_env(config.env_id, gym_id)
    started_at = time.monotonic()
    total_steps = 0
    try:
        n_states = int(env.observation_space.shape[0])
        n_actions = int(env.action_space.n)

        if resume_blob is not None:
            table, start_episode = _load_table(resume_blob, n_states, n_actions)
            rng = np.random.default_rng(config.seed + start_episode)
        else:
            start_episode = 0
            rng = np.random.default_rng(config.seed)
            table = np.zeros((n_states, n_actions), dtype=np.float64)

        total_episodes = start_episode + hp.episodes
        report_every = max(1, hp.episodes // _TARGET_REPORTS)
        report_index = 0
        # Rolling batch of returns/lengths since the last report → the reported means.
        batch_rewards: list[float] = []
        batch_lengths: list[int] = []

        # Show the (empty) starting table + greedy policy immediately, before any learning.
        on_policy(_make_predict(table.copy()))
        on_qtable(_qtable_frame(table, start_episode, total_episodes))

        for local in range(1, hp.episodes + 1):
            episode = start_episode + local
            # Park here while paused; bail out promptly (≈ one episode) on stop.
            control.wait_if_paused()
            if control.stop_requested:
                return "stopped"

            eps = _epsilon(episode, total_episodes, hp)
            obs, _ = env.reset(seed=config.seed + episode)
            state = _decode_state(obs)
            done = False
            ep_reward = 0.0
            ep_len = 0
            while not done:
                if rng.random() < eps:
                    action = int(rng.integers(n_actions))
                else:
                    action = int(np.argmax(table[state]))
                obs, reward, terminated, truncated, _ = env.step(action)
                next_state = _decode_state(obs)
                # Bootstrap from the next state unless the episode *terminated* (goal/hole/cliff);
                # a time-limit *truncation* is not a true terminal, so it still bootstraps.
                future = 0.0 if terminated else float(np.max(table[next_state]))
                td_target = float(reward) + hp.gamma * future
                table[state, action] += hp.learning_rate * (td_target - table[state, action])
                state = next_state
                ep_reward += float(reward)
                ep_len += 1
                total_steps += 1
                done = bool(terminated or truncated)

            batch_rewards.append(ep_reward)
            batch_lengths.append(ep_len)

            if local % report_every == 0 or local == hp.episodes:
                report_index += 1
                ep_rew_mean = float(np.mean(batch_rewards)) if batch_rewards else None
                ep_len_mean = float(np.mean(batch_lengths)) if batch_lengths else None
                batch_rewards, batch_lengths = [], []

                # Publish the current greedy policy (a copy, so further learning can't race the
                # preview's reads) + the heatmap + the metric frame, then snapshot for Save/resume.
                on_policy(_make_predict(table.copy()))
                on_qtable(_qtable_frame(table, episode, total_episodes))
                on_metrics(
                    QLearningMetrics(
                        iteration=report_index,
                        episode=episode,
                        total_episodes=total_episodes,
                        epsilon=eps,
                        ep_rew_mean=ep_rew_mean,
                        ep_len_mean=ep_len_mean,
                        timesteps=total_steps,
                        elapsed=time.monotonic() - started_at,
                    )
                )
                if on_snapshot is not None:
                    on_snapshot(_snapshot(table, episode, total_episodes, ep_rew_mean, total_steps))
        return "finished"
    finally:
        env.close()
