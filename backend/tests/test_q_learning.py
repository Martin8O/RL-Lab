"""G2b — Tabular Q-learning trainer: episodic frames, table fills + learns, reproducibility,
stop, manager wiring, status reconcile, and the AI-play policy round-trip (one-hot decode)."""

import numpy as np
from app.envs.factory import make_env
from app.schemas.training import QLearningHyperparams, QLearningMetrics, QTableFrame, TrainConfig
from app.services.checkpoints import CheckpointMeta, LoadedCheckpoint
from app.services.connection_manager import manager
from app.services.policy import predict_from_checkpoint
from app.services.train_control import TrainControl
from app.services.trainer_q import train_q_learning
from app.services.training_manager import TrainingManager


def _tiny_config(env_id: str = "frozenlake_noslip", episodes: int = 1500, seed: int = 42) -> TrainConfig:
    return TrainConfig(
        env_id=env_id,
        algo="q_learning",
        seed=seed,
        q_learning=QLearningHyperparams(episodes=episodes),
    )


def _run(
    config: TrainConfig, control: TrainControl | None = None
) -> tuple[list[QLearningMetrics], list[QTableFrame], list]:
    metrics: list[QLearningMetrics] = []
    tables: list[QTableFrame] = []
    policies: list = []
    # gym_id is ignored for a registered env (make_env resolves it from the registry).
    train_q_learning(
        config, "FrozenLake-v1", control or TrainControl(),
        metrics.append, tables.append, policies.append,
    )
    return metrics, tables, policies


# -- trainer ----------------------------------------------------------------


def test_q_learning_streams_frames_and_fills_table() -> None:
    metrics, tables, policies = _run(_tiny_config(episodes=1500))
    assert len(metrics) > 0
    last = metrics[-1]
    assert last.episode == 1500
    assert last.total_episodes == 1500
    assert last.epsilon <= metrics[0].epsilon  # ε anneals down
    # An initial (empty) table frame + one per report.
    assert len(tables) == len(metrics) + 1
    table = np.array(tables[-1].table.values)
    assert table.shape == (16, 4)
    assert np.count_nonzero(table) > 0  # the table filled in
    assert callable(policies[-1])


def test_q_learning_actually_learns_noslip_frozenlake() -> None:
    """The greedy policy from the final table should solve the deterministic maze."""
    _metrics, _tables, policies = _run(_tiny_config(episodes=1500))
    predict = policies[-1]
    env = make_env("frozenlake_noslip")
    wins = 0
    for ep in range(10):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        r = 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(predict(obs))
            done = term or trunc
        wins += int(r > 0)
    env.close()
    assert wins == 10  # solves every episode


def test_q_learning_reproducible_with_same_seed() -> None:
    a, _, _ = _run(_tiny_config(seed=7))
    b, _, _ = _run(_tiny_config(seed=7))
    assert [m.ep_rew_mean for m in a] == [m.ep_rew_mean for m in b]
    assert [m.epsilon for m in a] == [m.epsilon for m in b]


def test_q_learning_resume_continues_epsilon_schedule() -> None:
    """Resuming a checkpoint must continue annealing ε (keyed off the *global* episode), not
    re-explore from ε_start — ε is budget-relative to the total, not the per-run offset."""
    cfg = _tiny_config(episodes=1000)
    snaps: list = []
    first: list[QLearningMetrics] = []
    train_q_learning(
        cfg, "FrozenLake-v1", TrainControl(),
        first.append, lambda _t: None, lambda _p: None, snaps.append,
    )
    assert first[-1].epsilon <= 0.1  # annealed to the ε_end floor by the end of the first run

    resumed: list[QLearningMetrics] = []
    train_q_learning(
        cfg, "FrozenLake-v1", TrainControl(),
        resumed.append, lambda _t: None, lambda _p: None, lambda _s: None, snaps[-1].blob,
    )
    assert resumed[0].episode > 1000  # episode numbering continues from the checkpoint
    # The resumed run is past the decay window → ε stays at the floor; it never jumps back to ε_start.
    assert max(f.epsilon for f in resumed) <= 0.2


def test_q_learning_stop_aborts_promptly() -> None:
    control = TrainControl()
    seen: list[QLearningMetrics] = []

    def sink(m: QLearningMetrics) -> None:
        seen.append(m)
        control.request_stop()

    terminal = train_q_learning(
        _tiny_config(episodes=50_000), "FrozenLake-v1", control,
        sink, lambda _t: None, lambda _p: None,
    )
    assert terminal == "stopped"


# -- AI-play policy round-trip ----------------------------------------------


def test_q_learning_checkpoint_predicts_via_one_hot() -> None:
    """A saved qtable.npz reloads into a greedy predict fn that decodes the one-hot obs the play
    env produces (the discrete-obs seam for AI play)."""
    config = _tiny_config(episodes=1500)
    snaps: list = []
    train_q_learning(
        config, "FrozenLake-v1", TrainControl(),
        lambda _m: None, lambda _t: None, lambda _p: None, snaps.append,
    )
    snap = snaps[-1]
    assert snap.artifact_name == "qtable.npz"
    meta = CheckpointMeta(
        id="x", label="l", env_id=config.env_id, algo="q_learning", seed=config.seed,
        created_at="now", artifact="qtable.npz", iteration=snap.iteration,
    )
    predict = predict_from_checkpoint(LoadedCheckpoint(meta=meta, config=config, blob=snap.blob))
    env = make_env("frozenlake_noslip")  # one-hot wrapped, like the play session
    obs, _ = env.reset(seed=1000)
    action = predict(obs)
    assert isinstance(action, int) and 0 <= action < 4
    env.close()


# -- manager ----------------------------------------------------------------


def test_manager_routes_to_q_learning_and_stops_clean() -> None:
    mgr = TrainingManager(manager)  # no loop bound → broadcasts are skipped
    status = mgr.start(_tiny_config(episodes=200_000))  # long enough to still be running
    try:
        assert status.state == "running"
        assert status.algo == "q_learning"
    finally:
        mgr.stop()
        mgr.join(timeout=30)
    assert mgr.status().state == "stopped"


def test_status_retains_last_q_learning_and_table() -> None:
    mgr = TrainingManager(manager)
    mgr.start(_tiny_config(episodes=1500))
    mgr.join(timeout=60)

    status = mgr.status()
    assert status.state == "finished"
    assert status.last_q_learning is not None
    assert status.last_q_learning.episode == 1500
    assert status.last_qtable is not None
    assert status.last_qtable.table.n_states == 16
    # Survives serialization (what /api/train/status returns to a reconnecting client).
    dumped = status.model_dump()
    assert dumped["last_q_learning"]["type"] == "q_learning"
    assert dumped["last_qtable"]["type"] == "qtable"

    # A fresh run clears the previous snapshot until its first report lands.
    mgr2 = TrainingManager(manager)
    running = mgr2.start(_tiny_config(episodes=200_000))
    try:
        assert running.last_q_learning is None
        assert running.last_qtable is None
    finally:
        mgr2.stop()
        mgr2.join(timeout=30)
