"""AlphaZero-lite board trainer — CNN policy+value + neural-guided MCTS self-play (G6f, ADR-055).

The 4th learning algorithm and the *algorithm jump* of the board branch. Where G6b's
:mod:`app.services.trainer_board` learns a board game by playing the G6a MCTS *teacher* with
MaskablePPO, this trainer learns purely by **self-play**: a PyTorch CNN (policy + value) guides
OpenSpiel's MCTS (:mod:`app.services.az_net`), and the search's visit counts + game outcomes train the
net. No teacher, no human data — the AlphaZero recipe, scaled down to small boards and a tolerable GPU
budget so it can be **compared head-to-head with the MaskablePPO baseline on the same game**.

It keeps the ``trainer_board`` *shape* — a custom trainer the manager routes to (on
``is_board_game(spec) and algo=="alphazero"``), a decoupled CPU snapshot published to the live preview
(ADR-019), a packed checkpoint Save/Load round-trips — and reuses the board subsystem wholesale: the
honest learning curve is **eval-vs-reference-MCTS ∈ [−1, 1]** scored against the *same*
``board_engine.board_profile`` reference as the PPO baseline (a fair apples-to-apples yardstick),
reported as ``ep_rew_mean`` on the existing ``metrics`` + ``progress`` frames, and the trained net
exposes the same ``(obs, mask) -> action`` predict shape so the live preview, Play-vs-net and Save/Load
lanes are unchanged. The only AZ-specific code is here + :mod:`app.services.az_net`.

GPU/CUDA when available (the G4b device path); ``torch`` loads lazily via ``az_net``.
"""

import time
from collections import deque
from collections.abc import Callable
from typing import Any

from app.schemas.training import (
    AlphaZeroHyperparams,
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
)
from app.services import az_net, board_engine
from app.services.checkpoints import CheckpointArtifact
from app.services.train_control import TrainControl

MetricsSink = Callable[[TrainingMetrics], None]
ProgressSink = Callable[[TrainingProgress], None]
PredictPublisher = Callable[[Callable[[object], Any]], None]
SnapshotSink = Callable[[CheckpointArtifact], None]

# Games per eval — each eval move runs neural-MCTS (eval_simulations forwards), so this is kept modest
# to keep the per-iteration eval to a few seconds, but large enough to steady the noisy ±1 board metric.
_EVAL_GAMES = 18


def _dirichlet_alpha(n_actions: int) -> float:
    """Root-exploration noise scale ≈ 10 / typical-branching: looser for low-branching games (Connect
    Four ~7 moves → 1.0), tighter for high-branching ones (Breakthrough/Othello → 0.3)."""
    return 1.0 if n_actions < 50 else 0.3


def train_az(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: MetricsSink,
    on_progress: ProgressSink | None = None,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train a board-game net by AlphaZero-lite self-play (CNN + neural-guided MCTS).

    Blocks the calling thread; the manager runs this off the event loop. ``gym_id`` is the OpenSpiel
    short name (e.g. ``"connect_four"``). Each iteration generates ``games_per_iter`` self-play games,
    trains the net on the replay buffer, evaluates the snapshot vs the profile's fixed reference MCTS,
    and emits **both** a ``metrics`` frame (Loss tab) and a ``progress`` frame (Reward tab) carrying the
    eval-vs-MCTS score as ``ep_rew_mean`` — the same honest curve as the PPO board trainer, on the same
    yardstick. It publishes a decoupled preview policy + packs an ``board.zip``-equivalent AZ checkpoint.
    Resumes from a saved AZ blob (continuing the games counter; the budget is another full schedule).
    """
    import numpy as np
    import torch

    hp: AlphaZeroHyperparams = config.alphazero or AlphaZeroHyperparams()
    game = board_engine.load_game(gym_id)
    # CPU on purpose, even on a CUDA box: AlphaZero self-play is thousands of **single-position** MCTS
    # forwards on a small CNN over a tiny board — latency-bound, batch-of-1 work where a GPU's per-call
    # launch + host↔device transfer overhead makes it *slower* than the CPU (measured ~2× on Connect
    # Four: 17 ms/move CPU vs 35 ms/move GPU). The GPU only pays off with a big net + **batched**
    # self-play (a future chess-scale build), so `az_net.best_device()` is reserved for that.
    device = "cpu"
    started_at = time.monotonic()

    # Same reference MCTS the PPO baseline is scored against (board_engine.board_profile, G6c), so the
    # two algorithms' learning curves are directly comparable. AZ has no teacher — it self-plays.
    eval_sims = board_engine.STRENGTH_SIMS[board_engine.board_profile(gym_id).eval_strength]
    dir_alpha = _dirichlet_alpha(int(game.num_distinct_actions()))

    if resume_blob is not None:
        model, games_done = az_net.build_model_from_blob(resume_blob, game, device=device)
    else:
        model = az_net.AZModel(game, channels=hp.channels, blocks=hp.blocks, device=device)
        games_done = 0
    model.net.eval()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=hp.learning_rate, weight_decay=1e-4)

    buffer: deque[tuple[Any, Any, float]] = deque(maxlen=hp.buffer_size)
    rng = np.random.default_rng(config.seed)
    games_per_iter = max(1, hp.games_per_iter)
    # Budget in self-play games — the AZ analogue of "Total Steps" (reported as timesteps below). On
    # resume, continue the counter and run another full `iterations` schedule on top.
    total_target = games_done + hp.iterations * games_per_iter
    iteration = [0]  # the iteration index → the metrics frame's "iteration"
    last_eval = [0.0]
    should_stop = lambda: control.stop_requested  # noqa: E731

    def snapshot_blob() -> bytes:
        # The AZ checkpoint blob = net weights + geometry + the games counter (for resume), via az_net.
        import io

        buf = io.BytesIO()
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.net.state_dict().items()},
                "planes": model.planes, "rows": model.rows, "cols": model.cols,
                "n_actions": model.n_actions, "num_players": model.num_players,
                "channels": model.channels, "blocks": model.blocks,
                "games_played": games_done,
            },
            buf,
        )
        return buf.getvalue()

    def evaluate() -> float:
        # Score the net at its REAL strength — neural-MCTS, not the bare policy head — vs the reference,
        # so the curve reflects the move AZ would actually play (and clearly clears the PPO baseline).
        return board_engine.eval_vs_mcts(
            az_net.az_move_fn(model, hp.eval_simulations, hp.c_puct, config.seed),
            game, eval_sims, _EVAL_GAMES, config.seed, should_stop=should_stop,
        )

    def publish() -> None:
        # Decoupled preview policy: the net self-plays the board as it learns (sampled so games vary).
        if on_policy is not None:
            predict = az_net.build_az_predict(model, deterministic=False)
            on_policy(board_engine.board_move_fn(game, predict))

    def emit_progress() -> None:
        # The ~live progress frame (the Reward chart reads progressHistory). Emitted at every iteration
        # boundary AND after each self-play game, so the chart advances smoothly as games accrue instead
        # of jumping once per iteration (~every n×games seconds) — that lag read as a "stuck"/laggy chart.
        # ep_rew_mean holds the last eval between iterations (a step curve in y, smooth in x = games).
        if on_progress is None:
            return
        elapsed = time.monotonic() - started_at
        on_progress(
            TrainingProgress(
                iteration=iteration[0],
                timesteps=games_done,
                total_timesteps=total_target,
                steps_per_sec=games_done / elapsed if elapsed > 0 else 0.0,
                ep_rew_mean=last_eval[0],
                ep_len_mean=None,
                elapsed=elapsed,
            )
        )

    def emit() -> None:
        # The per-iteration metrics frame (Loss tab + high-score) + a progress frame.
        elapsed = time.monotonic() - started_at
        on_metrics(
            TrainingMetrics(
                iteration=iteration[0],
                timesteps=games_done,  # progress unit = self-play games played
                total_timesteps=total_target,
                ep_rew_mean=last_eval[0],  # eval-vs-reference-MCTS ∈ [−1, 1]
                ep_len_mean=None,
                loss=last_loss[0],
                learning_rate=hp.learning_rate,
                elapsed=elapsed,
            )
        )
        emit_progress()

    def take_snapshot() -> None:
        if on_snapshot is None:
            return
        on_snapshot(
            CheckpointArtifact(
                algo="alphazero",  # routes Play/Watch load to az_net.load_az_predict (not MaskablePPO)
                blob=snapshot_blob(),
                artifact_name="board.zip",
                reward=last_eval[0],
                timesteps=games_done,
                total_timesteps=total_target,
                iteration=iteration[0],
            )
        )

    last_loss: list[float | None] = [None]

    # Initial (untrained) preview policy + chart point + a savable snapshot — Save works from step 0.
    last_eval[0] = evaluate()
    publish()
    emit()
    take_snapshot()

    for it in range(hp.iterations):
        if control.stop_requested:
            break
        # Self-play: generate games_per_iter games into the replay buffer (pause/stop between games —
        # each board game is well under a second, so this is responsive enough without interrupting MCTS).
        for _ in range(games_per_iter):
            control.wait_if_paused()
            if control.stop_requested:
                break
            examples, returns = az_net.self_play_game(
                model, hp.simulations, hp.c_puct, dir_alpha, 0.25, hp.temp_moves,
                rng, seed=config.seed + games_done,
            )
            for obs, target, player in examples:
                buffer.append((obs, target, returns[player]))
            games_done += 1
            emit_progress()  # keep the chart moving smoothly between iteration boundaries
        if control.stop_requested:
            break
        # Train the net on the replay buffer (gentle: a few epochs, not a fixed large step count).
        steps = max(10, int(hp.train_epochs * len(buffer) / hp.batch_size))
        last_loss[0] = az_net.train_on_buffer(
            model, optimizer, list(buffer), hp.batch_size, steps, rng
        )
        if control.stop_requested:
            break
        # Iteration boundary (quiescent): eval the snapshot, refresh preview, snapshot the checkpoint.
        iteration[0] = it + 1
        last_eval[0] = evaluate()
        publish()
        emit()
        take_snapshot()

    return "stopped" if control.stop_requested else "finished"
