"""Neural board-game trainer — MaskablePPO vs the MCTS *teacher* (G6b, ADR-051).

G6a makes a board game playable against a training-free MCTS; G6b lets a neural net *learn* the game
so the user can then play against their own trained AI. A risk-gate
(``Local/_probe_maskable_selfplay.py``) found that **pure frozen self-play** (the ``trainer_tag`` /
ADR-048 pattern) barely learns Tic-Tac-Toe in a CPU budget, while **MaskablePPO trained against the
G6a MCTS opponent** reaches a near-optimal (drawing) policy in ~80k steps / ~2 min on CPU. So this
trainer's opponent is the MCTS, with an action mask from ``legal_actions()`` (``sb3-contrib``
``MaskablePPO``); pure self-play is parked.

It keeps the ``trainer_tag`` *shape* — a custom trainer the manager routes to, a decoupled numpy/CPU
snapshot published to the live preview (ADR-019), a packed checkpoint Save/Load round-trips — only the
opponent is the search bot rather than a frozen self. The honest learning curve is **eval-vs-reference-
MCTS ∈ [−1, 1]**, reported as ``ep_rew_mean`` on the existing ``metrics`` frame (``min_score=-1`` /
``solved_score=1`` already match), so the reward chart / high-score / run-archive / checkpoint-reward
all work unchanged — no new WS frame, TS type, chart tab or store field.

Game-agnostic: everything keys off the generic ``pyspiel`` API via :mod:`app.services.board_engine`
(a ``connect_four`` test drives the same functions). CPU-only; ``torch``/SB3 load lazily here.
"""

import io
import time
from collections.abc import Callable
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from app.schemas.training import (
    SelfPlayHyperparams,
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
)
from app.services import board_engine
from app.services.checkpoints import CheckpointArtifact
from app.services.train_control import TrainControl
from app.services.trainer_ppo import _ACTIVATIONS

MetricsSink = Callable[[TrainingMetrics], None]
ProgressSink = Callable[[TrainingProgress], None]
# Published to the decoupled preview: a (state) -> action move over a CPU snapshot (numpy-isolated),
# so the board preview can self-play the learning net without touching the live model (ADR-019).
PredictPublisher = Callable[[Callable[[object], Any]], None]
SnapshotSink = Callable[[CheckpointArtifact], None]

# The reference opponent the live skill curve is measured against (a fixed, moderately strong MCTS),
# and how many games per eval — cheap for a tiny game (sub-ms/move), enough to read a stable rate.
_EVAL_SIMS = board_engine.STRENGTH_SIMS["medium"]
_EVAL_GAMES = 20


def _train_sims(round_idx: int, rounds: int) -> int:
    """Training-opponent strength for a round — a gentle **easy → medium** MCTS curriculum so the net
    faces progressively stronger play (better final policy) without a slow, near-perfect bot from step 0."""
    easy, med = board_engine.STRENGTH_SIMS["easy"], board_engine.STRENGTH_SIMS["medium"]
    if rounds <= 1:
        return med
    return int(round(easy + (med - easy) * round_idx / (rounds - 1)))


def _opponent_move(game: Any, sims: int, seed: int) -> Callable[[Any], int]:
    """A ``(state) -> action`` move from a fresh MCTS bot at ``sims`` strength (the round's teacher)."""
    return board_engine.MctsOpponent(game, sims, seed).step


def _build_model(config: TrainConfig, game: Any, opponent_move: Callable[[Any], int]) -> Any:
    """One MaskablePPO over the action-masked self-play env (opponent baked in; reset per round)."""
    from sb3_contrib import MaskablePPO

    hp = config.hyperparams
    env = board_engine.make_self_play_env(game, opponent_move, seed=config.seed)
    return MaskablePPO(
        "MlpPolicy",
        env,
        seed=config.seed,
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        clip_range=hp.clip_range,
        ent_coef=hp.ent_coef,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        policy_kwargs={
            "net_arch": [hp.neurons_per_layer] * hp.n_hidden_layers,
            "activation_fn": _ACTIVATIONS[hp.activation],
        },
        device="cpu",
        verbose=0,
    )


def _load_model(config: TrainConfig, game: Any, resume_blob: bytes) -> Any:
    """Rebuild MaskablePPO from a saved ``board.zip`` and attach a fresh env (the loop resets it per
    round, so the opponent here only needs to match the obs/action spaces). ``num_timesteps`` restored."""
    from sb3_contrib import MaskablePPO

    env = board_engine.make_self_play_env(game, _opponent_move(game, _EVAL_SIMS, config.seed), config.seed)
    return MaskablePPO.load(io.BytesIO(resume_blob), env=env, device="cpu")


class _BoardCallback(BaseCallback):
    """Honours pause/stop during a round; fires ``on_rollout`` at each rollout boundary so the chart's
    timestep counter advances smoothly between the (per-round) eval points (TTT updates are tiny, so no
    between-epochs interrupt is needed — a plain stop check suffices)."""

    def __init__(self, control: TrainControl, on_rollout: Callable[[], None]) -> None:
        super().__init__()
        self._control = control
        self._on_rollout = on_rollout

    def _on_step(self) -> bool:
        self._control.wait_if_paused()
        return not self._control.stop_requested

    def _on_rollout_end(self) -> None:
        self._on_rollout()


def train_board(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: MetricsSink,
    on_progress: ProgressSink | None = None,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train a board-game net by playing the G6a MCTS teacher (MaskablePPO + action masking).

    Blocks the calling thread; the manager runs this off the event loop. ``gym_id`` is the OpenSpiel
    short name (e.g. ``"tic_tac_toe"``). Each round trains against an MCTS opponent (easy→medium
    curriculum), then evaluates the snapshot vs a fixed reference MCTS and emits **both** a ``metrics``
    frame (for the Loss tab) and a ``progress`` frame (which the Reward tab reads, exactly like PPO) —
    both carrying that eval-vs-MCTS score as ``ep_rew_mean`` (the honest skill curve). It also publishes
    a fresh decoupled snapshot to the preview and (if ``on_snapshot``) packs a ``board.zip`` checkpoint.
    Resumes from a saved ``board.zip`` (continuing the step counter; the budget is another full schedule).
    """
    game = board_engine.load_game(gym_id)
    rounds = max(1, (config.self_play or SelfPlayHyperparams()).rounds)
    started_at = time.monotonic()

    model = (
        _load_model(config, game, resume_blob)
        if resume_blob is not None
        else _build_model(config, game, _opponent_move(game, _train_sims(0, rounds), config.seed))
    )
    per_round = max(1, config.total_timesteps // rounds)
    # Resume-aware budget for the progress bar: where this run started + its budget (so a resumed run
    # reports total = start + budget and the bar never shows steps exceeding the total).
    total_target = int(model.num_timesteps) + config.total_timesteps

    rollout = [0]  # rollout counter → the metrics frame's "iteration" (work-done index)
    last_eval = [
        board_engine.eval_vs_mcts(
            board_engine.build_board_predict(model), game, _EVAL_SIMS, _EVAL_GAMES, config.seed
        )
    ]

    def publish() -> None:
        # Decoupled preview policy: the net self-plays the board as it learns. Sampled (non-deterministic)
        # so the watched games vary instead of replaying one deterministic line. Never the live model.
        if on_policy is not None:
            snapshot = board_engine.build_board_predict(model, deterministic=False)
            on_policy(board_engine.board_move_fn(game, snapshot))

    def emit() -> None:
        # The logger is only configured once learn() starts, so the pre-training frame has no loss yet.
        sb3_logger = getattr(model, "_logger", None)
        loss = sb3_logger.name_to_value.get("train/loss") if sb3_logger is not None else None
        steps = int(model.num_timesteps)
        elapsed = time.monotonic() - started_at
        on_metrics(
            TrainingMetrics(
                iteration=rollout[0],
                timesteps=steps,
                total_timesteps=total_target,
                ep_rew_mean=last_eval[0],  # the eval-vs-reference-MCTS score ∈ [−1, 1]
                ep_len_mean=None,
                loss=float(loss) if loss is not None else None,
                learning_rate=config.hyperparams.learning_rate,
                elapsed=elapsed,
            )
        )
        # Also emit a progress frame: the Reward chart tab reads progressHistory (like PPO), so without
        # this the board reward line would be blank while the Loss tab (metricsHistory) showed data.
        if on_progress is not None:
            on_progress(
                TrainingProgress(
                    iteration=rollout[0],
                    timesteps=steps,
                    total_timesteps=total_target,
                    steps_per_sec=steps / elapsed if elapsed > 0 else 0.0,
                    ep_rew_mean=last_eval[0],
                    ep_len_mean=None,
                    elapsed=elapsed,
                )
            )

    def on_rollout() -> None:
        rollout[0] += 1
        emit()

    publish()  # initial (untrained) preview policy + chart point
    emit()
    callback = _BoardCallback(control, on_rollout)

    try:
        for r in range(rounds):
            if control.stop_requested:
                break
            # Rebuild the env with this round's teacher (curriculum); reset the step counter only on a
            # fresh run's first round.
            model.set_env(board_engine.make_self_play_env(game, _opponent_move(game, _train_sims(r, rounds), config.seed + r), config.seed + r))
            model.learn(
                per_round,
                callback=callback,
                reset_num_timesteps=(r == 0 and resume_blob is None),
            )
            # Round boundary (quiescent): eval the snapshot, refresh the preview, snapshot the checkpoint.
            last_eval[0] = board_engine.eval_vs_mcts(
                board_engine.build_board_predict(model), game, _EVAL_SIMS, _EVAL_GAMES, config.seed
            )
            publish()
            emit()
            if on_snapshot is not None:
                buf = io.BytesIO()
                model.save(buf)
                on_snapshot(
                    CheckpointArtifact(
                        algo="ppo",  # surfaced as PPO (the board self-play precedent); routed by is_board_game
                        blob=buf.getvalue(),
                        artifact_name="board.zip",
                        reward=last_eval[0],
                        timesteps=int(model.num_timesteps),
                        total_timesteps=total_target,
                        iteration=r + 1,
                    )
                )
        return "stopped" if control.stop_requested else "finished"
    finally:
        if model.env is not None:
            model.env.close()
