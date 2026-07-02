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

import threading
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
from app.services import az_batch, az_net, az_parallel, board_engine
from app.services.checkpoints import CheckpointArtifact
from app.services.train_control import TrainControl

MetricsSink = Callable[[TrainingMetrics], None]
ProgressSink = Callable[[TrainingProgress], None]
PredictPublisher = Callable[[Callable[[object], Any]], None]
SnapshotSink = Callable[[CheckpointArtifact], None]

# Games per eval — each eval move runs neural-MCTS (eval_simulations forwards), so this is kept modest
# to keep the per-iteration eval to a few seconds, but large enough to steady the noisy ±1 board metric.
_EVAL_GAMES = 18

# The DISPLAYED score is the mean of the last N raw evals (not just the latest). A single eval is a few
# near-±1 games, so one flipped game whipsaws the % (the user saw 100% → 24% between iterations); averaging
# recent evals turns it into a readable trend. It only smooths the *display* — the raw evals still drive it.
_EVAL_SMOOTH_WINDOW = 3

# Live-progress cadence (G6h follow-up). The AZ self-play count is intrinsically *bursty*: a cohort of
# games runs in lockstep and finishes in a cluster, with no completions at all during the per-iteration
# eval + net update. Emitting a progress frame per finished game therefore arrived in spikes-then-gaps, and
# a frontend rate over those frames flickered (0, 8, 1, …). Instead a steady ticker thread emits ONE frame
# per second (like the PPO ticker) carrying a trailing-window games/s, so the display refreshes smoothly
# and the rate matches the count growth (it *is* the count delta over the window).
_PROGRESS_INTERVAL = 1.0  # seconds between live progress frames (1 Hz, matching trainer_ppo)
_RATE_WINDOW = 10.0  # trailing window (s) the games/s is averaged over — absorbs the per-cohort burstiness


def _eval_budget(n_actions: int, eval_simulations: int) -> tuple[int, int]:
    """``(#eval games, neural-MCTS sims/move)`` for the per-iteration eval-vs-reference (G6g).

    The eval pits the net (batched neural-MCTS) against a random-rollout reference MCTS — cheap on the
    small boards but heavy on **chess** (~4674 moves, ~100-ply games, and each reference rollout plays a
    full random game), where one eval game measured ~5–6 s. So a high-branching game gets fewer games + a
    capped search depth to keep the per-iteration eval a tolerable few-tens-of-seconds; the small boards
    keep the steady 18 games at full depth. 8 chess games (was 4) halves the per-eval variance, and the
    displayed score is further smoothed over recent evals (``_EVAL_SMOOTH_WINDOW``) so the ±1 readout reads
    as a trend, not a coin flip — the net self-plays through the longer eval, so the counter never stalls."""
    if n_actions > 1000:  # chess / go — expensive long-game random-rollout eval
        return 8, min(eval_simulations, 12)
    return _EVAL_GAMES, eval_simulations


def _self_play_ply_cap(n_actions: int) -> int | None:
    """Max plies per self-play game for an *unbounded* game (chess/go) → bounds the marathon games a weak
    early net produces, ``None`` (no cap) for the small bounded boards that always end first.

    A near-random early policy rarely converts a high-branching game, so without a cap chess self-play
    drags to the 75-move forced-draw ceiling (~hundreds of plies) — worst of all under Gumbel's exploratory
    play (G6h), making the first iterations needlessly slow. 160 plies (80 moves) lets a genuinely decisive
    game finish while cutting the long shuffles; the bounded boards (≤~9–42 plies) never reach it."""
    return 160 if n_actions > 1000 else None


def _dirichlet_alpha(n_actions: int, gym_id: str = "") -> float:
    """Root-exploration noise scale ≈ 10 / typical-branching: looser for low-branching games (Connect
    Four ~7 moves → 1.0), tighter for high-branching ones (Breakthrough/Othello/chess → 0.3).

    ``n_actions`` is only a *proxy* for branching, and **checkers** breaks it — 512 distinct actions but
    only ~5 legal moves/position (mandatory captures), so it takes the loose 1.0 like Connect Four despite
    the big action space (measured: AZ learns checkers with 1.0 in the G6-Dáma risk-gate)."""
    if gym_id == "checkers":
        return 1.0
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
    # GPU when available (G6g): self-play now runs `parallel_games` games concurrently and batches every
    # MCTS step's leaf evaluations into one wide forward (az_batch), so the GPU is the workhorse. G6f's
    # single-position (batch-1) forwards were the regime where a GPU sits idle/slower; the batched engine
    # measured ~6× faster than that sequential path on Connect Four. CPU is the graceful fallback.
    device = az_net.best_device()
    # G6i (ADR-062): >1 ⇒ run self-play in independent GPU worker PROCESSES instead of the in-process
    # actor thread (escapes the GIL the learner thread otherwise contends for; ~1.6× chess on the RTX
    # 5070). CUDA-only — the 128×10 net is too heavy for parallel CPU workers (measured ~10× slower), so
    # on CPU we keep the single in-process actor (=1 stays byte-identical to G6h on every machine).
    use_parallel = hp.actor_processes > 1 and device == "cuda"
    started_at = time.monotonic()

    # Same reference MCTS the PPO baseline is scored against (board_engine.board_profile, G6c), so the
    # two algorithms' learning curves are directly comparable. AZ has no teacher — it self-plays.
    eval_sims = board_engine.STRENGTH_SIMS[board_engine.board_profile(gym_id).eval_strength]
    n_actions = int(game.num_distinct_actions())
    dir_alpha = _dirichlet_alpha(n_actions, gym_id)
    # Lighter eval for high-branching games (chess) so the per-iteration eval stays a few-tens-of-seconds.
    eval_games, eval_move_sims = _eval_budget(n_actions, hp.eval_simulations)
    # Cap self-play game length for unbounded games (chess) so a weak early net's near-random play doesn't
    # drag every game to the forced-draw ceiling — None (uncapped) for the small bounded boards (G6h).
    ply_cap = _self_play_ply_cap(n_actions)

    if resume_blob is not None:
        model, games_done = az_net.build_model_from_blob(resume_blob, game, device=device)
        # Cumulative self-play plies at the checkpoint (the canonical env-steps axis, X1) — kept across
        # resume so env_steps stays cumulative like PPO's num_timesteps. Absent in pre-X1 blobs → 0.
        plies_done = int(az_net.load_az_model(resume_blob, device="cpu").get("plies_played", 0))
    else:
        model = az_net.AZModel(
            game, channels=hp.channels, blocks=hp.blocks, device=device, norm=hp.norm
        )
        games_done = 0
        plies_done = 0
    model.net.eval()
    optimizer = torch.optim.Adam(model.net.parameters(), lr=hp.learning_rate, weight_decay=1e-4)

    # Actor–learner split (G6h, ADR-059) — the profiled fix for the frozen counter. Self-play runs on a
    # background thread with its OWN copy of the net (``actor_model``), feeding the shared replay buffer
    # CONTINUOUSLY, while this (learner) thread trains ``model`` and periodically evaluates. So the games
    # counter — and the board preview — keep moving during training AND the eval, instead of freezing for
    # ~20 s every iteration (the dominant stall, profiled headless). The eval is CPU-bound (the reference's
    # random rollouts), so the GPU stays free for the actor's self-play *through* the eval. The actor
    # re-syncs from the learner after every training round (the decoupled-snapshot pattern, ADR-008/019).
    # Two threads ⇒ the run is policy-level reproducible (seeded), not bit-reproducible — like the SuperSuit
    # multi-agent path (ADR-038). (Skipped under use_parallel: the worker processes build their own nets.)
    actor_model = None
    if not use_parallel:
        actor_model = az_net.AZModel(
            game, channels=model.channels, blocks=model.blocks, device=device, norm=model.norm
        )
        actor_model.net.load_state_dict(model.net.state_dict())
        actor_model.net.eval()

    buffer: deque[tuple[Any, Any, float]] = deque(maxlen=hp.buffer_size)
    buffer_lock = threading.Lock()  # the actor extends it; the learner snapshots it per training round
    rng = np.random.default_rng(config.seed)
    games_per_iter = max(1, hp.games_per_iter)
    # Rolling self-play cohort width: keep up to this many games concurrent (one GPU forward per MCTS step).
    parallel = max(1, min(hp.parallel_games, games_per_iter))
    # G6i: with worker processes the GPU-batch budget (parallel_games) is split across them — 2 workers ×
    # 64 = the probe's measured-good chess config (peak VRAM ~4.5 GB, GPU 94 %). The =1 path is unchanged
    # (the `parallel`-wide in-process actor above).
    per_worker_parallel = max(1, hp.parallel_games // max(1, hp.actor_processes))
    games_base = games_done  # the count at run start (resume continues from here)
    # Budget in self-play games — the AZ analogue of "Total Steps" (reported as timesteps). The actor keeps
    # producing while the learner finishes its rounds, so the actual count may run a little past this.
    total_target = games_base + hp.iterations * games_per_iter
    search_sims = hp.gumbel_sims if hp.use_gumbel else hp.simulations
    # The eval is the expensive phase (~20 s on chess); run it every few rounds on high-branching games so
    # the learner stays faster than the actor (the actor is the producer → the counter tracks the budget).
    # The actor self-plays through the eval regardless, so the display stays smooth either way. Small boards
    # eval cheaply every round.
    eval_every = 3 if n_actions > 1000 else 1

    iteration = [0]  # the iteration index → the metrics frame's "iteration"
    # The displayed eval score: None until the FIRST eval lands (the panel then reads "—" instead of a
    # fake 0.0 = 50%, which looked like a real measurement before any game had been scored — user-flagged).
    # Once real, it's the mean of the last few raw evals (eval_window) so the noisy ±1 readout reads as a
    # trend, not a per-iteration coin flip.
    last_eval: list[float | None] = [None]
    eval_window: deque[float] = deque(maxlen=_EVAL_SMOOTH_WINDOW)  # recent raw evals → the smoothed display
    live_games = [games_base]  # cumulative games produced by the actor — bumped per game, read by the ticker
    # Cumulative self-play plies (moves) — the canonical env-steps axis (X1). AZ's ``timesteps`` is games
    # (its progress unit), so ``env_steps`` tracks plies instead, making AZ directly comparable to the board
    # MaskablePPO trainer (which counts moves). Bumped per finished game by len(game_examples) == its plies.
    live_plies = [plies_done]
    last_loss: list[float | None] = [None]
    stop_event = threading.Event()  # retires the actor + ticker on every exit path
    stop_ticker = threading.Event()
    pending_sync = [False]  # the learner has published a newer net for the actor to pick up
    shared_sd: list[Any] = [None]  # CPU state_dict the actor reloads between games (decoupled snapshot)
    sync_lock = threading.Lock()
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
                "channels": model.channels, "blocks": model.blocks, "norm": model.norm,
                "games_played": live_games[0],
                "plies_played": live_plies[0],  # cumulative env-steps axis (X1) — restored on resume
            },
            buf,
        )
        return buf.getvalue()

    def evaluate() -> float:
        # Score the net at its REAL strength — neural-MCTS, not the bare policy head — vs the reference,
        # so the curve reflects the move AZ would actually play (and clearly clears the PPO baseline). The
        # net's eval moves are BATCHED across the eval games (az_batch). Capped (chess) so one marathon
        # eval game can't stall the round — a capped game is scored by material (unresolved_value_fn), not a
        # flat 0, so a weak early net's score isn't pinned at 0.0 (G6h). Runs here while the actor self-plays.
        return az_batch.eval_vs_mcts_parallel(
            model, game, eval_sims, eval_games, eval_move_sims, hp.c_puct,
            config.seed, should_stop=should_stop, max_game_plies=ply_cap,
            unresolved_value=board_engine.unresolved_value_fn(gym_id),
        )

    def publish() -> None:
        # Decoupled preview policy: the net self-plays the board as it learns (sampled so games vary).
        if on_policy is not None:
            predict = az_net.build_az_predict(model, deterministic=False)
            on_policy(board_engine.board_move_fn(game, predict))

    def _actor() -> None:
        # The background self-play producer: a continuous rolling cohort (games complete one-by-one and are
        # replaced → a smooth counter + a full GPU batch) feeding the shared buffer + the live counter. It
        # picks up the learner's latest net between games (decoupled snapshot). Runs until stop.
        # (Only started when not use_parallel, so actor_model is always built here.)
        assert actor_model is not None
        actor_rng = np.random.default_rng(config.seed + 99_991)  # a distinct, seeded stream
        for game_examples, _ret in az_batch.self_play_rolling(
            actor_model, parallel, search_sims, hp.c_puct, dir_alpha, 0.25, hp.temp_moves, actor_rng,
            gumbel=hp.use_gumbel, gumbel_considered=hp.gumbel_considered, max_game_plies=ply_cap,
            should_stop=lambda: stop_event.is_set() or control.stop_requested,
        ):
            with buffer_lock:
                buffer.extend(game_examples)
            live_games[0] += 1
            live_plies[0] += len(game_examples)  # each example is one ply → the env-steps axis (X1)
            if pending_sync[0]:  # the learner trained → adopt the new net for the next games
                with sync_lock:
                    sd, pending_sync[0] = shared_sd[0], False
                if sd is not None:
                    actor_model.net.load_state_dict(sd)
                    actor_model.net.eval()
            while control.paused and not (stop_event.is_set() or control.stop_requested):
                time.sleep(0.05)  # idle while paused (the preview is frozen too)

    def _progress_ticker() -> None:
        # Emit one progress frame per _PROGRESS_INTERVAL with a trailing-window games/s (decoupled from the
        # bursty per-game completions — see _PROGRESS_INTERVAL). Reads shared counters only; never mutates
        # training state, so it can't affect reproducibility (mirrors trainer_ppo._progress_ticker). Frames
        # are emitted immediately (t=0, run is alive), every tick, and once more on stop (the final count,
        # never stale) — so even a sub-second run is bookended by a start + end frame.
        if on_progress is None:
            return
        samples: deque[tuple[float, int]] = deque()  # (monotonic time, games) over the last _RATE_WINDOW s

        def _emit() -> None:
            now = time.monotonic()
            games = live_games[0]
            samples.append((now, games))
            while len(samples) > 1 and now - samples[0][0] > _RATE_WINDOW:
                samples.popleft()
            span = now - samples[0][0]
            rate = (games - samples[0][1]) / span if span > 0 else 0.0
            on_progress(
                TrainingProgress(
                    iteration=iteration[0],
                    timesteps=games,
                    total_timesteps=total_target,
                    steps_per_sec=rate,  # a smooth RECENT games/s, matching the count growth
                    ep_rew_mean=last_eval[0],  # last eval ∈ [−1, 1] — a step curve in y, smooth in x (games)
                    ep_len_mean=None,
                    elapsed=now - started_at,
                )
            )

        _emit()  # an immediate frame so the panel is live from t=0 (before the slow initial eval)
        while True:
            stopped = stop_ticker.wait(_PROGRESS_INTERVAL)
            if control.paused and not stopped:
                continue  # hold steady while paused (the preview is frozen too)
            _emit()  # every tick AND once on stop (so the final games count is never left stale)
            if stopped:
                break

    def emit() -> None:
        # The per-iteration metrics frame (Loss tab + high-score). The progress frame is the ticker's job.
        elapsed = time.monotonic() - started_at
        on_metrics(
            TrainingMetrics(
                iteration=iteration[0],
                timesteps=live_games[0],  # progress unit = self-play games played (the actor's live count)
                total_timesteps=total_target,
                # env_steps ≠ timesteps for AZ: cumulative self-play plies (moves), the canonical
                # env-interactions axis (X1) — comparable to the board MaskablePPO trainer's move count.
                env_steps=live_plies[0],
                ep_rew_mean=last_eval[0],  # eval-vs-reference-MCTS ∈ [−1, 1]
                ep_len_mean=None,
                loss=last_loss[0],
                learning_rate=hp.learning_rate,
                elapsed=elapsed,
            )
        )

    def take_snapshot() -> None:
        if on_snapshot is None:
            return
        on_snapshot(
            CheckpointArtifact(
                algo="alphazero",  # routes Play/Watch load to az_net.load_az_predict (not MaskablePPO)
                blob=snapshot_blob(),
                artifact_name="board.zip",
                reward=last_eval[0],
                timesteps=live_games[0],
                total_timesteps=total_target,
                iteration=iteration[0],
            )
        )

    # Start the self-play actor (games flow from t≈0) + the 1 Hz progress ticker, then run the learner loop.
    # The actor produces continuously; the learner trains + evals between game milestones. Because the actor
    # never stops, the games counter + board preview stay live the whole run — no startup or per-iteration
    # freeze (the headless profile showed ~57 % of wall-clock frozen before this).
    publish()  # initial preview from the (random) net, immediately
    ticker = threading.Thread(target=_progress_ticker, name="az-progress-ticker", daemon=True)
    ticker.start()
    # The self-play producer: either today's in-process actor thread (=1) or, under use_parallel, a pool of
    # independent GPU worker processes (G6i). Both feed the SAME shared buffer + live_games counter the
    # learner loop, ticker, eval and snapshot read, so only the producer differs — the rest is identical.
    parallel_actor: az_parallel.ParallelActor | None = None
    actor: threading.Thread | None = None
    if use_parallel:
        parallel_actor = az_parallel.ParallelActor(
            gym_id=gym_id, n_workers=hp.actor_processes, per_worker_parallel=per_worker_parallel,
            build={"channels": model.channels, "blocks": model.blocks, "norm": model.norm},
            search=az_parallel.build_search(
                sims=search_sims, c_puct=hp.c_puct, dir_alpha=dir_alpha, dir_frac=0.25,
                temp_moves=hp.temp_moves, gumbel=hp.use_gumbel,
                gumbel_considered=hp.gumbel_considered, ply_cap=ply_cap,
            ),
            base_seed=config.seed + 99_991, initial_state_dict=az_parallel.cpu_state_dict(model.net),
            buffer=buffer, buffer_lock=buffer_lock, live_games=live_games, live_plies=live_plies,
            control=control,
        )
        parallel_actor.start()
        actor_alive = parallel_actor.is_alive
    else:
        actor = threading.Thread(target=_actor, name="az-actor", daemon=True)
        actor.start()
        actor_alive = actor.is_alive
    try:
        for it in range(hp.iterations):
            # Wait for the actor to produce this round's games (it keeps playing, so the counter keeps
            # moving — the wait never freezes the UI). Poll cheaply; honour pause/stop.
            target_games = games_base + (it + 1) * games_per_iter
            while live_games[0] < target_games and not control.stop_requested and actor_alive():
                control.wait_if_paused()
                if stop_event.is_set():
                    break
                time.sleep(0.1)
            if control.stop_requested:
                break
            # Train the learner on the buffer the actor has been filling (gentle: a few epochs over a
            # consistent snapshot taken under the lock; the actor keeps appending meanwhile).
            with buffer_lock:
                data = list(buffer)
            if data:
                steps = max(10, int(hp.train_epochs * len(data) / hp.batch_size))
                # should_stop makes the (long, ~1000-step on chess) update interruptible so a user Stop
                # doesn't wait it out — the dominant Stop-latency term once self-play runs in parallel.
                last_loss[0] = az_net.train_on_buffer(
                    model, optimizer, data, hp.batch_size, steps, rng, should_stop=should_stop
                )
            # Hand the freshly trained net to the actor (a decoupled CPU snapshot it reloads between games):
            # an in-memory state_dict for the in-process actor, or an atomic shared-file publish the worker
            # processes pick up by mtime (G6i). Both are infrequent (once per iteration), never per-round.
            if parallel_actor is not None:
                parallel_actor.publish_net(az_parallel.cpu_state_dict(model.net))
            else:
                with sync_lock:
                    shared_sd[0] = {k: v.detach().cpu() for k, v in model.net.state_dict().items()}
                    pending_sync[0] = True
            iteration[0] = it + 1
            # Eval periodically (the actor self-plays through it on the free GPU). Always eval the first +
            # last round so the curve starts and ends on a real point.
            if it == 0 or (it + 1) % eval_every == 0 or it + 1 == hp.iterations:
                eval_window.append(evaluate())
                last_eval[0] = float(np.mean(eval_window))  # smoothed over recent evals (each is noisy)
            publish()
            emit()
            take_snapshot()
    finally:
        stop_event.set()  # retire the in-process actor (it checks this between plies)
        if parallel_actor is not None:
            parallel_actor.stop()  # Windows-hardened: drain → cancel_join_thread → terminate the workers
        if actor is not None:
            actor.join(timeout=10.0)
        stop_ticker.set()  # retire the ticker
        ticker.join(timeout=2.0)

    return "stopped" if control.stop_requested else "finished"
