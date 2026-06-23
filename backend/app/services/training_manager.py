"""Training lifecycle manager: one active run, controllable, streaming over WS.

SB3 runs synchronously on a daemon thread; metric/status frames are marshalled back onto
the FastAPI event loop with ``run_coroutine_threadsafe`` so the async connection manager
can broadcast them. No ML imports here — the trainer (and torch) is imported lazily inside
:meth:`_run` only when a run actually starts.
"""

import asyncio
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from app.core.logging import get_logger
from app.envs.registry import get_env
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import (
    EvolutionMetrics,
    HwStatsFrame,
    MultiAgentMetrics,
    QLearningMetrics,
    QTableFrame,
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
    TrainStatus,
)
from app.services.board_engine import is_board_game
from app.services.checkpoints import CheckpointArtifact, CheckpointStore, checkpoint_store
from app.services.connection_manager import ConnectionManager, manager
from app.services.highscores import HighScoreStore, highscores, make_meta
from app.services.ma_env import is_competitive_ma
from app.services.preview_streamer import preview_streamer
from app.services.runs import RunStore, final_score, run_store, should_archive
from app.services.system_info import gpu_available
from app.services.train_control import TrainControl

logger = get_logger(__name__)

# Cap on metric frames retained for a checkpoint's metrics.json (per-rollout / per-generation
# frames only — the high-frequency progress ticks are not logged). Generous for any CartPole run.
_METRICS_LOG_CAP = 10_000
# Cadence of the algorithm-independent hardware-telemetry frame (the HW panel), in seconds.
_HW_STATS_INTERVAL = 1.0


class AlreadyRunningError(RuntimeError):
    """Raised when a start is attempted while a run is already active."""


class InvalidConfigError(ValueError):
    """Raised when the requested env/algo is unknown or unsupported."""


class CheckpointNotFoundError(RuntimeError):
    """Raised when a load/save targets a checkpoint id that does not exist."""


class NothingToSaveError(RuntimeError):
    """Raised when a save is attempted before any model snapshot exists."""


class TrainingManager:
    """Owns the single active training run and mirrors its state over WebSocket."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        hi_scores: HighScoreStore = highscores,
        checkpoints: CheckpointStore = checkpoint_store,
        runs: RunStore = run_store,
    ) -> None:
        self._cm = connection_manager
        self._hi = hi_scores
        self._ckpt = checkpoints
        self._runs = runs
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._control: TrainControl | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._state: TrainState = "idle"
        self._config: TrainConfig | None = None
        self._timesteps = 0
        self._last_metrics: TrainingMetrics | None = None
        # Latest evolution frame, kept so a late-joining client reconciling via
        # /api/train/status sees the current generation (leaderboard / stats / Fitness)
        # without waiting for the next frame — and still sees it after the run finished.
        self._last_evolution: EvolutionMetrics | None = None
        # Latest tabular Q-learning frame + Q-table snapshot, kept so a late-joining client (or one
        # connecting after a finished run) sees the current chart/stats/heatmap immediately.
        self._last_q_learning: QLearningMetrics | None = None
        self._last_qtable: QTableFrame | None = None
        # Latest competitive self-play frame (simple_tag, G7b-2), kept so a late-joining client sees
        # the current two-line ecosystem chart immediately. None for every single-policy run.
        self._last_ma_metrics: MultiAgentMetrics | None = None
        self._error: str | None = None
        # Latest model snapshot from the trainer + the run's metric frames — the raw material
        # "Save" persists into a checkpoint slot. Both reset when a fresh run launches.
        self._snapshot: CheckpointArtifact | None = None
        self._metrics_log: list[dict] = []
        # When the active run started (ISO-8601 UTC) — stamped onto its run-history record
        # when it reaches a terminal state.
        self._run_started_at: str | None = None

    # -- wiring -----------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the event loop so worker-thread broadcasts can reach it."""
        self._loop = loop

    # -- lifecycle --------------------------------------------------------------

    def start(self, config: TrainConfig) -> TrainStatus:
        spec = get_env(config.env_id)
        if spec is None:
            raise InvalidConfigError(f"Unknown environment '{config.env_id}'")
        if config.algo not in spec.supported_algos:
            raise InvalidConfigError(
                f"Environment '{config.env_id}' does not support algo '{config.algo}'"
            )
        # Backstop for an env whose trainer isn't built yet: reject before the manager would build the
        # wrong policy on its obs and crash. This holds **even on a CUDA machine**, so a GPU desktop
        # (or someone building from source on a GPU) can't un-gate it via the gpu check below. Every
        # shipped env now trains — the image-obs CnnPolicy path landed for Atari (G4b) and CarRacing
        # (G3c-train) — so this guard is currently inert, kept for any future not-yet-built family.
        # Human play needs no net and stays available regardless.
        if not spec.train_implemented:
            raise InvalidConfigError(
                f"Training '{config.env_id}' isn't available yet — pixel-based games need the GPU "
                f"(CnnPolicy) trainer that's coming in a later version. You can still play it by hand now."
            )
        # GPU-gated *vector* envs (BipedalWalker, MuJoCo) train correctly with the existing MlpPolicy
        # path — they're gated only because a gait needs millions of steps, too slow on a CPU. Reject
        # on a machine with no CUDA device; a GPU machine un-gates them (the trainer runs as-is). The UI
        # also disables Run, so this is a defensive backstop. (gpu_available() is cached — see /api/system.)
        if spec.hw_requirement == "gpu" and not gpu_available():
            raise InvalidConfigError(
                f"Training '{config.env_id}' needs a CUDA GPU, which isn't available on this "
                f"machine. You can still play it by hand now; GPU training runs on a CUDA desktop."
            )
        return self._launch(config, spec.gym_id, resume=None)

    def _launch(
        self, config: TrainConfig, gym_id: str, resume: bytes | None
    ) -> TrainStatus:
        """Spin up a worker thread for a fresh or resumed run (shared by start + load)."""
        with self._lock:
            if self._state in ("running", "paused", "stopping"):
                raise AlreadyRunningError("A training run is already active")
            self._control = TrainControl()
            self._config = config
            self._timesteps = 0
            self._last_metrics = None
            self._last_evolution = None  # clear the previous run's evolution panels
            self._last_q_learning = None  # clear the previous run's Q-learning chart/stats
            self._last_qtable = None  # clear the previous run's heatmap
            self._last_ma_metrics = None  # clear the previous run's ecosystem (self-play) chart
            self._error = None
            self._snapshot = None  # don't let a previous run's model be saved as this one's
            self._metrics_log = []
            self._run_started_at = datetime.now(UTC).isoformat()
            self._state = "running"
            # Snapshot the freshly-launched state *under the lock*, before the worker thread can
            # emit its first frame. Reading status() after thread.start() would race a fast run
            # (e.g. tiny CartPole evolution lands generation 1 in milliseconds), so the returned
            # status could already carry that frame instead of the clean "just started" state.
            initial_status = self._status_locked()

        self._broadcast_status()
        # Image-obs runs (Atari/CarRacing) spin up two SB3-importing threads at once — the trainer
        # (`from stable_baselines3 import PPO`) and the image-obs preview (env_util via image_vec) —
        # whose *divergent* first-time entries into the stable_baselines3 package, run concurrently on
        # the process's first such run, can deadlock Python's per-module import locks (observed with
        # CarRacing: a `_DeadlockError` on `_ModuleLock('…env_util')`; Atari only dodged it because its
        # `import ale_py` happened to serialize the two threads). Preload the package HERE — on this
        # single thread, before either thread spawns — so both then hit a fully-initialised cache.
        # Idempotent + cheap after the first run; scoped to image envs so non-image algos (evolution /
        # Q-learning, which never import SB3) don't pull it in. Outside the lock: the first import is
        # ~1 s and must not block status() readers.
        launch_spec = get_env(config.env_id)
        if launch_spec is not None and launch_spec.obs_type == "image":
            import stable_baselines3  # noqa: F401 — single-threaded preload to avoid the import deadlock
        # Algo-independent HW telemetry for the lifetime of THIS run. The stop event is created per
        # run and handed to both the ticker and _run, so a back-to-back start can't have the finishing
        # run's teardown stop the next run's ticker (a shared field would race; see _run's finally).
        hw_stop = threading.Event()
        self._start_hw_ticker(hw_stop)
        self._thread = threading.Thread(
            target=self._run,
            args=(config, gym_id, self._control, resume, hw_stop),
            name="ppo-trainer",
            daemon=True,
        )
        self._thread.start()
        # Let the (decoupled) preview streamer begin watching this run, if visual is on. It builds
        # the env from the registry id (same factory + wrappers as the trainer), so its decoupled
        # predict policy sees the obs shape the model was trained on.
        preview_streamer.attach_run(config.env_id)
        return initial_status

    # -- checkpoints ------------------------------------------------------------

    def save_checkpoint(self, label: str | None = None) -> CheckpointMeta:
        """Persist the latest model snapshot + config + metrics into a new slot."""
        with self._lock:
            snapshot = self._snapshot
            config = self._config
            metrics = list(self._metrics_log)
        if snapshot is None or config is None:
            raise NothingToSaveError("No trained model to save yet — start a run first")
        return self._ckpt.save(config, snapshot, metrics, label)

    def load_checkpoint(
        self, checkpoint_id: str, new_config: TrainConfig | None = None
    ) -> TrainStatus:
        """Resume training from a saved checkpoint (PPO continues; evolution continues).

        By default the resumed run keeps the **saved** config. If ``new_config`` is supplied (the
        sidebar's current settings) **and it targets the same game + algorithm**, the run adopts its
        hyperparameters + seed instead — so the user can *extend or retune* a run (e.g. raise AlphaZero's
        Iterations to keep training for hours) while picking up from the saved net. The env/algorithm
        always come from the checkpoint and the saved **weights** always win (the architecture is read
        from the blob), so a mismatched ``new_config`` is simply ignored — Load never fails on it.
        """
        loaded = self._ckpt.load(checkpoint_id)
        if loaded is None:
            raise CheckpointNotFoundError(f"Checkpoint '{checkpoint_id}' not found")

        spec = get_env(loaded.config.env_id)
        if spec is None:
            raise InvalidConfigError(
                f"Checkpoint environment '{loaded.config.env_id}' is no longer available"
            )
        if loaded.config.algo not in spec.supported_algos:
            raise InvalidConfigError(
                f"Environment '{loaded.config.env_id}' does not support algo "
                f"'{loaded.config.algo}'"
            )

        # Adopt the current sidebar settings only when they continue the *same* run (same game + algo);
        # otherwise fall back to the saved config so Load is always safe (the sidebar may be elsewhere).
        config = loaded.config
        if (
            new_config is not None
            and new_config.env_id == loaded.config.env_id
            and new_config.algo == loaded.config.algo
        ):
            config = new_config
        # Both PPO and AlphaZero target ANOTHER full budget on top of where the checkpoint left off, so
        # the recorded total must add the elapsed steps/games (else a just-loaded finished AZ model reads
        # as ~50%: the sidebar shows the bare new schedule, e.g. 960, while the bar climbs to 1920).
        #  • PPO continues the global step counter → games_done + new budget.
        #  • AZ runs another full `iterations` schedule → total_target = games_done + iterations × games.
        # Evolution continues by generation (in-trainer). The exceptions that must NOT add elapsed steps
        # (their "rounds × per-round" budget would inflate): competitive self-play (simple_tag) and the
        # MaskablePPO *board* trainer (G6b) — hence the PPO guard excludes them; AZ is always a board game
        # but computes its own schedule, so it's added explicitly.
        if config.algo == "alphazero" or (
            config.algo == "ppo" and not is_competitive_ma(spec) and not is_board_game(spec)
        ):
            config = config.model_copy(
                update={"total_timesteps": loaded.meta.timesteps + config.total_timesteps}
            )
        return self._launch(config, spec.gym_id, resume=loaded.blob)

    def pause(self) -> TrainStatus:
        with self._lock:
            if self._state != "running" or self._control is None:
                return self._status_locked()
            self._control.pause()
            self._state = "paused"
        preview_streamer.set_paused(True)  # freeze the live preview alongside training
        self._broadcast_status()
        return self.status()

    def resume(self) -> TrainStatus:
        with self._lock:
            if self._state != "paused" or self._control is None:
                return self._status_locked()
            self._control.resume()
            self._state = "running"
        preview_streamer.set_paused(False)  # unfreeze the live preview
        self._broadcast_status()
        return self.status()

    def stop(self) -> TrainStatus:
        with self._lock:
            if self._state not in ("running", "paused") or self._control is None:
                return self._status_locked()
            self._control.request_stop()
            self._state = "stopping"
        self._broadcast_status()
        return self.status()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to finish (used by tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- hardware telemetry (the HW panel, G4b) ---------------------------------

    def _start_hw_ticker(self, stop: threading.Event) -> None:
        """Broadcast a 1 Hz hardware-telemetry frame while the run is active — independent of the
        algorithm (PPO / neuroevolution / Q-learning all light up the HW panel). The daemon thread
        owns ``stop`` (this run's event, set in ``_run``'s finally) and also self-retires if the run
        leaves an active state, so it can never broadcast for a *different* run than the one it began."""

        def loop() -> None:
            from app.services.hw_stats import sample  # lazy: psutil/pynvml off the boot path

            while not stop.wait(_HW_STATS_INTERVAL):
                with self._lock:
                    active = self._state in ("running", "paused", "stopping")
                if not active:
                    break
                try:
                    frame = HwStatsFrame(stats=sample()).model_dump()
                except Exception:  # noqa: BLE001 — telemetry must never disturb the run
                    logger.debug("HW stats sample failed", exc_info=True)
                    continue
                self._broadcast(frame)

        threading.Thread(target=loop, name="hw-stats", daemon=True).start()

    # -- worker thread ----------------------------------------------------------

    def _run(
        self,
        config: TrainConfig,
        gym_id: str,
        control: TrainControl,
        resume: bytes | None = None,
        hw_stop: threading.Event | None = None,
    ) -> None:
        try:
            if config.algo == "neuroevolution":
                from app.services.trainer_evolution import train_evolution  # lazy: numpy/gym

                terminal = train_evolution(
                    config,
                    gym_id,
                    control,
                    self._emit_evolution,
                    self._publish_predict,
                    self._on_snapshot,
                    resume,
                )
            elif config.algo == "q_learning":
                from app.services.trainer_q import train_q_learning  # lazy: numpy/gym

                terminal = train_q_learning(
                    config,
                    gym_id,
                    control,
                    self._emit_q_learning,
                    self._emit_qtable,
                    self._publish_predict,
                    self._on_snapshot,
                    resume,
                )
            elif is_competitive_ma(get_env(config.env_id)):
                # Heterogeneous, competitive multi-agent (simple_tag) → per-species frozen self-play
                # (G7b-2, ADR-048). Still algo=="ppo" in the UI, but a different trainer: two shared
                # policies (one per species) alternating against each other's frozen snapshot. Publishes
                # BOTH species' preview policies and the two-line ecosystem metrics frame.
                from app.services.trainer_tag import train_tag  # lazy: loads torch/SB3

                terminal = train_tag(
                    config,
                    gym_id,
                    control,
                    self._emit_ma_metrics,
                    self._publish_predicts,
                    self._on_snapshot,
                    resume,
                )
            elif is_board_game(get_env(config.env_id)):
                # Board game → routed by algo to one of the two board trainers (both reuse the standard
                # metrics+progress frames = the eval-vs-reference-MCTS skill curve, the single-policy
                # preview publish, and the snapshot). This is the parallel to is_competitive_ma → train_tag.
                if config.algo == "alphazero":
                    # AlphaZero-lite (G6f, ADR-055): a CNN policy+value net guides MCTS and learns by
                    # pure self-play (no teacher) — the board branch's algorithm jump, GPU when available.
                    from app.services.trainer_az import train_az  # lazy: loads torch via az_net

                    terminal = train_az(
                        config,
                        gym_id,
                        control,
                        self._emit_metrics,
                        self._emit_progress,
                        self._publish_predict,
                        self._on_snapshot,
                        resume,
                    )
                else:
                    # MaskablePPO-vs-MCTS-teacher (G6b, ADR-051): masked PPO learns by playing the G6a
                    # MCTS teacher (action mask from legal_actions()). Surfaced as algo=="ppo".
                    from app.services.trainer_board import (
                        train_board,  # lazy: loads torch/SB3 + sb3-contrib
                    )

                    terminal = train_board(
                        config,
                        gym_id,
                        control,
                        self._emit_metrics,
                        self._emit_progress,  # board emits progress too → the Reward tab (progressHistory) fills
                        self._publish_predict,
                        self._on_snapshot,
                        resume,
                    )
            else:
                from app.services.trainer_ppo import train_ppo  # lazy: loads torch/SB3

                terminal = train_ppo(
                    config,
                    gym_id,
                    control,
                    self._emit_metrics,
                    self._emit_progress,
                    self._publish_predict,
                    self._on_snapshot,
                    resume,
                )
            with self._lock:
                self._state = terminal
        except Exception as exc:  # noqa: BLE001 — surface any trainer failure as state
            logger.exception("Training run failed")
            with self._lock:
                self._state = "error"
                self._error = str(exc)
        finally:
            preview_streamer.detach_run()  # stop the preview + close its render env
            if hw_stop is not None:
                hw_stop.set()  # retire *this run's* HW-telemetry ticker promptly (own event, no race)
        self._persist_run()  # archive the finished run for history / comparison (D2)
        self._broadcast_status()

    def _persist_run(self) -> None:
        """Record a completed run (config + metric frames) for later comparison.

        Only runs that finished/stopped *and* reached ≥10% of the env's solved score are
        archived — sub-10% noise is dropped (see :func:`should_archive`). Best-effort: a
        storage hiccup must never crash the worker thread or mask the run's terminal state.
        """
        with self._lock:
            state = self._state
            config = self._config
            metrics = list(self._metrics_log)
            started_at = self._run_started_at
        if config is None or started_at is None or not metrics:
            return
        spec = get_env(config.env_id)
        solved_score = spec.solved_score if spec is not None else 0.0
        final = final_score(config, metrics)
        if not should_archive(state, final, solved_score):
            logger.info(
                "Run not archived (state=%s, final=%s, <10%% of %s)", state, final, solved_score
            )
            return
        try:
            self._runs.save(
                config, metrics, state=state, started_at=started_at, solved_score=solved_score
            )
        except Exception:  # noqa: BLE001 — never let history IO disturb the run lifecycle
            logger.exception("Failed to record run history")

    def _on_snapshot(self, artifact: CheckpointArtifact) -> None:
        """Hold the trainer's latest serialized model so a Save can persist it."""
        with self._lock:
            self._snapshot = artifact

    def _publish_predict(self, predict: Callable[[object], int]) -> None:
        """Hand the preview streamer a decoupled predict fn (numpy snapshot for PPO, evolution
        leader for neuroevolution) — never the live SB3 model, which would perturb training."""
        preview_streamer.set_policy(predict)

    def _publish_predicts(self, predicts: dict[str, Callable[[object], object]]) -> None:
        """Hand the preview a per-species {role -> predict} map for competitive self-play (simple_tag):
        each agent is driven by its own species' decoupled snapshot, so the swarm renders real
        predators vs. real prey as the two co-evolve (ADR-019 still holds — both are numpy snapshots)."""
        preview_streamer.set_policies(predicts)

    def _emit_metrics(self, metrics: TrainingMetrics) -> None:
        frame = metrics.model_dump()
        with self._lock:
            self._last_metrics = metrics
            self._timesteps = metrics.timesteps
            self._metrics_log.append(frame)
            del self._metrics_log[:-_METRICS_LOG_CAP]
        self._broadcast(frame)
        if metrics.ep_rew_mean is not None:
            self._record_highscore(metrics.ep_rew_mean, iteration=metrics.iteration)

    def _emit_evolution(self, ev: EvolutionMetrics) -> None:
        frame = ev.model_dump()
        with self._lock:
            self._timesteps = ev.timesteps
            self._last_evolution = ev
            self._metrics_log.append(frame)
            del self._metrics_log[:-_METRICS_LOG_CAP]
        self._broadcast(frame)
        self._record_highscore(ev.best_fitness, generation=ev.generation)

    def _emit_q_learning(self, m: QLearningMetrics) -> None:
        frame = m.model_dump()
        with self._lock:
            self._timesteps = m.timesteps
            self._last_q_learning = m
            self._metrics_log.append(frame)
            del self._metrics_log[:-_METRICS_LOG_CAP]
        self._broadcast(frame)
        if m.ep_rew_mean is not None:
            self._record_highscore(m.ep_rew_mean, iteration=m.episode)

    def _emit_ma_metrics(self, m: MultiAgentMetrics) -> None:
        frame = m.model_dump()
        with self._lock:
            self._timesteps = m.timesteps
            self._last_ma_metrics = m
            self._metrics_log.append(frame)
            del self._metrics_log[:-_METRICS_LOG_CAP]
        self._broadcast(frame)
        # The predator headline drives the high-score board (its solved_score is predator-side).
        if m.ep_rew_mean is not None:
            self._record_highscore(m.ep_rew_mean, iteration=m.round)

    def _emit_qtable(self, frame: QTableFrame) -> None:
        # The heatmap snapshot is *not* logged into _metrics_log (it is large and the chart/history
        # only need the QLearningMetrics frame); just retain the latest for late-join reconcile.
        with self._lock:
            self._last_qtable = frame
        self._broadcast(frame.model_dump())

    def _record_highscore(
        self, score: float, *, generation: int | None = None, iteration: int | None = None
    ) -> None:
        """Persist a new all-time best for the active env and push it to clients live.

        Called from the trainer thread on each metrics/evolution frame; the store only writes
        (and only returns a record) when ``score`` actually beats the stored best, so this is
        a cheap no-op for the common case. File IO runs outside the manager lock.
        """
        with self._lock:
            cfg = self._config
        if cfg is None:
            return
        meta = make_meta(cfg.algo, cfg.seed, generation=generation, iteration=iteration)
        record = self._hi.record(cfg.env_id, score, meta)
        if record is not None:
            self._broadcast(record.model_dump())

    def _emit_progress(self, progress: TrainingProgress) -> None:
        with self._lock:
            self._timesteps = progress.timesteps
        self._broadcast(progress.model_dump())

    # -- status / broadcast -----------------------------------------------------

    def status(self) -> TrainStatus:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> TrainStatus:
        cfg = self._config
        return TrainStatus(
            state=self._state,
            env_id=cfg.env_id if cfg else None,
            algo=cfg.algo if cfg else None,
            seed=cfg.seed if cfg else None,
            timesteps=self._timesteps,
            total_timesteps=cfg.total_timesteps if cfg else 0,
            config=cfg,
            last_metrics=self._last_metrics,
            last_evolution=self._last_evolution,
            last_q_learning=self._last_q_learning,
            last_qtable=self._last_qtable,
            last_ma_metrics=self._last_ma_metrics,
            error=self._error,
        )

    def _broadcast_status(self) -> None:
        self._broadcast(self.status().model_dump())

    def _broadcast(self, frame: dict) -> None:
        """Schedule a WS broadcast on the bound loop from any thread (best-effort)."""
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._cm.broadcast(frame), loop)
        except Exception:  # noqa: BLE001 — never let a dead loop kill training
            logger.debug("WS broadcast skipped (loop unavailable)", exc_info=True)


# Module singleton, wired to the shared connection manager.
training_manager = TrainingManager(manager)
