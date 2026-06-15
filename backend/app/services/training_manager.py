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
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
    TrainStatus,
)
from app.services.checkpoints import CheckpointArtifact, CheckpointStore, checkpoint_store
from app.services.connection_manager import ConnectionManager, manager
from app.services.highscores import HighScoreStore, highscores, make_meta
from app.services.preview_streamer import preview_streamer
from app.services.runs import RunStore, final_score, run_store, should_archive
from app.services.train_control import TrainControl

logger = get_logger(__name__)

# Cap on metric frames retained for a checkpoint's metrics.json (per-rollout / per-generation
# frames only — the high-frequency progress ticks are not logged). Generous for any CartPole run.
_METRICS_LOG_CAP = 10_000


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
        self._thread = threading.Thread(
            target=self._run,
            args=(config, gym_id, self._control, resume),
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

    def load_checkpoint(self, checkpoint_id: str) -> TrainStatus:
        """Resume training from a saved checkpoint (PPO continues; evolution continues)."""
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

        config = loaded.config
        # PPO continues the global step counter, so target an additional full budget on top of
        # where the checkpoint left off. Evolution continues by generation (handled in-trainer).
        if config.algo == "ppo":
            config = config.model_copy(
                update={"total_timesteps": loaded.meta.timesteps + loaded.config.total_timesteps}
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

    # -- worker thread ----------------------------------------------------------

    def _run(
        self,
        config: TrainConfig,
        gym_id: str,
        control: TrainControl,
        resume: bytes | None = None,
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
