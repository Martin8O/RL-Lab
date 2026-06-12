"""Training lifecycle manager: one active run, controllable, streaming over WS.

SB3 runs synchronously on a daemon thread; metric/status frames are marshalled back onto
the FastAPI event loop with ``run_coroutine_threadsafe`` so the async connection manager
can broadcast them. No ML imports here — the trainer (and torch) is imported lazily inside
:meth:`_run` only when a run actually starts.
"""

import asyncio
import threading

from app.core.logging import get_logger
from app.envs.registry import get_env
from app.schemas.training import (
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
    TrainStatus,
)
from app.services.connection_manager import ConnectionManager, manager
from app.services.preview_streamer import preview_streamer
from app.services.train_control import TrainControl

logger = get_logger(__name__)


class AlreadyRunningError(RuntimeError):
    """Raised when a start is attempted while a run is already active."""


class InvalidConfigError(ValueError):
    """Raised when the requested env/algo is unknown or unsupported."""


class TrainingManager:
    """Owns the single active training run and mirrors its state over WebSocket."""

    def __init__(self, connection_manager: ConnectionManager) -> None:
        self._cm = connection_manager
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._control: TrainControl | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        self._state: TrainState = "idle"
        self._config: TrainConfig | None = None
        self._timesteps = 0
        self._last_metrics: TrainingMetrics | None = None
        self._error: str | None = None

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

        with self._lock:
            if self._state in ("running", "paused", "stopping"):
                raise AlreadyRunningError("A training run is already active")
            self._control = TrainControl()
            self._config = config
            self._timesteps = 0
            self._last_metrics = None
            self._error = None
            self._state = "running"

        self._broadcast_status()
        self._thread = threading.Thread(
            target=self._run,
            args=(config, spec.gym_id, self._control),
            name="ppo-trainer",
            daemon=True,
        )
        self._thread.start()
        # Let the (decoupled) preview streamer begin watching this run, if visual is on.
        preview_streamer.attach_run(spec.gym_id)
        return self.status()

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

    def _run(self, config: TrainConfig, gym_id: str, control: TrainControl) -> None:
        from app.services.trainer_ppo import train_ppo  # lazy: loads torch/SB3

        try:
            terminal = train_ppo(
                config,
                gym_id,
                control,
                self._emit_metrics,
                self._emit_progress,
                self._publish_model,
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
        self._broadcast_status()

    def _publish_model(self, model: object) -> None:
        """Hand the preview streamer a deterministic predict fn over the live model."""

        def predict(obs: object) -> object:
            action, _ = model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
            return action

        preview_streamer.set_policy(predict)

    def _emit_metrics(self, metrics: TrainingMetrics) -> None:
        with self._lock:
            self._last_metrics = metrics
            self._timesteps = metrics.timesteps
        self._broadcast(metrics.model_dump())

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
