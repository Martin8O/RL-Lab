"""SB3 PPO trainer for CartPole — runs synchronously on a background thread.

Imported lazily by the training manager so that torch/SB3 are only loaded when a run
actually starts (keeps /health, /envs and the WS echo torch-free and fast to boot).
"""

import threading
import time
from collections.abc import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn

from app.schemas.training import (
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
)
from app.services.train_control import TrainControl

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}
_PROGRESS_INTERVAL = 1.0  # seconds between live progress frames

MetricsSink = Callable[[TrainingMetrics], None]
ProgressSink = Callable[[TrainingProgress], None]


def _ep_means(model: BaseAlgorithm) -> tuple[float | None, float | None]:
    """Mean reward/length over SB3's recent-episode buffer, or ``(None, None)``.

    Read from the progress-ticker thread while ``learn()`` runs on another thread. The
    buffer is a deque that ``learn`` appends to, so we snapshot defensively and treat a
    rare concurrent mutation as "no update this tick".
    """
    buf = getattr(model, "ep_info_buffer", None)
    if not buf:
        return None, None
    try:
        episodes = list(buf)  # snapshot; may raise if mutated mid-iteration
    except RuntimeError:
        return None, None
    if not episodes:
        return None, None
    n = len(episodes)
    return sum(e["r"] for e in episodes) / n, sum(e["l"] for e in episodes) / n


class _MetricsCallback(BaseCallback):
    """Emits a metrics frame after each PPO rollout and honours pause/stop."""

    def __init__(
        self,
        control: TrainControl,
        on_metrics: MetricsSink,
        total_timesteps: int,
        started_at: float,
    ) -> None:
        super().__init__()
        self._control = control
        self._on_metrics = on_metrics
        self._total = total_timesteps
        self._started_at = started_at
        self.iteration_count = 0  # read by the progress ticker (a separate thread)

    def _on_step(self) -> bool:
        # Park here while paused; wake and abort if a stop was requested.
        self._control.wait_if_paused()
        return not self._control.stop_requested

    def _on_rollout_end(self) -> None:
        self.iteration_count += 1
        ep_rew_mean, ep_len_mean = _ep_means(self.model)

        # loss / lr are recorded during the previous update; absent on the first rollout.
        recorded = self.model.logger.name_to_value
        loss = recorded.get("train/loss")
        lr = recorded.get("train/learning_rate")

        self._on_metrics(
            TrainingMetrics(
                iteration=self.iteration_count,
                timesteps=int(self.model.num_timesteps),
                total_timesteps=self._total,
                ep_rew_mean=ep_rew_mean,
                ep_len_mean=ep_len_mean,
                loss=float(loss) if loss is not None else None,
                learning_rate=float(lr) if lr is not None else None,
                elapsed=time.monotonic() - self._started_at,
            )
        )


def _progress_ticker(
    model: PPO,
    callback: _MetricsCallback,
    control: TrainControl,
    on_progress: ProgressSink,
    total_timesteps: int,
    started_at: float,
    stop_event: threading.Event,
) -> None:
    """Emit a progress frame every ``_PROGRESS_INTERVAL`` seconds until stopped.

    Decoupled from SB3's per-step callback (which is dormant during the PPO update phase),
    so the live stats refresh at a steady ~1 Hz regardless of training phase. Mirrors the
    decoupled preview streamer (ADR-008). Reads model counters/buffers only — it never
    mutates model state — so it cannot affect training reproducibility.
    """
    last_t = started_at
    last_steps = 0
    sps_ema: float | None = None
    last_rew: float | None = None
    last_len: float | None = None

    while not stop_event.wait(_PROGRESS_INTERVAL):
        now = time.monotonic()
        if control.paused:
            # Hold steady while paused (the preview is frozen too); keep the throughput
            # baseline fresh so steps/s doesn't lurch on resume.
            last_t, last_steps = now, int(model.num_timesteps)
            continue

        steps = int(model.num_timesteps)
        dt = now - last_t
        gained = steps - last_steps
        # Update the throughput EMA only when steps actually advanced, so the displayed
        # rate stays at the collection speed instead of dropping to ~0 during the (step-less)
        # update phase — while still emitting a frame every tick so the UI keeps refreshing.
        if dt > 0 and gained > 0:
            instant = gained / dt
            sps_ema = instant if sps_ema is None else 0.3 * instant + 0.7 * sps_ema

        rew, length = _ep_means(model)
        if rew is not None:
            last_rew, last_len = rew, length

        on_progress(
            TrainingProgress(
                iteration=callback.iteration_count,
                timesteps=steps,
                total_timesteps=total_timesteps,
                steps_per_sec=sps_ema or 0.0,
                ep_rew_mean=last_rew,
                ep_len_mean=last_len,
                elapsed=now - started_at,
            )
        )
        last_t, last_steps = now, steps


def _build_model(config: TrainConfig, gym_id: str) -> PPO:
    hp = config.hyperparams
    policy_kwargs = {
        "net_arch": [hp.neurons_per_layer] * hp.n_hidden_layers,
        "activation_fn": _ACTIVATIONS[hp.activation],
    }
    # Passing seed= makes SB3 seed python/numpy/torch + the env action space, so the
    # same seed reproduces the early metrics on CPU.
    return PPO(
        "MlpPolicy",
        gym_id,
        seed=config.seed,
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        clip_range=hp.clip_range,
        ent_coef=hp.ent_coef,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
    )


def train_ppo(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_model_ready: Callable[[PPO], None] | None = None,
) -> TrainState:
    """Train PPO to completion (or until stopped). Returns the terminal state.

    Blocks the calling thread; the manager runs this off the event loop. ``on_model_ready``
    (if given) is called once the model is built so the preview streamer can read its policy.
    A daemon ticker thread emits ~1 Hz progress frames for the duration of ``learn()``.
    """
    model = _build_model(config, gym_id)
    if on_model_ready is not None:
        on_model_ready(model)

    started_at = time.monotonic()
    callback = _MetricsCallback(control, on_metrics, config.total_timesteps, started_at)
    stop_event = threading.Event()
    ticker = threading.Thread(
        target=_progress_ticker,
        args=(
            model,
            callback,
            control,
            on_progress,
            config.total_timesteps,
            started_at,
            stop_event,
        ),
        name="ppo-progress",
        daemon=True,
    )
    ticker.start()
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
    finally:
        stop_event.set()  # wake + retire the ticker
        ticker.join(timeout=2.0)
        if model.env is not None:
            model.env.close()
    return "stopped" if control.stop_requested else "finished"
