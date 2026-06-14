"""SB3 PPO trainer for CartPole — runs synchronously on a background thread.

Imported lazily by the training manager so that torch/SB3 are only loaded when a run
actually starts (keeps /health, /envs and the WS echo torch-free and fast to boot).
"""

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
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
from app.services.checkpoints import CheckpointArtifact
from app.services.train_control import TrainControl

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}
_PROGRESS_INTERVAL = 1.0  # seconds between live progress frames

MetricsSink = Callable[[TrainingMetrics], None]
ProgressSink = Callable[[TrainingProgress], None]
SnapshotSink = Callable[[CheckpointArtifact], None]
# Hands the decoupled preview a self-contained predict fn (obs → action) over a weight snapshot.
PredictPublisher = Callable[[Callable[[object], int]], None]


def _build_numpy_predict(model: BaseAlgorithm) -> Callable[[object], int]:
    """A standalone **numpy** forward over a snapshot of the policy's action path.

    The preview must never call ``model.predict`` on the *live* model: doing so concurrently with
    ``learn()`` measurably perturbs PPO's training trajectory (proven empirically — concurrent SB3
    model access diverges a same-seed run, while pure compute does not). A numpy forward over
    copied weights cannot touch the trainer's torch state, so the preview stays a true read-only
    observer (mirrors how the neuroevolution trainer already publishes its preview policy).

    Built at a rollout boundary on the trainer thread (a quiescent point), so the weight copy
    never races the optimizer. Handles any ``net_arch`` depth + tanh/relu; discrete actions only
    (argmax of the action logits == SB3's ``deterministic=True``).
    """
    policy: Any = model.policy  # torch dynamic attrs (mlp_extractor/action_net) aren't typed

    def arr(t: Any) -> np.ndarray:
        return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)

    pi_net = policy.mlp_extractor.policy_net
    layers = [(arr(m.weight), arr(m.bias)) for m in pi_net if isinstance(m, nn.Linear)]
    act_w, act_b = arr(policy.action_net.weight), arr(policy.action_net.bias)
    relu = any(isinstance(m, nn.ReLU) for m in pi_net)

    def predict(obs: object) -> int:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in layers:
            x = x @ w.T + b
            x = np.maximum(0.0, x) if relu else np.tanh(x)
        return int(np.argmax(x @ act_w.T + act_b))

    return predict


def _snapshot(model: BaseAlgorithm, total_timesteps: int, iteration: int) -> CheckpointArtifact:
    """Serialize the model to an in-memory ``model.zip`` for the checkpoint store.

    Called at a rollout boundary (or after ``learn`` returns) — both quiescent points on the
    trainer thread — so it never races SB3's optimizer. CartPole's net is tiny, so doing this
    each rollout is negligible; for heavy GPU envs (Phase G) this would move to an on-demand
    barrier snapshot instead.
    """
    rew, _ = _ep_means(model)
    buf = io.BytesIO()
    model.save(buf)
    return CheckpointArtifact(
        algo="ppo",
        blob=buf.getvalue(),
        artifact_name="model.zip",
        reward=rew,
        timesteps=int(model.num_timesteps),
        total_timesteps=total_timesteps,
        iteration=iteration,
    )


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
        on_snapshot: SnapshotSink | None = None,
        on_policy: PredictPublisher | None = None,
    ) -> None:
        super().__init__()
        self._control = control
        self._on_metrics = on_metrics
        self._on_snapshot = on_snapshot
        self._on_policy = on_policy
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
        # Capture a snapshot at this rollout boundary so "Save" can persist the live model
        # mid-run (the terminal snapshot below captures the final/stopped model).
        if self._on_snapshot is not None:
            self._on_snapshot(_snapshot(self.model, self._total, self.iteration_count))
        # Refresh the preview's decoupled (numpy) policy with this rollout's weights, so it shows
        # the learning progress without the live model.predict that would perturb training.
        if self._on_policy is not None:
            self._on_policy(_build_numpy_predict(self.model))


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


def _load_model(gym_id: str, resume_blob: bytes) -> PPO:
    """Rebuild a PPO model from a saved ``model.zip`` and attach a fresh env.

    ``PPO.load`` runs ``check_for_correct_spaces`` against the env, so loading a checkpoint
    whose observation/action space no longer matches the env raises (surfaced as a clear
    error by the manager). ``num_timesteps`` is restored, so ``reset_num_timesteps=False``
    continues the global step counter.
    """
    import gymnasium as gym  # lazy: keep gym out of startup

    env = gym.make(gym_id)
    return PPO.load(io.BytesIO(resume_blob), env=env, device="cpu")


def train_ppo(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train PPO to completion (or until stopped). Returns the terminal state.

    Blocks the calling thread; the manager runs this off the event loop. ``on_policy`` (if given)
    is handed a self-contained numpy predict fn over the current weights — initially and at every
    rollout boundary — so the decoupled preview can render the live policy *without* calling into
    the live SB3 model (which would perturb training). A daemon ticker thread emits ~1 Hz progress
    frames for the duration of ``learn()``.

    ``resume_blob`` resumes from a saved ``model.zip`` (continuing the timestep counter, so
    ``config.total_timesteps`` is the *absolute* target). ``on_snapshot`` receives a
    serialized model at each rollout boundary and once more after ``learn`` returns, so the
    checkpoint store can persist the current (or final) model.
    """
    resuming = resume_blob is not None
    model = (
        _load_model(gym_id, resume_blob)
        if resume_blob is not None
        else _build_model(config, gym_id)
    )
    if on_policy is not None:
        on_policy(_build_numpy_predict(model))  # initial preview policy (before the first rollout)

    started_at = time.monotonic()
    callback = _MetricsCallback(
        control, on_metrics, config.total_timesteps, started_at, on_snapshot, on_policy
    )
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
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            reset_num_timesteps=not resuming,
        )
    finally:
        stop_event.set()  # wake + retire the ticker
        ticker.join(timeout=2.0)
        # Terminal snapshot — captures the final (or stopped) model accurately, even if the
        # last rollout-boundary snapshot predated the final update phase.
        if on_snapshot is not None:
            on_snapshot(_snapshot(model, config.total_timesteps, callback.iteration_count))
        if model.env is not None:
            model.env.close()
    return "stopped" if control.stop_requested else "finished"
