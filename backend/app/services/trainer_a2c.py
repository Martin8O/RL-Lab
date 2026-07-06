"""SB3 Advantage Actor-Critic trainer (S5d) — the 8th algorithm, on-policy actor-critic.

A2C is **PPO's simpler predecessor** and its on-policy sibling: the same actor-critic shape (one
network with a policy head + a value head, trained with gradients on freshly-collected rollouts —
**no replay buffer**), but *without* PPO's clipped-surrogate objective. It does one plain
policy-gradient update per rollout instead of PPO's several clipped epochs, which makes it the
natural **PPO-vs-A2C teaching comparison**: same family, so any gap is down to PPO's clipping +
multi-epoch reuse. It is a **peer trainer** behind the same manager (ADR-004/028) and reuses the
whole rest of the on-policy lane — the rollout-boundary metrics, the ~1 Hz ``_progress_ticker``, the
decoupled numpy preview policy, the ``[min_score, solved_score]`` skill meter, the reward chart
(``ep_rew_mean``) — with **no new WS frame / TS type**.

Because A2C *is* an on-policy actor-critic just like PPO, this module is deliberately a thin near-copy
of :mod:`app.services.trainer_ppo` and imports its stable helpers verbatim (``_ep_means``,
``_progress_ticker``, ``_build_preview_predict``, and the sink types). What differs from PPO:

* **No clip / no multi-epoch update.** A2C's ``train()`` is a single fast pass per rollout, so there is
  no ``_InterruptiblePPO`` multi-epoch stop hack — a Stop is observed at the next rollout step (fast,
  since A2C's rollouts are tiny). A2C is offered only on single-agent vector envs, so there is no heavy
  parameter-sharing update to strand a Stop on either.
* **Short rollout.** ``n_steps`` defaults to 5 (PPO uses 2048): A2C updates after only a handful of
  steps. The classic recipe leans on many parallel envs for a stable gradient; we run a single env, so
  the registry nudges the ★ ``n_steps`` up a little per env to steady it.
* **Both action types.** A2C handles discrete **and** continuous (``Box``) actions (unlike DQN's
  discrete-only or SAC/TD3's continuous-only gates), so ``_build_preview_predict`` — which already
  dispatches int (arg-max) vs box (clipped mean) by action space — covers every A2C env unchanged.
* **Loss signal.** SB3's A2C logs ``train/value_loss`` / ``train/policy_loss`` / ``train/entropy_loss``
  separately rather than PPO's single ``train/loss``; the metrics frame's ``loss`` field carries the
  **value loss** (the critic's TD error — the most loss-like scalar, and it trends down as the critic
  improves), falling back to a combined ``train/loss`` if a future SB3 version records one.

The decoupled preview policy (ADR-019) is a standalone numpy forward over copied weights, never the
live model — identical to PPO's, and load-bearing for the same proven reason (concurrent SB3 access
perturbs an on-policy run). A2C is offered only on classic-control vector envs (no image, no MuJoCo),
so the CnnPolicy / VecNormalize branches PPO carries are unreachable here and deliberately omitted.
"""

import io
import threading
import time
from typing import Any

from stable_baselines3 import A2C
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn

from app.envs.factory import make_env
from app.schemas.training import TrainConfig, TrainingMetrics, TrainState
from app.services.checkpoints import CheckpointArtifact

# Reuse PPO's stable, load-bearing on-policy helpers rather than duplicate them: the recent-episode
# mean reader, the decoupled ~1 Hz progress ticker, the numpy/CPU preview-policy builder, and the sink
# types. All read only counters/buffers/weights present on any SB3 on-policy model, so they work for
# A2C unchanged (the same way trainer_dqn reuses them for the off-policy lane).
from app.services.trainer_ppo import (
    MetricsSink,
    PredictPublisher,
    ProgressSink,
    SnapshotSink,
    _build_preview_predict,
    _ep_means,
    _progress_ticker,
)

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}


def _snapshot(model: BaseAlgorithm, total_timesteps: int, iteration: int) -> CheckpointArtifact:
    """Serialize the model to an in-memory ``model.zip`` for the checkpoint store (algo="a2c").

    Called at a rollout boundary (or after ``learn`` returns) — quiescent points on the trainer
    thread — so it never races SB3's optimizer. A2C's net is tiny, so snapshotting each rollout is
    negligible. No VecNormalize embedding (A2C is not offered on the MuJoCo family), so the blob is
    just the policy + value net + params, like an un-normalized PPO checkpoint.
    """
    rew, _ = _ep_means(model)
    buf = io.BytesIO()
    model.save(buf)
    return CheckpointArtifact(
        algo="a2c",
        blob=buf.getvalue(),
        artifact_name="model.zip",
        reward=rew,
        timesteps=int(model.num_timesteps),
        total_timesteps=total_timesteps,
        iteration=iteration,
    )


class _MetricsCallback(BaseCallback):
    """Emits a metrics frame after each A2C rollout and honours pause/stop (same shape as PPO's)."""

    def __init__(
        self,
        control: Any,
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
        # Park here while paused; wake and abort if a stop was requested (A2C's short rollouts make
        # this responsive without PPO's between-epochs stop hack).
        self._control.wait_if_paused()
        return not self._control.stop_requested

    def _on_rollout_end(self) -> None:
        self.iteration_count += 1
        ep_rew_mean, ep_len_mean = _ep_means(self.model)

        # A2C logs value/policy/entropy losses separately (not PPO's single train/loss). Surface the
        # value loss as the chart's "loss" — the critic's TD error, the most loss-like scalar, trending
        # down as the value head improves. Prefer a combined train/loss if a future SB3 records one.
        # All are absent on the very first rollout (before the first update).
        recorded = self.model.logger.name_to_value
        loss = recorded.get("train/loss")
        if loss is None:
            loss = recorded.get("train/value_loss")
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
        # Snapshot at this rollout boundary so "Save" can persist the live model; refresh the decoupled
        # numpy preview policy with this rollout's weights (never the live model — ADR-019).
        if self._on_snapshot is not None:
            self._on_snapshot(_snapshot(self.model, self._total, self.iteration_count))
        if self._on_policy is not None:
            self._on_policy(_build_preview_predict(self.model))


def _build_model(config: TrainConfig, gym_id: str) -> A2C:
    """Build an A2C model for a single-agent vector env (MlpPolicy on CPU).

    A2C is offered only on the classic-control vector envs (discrete + continuous Box), so this is
    always the MlpPolicy/CPU path — no image (CnnPolicy) or MuJoCo (VecNormalize) branch. SB3 wraps the
    factory env in Monitor + DummyVecEnv, so ``ep_rew_mean`` stays raw (the skill meter reads like PPO's).
    """
    hp = config.a2c
    assert hp is not None, "A2C run without an a2c hyperparam block"
    env = make_env(config.env_id, gym_id)
    policy_kwargs = {
        "net_arch": [hp.neurons_per_layer] * hp.n_hidden_layers,
        "activation_fn": _ACTIVATIONS[hp.activation],
    }
    return A2C(
        "MlpPolicy",
        env,
        seed=config.seed,
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        n_steps=hp.n_steps,
        gae_lambda=hp.gae_lambda,
        ent_coef=hp.ent_coef,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=0,
    )


def _load_model(config: TrainConfig, gym_id: str, resume_blob: bytes) -> A2C:
    """Rebuild an A2C model from a saved ``model.zip`` and attach a fresh env (resume).

    The env is built through the shared factory exactly as in training, so ``A2C.load``'s
    ``check_for_correct_spaces`` matches. ``num_timesteps`` is restored so ``reset_num_timesteps=False``
    continues the counter. A2C is on-policy (no replay buffer), so resume is clean — no empty-buffer
    warmup gate is needed (unlike the off-policy SAC/TD3/DQN lane).
    """
    env = make_env(config.env_id, gym_id)
    return A2C.load(io.BytesIO(resume_blob), env=env, device="cpu")


def train_a2c(
    config: TrainConfig,
    gym_id: str,
    control: Any,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train A2C to completion (or until stopped). Returns the terminal state.

    Same contract as :func:`app.services.trainer_ppo.train_ppo` — blocks the calling thread (the
    manager runs it off the event loop), emits the standard ``metrics``/``progress`` frames, publishes a
    decoupled numpy preview policy at every rollout boundary, and snapshots at quiescent points.
    ``resume_blob`` continues the timestep counter (so ``config.total_timesteps`` is the *absolute*
    target).
    """
    resuming = resume_blob is not None
    model = (
        _load_model(config, gym_id, resume_blob)
        if resume_blob is not None
        else _build_model(config, gym_id)
    )
    if on_policy is not None:
        on_policy(_build_preview_predict(model))  # initial preview policy (before the first rollout)

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
        name="a2c-progress",
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
        # Terminal snapshot — captures the final (or stopped) model accurately even if the last
        # rollout-boundary snapshot predated the final update.
        if on_snapshot is not None:
            on_snapshot(_snapshot(model, config.total_timesteps, callback.iteration_count))
        if model.env is not None:
            model.env.close()
    return "stopped" if control.stop_requested else "finished"
