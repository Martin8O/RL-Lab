"""SB3 Soft Actor-Critic trainer (S5a) — the 5th algorithm, off-policy continuous control.

SAC is the off-policy, entropy-regularized continuous-control algorithm that actually *solves* the
high-DoF MuJoCo robots (≈5000–6000 on Humanoid in 1–3M steps — far more sample-efficient than PPO's
20M+ and a higher ceiling). It is a **peer trainer** behind the same manager as PPO / neuroevolution /
Q-learning / AlphaZero (ADR-004/028) and reuses the whole rest of the lane — the box-action
play/predict, the ``[min_score, solved_score]`` skill meter, the reward chart (``ep_rew_mean``) — with
**no new WS frame / TS type**. Only the off-policy *shape* and the param surface differ from PPO.

Two things differ from :mod:`app.services.trainer_ppo`:

* **Off-policy ⇒ no rollout boundary.** PPO emits a metrics frame per rollout; SAC collects single
  transitions into a replay buffer and updates continuously, so there is no natural boundary. Instead
  the metrics callback emits on a **step interval** (``_METRICS_INTERVAL_STEPS``) — a snapshot + a
  fresh decoupled preview policy ride the same interval. A ~1 Hz progress ticker (shared with PPO) keeps
  the live stats + reward curve smooth between those frames.
* **Raw obs/rewards (NO VecNormalize).** Unlike the PPO MuJoCo path (G5c), SAC does not wrap the env in
  VecNormalize: its reward normalization is on-policy-shaped (a running return scaling) and would drift
  against a replay buffer of rewards stamped with *old* statistics, and the standard SAC recipe needs
  neither obs nor reward normalization. So ``ep_rew_mean`` is raw and the skill meter reads exactly like
  PPO's — and a PPO-vs-SAC comparison on one robot is apples-to-apples on the same raw scale.

Imports torch/SB3 lazily (via this module being imported only when a SAC run starts), like the other
trainers. The decoupled preview policy (ADR-019) is a CPU save/load copy of the actor, never the live
model — concurrent access to the training model perturbs it (proven for PPO; the same discipline holds).
"""

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from app.envs.factory import make_env
from app.schemas.training import TrainConfig, TrainingMetrics, TrainState
from app.services.checkpoints import CheckpointArtifact

# Reuse PPO's stable, load-bearing helpers rather than duplicate them: the recent-episode mean reader
# and the decoupled ~1 Hz progress ticker both read only `num_timesteps` / `ep_info_buffer` (present on
# SAC) + the callback's `iteration_count`, so they work unchanged for an off-policy run.
from app.services.trainer_ppo import (
    MetricsSink,
    PredictPublisher,
    ProgressSink,
    SnapshotSink,
    _ep_means,
    _progress_ticker,
)

# How many env steps between metrics frames (+ a snapshot + a refreshed preview policy). SAC has no
# rollout boundary, so this sets the chart/checkpoint cadence: ~100 frames over a 200k Pendulum run,
# ~500 over a 1M MuJoCo run — frequent enough to plot, cheap enough not to slow the GPU update loop.
_METRICS_INTERVAL_STEPS = 2_000

# Minimum completed episodes before the LIVE chart plots a reward (the off-policy fix, shared with TD3).
# The 1 Hz ticker fires within a few hundred steps, when the episode buffer holds only one or two
# high-variance episodes (often a lucky random-warmup one) — which read as a misleading "starts high then
# dips" before the rolling mean settled. Gating to a few episodes makes the curve start at the settled
# baseline and climb cleanly. Snapshots/checkpoints still use any available reward (the default min of 1).
_MIN_REPORT_EPISODES = 5


def _build_sac_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A read-only CPU forward over a save/load **copy** of SAC's policy (the decoupled preview).

    SAC's actor is a squashed-Gaussian MLP (``latent_pi`` → ``mu`` → tanh → action-space scaling), not
    PPO's ``mlp_extractor.policy_net`` + ``action_net``, so the PPO numpy forward doesn't apply. Rather
    than re-derive SAC's squashing/unscaling by hand (error-prone), round-trip the policy through SB3's
    own ``save``/``load`` into an **independent CPU policy** and let SB3 compute the deterministic action
    — the same isolation the CnnPolicy preview uses (ADR-019): the copy shares no tensor storage with the
    trainer, so forwarding it cannot perturb training. Built at a metrics-interval boundary (a quiescent
    point on the trainer thread). SAC's action space is always ``Box``, so the result is a clipped float
    vector the play/preview loop steps directly (the ADR-021 int|box contract — always the box arm here).
    """
    import torch

    buf = io.BytesIO()
    # save/load are typed for a str path but accept a file-like at runtime (same in-memory trick the
    # CnnPolicy snapshot uses); round-trips state_dict + constructor params, mapping tensors → CPU.
    model.policy.save(buf)  # type: ignore[arg-type]
    buf.seek(0)
    policy = model.policy.__class__.load(buf, device="cpu")  # type: ignore[arg-type]
    policy.set_training_mode(False)
    low = np.asarray(getattr(model.action_space, "low", -1.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 1.0), dtype=np.float32)

    def predict(obs: object) -> Any:
        with torch.no_grad():
            action, _ = policy.predict(np.asarray(obs), deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), low, high)

    return predict


def _snapshot(model: BaseAlgorithm, total_timesteps: int, iteration: int) -> CheckpointArtifact:
    """Serialize the model to an in-memory ``model.zip`` for the checkpoint store (algo="sac").

    Called at a metrics-interval boundary (or after ``learn`` returns) — quiescent points on the
    trainer thread — so it never races SB3's optimizer. SB3's ``save`` excludes the replay buffer by
    default, so the blob is light (policy + critics + params); resume rebuilds a fresh buffer.
    """
    rew, _ = _ep_means(model)
    buf = io.BytesIO()
    model.save(buf)
    return CheckpointArtifact(
        algo="sac",
        blob=buf.getvalue(),
        artifact_name="model.zip",
        reward=rew,
        timesteps=int(model.num_timesteps),
        total_timesteps=total_timesteps,
        iteration=iteration,
    )


class _MetricsCallback(BaseCallback):
    """Emits a metrics frame on a step interval and honours pause/stop (off-policy: no rollout end)."""

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
        self._next_emit = _METRICS_INTERVAL_STEPS

    def _on_step(self) -> bool:
        # Called every env step in collect_rollouts. Park here while paused; abort if a stop was asked.
        self._control.wait_if_paused()
        if self._control.stop_requested:
            return False
        if self.num_timesteps >= self._next_emit:
            self._next_emit += _METRICS_INTERVAL_STEPS
            self._emit()
        return True

    def _emit(self) -> None:
        self.iteration_count += 1
        # Gate the live chart reward to a few episodes so a 1–2-episode early fluke isn't plotted.
        ep_rew_mean, ep_len_mean = _ep_means(self.model, _MIN_REPORT_EPISODES)
        # SAC logs train/critic_loss (the headline learning signal — no single "train/loss" like PPO)
        # + train/learning_rate; both absent until the first gradient update (after learning_starts).
        recorded = self.model.logger.name_to_value
        loss = recorded.get("train/critic_loss")
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
        # A mid-run snapshot so "Save" can persist the live model; refresh the decoupled preview policy.
        if self._on_snapshot is not None:
            self._on_snapshot(_snapshot(self.model, self._total, self.iteration_count))
        if self._on_policy is not None:
            self._on_policy(_build_sac_predict(self.model))


def _sac_kwargs(config: TrainConfig) -> dict[str, Any]:
    """Map the SAC hyperparam block onto SB3 SAC constructor kwargs.

    ``ent_coef`` is "auto" (SAC self-tunes the entropy temperature — the recommended default) or a
    pinned numeric string. ``gradient_steps`` tracks ``train_freq`` so the update:collection ratio
    stays 1:1 (SB3's standard SAC shape). The net size is left at SB3's default ([256, 256]) — the PPO
    net sliders describe an on-policy MLP and don't map cleanly to SAC's actor/critic pair.
    """
    hp = config.sac
    assert hp is not None, "SAC run without a sac hyperparam block"
    ent_coef: Any = "auto" if hp.ent_coef == "auto" else float(hp.ent_coef)
    # learning_starts is the random warmup that fills the replay buffer before the first gradient update.
    # The default (10k) is SB3's MuJoCo value — ~2% of a 500k–2M robot budget, but a full 20% of a 50k
    # classic-control budget, so a short run would burn a fifth of itself on random steps + start learning
    # far too late (the "progress jumps to 25% then crawls" symptom). Scale it to ~2% of the budget, capped
    # at the configured value and floored at one batch so the first update has enough samples.
    warmup = min(hp.learning_starts, max(hp.batch_size, config.total_timesteps // 50))
    return {
        "learning_rate": hp.learning_rate,
        "gamma": hp.gamma,
        "tau": hp.tau,
        "buffer_size": hp.buffer_size,
        "batch_size": hp.batch_size,
        "learning_starts": warmup,
        "train_freq": hp.train_freq,
        "gradient_steps": hp.train_freq,  # 1:1 update:collection ratio
        "ent_coef": ent_coef,
    }


def _build_model(config: TrainConfig, gym_id: str) -> SAC:
    # device="cpu": SAC's small MLPs (256×256) make each per-step gradient update a tiny batch-256 forward,
    # and shuttling that to a GPU is latency-bound — the card idles while kernel-launch + sync overhead
    # dominates. Measured (Local/_probe_sac_device.py): HalfCheetah CPU 163 vs CUDA 120 steps/s (CPU +36%),
    # Pendulum a wash — the same MlpPolicy-faster-on-CPU result PPO has (ADR-056). So CPU is faster AND frees
    # the GPU. SB3 auto-wraps the single env in Monitor + DummyVecEnv, so ep_rew_mean stays raw — no
    # VecNormalize (see the module docstring).
    return SAC(
        "MlpPolicy",
        make_env(config.env_id, gym_id),
        seed=config.seed,
        device="cpu",
        verbose=0,
        **_sac_kwargs(config),
    )


def _load_model(config: TrainConfig, gym_id: str, resume_blob: bytes) -> SAC:
    """Rebuild a SAC model from a saved ``model.zip`` and attach a fresh env (resume).

    ``num_timesteps`` is restored so ``reset_num_timesteps=False`` continues the counter. The replay
    buffer is **not** in the blob (SB3 excludes it), so a resumed run starts with an empty buffer and
    refills it — the policy weights continue, training re-stabilises within a short window.
    """
    env = make_env(config.env_id, gym_id)
    return SAC.load(io.BytesIO(resume_blob), env=env, device="cpu")  # CPU is faster for the small MLP (ADR-056)


def train_sac(
    config: TrainConfig,
    gym_id: str,
    control: Any,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train SAC to completion (or until stopped). Returns the terminal state.

    Same contract as :func:`app.services.trainer_ppo.train_ppo` — blocks the calling thread (the
    manager runs it off the event loop), emits the standard ``metrics``/``progress`` frames, publishes a
    decoupled preview policy, and snapshots at quiescent points. ``resume_blob`` continues the timestep
    counter (so ``config.total_timesteps`` is the *absolute* target).
    """
    resuming = resume_blob is not None
    model = (
        _load_model(config, gym_id, resume_blob)
        if resume_blob is not None
        else _build_model(config, gym_id)
    )
    if on_policy is not None:
        on_policy(_build_sac_predict(model))  # initial preview policy (before the first step)

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
            _MIN_REPORT_EPISODES,  # gate the live reward to a stable episode count (off-policy fix)
        ),
        name="sac-progress",
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
        # interval boundary predated the final updates.
        if on_snapshot is not None:
            on_snapshot(_snapshot(model, config.total_timesteps, callback.iteration_count))
        if model.env is not None:
            model.env.close()
    return "stopped" if control.stop_requested else "finished"
