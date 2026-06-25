"""SB3 Twin Delayed DDPG trainer (S5b) — the 6th algorithm, off-policy continuous control.

TD3 is SAC's sibling and the second off-policy continuous-control method here. It shares the whole
off-policy machinery — a replay buffer, **twin** clipped-double-Q critics, slow target networks — and
the same continuous-``Box`` gate (MuJoCo + BipedalWalker + Pendulum + MountainCarContinuous). Like
:mod:`app.services.trainer_sac` it is a **peer trainer** behind the same manager (ADR-004/028) and
reuses the whole rest of the lane — the box-action play/predict, the ``[min_score, solved_score]``
skill meter, the reward chart (``ep_rew_mean``) — with **no new WS frame / TS type**.

What differs from SAC is the **policy**: TD3's actor is *deterministic* (one action per state), so it
has no entropy term. Its three signature tricks are SB3 defaults, kept fixed here:

* **Twin clipped critics** — take the *minimum* of two Q-estimates for the target, curbing the
  overestimation bias DDPG suffers from.
* **Delayed policy updates** (``policy_delay=2``) — update the actor + targets once per two critic
  updates, so the policy chases a more stable value estimate.
* **Target-policy smoothing** (``target_policy_noise``/``target_noise_clip``) — add a little clipped
  noise to the target action so the critic can't exploit sharp peaks.

Because the policy is deterministic it must **inject** noise into the actions it collects in order to
explore (SAC explores via its entropy bonus instead — the conceptual analogue). ``train_noise`` is the
std of that Gaussian exploration noise; the TD3 paper / rl-zoo3 use 0.1.

Everything else is identical to the SAC trainer's off-policy shape — and deliberately so:

* **Off-policy ⇒ no rollout boundary.** The metrics callback emits on a **step interval**
  (``_METRICS_INTERVAL_STEPS``); a snapshot + a fresh decoupled preview policy ride the same interval,
  and the shared ~1 Hz ``_progress_ticker`` (reused from ``trainer_ppo``) keeps the live stats + reward
  curve smooth between frames.
* **Raw obs/rewards (NO VecNormalize)** — same reasoning as SAC: its running reward scaling is
  on-policy-shaped and would drift against a replay buffer, and the standard recipe needs neither. So
  ``ep_rew_mean`` is raw and PPO-vs-SAC-vs-TD3 on one robot is apples-to-apples on the same scale.
* **Device ``cpu``** — TD3's small MLP (default ``[400, 300]``) makes each per-step gradient update a
  tiny batch-256 forward that is latency-bound on a GPU; the CPU is faster and frees the card (the
  ADR-056 result for MlpPolicy / off-policy training).

The decoupled preview policy (ADR-019) is a CPU save/load copy of the actor, never the live model.
"""

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from app.envs.factory import make_env
from app.schemas.training import TrainConfig, TrainingMetrics, TrainState
from app.services.checkpoints import CheckpointArtifact
from app.services.offpolicy import ResumeBufferGate

# Reuse PPO's stable, load-bearing helpers rather than duplicate them — exactly as trainer_sac does. The
# recent-episode mean reader and the decoupled ~1 Hz progress ticker both read only `num_timesteps` /
# `ep_info_buffer` (present on TD3) + the callback's `iteration_count`, so they work unchanged off-policy.
from app.services.trainer_ppo import (
    MetricsSink,
    PredictPublisher,
    ProgressSink,
    SnapshotSink,
    _ep_means,
    _progress_ticker,
)

# How many env steps between metrics frames (+ a snapshot + a refreshed preview policy). TD3, like SAC,
# has no rollout boundary, so this sets the chart/checkpoint cadence (matches trainer_sac's interval).
_METRICS_INTERVAL_STEPS = 2_000

# Minimum completed episodes before the LIVE chart plots a reward (the off-policy fix). The 1 Hz ticker
# fires within a few hundred steps, when the episode buffer holds only one or two high-variance episodes
# (often a lucky random-warmup one) — which read as a misleading "starts high then dips" before the
# rolling mean settled. Gating to a few episodes makes the curve start at the settled baseline and climb
# cleanly. Snapshots/checkpoints still use any available reward (the default min of 1), not this gate.
_MIN_REPORT_EPISODES = 5


def _build_td3_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A read-only CPU forward over a save/load **copy** of TD3's policy (the decoupled preview).

    TD3's actor is a deterministic MLP (``mu`` → tanh → action-space scaling). Rather than re-derive its
    unscaling by hand, round-trip ``model.policy`` through SB3's own ``save``/``load`` into an
    independent CPU policy and let SB3 compute the action — the same isolation the CnnPolicy and SAC
    previews use (ADR-019): the copy shares no tensor storage with the trainer, so forwarding it cannot
    perturb training. Built at a metrics-interval boundary (a quiescent point on the trainer thread).
    TD3's action space is always ``Box``, so the result is a clipped float vector the play/preview loop
    steps directly (the ADR-021 int|box contract — always the box arm here). ``deterministic=True`` drops
    the exploration noise, so preview/AI-play show the policy's actual learned action.
    """
    import torch

    buf = io.BytesIO()
    # save/load are typed for a str path but accept a file-like at runtime (the in-memory trick the
    # CnnPolicy/SAC snapshots use); round-trips state_dict + constructor params, mapping tensors → CPU.
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
    """Serialize the model to an in-memory ``model.zip`` for the checkpoint store (algo="td3").

    Called at a metrics-interval boundary (or after ``learn`` returns) — quiescent points on the
    trainer thread — so it never races SB3's optimizer. SB3's ``save`` excludes the replay buffer by
    default, so the blob is light (policy + critics + params); resume rebuilds a fresh buffer.
    """
    rew, _ = _ep_means(model)
    buf = io.BytesIO()
    model.save(buf)
    return CheckpointArtifact(
        algo="td3",
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
        # Seeded on the FIRST step relative to the (possibly resumed) counter, not to a fixed 2000: a
        # RESUMED run starts num_timesteps at the restored total, which already exceeds a fixed 2000, so
        # _emit() (a full model.save() snapshot + preview rebuild) would fire EVERY step until the
        # threshold caught up — crippling resume to a crawl for a long stretch (the DQN-reported bug; TD3
        # shares this callback). Seeding to num_timesteps + interval emits once per interval from the
        # resume point. Fresh runs start at 0, unchanged.
        self._next_emit: int | None = None

    def _on_step(self) -> bool:
        # Called every env step in collect_rollouts. Park here while paused; abort if a stop was asked.
        self._control.wait_if_paused()
        if self._control.stop_requested:
            return False
        if self._next_emit is None:
            self._next_emit = self.num_timesteps + _METRICS_INTERVAL_STEPS
        if self.num_timesteps >= self._next_emit:
            self._next_emit += _METRICS_INTERVAL_STEPS
            self._emit()
        return True

    def _emit(self) -> None:
        self.iteration_count += 1
        # Gate the live chart reward to a few episodes so a 1–2-episode early fluke isn't plotted.
        ep_rew_mean, ep_len_mean = _ep_means(self.model, _MIN_REPORT_EPISODES)
        # TD3 logs train/critic_loss (the headline learning signal — like SAC, no single "train/loss")
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
            self._on_policy(_build_td3_predict(self.model))


def _td3_kwargs(config: TrainConfig, n_actions: int) -> dict[str, Any]:
    """Map the TD3 hyperparam block onto SB3 TD3 constructor kwargs.

    ``gradient_steps`` tracks ``train_freq`` so the update:collection ratio stays 1:1 (SB3's standard
    shape). The net size is left at SB3's default (``[400, 300]`` for TD3/DDPG — the TD3 paper's net,
    distinct from SAC's ``[256, 256]``). The deterministic policy explores via injected Gaussian action
    noise (``train_noise``); a std of 0 means no exploration noise (warmup-only exploration). The three
    TD3 tricks (``policy_delay`` / ``target_policy_noise`` / ``target_noise_clip``) ride the config as
    fixed defaults. ``learning_starts`` is budget-scaled exactly like SAC's so a short run doesn't burn
    a fifth of itself on random warmup.
    """
    hp = config.td3
    assert hp is not None, "TD3 run without a td3 hyperparam block"
    # Same budget-scaled warmup as SAC: the default 10k is ~2% of a 500k–2M robot budget but a full 20%
    # of a 50k classic-control budget, so scale it to ~2% of the budget, capped at the configured value
    # and floored at one batch so the first update has enough samples.
    warmup = min(hp.learning_starts, max(hp.batch_size, config.total_timesteps // 50))
    action_noise = (
        NormalActionNoise(mean=np.zeros(n_actions), sigma=hp.train_noise * np.ones(n_actions))
        if hp.train_noise > 0
        else None
    )
    return {
        "learning_rate": hp.learning_rate,
        "gamma": hp.gamma,
        "tau": hp.tau,
        "buffer_size": hp.buffer_size,
        "batch_size": hp.batch_size,
        "learning_starts": warmup,
        "train_freq": hp.train_freq,
        "gradient_steps": hp.train_freq,  # 1:1 update:collection ratio
        "action_noise": action_noise,
        "policy_delay": hp.policy_delay,
        "target_policy_noise": hp.target_policy_noise,
        "target_noise_clip": hp.target_noise_clip,
    }


def _build_model(config: TrainConfig, gym_id: str) -> TD3:
    # device="cpu" for the same reason as SAC (ADR-056): TD3's small MLP makes each per-step gradient
    # update a tiny batch-256 forward that is latency-bound on a GPU, so the CPU is faster AND frees the
    # card. SB3 auto-wraps the single env in Monitor + DummyVecEnv, so ep_rew_mean stays raw — no
    # VecNormalize (see the module docstring). The env is built first so we can size the action noise.
    env = make_env(config.env_id, gym_id)
    n_actions = int(np.asarray(env.action_space.shape).prod())
    return TD3(
        "MlpPolicy",
        env,
        seed=config.seed,
        device="cpu",
        verbose=0,
        **_td3_kwargs(config, n_actions),
    )


class _ResumeTD3(ResumeBufferGate, TD3):
    """TD3 with the off-policy resume guard (gradient updates wait for the buffer to refill)."""


def _load_model(config: TrainConfig, gym_id: str, resume_blob: bytes) -> TD3:
    """Rebuild a TD3 model from a saved ``model.zip`` and attach a fresh env (resume).

    ``num_timesteps`` is restored so ``reset_num_timesteps=False`` continues the counter. The replay
    buffer is **not** in the blob (SB3 excludes it), so a resumed run starts with an empty buffer; the
    :class:`ResumeBufferGate` stops the first updates from overfitting the twin critics to that
    near-empty buffer and degrading the restored policy (see :mod:`app.services.offpolicy`).
    """
    env = make_env(config.env_id, gym_id)
    model = _ResumeTD3.load(io.BytesIO(resume_blob), env=env, device="cpu")  # CPU is faster for the small MLP
    # Hold gradient updates until the empty-on-resume buffer refills to the same warmup the fresh run
    # used; collection meanwhile rides the restored policy (num_timesteps > learning_starts ⇒ not random).
    n_actions = int(np.asarray(env.action_space.shape).prod())
    model.grad_start_size = _td3_kwargs(config, n_actions)["learning_starts"]
    return model


def train_td3(
    config: TrainConfig,
    gym_id: str,
    control: Any,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train TD3 to completion (or until stopped). Returns the terminal state.

    Same contract as :func:`app.services.trainer_sac.train_sac` / ``train_ppo`` — blocks the calling
    thread (the manager runs it off the event loop), emits the standard ``metrics``/``progress`` frames,
    publishes a decoupled preview policy, and snapshots at quiescent points. ``resume_blob`` continues
    the timestep counter (so ``config.total_timesteps`` is the *absolute* target).
    """
    resuming = resume_blob is not None
    model = (
        _load_model(config, gym_id, resume_blob)
        if resume_blob is not None
        else _build_model(config, gym_id)
    )
    if on_policy is not None:
        on_policy(_build_td3_predict(model))  # initial preview policy (before the first step)

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
        name="td3-progress",
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
