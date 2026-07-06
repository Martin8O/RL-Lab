"""SB3-contrib Quantile-Regression DQN trainer (S5e) — the 9th algorithm, distributional value-based.

QR-DQN is **DQN made distributional**. Plain DQN (``trainer_dqn``) learns a single number per action —
the *mean* expected return (Q) — and acts by ``argmax``. QR-DQN instead learns the whole **return
distribution** for each action, represented as a fixed set of ``n_quantiles`` values (the median, the
quartiles, …), trained with a **quantile-Huber** regression loss; it still acts on the *mean* of those
quantiles, so the greedy policy is directly comparable to DQN's, but the richer learning target is often
a more stable signal. It is one of the ingredients of Rainbow, and the natural **DQN-vs-QR-DQN teaching
comparison** — the same off-policy value-based machinery, so any difference isolates *distributional*
learning, not a change of family.

Because QR-DQN *is* an off-policy value-based method just like DQN, this module is deliberately a thin
near-copy of :mod:`app.services.trainer_dqn` and everything off-policy-shaped is identical:

* **Off-policy ⇒ no rollout boundary.** The metrics callback emits on a **step interval**
  (``_METRICS_INTERVAL_STEPS``); a snapshot + a fresh decoupled preview policy ride the same interval,
  and the shared ~1 Hz ``_progress_ticker`` (reused from ``trainer_ppo``) keeps the live stats + reward
  curve smooth between frames.
* **Raw obs/rewards (NO VecNormalize)** — same reasoning as DQN/SAC/TD3: a replay buffer + on-policy-
  shaped reward scaling don't mix. So ``ep_rew_mean`` is raw and DQN-vs-QR-DQN on one game is apples-to-
  apples on the same scale (the S5e teaching payoff).
* **Off-policy early-curve gate** — ``_ep_means(min_episodes>=5)`` (ADR-068): the 1 Hz ticker fires
  within a few hundred steps, when the buffer holds only one or two high-variance early episodes, which
  read as a misleading "starts high then dips"; gating to a few episodes makes the curve start settled.
* **ε-greedy exploration + hard target sync + the empty-buffer resume gate** — all identical to DQN.

What differs from DQN:

* **A distribution, not a scalar** — the one new knob is ``n_quantiles`` (how many quantiles represent
  each action's return distribution). In SB3-contrib it is a **policy** kwarg (not a model kwarg), so it
  is passed through ``policy_kwargs`` alongside the vector ``net_arch``. Its greedy action is still a
  plain ``int`` (the ADR-021 discrete arm — ``argmax`` over the mean of the quantiles).
* **The class lives in ``sb3_contrib``** (``QRDQN``), the same dependency that already ships MaskablePPO
  for the board trainer — no new dependency.

The decoupled preview policy (ADR-019) is a CPU save/load copy of the quantile net, never the live model.
"""

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from sb3_contrib import QRDQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from app.envs.factory import make_env
from app.envs.registry import get_env
from app.schemas.training import TrainConfig, TrainingMetrics, TrainState
from app.services.checkpoints import CheckpointArtifact
from app.services.offpolicy import ResumeBufferGate

# Reuse PPO's stable, load-bearing helpers rather than duplicate them — exactly as trainer_dqn does. The
# recent-episode mean reader and the decoupled ~1 Hz progress ticker both read only `num_timesteps` /
# `ep_info_buffer` (present on QRDQN) + the callback's `iteration_count`, so they work unchanged.
from app.services.trainer_ppo import (
    MetricsSink,
    PredictPublisher,
    ProgressSink,
    SnapshotSink,
    _ep_means,
    _progress_ticker,
)

# How many env steps between metrics frames (+ a snapshot + a refreshed preview policy). QR-DQN, like
# DQN, has no rollout boundary, so this sets the chart/checkpoint cadence (matches the off-policy lane).
_METRICS_INTERVAL_STEPS = 2_000

# Minimum completed episodes before the LIVE chart plots a reward (the off-policy fix, shared with
# DQN/SAC/TD3, ADR-068). The 1 Hz ticker fires within a few hundred steps, when the episode buffer holds
# only one or two high-variance early episodes — which read as a misleading "starts high then dips"
# before the rolling mean settled. Gating to a few episodes makes the curve start settled and climb.
_MIN_REPORT_EPISODES = 5

# Parallel image envs are 1 for QR-DQN (a single-env replay-buffer method, like DQN): a single env keeps
# the buffer memory bounded (Atari frames are RAM-heavy) and matches the preview/AI-play env.
_IMAGE_N_ENVS = 1
# Net width for the vector MlpPolicy (rl-zoo3 classic-control QR-DQN uses [256, 256], like DQN). The
# CnnPolicy (Atari) keeps SB3's NatureCNN default, so this applies to the vector path only.
_MLP_NET_ARCH = [256, 256]


def _is_image_env(config: TrainConfig) -> bool:
    """True for an image-obs env (Atari) — the CnnPolicy/CUDA/frame-stack path (distributional Rainbow)."""
    spec = get_env(config.env_id)
    return spec is not None and spec.obs_type == "image"


def _build_qrdqn_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A read-only CPU forward over a save/load **copy** of QR-DQN's quantile net (the decoupled preview).

    QR-DQN acts greedily — ``argmax`` over the **mean** of each action's quantiles — so
    ``deterministic=True`` drops the ε-exploration and returns the policy's actual learned action (a
    plain ``int``). Rather than re-derive the quantile-mean forward by hand, round-trip ``model.policy``
    through SB3's own ``save``/``load`` into an independent CPU policy and let it compute the action —
    the same isolation the DQN/SAC/TD3/CnnPolicy previews use (ADR-019): the copy shares no tensor storage
    with the trainer, so forwarding it cannot perturb training. Built at a metrics-interval boundary (a
    quiescent point on the trainer thread). Works for both the MLP (vector) and CNN (Atari) quantile net.
    """
    import torch

    buf = io.BytesIO()
    # save/load are typed for a str path but accept a file-like at runtime (the in-memory trick the
    # CnnPolicy/DQN/SAC/TD3 snapshots use); round-trips state_dict + constructor params, tensors → CPU.
    model.policy.save(buf)  # type: ignore[arg-type]
    buf.seek(0)
    policy = model.policy.__class__.load(buf, device="cpu")  # type: ignore[arg-type]
    policy.set_training_mode(False)

    def predict(obs: object) -> Any:
        with torch.no_grad():
            action, _ = policy.predict(np.asarray(obs), deterministic=True)
        return int(np.asarray(action).flatten()[0])

    return predict


def _snapshot(model: BaseAlgorithm, total_timesteps: int, iteration: int) -> CheckpointArtifact:
    """Serialize the model to an in-memory ``model.zip`` for the checkpoint store (algo="qrdqn").

    Called at a metrics-interval boundary (or after ``learn`` returns) — quiescent points on the
    trainer thread — so it never races SB3's optimizer. SB3's ``save`` excludes the replay buffer by
    default, so the blob is light (quantile net + target + params); resume rebuilds a fresh buffer.
    """
    rew, _ = _ep_means(model)
    buf = io.BytesIO()
    model.save(buf)
    return CheckpointArtifact(
        algo="qrdqn",
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
        # Seeded on the FIRST step (relative to the possibly-resumed counter), not to a fixed 2000 — the
        # same off-policy resume fix as DQN/SAC/TD3 (ADR-069): on a RESUMED run num_timesteps already
        # starts at the restored total, so a fixed threshold would fire _emit (a full model.save + a
        # preview rebuild) on EVERY step until the threshold crawled up. Fresh runs start at 0, unchanged.
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
        # QR-DQN logs train/loss (the quantile-Huber regression loss — the headline learning signal, the
        # distributional analogue of DQN's TD/Bellman loss) + train/learning_rate; both absent until the
        # first gradient update (after learning_starts).
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
        # A mid-run snapshot so "Save" can persist the live model; refresh the decoupled preview policy.
        if self._on_snapshot is not None:
            self._on_snapshot(_snapshot(self.model, self._total, self.iteration_count))
        if self._on_policy is not None:
            self._on_policy(_build_qrdqn_predict(self.model))


def _qrdqn_kwargs(config: TrainConfig, is_image: bool) -> dict[str, Any]:
    """Map the QR-DQN hyperparam block onto SB3-contrib QRDQN **model** constructor kwargs.

    Identical shape to :func:`app.services.trainer_dqn._dqn_kwargs` — QR-DQN's model constructor takes
    the same off-policy arguments as DQN. The one distinctive knob, ``n_quantiles``, is a *policy* kwarg
    (not a model kwarg in SB3-contrib), so it is applied in :func:`_build_model` via ``policy_kwargs``,
    NOT here. ``gradient_steps`` is 1 on the Atari image path (the Nature recipe's single update per
    ``train_freq`` collected steps) and equals ``train_freq`` on the vector path (one update per collected
    step). ``batch_size`` is the rl-zoo3 classic-control 128 for vector and the Nature 32 for Atari.
    ``learning_starts`` is budget-scaled exactly like DQN so a short run doesn't burn a fifth of itself on
    random warmup.
    """
    hp = config.qrdqn
    assert hp is not None, "QR-DQN run without a qrdqn hyperparam block"
    # Same budget-scaled warmup as DQN: the configured value is the cap, ~2% of the budget the target,
    # floored at one batch so the first update has enough samples.
    warmup = min(hp.learning_starts, max(hp.batch_size, config.total_timesteps // 50))
    # Vector envs: one update per collected env step (gradient_steps == train_freq). Atari: the Nature
    # recipe's single update per train_freq collected steps.
    gradient_steps = 1 if is_image else hp.train_freq
    batch_size = 32 if is_image else hp.batch_size
    kwargs: dict[str, Any] = {
        "learning_rate": hp.learning_rate,
        "gamma": hp.gamma,
        "buffer_size": hp.buffer_size,
        "batch_size": batch_size,
        "learning_starts": warmup,
        "train_freq": hp.train_freq,
        "gradient_steps": gradient_steps,
        "target_update_interval": hp.target_update_interval,
        "exploration_fraction": hp.exploration_fraction,
        "exploration_final_eps": hp.exploration_final_eps,
    }
    if is_image:
        # Atari's replay buffer stores STACKED 84×84×4 frames, so it is RAM-heavy — the same memory
        # handling as DQN (ADR-069): optimize_memory_usage drops the duplicate next-obs array (~halves
        # it), which also makes DQN.load-on-resume re-allocate the smaller buffer instead of MemoryError-ing.
        # It requires handle_timeout_termination=False (SB3 constraint), fine for Atari (episodes end on
        # game-over, not a TimeLimit). VECTOR envs keep the default buffer AND timeout handling — CartPole's
        # TimeLimit truncation must NOT be treated as terminal, or value bootstrapping is wrong.
        kwargs["optimize_memory_usage"] = True
        kwargs["replay_buffer_kwargs"] = {"handle_timeout_termination": False}
    return kwargs


def _policy_kwargs(config: TrainConfig, is_image: bool) -> dict[str, Any]:
    """Build QR-DQN's ``policy_kwargs`` — where ``n_quantiles`` lives (a policy kwarg in SB3-contrib).

    ``n_quantiles`` is QR-DQN's one knob beyond DQN: how many quantiles represent each action's return
    distribution. The vector MlpPolicy also carries the ``net_arch`` [256, 256] (like DQN's vector net);
    the Atari CnnPolicy keeps SB3's NatureCNN default, so only ``n_quantiles`` is set there.
    """
    hp = config.qrdqn
    assert hp is not None, "QR-DQN run without a qrdqn hyperparam block"
    kwargs: dict[str, Any] = {"n_quantiles": hp.n_quantiles}
    if not is_image:
        kwargs["net_arch"] = _MLP_NET_ARCH
    return kwargs


def _build_model(config: TrainConfig, gym_id: str) -> QRDQN:
    is_image = _is_image_env(config)
    if is_image:
        # Atari (image obs): a CnnPolicy on CUDA over the AtariWrapper + frame stack — distributional RL's
        # historical home (Rainbow). Built through the SAME shared make_image_vec dispatcher PPO/DQN-Atari
        # use (n_envs=1: QR-DQN is a single-env replay-buffer method, and one env keeps the RAM-heavy frame
        # buffer bounded), so the net sees the obs shape the preview/AI-play will match. device="cuda" is
        # gated upstream.
        from app.envs.image_vec import make_image_vec

        spec = get_env(config.env_id)
        assert spec is not None  # _is_image_env already established this
        env: Any = make_image_vec(spec, _IMAGE_N_ENVS, seed=config.seed)
        policy, device = "CnnPolicy", "cuda"
    else:
        # Vector envs: an MlpPolicy [256, 256] on CPU (ADR-056 — the tiny batched MLP update is faster on
        # CPU than a latency-bound GPU shuttle, and it frees the card). SB3 auto-wraps the single env in
        # Monitor + DummyVecEnv, so ep_rew_mean stays raw — no VecNormalize (see the module docstring).
        env = make_env(config.env_id, gym_id)
        policy, device = "MlpPolicy", "cpu"
    return QRDQN(
        policy,
        env,
        seed=config.seed,
        device=device,
        policy_kwargs=_policy_kwargs(config, is_image),
        verbose=0,
        **_qrdqn_kwargs(config, is_image),
    )


class _ResumeQRDQN(ResumeBufferGate, QRDQN):
    """QR-DQN with the off-policy resume guard (gradient updates wait for the empty buffer to refill)."""


def _load_model(config: TrainConfig, gym_id: str, resume_blob: bytes) -> QRDQN:
    """Rebuild a QR-DQN model from a saved ``model.zip`` and attach a fresh env (resume).

    ``num_timesteps`` is restored so ``reset_num_timesteps=False`` continues the counter. The replay
    buffer is **not** in the blob (SB3 excludes it), so a resumed run starts with an empty buffer; the
    :class:`ResumeBufferGate` stops the first updates from overfitting the net to that near-empty buffer
    and degrading the restored policy (see :mod:`app.services.offpolicy`) — the same fix as DQN/SAC/TD3.
    """
    is_image = _is_image_env(config)
    load_kwargs: dict[str, Any] = {}
    if is_image:
        from app.envs.image_vec import make_image_vec

        spec = get_env(config.env_id)
        assert spec is not None
        env: Any = make_image_vec(spec, _IMAGE_N_ENVS, seed=config.seed)
        device = "cuda"
        # Force the memory-safe replay buffer on resume too (the DQN ADR-069 fix): these kwargs override
        # the saved values, so any resumed Atari run gets the optimized, config-sized buffer rather than
        # re-allocating a full one (MemoryError/10-steps-per-s risk). SB3 excludes the buffer from
        # model.zip, so it is rebuilt fresh on load anyway — only its size/layout is restored here.
        hp = config.qrdqn
        if hp is not None:
            load_kwargs["buffer_size"] = hp.buffer_size
        load_kwargs["optimize_memory_usage"] = True
        load_kwargs["replay_buffer_kwargs"] = {"handle_timeout_termination": False}
    else:
        env = make_env(config.env_id, gym_id)
        device = "cpu"  # CPU is faster for the small MLP (ADR-056)
    model = _ResumeQRDQN.load(io.BytesIO(resume_blob), env=env, device=device, **load_kwargs)
    # Hold gradient updates until the empty-on-resume buffer refills to the same warmup the fresh run
    # used; collection meanwhile rides the restored ε-greedy policy (num_timesteps > learning_starts).
    model.grad_start_size = _qrdqn_kwargs(config, is_image)["learning_starts"]
    return model


def train_qrdqn(
    config: TrainConfig,
    gym_id: str,
    control: Any,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train QR-DQN to completion (or until stopped). Returns the terminal state.

    Same contract as :func:`app.services.trainer_dqn.train_dqn` / ``train_ppo`` — blocks the calling
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
        on_policy(_build_qrdqn_predict(model))  # initial preview policy (before the first step)

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
        name="qrdqn-progress",
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
