"""SB3 Deep Q-Network trainer (S5c) — the 7th algorithm, off-policy value-based, discrete actions.

DQN is the **value-based** counterpart to policy-gradient PPO and the original deep-RL breakthrough
(Atari, Mnih et al. 2015). It is the **discrete-action mirror of SAC/TD3**: the same off-policy
machinery — a replay buffer + a slow target network — but it learns an **action-value function** (Q)
with a neural net and acts by taking the highest-Q action, rather than learning a policy directly. It
is a **peer trainer** behind the same manager (ADR-004/028) and reuses the whole rest of the lane —
the discrete play/predict, the ``[min_score, solved_score]`` skill meter, the reward chart
(``ep_rew_mean``) — with **no new WS frame / TS type**. This module is a near-copy of
:mod:`app.services.trainer_td3`; everything off-policy-shaped is deliberately identical:

* **Off-policy ⇒ no rollout boundary.** The metrics callback emits on a **step interval**
  (``_METRICS_INTERVAL_STEPS``); a snapshot + a fresh decoupled preview policy ride the same interval,
  and the shared ~1 Hz ``_progress_ticker`` (reused from ``trainer_ppo``) keeps the live stats + reward
  curve smooth between frames.
* **Raw obs/rewards (NO VecNormalize)** — same reasoning as SAC/TD3: a replay buffer + on-policy-shaped
  reward scaling don't mix, and the standard DQN recipe needs neither. So ``ep_rew_mean`` is raw and
  PPO-vs-DQN on one game is apples-to-apples on the same scale (the S5c teaching payoff).
* **Off-policy early-curve gate** — ``_ep_means(min_episodes>=5)`` (ADR-068): the 1 Hz ticker fires
  within a few hundred steps, when the buffer holds only one or two high-variance early episodes, which
  read as a misleading "starts high then dips"; gating to a few episodes makes the curve start settled.

What differs from SAC/TD3:

* **Discrete actions** — DQN's policy is ``argmax`` over the Q-net's per-action values, so the preview
  /AI-play action is a plain ``int`` (the ADR-021 contract — always the discrete arm here), never a Box.
* **Exploration is ε-greedy** — not entropy (SAC's ``ent_coef``) or injected action noise (TD3's
  ``train_noise``), but an ε schedule: a random action with probability ε, annealing from 1.0 to
  ``exploration_final_eps`` over the first ``exploration_fraction`` of the budget, then held. The slow
  target net is **hard-copied** every ``target_update_interval`` steps (DQN's analogue of τ).
* **Two policies, two devices.** Vector envs (CartPole + classic-control discretes + LunarLander) use an
  ``MlpPolicy`` on the **CPU** — the ADR-056 result (tiny batched MLP updates are latency-bound on a GPU,
  so CPU is faster + frees the card). Atari is image-obs → a ``CnnPolicy`` on **CUDA** (DQN's literal
  birthplace), built over the shared ``image_vec.make_image_vec`` dispatcher exactly like PPO-Atari.

The decoupled preview policy (ADR-019) is a CPU save/load copy of the Q-net, never the live model.
"""

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from app.envs.factory import make_env
from app.envs.registry import get_env
from app.schemas.training import TrainConfig, TrainingMetrics, TrainState
from app.services.checkpoints import CheckpointArtifact

# Reuse PPO's stable, load-bearing helpers rather than duplicate them — exactly as trainer_td3 does. The
# recent-episode mean reader and the decoupled ~1 Hz progress ticker both read only `num_timesteps` /
# `ep_info_buffer` (present on DQN) + the callback's `iteration_count`, so they work unchanged off-policy.
from app.services.trainer_ppo import (
    MetricsSink,
    PredictPublisher,
    ProgressSink,
    SnapshotSink,
    _ep_means,
    _progress_ticker,
)

# How many env steps between metrics frames (+ a snapshot + a refreshed preview policy). DQN, like
# SAC/TD3, has no rollout boundary, so this sets the chart/checkpoint cadence (matches their interval).
_METRICS_INTERVAL_STEPS = 2_000

# Minimum completed episodes before the LIVE chart plots a reward (the off-policy fix, shared with
# SAC/TD3, ADR-068). The 1 Hz ticker fires within a few hundred steps, when the episode buffer holds
# only one or two high-variance early episodes (often a lucky random-warmup one) — which read as a
# misleading "starts high then dips" before the rolling mean settled. Gating to a few episodes makes the
# curve start at the settled baseline and climb cleanly. Snapshots/checkpoints still use any available
# reward (the default min of 1), not this gate.
_MIN_REPORT_EPISODES = 5

# Parallel image envs are 1 for DQN (the Nature recipe is a single-env replay-buffer method): a single
# env keeps the buffer memory bounded (Atari frames are RAM-heavy) and matches the preview/AI-play env.
_IMAGE_N_ENVS = 1
# Net width for the vector MlpPolicy (rl-zoo3 classic-control DQN uses [256, 256], bigger than PPO's
# [64, 64] default). The CnnPolicy (Atari) keeps SB3's NatureCNN default, so this applies to vector only.
_MLP_NET_ARCH = [256, 256]


def _is_image_env(config: TrainConfig) -> bool:
    """True for an image-obs env (Atari) — the CnnPolicy/CUDA/frame-stack path (DQN's birthplace)."""
    spec = get_env(config.env_id)
    return spec is not None and spec.obs_type == "image"


def _build_dqn_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A read-only CPU forward over a save/load **copy** of DQN's Q-net (the decoupled preview).

    DQN acts greedily — ``argmax`` over the Q-net's per-action values — so ``deterministic=True``
    drops the ε-exploration and returns the policy's actual learned action (a plain ``int``). Rather
    than re-derive the Q-head forward by hand, round-trip ``model.policy`` through SB3's own
    ``save``/``load`` into an independent CPU policy and let SB3 compute the action — the same isolation
    the SAC/TD3/CnnPolicy previews use (ADR-019): the copy shares no tensor storage with the trainer, so
    forwarding it cannot perturb training. Built at a metrics-interval boundary (a quiescent point on the
    trainer thread). Works for both the MLP (vector) and CNN (Atari) Q-net — ``policy.predict`` handles
    the stacked image obs the same way the PPO-Atari preview does.
    """
    import torch

    buf = io.BytesIO()
    # save/load are typed for a str path but accept a file-like at runtime (the in-memory trick the
    # CnnPolicy/SAC/TD3 snapshots use); round-trips state_dict + constructor params, mapping tensors → CPU.
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
    """Serialize the model to an in-memory ``model.zip`` for the checkpoint store (algo="dqn").

    Called at a metrics-interval boundary (or after ``learn`` returns) — quiescent points on the
    trainer thread — so it never races SB3's optimizer. SB3's ``save`` excludes the replay buffer by
    default, so the blob is light (Q-net + target + params); resume rebuilds a fresh buffer.
    """
    rew, _ = _ep_means(model)
    buf = io.BytesIO()
    model.save(buf)
    return CheckpointArtifact(
        algo="dqn",
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
        # Seeded on the FIRST step (relative to the possibly-resumed counter), not to a fixed 2000. On a
        # RESUMED run num_timesteps already starts at the restored total (e.g. 1.4M), so a fixed threshold
        # of 2000 is immediately exceeded → _emit() (a full model.save() snapshot + a preview-policy
        # rebuild) would fire EVERY step until the threshold crawled up to 1.4M — crippling resume to
        # ~12 steps/s for the first minute, then "jumping" to full speed once it caught up (the reported
        # load-a-DQN-save bug). Seeding to num_timesteps + interval makes it fire once per interval from
        # the resume point. Fresh runs start at 0, so this is unchanged for them.
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
        # DQN logs train/loss (the headline learning signal, like PPO — the TD/Bellman loss) +
        # train/learning_rate; both absent until the first gradient update (after learning_starts).
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
            self._on_policy(_build_dqn_predict(self.model))


def _dqn_kwargs(config: TrainConfig, is_image: bool) -> dict[str, Any]:
    """Map the DQN hyperparam block onto SB3 DQN constructor kwargs.

    ``gradient_steps`` is set here rather than tracking ``train_freq`` 1:1 (the SAC/TD3 shape, where
    ``train_freq`` is 1): DQN's ``train_freq`` is large (CartPole's tuned recipe is 256), so a 1:1 ratio
    would be a sane "one update per collected step" for the vector envs — but **Atari** follows the
    Nature recipe of a *single* gradient step per ``train_freq`` collected steps (4 env steps → 1
    update), so the image path pins ``gradient_steps=1`` to keep training stable + fast. ``batch_size``
    is the rl-zoo3 classic-control 128 for vector and the Nature 32 for Atari. ``learning_starts`` is
    budget-scaled exactly like SAC/TD3 so a short run doesn't burn a fifth of itself on random warmup.
    """
    hp = config.dqn
    assert hp is not None, "DQN run without a dqn hyperparam block"
    # Same budget-scaled warmup as SAC/TD3: the configured value is the cap, ~2% of the budget the target,
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
        # Atari's replay buffer stores STACKED 84×84×4 frames, so it is RAM-heavy. SB3's default keeps a
        # separate next-obs array too, so a 50k buffer costs ~2.8 GB (and 100k ~5.6 GB). optimize_memory_usage
        # drops that duplicate (reuses obs[i+1] as next_obs) → ~halves it (~1.4 GB at the 50k Atari ★). This
        # matters most on RESUME: DQN.load re-allocates the whole buffer, and the full default size could
        # MemoryError (run "breaks") or thrash paging to ~10 steps/s — the reported S5c resume bug. The flag
        # rides inside model.zip, so a resumed run rebuilds the smaller buffer too. It requires
        # handle_timeout_termination=False (SB3 constraint), fine for Atari (episodes end on game-over, not a
        # TimeLimit). VECTOR envs keep the default buffer (tiny) AND timeout handling — CartPole's TimeLimit
        # truncation must NOT be treated as terminal, or value bootstrapping is wrong.
        kwargs["optimize_memory_usage"] = True
        kwargs["replay_buffer_kwargs"] = {"handle_timeout_termination": False}
    return kwargs


def _build_model(config: TrainConfig, gym_id: str) -> DQN:
    is_image = _is_image_env(config)
    if is_image:
        # Atari (image obs): a CnnPolicy on CUDA over the AtariWrapper + frame stack — DQN's birthplace.
        # Built through the SAME shared make_image_vec dispatcher PPO-Atari uses (n_envs=1: DQN is a
        # single-env replay-buffer method, and one env keeps the RAM-heavy frame buffer bounded), so the
        # Q-net sees the obs shape the preview/AI-play will match. device="cuda" is gated upstream.
        from app.envs.image_vec import make_image_vec

        spec = get_env(config.env_id)
        assert spec is not None  # _is_image_env already established this
        env: Any = make_image_vec(spec, _IMAGE_N_ENVS, seed=config.seed)
        policy, device, policy_kwargs = "CnnPolicy", "cuda", None
    else:
        # Vector envs: an MlpPolicy [256, 256] on CPU (ADR-056 — the tiny batched MLP update is faster on
        # CPU than a latency-bound GPU shuttle, and it frees the card). SB3 auto-wraps the single env in
        # Monitor + DummyVecEnv, so ep_rew_mean stays raw — no VecNormalize (see the module docstring).
        env = make_env(config.env_id, gym_id)
        policy, device = "MlpPolicy", "cpu"
        policy_kwargs = {"net_arch": _MLP_NET_ARCH}
    return DQN(
        policy,
        env,
        seed=config.seed,
        device=device,
        policy_kwargs=policy_kwargs,
        verbose=0,
        **_dqn_kwargs(config, is_image),
    )


def _load_model(config: TrainConfig, gym_id: str, resume_blob: bytes) -> DQN:
    """Rebuild a DQN model from a saved ``model.zip`` and attach a fresh env (resume).

    ``num_timesteps`` is restored so ``reset_num_timesteps=False`` continues the counter. The replay
    buffer is **not** in the blob (SB3 excludes it), so a resumed run starts with an empty buffer and
    refills it — the Q-net weights continue, training re-stabilises within a short window.
    """
    is_image = _is_image_env(config)
    load_kwargs: dict[str, Any] = {}
    if is_image:
        from app.envs.image_vec import make_image_vec

        spec = get_env(config.env_id)
        assert spec is not None
        env: Any = make_image_vec(spec, _IMAGE_N_ENVS, seed=config.seed)
        device = "cuda"
        # Force the memory-safe replay buffer on resume too — even for an Atari DQN saved BEFORE this fix
        # (which baked in the full non-optimized buffer ⇒ ~5.6 GB re-allocated on load, the reported
        # MemoryError/10-steps-per-s bug). These kwargs override the saved values, so any resumed run gets
        # the optimized, config-sized buffer. (SB3 excludes the buffer from model.zip, so it is rebuilt
        # fresh on load anyway — only its size/layout is restored from the saved params we override here.)
        hp = config.dqn
        if hp is not None:
            load_kwargs["buffer_size"] = hp.buffer_size
        load_kwargs["optimize_memory_usage"] = True
        load_kwargs["replay_buffer_kwargs"] = {"handle_timeout_termination": False}
    else:
        env = make_env(config.env_id, gym_id)
        device = "cpu"  # CPU is faster for the small MLP (ADR-056)
    return DQN.load(io.BytesIO(resume_blob), env=env, device=device, **load_kwargs)


def train_dqn(
    config: TrainConfig,
    gym_id: str,
    control: Any,
    on_metrics: MetricsSink,
    on_progress: ProgressSink,
    on_policy: PredictPublisher | None = None,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train DQN to completion (or until stopped). Returns the terminal state.

    Same contract as :func:`app.services.trainer_td3.train_td3` / ``train_ppo`` — blocks the calling
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
        on_policy(_build_dqn_predict(model))  # initial preview policy (before the first step)

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
        name="dqn-progress",
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
