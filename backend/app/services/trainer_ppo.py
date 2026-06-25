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

from app.envs.factory import make_env
from app.envs.registry import get_env
from app.schemas.training import (
    TrainConfig,
    TrainingMetrics,
    TrainingProgress,
    TrainState,
)
from app.services import vecnorm
from app.services.checkpoints import CheckpointArtifact
from app.services.ma_env import is_multi_agent, make_vec_env
from app.services.train_control import TrainControl

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}
_PROGRESS_INTERVAL = 1.0  # seconds between live progress frames
_IMAGE_N_ENVS = 8  # parallel image envs for the CnnPolicy rollout (Atari + CarRacing alike)


class _InterruptiblePPO(PPO):
    """PPO whose multi-epoch ``train()`` update can be stopped *between epochs*.

    SB3 only invokes the training callback during **rollout collection** (``_on_step``), never
    during the ``train()`` update — so a Stop pressed mid-update isn't observed until the whole
    update finishes and the next collection's first step runs. For a **multi-agent** (SuperSuit
    parameter-sharing) run that update is ``n_epochs`` passes over the N×-bigger shared batch and
    measured ~24 s on the CPU laptop (6-agent swarm), so Stop looked frozen — the "Zastavuji" hang
    the user hit (ADR-038 follow-up).

    Fix: run the epochs one at a time (set ``n_epochs=1`` and call the stock ``train()`` repeatedly)
    and check the stop flag between them. This is **behaviourally identical** to the stock 10-epoch
    update — each ``super().train()`` call does one full shuffled pass, so the RNG draws (one buffer
    permutation per epoch), the per-minibatch advantage normalisation, the gradient steps and the
    ``_n_updates``/loss logging are exactly the same; only ``clip_range``/learning-rate recompute and
    the metrics ``logger.record`` run once per epoch instead of once per update (same values, the
    metrics callback reads them only after the full update). Worst-case stop latency drops from a
    whole update to a single epoch (~2–3 s here). Harmless for single-agent runs (tiny updates), so
    it's used for every PPO run rather than branched on family.
    """

    stop_check: Callable[[], bool] | None = None

    def _excluded_save_params(self) -> list[str]:
        # ``stop_check`` closes over the run's TrainControl (a threading primitive) — unpicklable and
        # runtime-only — so keep it out of the saved model.zip; it's re-attached on the next run.
        return [*super()._excluded_save_params(), "stop_check"]

    def train(self) -> None:
        epochs = self.n_epochs
        self.n_epochs = 1
        try:
            for _ in range(epochs):
                super().train()
                if self.stop_check is not None and self.stop_check():
                    break  # bail out of the update; the next collect_rollouts step ends learn()
        finally:
            self.n_epochs = epochs

MetricsSink = Callable[[TrainingMetrics], None]
ProgressSink = Callable[[TrainingProgress], None]
SnapshotSink = Callable[[CheckpointArtifact], None]
# Hands the decoupled preview a self-contained predict fn (obs → action) over a weight snapshot.
# The action is an int (discrete) or a numpy float vector (continuous box) — Any.
PredictPublisher = Callable[[Callable[[object], Any]], None]


def _build_numpy_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A standalone **numpy** forward over a snapshot of the policy's action path.

    The preview must never call ``model.predict`` on the *live* model: doing so concurrently with
    ``learn()`` measurably perturbs PPO's training trajectory (proven empirically — concurrent SB3
    model access diverges a same-seed run, while pure compute does not). A numpy forward over
    copied weights cannot touch the trainer's torch state, so the preview stays a true read-only
    observer (mirrors how the neuroevolution trainer already publishes its preview policy).

    Built at a rollout boundary on the trainer thread (a quiescent point), so the weight copy
    never races the optimizer. Handles any ``net_arch`` depth + tanh/relu. For a **discrete** env
    the head is action logits → arg-max (== SB3's ``deterministic=True``); for a **continuous**
    (box) env the head is the Gaussian mean → clipped to the action bounds (what the env does).
    """
    policy: Any = model.policy  # torch dynamic attrs (mlp_extractor/action_net) aren't typed

    def arr(t: Any) -> np.ndarray:
        return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)

    pi_net = policy.mlp_extractor.policy_net
    layers = [(arr(m.weight), arr(m.bias)) for m in pi_net if isinstance(m, nn.Linear)]
    act_w, act_b = arr(policy.action_net.weight), arr(policy.action_net.bias)
    relu = any(isinstance(m, nn.ReLU) for m in pi_net)
    is_box = getattr(model.action_space, "n", None) is None
    low = np.asarray(getattr(model.action_space, "low", 0.0), dtype=np.float64)
    high = np.asarray(getattr(model.action_space, "high", 0.0), dtype=np.float64)
    # G5c: when the env is VecNormalize-wrapped (the MuJoCo family) the policy was trained on
    # normalized obs, so the preview — which feeds RAW obs from its own throwaway env — must apply the
    # same running-stat normalization before the MLP forward. None for every un-normalized env, leaving
    # the forward byte-identical to before.
    vec_norm = model.get_vec_normalize_env()
    normalize = vecnorm.obs_normalizer_from_env(vec_norm) if vec_norm is not None else None

    def predict(obs: object) -> Any:
        x = np.asarray(obs, dtype=np.float64)
        if normalize is not None:
            x = normalize(x)
        for w, b in layers:
            x = x @ w.T + b
            x = np.maximum(0.0, x) if relu else np.tanh(x)
        out = x @ act_w.T + act_b
        if is_box:  # continuous: the mean action, clipped into [low, high]
            return np.clip(out, low, high).astype(np.float32)
        return int(np.argmax(out))

    return predict


def _build_cnn_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A read-only **CPU torch** forward over a snapshot of an image-obs CnnPolicy (G4b/G3c-train).

    The numpy forward above only covers an MLP (``mlp_extractor.policy_net`` + ``action_net``); a
    CnnPolicy's NatureCNN feature extractor has no such path, so the preview needs a real torch
    forward. ADR-019 still holds — the preview must never touch the *live* CUDA model: a deepcopy
    of the policy trips torch's non-leaf-tensor guard, so we round-trip the policy through SB3's own
    ``save``/``load`` into an **independent CPU policy**. That copy shares no tensor storage with the
    trainer, so forwarding it cannot perturb training (the same isolation the numpy snapshot gives).

    Built at a rollout boundary (a quiescent point on the trainer thread). Returns a ``predict(obs)``
    over the stacked observation the preview's matching vec env yields: an **int** for a discrete env
    (Atari ``Discrete(18)``), or a **clipped float vector** for a continuous env (CarRacing's
    ``Box(3)`` steer/gas/brake) — the same int|box duality the numpy/PPO predict fns already use.
    """
    import io

    import torch

    buf = io.BytesIO()
    # SB3's BaseModel.save/load are typed for a str path but accept a file-like at runtime (the same
    # in-memory round-trip _snapshot uses for the whole model) — round-trips state_dict + constructor
    # params (tensors only → picklable), avoiding the non-leaf-tensor deepcopy trap.
    model.policy.save(buf)  # type: ignore[arg-type]
    buf.seek(0)
    policy = model.policy.__class__.load(buf, device="cpu")  # type: ignore[arg-type]  # maps tensors → CPU
    policy.set_training_mode(False)
    is_box = getattr(model.action_space, "n", None) is None  # Box has low/high, Discrete has n
    low = np.asarray(getattr(model.action_space, "low", 0.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 0.0), dtype=np.float32)

    def predict(obs: object) -> Any:
        with torch.no_grad():
            action, _ = policy.predict(np.asarray(obs), deterministic=True)
        arr = np.asarray(action)
        if is_box:  # continuous: the deterministic mean action, clipped into [low, high]
            return np.clip(arr.astype(np.float32).reshape(-1), low, high)
        return int(arr.flatten()[0])

    return predict


def _build_preview_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """Dispatch the decoupled preview policy by observation rank: a 3-D obs (H, W, C) is an image
    (CnnPolicy → CPU torch snapshot); anything else is a vector/one-hot obs (the numpy MLP forward).
    Either way the result is a self-contained predict fn over copied weights — never the live model."""
    if len(getattr(model.observation_space, "shape", ()) or ()) == 3:
        return _build_cnn_predict(model)
    return _build_numpy_predict(model)


def load_preview_predict(blob: bytes) -> Callable[[object], Any]:
    """Load a saved **cooperative** PPO ``model.zip`` and return its decoupled preview predict fn.

    The Watch-AI loader for a cooperative multi-agent checkpoint — a single shared brain over
    homogeneous agents (simple_spread, pursuit). The preview streamer applies the one returned fn to
    **every** agent (parameter sharing). (The *competitive* simple_tag case packs a per-species
    ``species.zip`` and is loaded by :func:`app.services.trainer_tag.load_species_predicts` instead.)

    Loaded for inference only (``env=None`` — the saved zip carries the obs/action spaces, so no env is
    built and no pygame is touched). Dispatches exactly like the live preview: pursuit's 3-D local-view
    obs → the CPU torch snapshot, a vector obs → the numpy MLP forward (ADR-019)."""
    model = _InterruptiblePPO.load(io.BytesIO(blob), env=None, device="cpu")
    return _build_preview_predict(model)


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
    blob = buf.getvalue()
    # G5c: embed the VecNormalize stats inside model.zip (MuJoCo) so they travel with the single
    # checkpoint blob to every inference + resume path. None for un-normalized envs → blob unchanged.
    vec_norm = model.get_vec_normalize_env()
    if vec_norm is not None:
        blob = vecnorm.embed_stats(blob, vec_norm)
    return CheckpointArtifact(
        algo="ppo",
        blob=blob,
        artifact_name="model.zip",
        reward=rew,
        timesteps=int(model.num_timesteps),
        total_timesteps=total_timesteps,
        iteration=iteration,
    )


def _ep_means(
    model: BaseAlgorithm, min_episodes: int = 1
) -> tuple[float | None, float | None]:
    """Mean reward/length over SB3's recent-episode buffer, or ``(None, None)``.

    Read from the progress-ticker thread while ``learn()`` runs on another thread. The
    buffer is a deque that ``learn`` appends to, so we snapshot defensively and treat a
    rare concurrent mutation as "no update this tick".

    ``min_episodes`` requires at least that many completed episodes before a mean is
    returned (else ``(None, None)``). The default 1 is the historical behaviour (PPO emits
    at rollout end, where the buffer is already full, and snapshots want any available
    reward). The **off-policy** trainers (SAC/TD3) pass a higher value for their live chart
    frames: their 1 Hz ticker fires within a few hundred steps, when the buffer holds only
    one or two high-variance episodes (often a lucky *random-warmup* one), which plotted as
    a misleading "starts high then dips" before the rolling mean settled. Gating to a few
    episodes makes the live curve start at the settled baseline and climb cleanly.
    """
    buf = getattr(model, "ep_info_buffer", None)
    if not buf:
        return None, None
    try:
        episodes = list(buf)  # snapshot; may raise if mutated mid-iteration
    except RuntimeError:
        return None, None
    if len(episodes) < min_episodes:
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
            self._on_policy(_build_preview_predict(self.model))


def _progress_ticker(
    model: PPO,
    callback: _MetricsCallback,
    control: TrainControl,
    on_progress: ProgressSink,
    total_timesteps: int,
    started_at: float,
    stop_event: threading.Event,
    min_report_episodes: int = 1,
) -> None:
    """Emit a progress frame every ``_PROGRESS_INTERVAL`` seconds until stopped.

    Decoupled from SB3's per-step callback (which is dormant during the PPO update phase),
    so the live stats refresh at a steady ~1 Hz regardless of training phase. Mirrors the
    decoupled preview streamer (ADR-008). Reads model counters/buffers only — it never
    mutates model state — so it cannot affect training reproducibility.
    """
    last_t = started_at
    # Seed the step baseline at the model's CURRENT counter, not 0. On a *resumed* run num_timesteps is
    # already the restored total (e.g. 1.4M) before the first tick, so a 0 baseline would make the first
    # delta the entire resumed total → an absurd steps/s spike (≈1.4M/s) that the EMA then takes ~15 ticks
    # to bleed off (the reported "crazy numbers then it settles" on load). Fresh runs start at 0, so this
    # is unchanged for them. (Reads the counter only — never mutates model state, so repro is unaffected.)
    last_steps = int(model.num_timesteps)
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

        rew, length = _ep_means(model, min_report_episodes)
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


def _is_image_env(config: TrainConfig) -> bool:
    """True for an image-obs env (Atari + CarRacing) — the CnnPolicy/CUDA/frame-stack path (G4b/G3c-train)."""
    spec = get_env(config.env_id)
    return spec is not None and spec.obs_type == "image"


def _make_train_env(config: TrainConfig, gym_id: str) -> tuple[Any, int | None]:
    """Build the training env + the seed to hand SB3 — vector, multi-agent (5th seam) or image (G4b).

    Vector single-agent envs go through the shared factory (variant kwargs + the discrete-obs
    one-hot wrapper) and let SB3 seed python/numpy/torch + the env from ``config.seed``.
    **Image** envs (Atari + CarRacing) go through the shared ``make_image_vec`` dispatcher (the
    AtariWrapper+frame-stack or the CarRacing raw-RGB+frame-stack builder, ``n_envs=8``) so the
    CnnPolicy sees the obs shape the preview will match; its DummyVecEnv exposes ``seed()`` so SB3
    seeds it normally. **Multi-agent** (PettingZoo) envs go
    through the SuperSuit parameter-sharing bridge (``ma_env.make_vec_env``): its ``ConcatVecEnv``
    exposes no ``seed()``, so we seed the policy globally here and pass ``seed=None`` to PPO
    (otherwise SB3 calls ``env.seed()`` and crashes). MA reproducibility is therefore policy-level.
    """
    if is_multi_agent(get_env(config.env_id)):
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(config.seed)  # seed numpy/torch/python (the SuperSuit vec env can't be seeded)
        return make_vec_env(config.env_id), None
    if _is_image_env(config):
        from app.envs.image_vec import make_image_vec

        spec = get_env(config.env_id)
        assert spec is not None  # _is_image_env already established this
        return make_image_vec(spec, _IMAGE_N_ENVS, seed=config.seed), config.seed
    if vecnorm.should_normalize(config.env_id):
        # MuJoCo (G5c): wrap the vector env in VecNormalize (obs + reward) — the rl-zoo3 recipe and the
        # single biggest lever for PPO to climb on MuJoCo's wildly-scaled obs. The Monitor sits inside
        # the wrapper so ep_rew_mean stays raw (skill meter unchanged); the inner DummyVecEnv exposes
        # seed(), so SB3 seeds the run normally (config.seed below).
        env = vecnorm.wrap_train_env(
            lambda: make_env(config.env_id, gym_id), config.hyperparams.gamma
        )
        return env, config.seed
    return make_env(config.env_id, gym_id), config.seed


def _build_model(config: TrainConfig, gym_id: str) -> _InterruptiblePPO:
    hp = config.hyperparams
    env, seed = _make_train_env(config, gym_id)
    if _is_image_env(config):
        # Image obs (Atari 84×84×4 / CarRacing 96×96×6, G4b/G3c-train): a CnnPolicy on CUDA over the
        # frame stack. The net_arch / activation sliders describe an MLP and don't apply to the fixed
        # NatureCNN feature extractor, so leave policy_kwargs at SB3's default; the lr/γ/clip/ent/
        # n_steps/batch knobs still tune the run. device="cuda" is gated upstream (manager rejects no GPU).
        policy, device, policy_kwargs = "CnnPolicy", "cuda", None
    else:
        # One MlpPolicy on CPU serves the rest: a single-agent factory env, or the multi-agent
        # SuperSuit vec env where the same policy is shared across all N homogeneous agents
        # (parameter sharing, ADR-038). SB3 sees a standard vector-obs env, so the callback, ticker
        # and numpy-predict snapshot are unchanged.
        policy, device = "MlpPolicy", "cpu"
        policy_kwargs = {
            "net_arch": [hp.neurons_per_layer] * hp.n_hidden_layers,
            "activation_fn": _ACTIVATIONS[hp.activation],
        }
    return _InterruptiblePPO(
        policy,
        env,
        seed=seed,
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        clip_range=hp.clip_range,
        ent_coef=hp.ent_coef,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        n_epochs=hp.n_epochs,  # passes per rollout; Atari uses 4 (zoo recipe), vector envs keep 10
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=0,
    )


def _load_model(config: TrainConfig, gym_id: str, resume_blob: bytes) -> _InterruptiblePPO:
    """Rebuild a PPO model from a saved ``model.zip`` and attach a fresh env.

    The env is built through the shared factory — or the multi-agent SuperSuit bridge — exactly as
    in training, so ``PPO.load``'s ``check_for_correct_spaces`` matches; loading a checkpoint whose
    observation/action space no longer fits the env raises (surfaced as a clear error by the
    manager). ``num_timesteps`` is restored, so ``reset_num_timesteps=False`` continues the counter.
    Loaded into the interruptible subclass (``load`` instantiates ``cls``) so a resumed run's update
    phase is stoppable too; the subclass adds no persisted state, so the zip stays compatible.
    """
    env, _ = _make_train_env(config, gym_id)
    device = "cuda" if _is_image_env(config) else "cpu"  # image (CnnPolicy) resumes on the GPU (G4b)
    # G5c: _make_train_env returns a FRESH VecNormalize for MuJoCo (identity stats). Restore the saved
    # running obs/reward stats so the resumed policy keeps seeing normalized obs (else it degrades while
    # the stats re-converge from scratch). No-op for un-normalized envs / pre-G5c checkpoints.
    if vecnorm.should_normalize(config.env_id):
        vecnorm.restore_into(env, resume_blob)
    return _InterruptiblePPO.load(io.BytesIO(resume_blob), env=env, device=device)


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
        _load_model(config, gym_id, resume_blob)
        if resume_blob is not None
        else _build_model(config, gym_id)
    )
    # Let the update phase (train()) bail between epochs the moment a Stop is requested, so a heavy
    # multi-agent update doesn't strand the run in "stopping" for tens of seconds (see _InterruptiblePPO).
    model.stop_check = lambda: control.stop_requested
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
