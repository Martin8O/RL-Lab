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
from app.services.checkpoints import CheckpointArtifact
from app.services.ma_env import is_multi_agent, make_vec_env
from app.services.train_control import TrainControl

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}
_PROGRESS_INTERVAL = 1.0  # seconds between live progress frames
_ATARI_N_ENVS = 8  # parallel image envs for the CnnPolicy rollout (the standard Atari setup)


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

    def predict(obs: object) -> Any:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in layers:
            x = x @ w.T + b
            x = np.maximum(0.0, x) if relu else np.tanh(x)
        out = x @ act_w.T + act_b
        if is_box:  # continuous: the mean action, clipped into [low, high]
            return np.clip(out, low, high).astype(np.float32)
        return int(np.argmax(out))

    return predict


def _build_cnn_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """A read-only **CPU torch** forward over a snapshot of an image-obs CnnPolicy (G4b).

    The numpy forward above only covers an MLP (``mlp_extractor.policy_net`` + ``action_net``); a
    CnnPolicy's NatureCNN feature extractor has no such path, so the preview needs a real torch
    forward. ADR-019 still holds — the preview must never touch the *live* CUDA model: a deepcopy
    of the policy trips torch's non-leaf-tensor guard, so we round-trip the policy through SB3's own
    ``save``/``load`` into an **independent CPU policy**. That copy shares no tensor storage with the
    trainer, so forwarding it cannot perturb training (the same isolation the numpy snapshot gives).

    Built at a rollout boundary (a quiescent point on the trainer thread). Returns ``predict(obs) ->
    int`` over the stacked 84×84×4 observation the preview's matching vec env yields.
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

    def predict(obs: object) -> Any:
        with torch.no_grad():
            action, _ = policy.predict(np.asarray(obs), deterministic=True)
        return int(np.asarray(action).flatten()[0])

    return predict


def _build_preview_predict(model: BaseAlgorithm) -> Callable[[object], Any]:
    """Dispatch the decoupled preview policy by observation rank: a 3-D obs (H, W, C) is an image
    (CnnPolicy → CPU torch snapshot); anything else is a vector/one-hot obs (the numpy MLP forward).
    Either way the result is a self-contained predict fn over copied weights — never the live model."""
    if len(getattr(model.observation_space, "shape", ()) or ()) == 3:
        return _build_cnn_predict(model)
    return _build_numpy_predict(model)


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
            self._on_policy(_build_preview_predict(self.model))


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


def _is_image_env(config: TrainConfig) -> bool:
    """True for an image-observation env (Atari) — the CnnPolicy + CUDA + frame-stack path (G4b)."""
    spec = get_env(config.env_id)
    return spec is not None and spec.obs_type == "image"


def _make_train_env(config: TrainConfig, gym_id: str) -> tuple[Any, int | None]:
    """Build the training env + the seed to hand SB3 — vector, multi-agent (5th seam) or image (G4b).

    Vector single-agent envs go through the shared factory (variant kwargs + the discrete-obs
    one-hot wrapper) and let SB3 seed python/numpy/torch + the env from ``config.seed``.
    **Image** envs (Atari) go through the shared ``make_atari`` vec builder (AtariWrapper +
    frame-stack, ``n_envs=8``) so the CnnPolicy sees the obs shape the preview will match; its
    DummyVecEnv exposes ``seed()`` so SB3 seeds it normally. **Multi-agent** (PettingZoo) envs go
    through the SuperSuit parameter-sharing bridge (``ma_env.make_vec_env``): its ``ConcatVecEnv``
    exposes no ``seed()``, so we seed the policy globally here and pass ``seed=None`` to PPO
    (otherwise SB3 calls ``env.seed()`` and crashes). MA reproducibility is therefore policy-level.
    """
    if is_multi_agent(get_env(config.env_id)):
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(config.seed)  # seed numpy/torch/python (the SuperSuit vec env can't be seeded)
        return make_vec_env(config.env_id), None
    if _is_image_env(config):
        from app.envs.atari import make_atari

        spec = get_env(config.env_id)
        assert spec is not None  # _is_image_env already established this
        return make_atari(
            gym_id, _ATARI_N_ENVS, make_kwargs=spec.make_kwargs, seed=config.seed
        ), config.seed
    return make_env(config.env_id, gym_id), config.seed


def _build_model(config: TrainConfig, gym_id: str) -> _InterruptiblePPO:
    hp = config.hyperparams
    env, seed = _make_train_env(config, gym_id)
    if _is_image_env(config):
        # Image obs (Atari, G4b): a CnnPolicy on CUDA over the 84×84×4 frame stack. The net_arch /
        # activation sliders describe an MLP and don't apply to the fixed NatureCNN feature
        # extractor, so leave policy_kwargs at SB3's default; the lr/γ/clip/ent/n_steps/batch knobs
        # still tune the run. device="cuda" is gated upstream (training_manager rejects on no GPU).
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
