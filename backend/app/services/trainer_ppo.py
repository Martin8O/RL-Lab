"""SB3 PPO trainer for CartPole — runs synchronously on a background thread.

Imported lazily by the training manager so that torch/SB3 are only loaded when a run
actually starts (keeps /health, /envs and the WS echo torch-free and fast to boot).
"""

import time
from collections.abc import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn

from app.schemas.training import TrainConfig, TrainingMetrics, TrainState
from app.services.train_control import TrainControl

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}

MetricsSink = Callable[[TrainingMetrics], None]


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
        self._iteration = 0

    def _on_step(self) -> bool:
        # Park here while paused; wake and abort if a stop was requested.
        self._control.wait_if_paused()
        return not self._control.stop_requested

    def _on_rollout_end(self) -> None:
        self._iteration += 1
        buf = self.model.ep_info_buffer
        if buf:
            ep_rew_mean: float | None = sum(e["r"] for e in buf) / len(buf)
            ep_len_mean: float | None = sum(e["l"] for e in buf) / len(buf)
        else:
            ep_rew_mean = ep_len_mean = None

        # loss / lr are recorded during the previous update; absent on the first rollout.
        recorded = self.model.logger.name_to_value
        loss = recorded.get("train/loss")
        lr = recorded.get("train/learning_rate")

        self._on_metrics(
            TrainingMetrics(
                iteration=self._iteration,
                timesteps=int(self.model.num_timesteps),
                total_timesteps=self._total,
                ep_rew_mean=ep_rew_mean,
                ep_len_mean=ep_len_mean,
                loss=float(loss) if loss is not None else None,
                learning_rate=float(lr) if lr is not None else None,
                elapsed=time.monotonic() - self._started_at,
            )
        )


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
) -> TrainState:
    """Train PPO to completion (or until stopped). Returns the terminal state.

    Blocks the calling thread; the manager runs this off the event loop.
    """
    model = _build_model(config, gym_id)
    callback = _MetricsCallback(
        control, on_metrics, config.total_timesteps, time.monotonic()
    )
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=callback)
    finally:
        if model.env is not None:
            model.env.close()
    return "stopped" if control.stop_requested else "finished"
