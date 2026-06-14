"""Build a deterministic inference fn from a saved checkpoint — for AI play sessions.

The play session (mode ``"ai"``) needs to *act* with a previously trained model without
resuming training. Given a :class:`~app.services.checkpoints.LoadedCheckpoint`, this returns a
plain ``predict(obs) -> int`` over the frozen policy:

* **PPO** — ``PPO.load`` the saved ``model.zip`` and predict deterministically (no env needed
  for inference; we never call ``learn``).
* **neuroevolution** — reconstruct the champion genome (``population[0]``, the elite carried
  over by the trainer's ``_breed``) as the same numpy :class:`~app.services.trainer_evolution._Policy`
  used during evolution, reading the obs/act/hidden dims the trainer stored in ``population.npz``.

ML imports (torch/SB3, numpy) are **lazy**, inside the function, so importing this module — and
the play session that uses it — stays torch-free and fast to boot, matching the trainers and the
preview streamer.
"""

from collections.abc import Callable
from io import BytesIO
from typing import Any

from app.services.checkpoints import LoadedCheckpoint

# Returns a discrete action (int) for a Discrete env, or a continuous action vector
# (numpy float array) for a Box env — the play loop steps the env with whatever it gets.
PredictFn = Callable[[object], Any]


class PolicyLoadError(RuntimeError):
    """Raised when a checkpoint cannot be turned into a usable policy (corrupt/mismatched)."""


def predict_from_checkpoint(loaded: LoadedCheckpoint) -> PredictFn:
    """Return a deterministic ``predict(obs) -> int`` over the checkpoint's frozen policy."""
    try:
        if loaded.config.algo == "ppo":
            return _ppo_predict(loaded.blob)
        return _evolution_predict(loaded.blob)
    except PolicyLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 — any deserialize failure → clear, typed error
        raise PolicyLoadError(f"Could not load policy from checkpoint: {exc}") from exc


def _ppo_predict(blob: bytes) -> PredictFn:
    import numpy as np
    from stable_baselines3 import PPO

    model = PPO.load(BytesIO(blob), device="cpu")  # env not needed for inference
    # A Discrete action space has `.n`; a Box (continuous) one has `.low`/`.high` instead.
    is_box = getattr(model.action_space, "n", None) is None
    low = np.asarray(getattr(model.action_space, "low", 0.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 0.0), dtype=np.float32)

    def predict(obs: object) -> Any:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        arr = np.asarray(action)
        if is_box:  # continuous: the deterministic action is the Gaussian mean, clipped to bounds
            return np.clip(arr.astype(np.float32).reshape(-1), low, high)
        return int(arr.flatten()[0])

    return predict


def _evolution_predict(blob: bytes) -> PredictFn:
    import numpy as np

    from app.services.trainer_evolution import _Policy

    data = np.load(BytesIO(blob))
    population = np.asarray(data["population"], dtype=np.float64)
    obs_dim = int(data["obs_dim"])
    act_dim = int(data["act_dim"])
    hidden = int(data["hidden"])
    # Continuous checkpoints store the action bounds so the champion squashes its output into
    # [low, high]; discrete checkpoints omit them (the genome argmaxes instead) — back-compatible.
    act_low = data.get("act_low")
    act_high = data.get("act_high")
    champion = _Policy(  # population[0] = elite carried over by the trainer's _breed
        obs_dim, hidden, act_dim, population[0], act_low=act_low, act_high=act_high
    )

    def predict(obs: object) -> Any:
        return champion.act(np.asarray(obs, dtype=np.float64))

    return predict
