"""Build a deterministic inference fn from a saved checkpoint — for AI play sessions.

The play session (mode ``"ai"``) needs to *act* with a previously trained model without
resuming training. Given a :class:`~app.services.checkpoints.LoadedCheckpoint`, this returns a
plain ``predict(obs) -> int`` over the frozen policy:

* **PPO** — ``PPO.load`` the saved ``model.zip`` and predict deterministically (no env needed
  for inference; we never call ``learn``). **SAC / TD3 / DQN / A2C / QR-DQN** load their own SB3(-contrib)
  class the same way (SAC/TD3 box-only, DQN + QR-DQN discrete-only, A2C either — QR-DQN acts on its
  quantile-mean, so from the caller's side it returns a plain int exactly like DQN).
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
        if loaded.config.algo == "sac":
            return _sac_predict(loaded.blob)
        if loaded.config.algo == "td3":
            return _td3_predict(loaded.blob)
        if loaded.config.algo == "dqn":
            return _dqn_predict(loaded.blob)
        if loaded.config.algo == "qrdqn":
            return _qrdqn_predict(loaded.blob)
        if loaded.config.algo == "a2c":
            return _a2c_predict(loaded.blob)
        if loaded.config.algo == "q_learning":
            return _q_learning_predict(loaded.blob)
        return _evolution_predict(loaded.blob)
    except PolicyLoadError:
        raise
    except Exception as exc:  # noqa: BLE001 — any deserialize failure → clear, typed error
        raise PolicyLoadError(f"Could not load policy from checkpoint: {exc}") from exc


def _ppo_predict(blob: bytes) -> PredictFn:
    import numpy as np
    from stable_baselines3 import PPO

    from app.services import vecnorm

    model = PPO.load(BytesIO(blob), device="cpu")  # env not needed for inference
    # A Discrete action space has `.n`; a Box (continuous) one has `.low`/`.high` instead.
    is_box = getattr(model.action_space, "n", None) is None
    low = np.asarray(getattr(model.action_space, "low", 0.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 0.0), dtype=np.float32)
    # G5c: a MuJoCo checkpoint embeds its VecNormalize obs stats in the blob. AI-play feeds the policy
    # RAW obs (the play env is un-normalized), so normalize them the same way training did before
    # predicting — otherwise a competently-trained agent gets unscaled obs and plays like it never
    # trained. None for un-normalized envs / pre-G5c checkpoints → obs passes through unchanged. Reward
    # normalization is training-only and is never applied here, so play scores stay on the raw scale.
    normalize = vecnorm.load_obs_normalizer(blob)

    def predict(obs: object) -> Any:
        x = np.asarray(obs)
        if normalize is not None:
            x = normalize(x)
        action, _ = model.predict(x, deterministic=True)
        arr = np.asarray(action)
        if is_box:  # continuous: the deterministic action is the Gaussian mean, clipped to bounds
            return np.clip(arr.astype(np.float32).reshape(-1), low, high)
        return int(arr.flatten()[0])

    return predict


def _a2c_predict(blob: bytes) -> PredictFn:
    """Deterministic A2C inference for AI-play (S5d). A2C is an on-policy actor-critic just like PPO, so
    this mirrors the PPO loader exactly — ``deterministic=True`` returns the greedy action (an ``int``
    for a discrete env, the Gaussian mean clipped to bounds for a continuous Box env), covering both
    action types A2C is offered on. A2C trains on raw obs (it is offered only on the un-normalized
    classic-control envs, never the VecNormalize'd MuJoCo family), so there are no running stats to
    apply; ``load_obs_normalizer`` returns None for its blobs, leaving obs unchanged."""
    import numpy as np
    from stable_baselines3 import A2C

    from app.services import vecnorm

    model = A2C.load(BytesIO(blob), device="cpu")  # env not needed for inference
    is_box = getattr(model.action_space, "n", None) is None
    low = np.asarray(getattr(model.action_space, "low", 0.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 0.0), dtype=np.float32)
    normalize = vecnorm.load_obs_normalizer(blob)  # None for A2C's un-normalized classic-control blobs

    def predict(obs: object) -> Any:
        x = np.asarray(obs)
        if normalize is not None:
            x = normalize(x)
        action, _ = model.predict(x, deterministic=True)
        arr = np.asarray(action)
        if is_box:  # continuous: the deterministic mean action, clipped to bounds
            return np.clip(arr.astype(np.float32).reshape(-1), low, high)
        return int(arr.flatten()[0])

    return predict


def _sac_predict(blob: bytes) -> PredictFn:
    """Deterministic SAC inference for AI-play (S5a). SAC is continuous-action only, so the action is
    always the actor's deterministic mean, clipped into the env's box bounds. Unlike the PPO loader,
    there is no VecNormalize to apply: SAC trains on raw obs/rewards, so the saved blob carries no
    running stats and play feeds the policy raw obs exactly as training did."""
    import numpy as np
    from stable_baselines3 import SAC

    model = SAC.load(BytesIO(blob), device="cpu")  # env not needed for inference
    low = np.asarray(getattr(model.action_space, "low", -1.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 1.0), dtype=np.float32)

    def predict(obs: object) -> Any:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), low, high)

    return predict


def _td3_predict(blob: bytes) -> PredictFn:
    """Deterministic TD3 inference for AI-play (S5b). TD3 is continuous-action only, and its actor is
    already deterministic, so ``predict(deterministic=True)`` drops the exploration noise and returns the
    learned action, clipped into the env's box bounds. Like SAC there is no VecNormalize to apply: TD3
    trains on raw obs/rewards, so play feeds the policy raw obs exactly as training did."""
    import numpy as np
    from stable_baselines3 import TD3

    model = TD3.load(BytesIO(blob), device="cpu")  # env not needed for inference
    low = np.asarray(getattr(model.action_space, "low", -1.0), dtype=np.float32)
    high = np.asarray(getattr(model.action_space, "high", 1.0), dtype=np.float32)

    def predict(obs: object) -> Any:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), low, high)

    return predict


def _dqn_predict(blob: bytes) -> PredictFn:
    """Deterministic DQN inference for AI-play (S5c). DQN is discrete-action and value-based: it acts by
    taking the highest-Q action, so ``predict(deterministic=True)`` drops the ε-exploration and returns
    the greedy ``argmax`` action as a plain ``int`` (the ADR-021 discrete contract). One loader serves
    both the vector MlpPolicy and the Atari CnnPolicy — ``model.predict`` handles the stacked image obs
    the image-AI loop feeds it. Like SAC/TD3 there is no VecNormalize: DQN trains on raw obs."""
    import numpy as np
    from stable_baselines3 import DQN

    # buffer_size=1 overrides the saved replay-buffer size: inference never touches the buffer, but
    # DQN.load would otherwise allocate the full trained size — for an Atari CnnPolicy that is GBs of
    # stacked frames (the S5c memory bug). Forcing it to 1 keeps AI-play memory-trivial for every DQN
    # save, including older Atari checkpoints saved before the buffer was trimmed. (Vector DQN buffers
    # are tiny anyway, so this is harmless there.)
    model = DQN.load(BytesIO(blob), device="cpu", buffer_size=1)  # env not needed for inference

    def predict(obs: object) -> Any:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        return int(np.asarray(action).flatten()[0])

    return predict


def _qrdqn_predict(blob: bytes) -> PredictFn:
    """Deterministic QR-DQN inference for AI-play (S5e). QR-DQN is discrete-action and distributional: it
    learns each action's return distribution but acts on its mean, so ``predict(deterministic=True)``
    drops the ε-exploration and returns the greedy ``argmax`` action as a plain ``int`` (the ADR-021
    discrete contract) — identical to DQN from the caller's side. One loader serves both the vector
    MlpPolicy and the Atari CnnPolicy. Like DQN there is no VecNormalize: QR-DQN trains on raw obs."""
    import numpy as np
    from sb3_contrib import QRDQN

    # buffer_size=1 overrides the saved replay-buffer size: inference never touches the buffer, but
    # QRDQN.load would otherwise allocate the full trained size — for an Atari CnnPolicy that is GBs of
    # stacked frames (the S5c memory bug, shared with DQN). Forcing it to 1 keeps AI-play memory-trivial.
    model = QRDQN.load(BytesIO(blob), device="cpu", buffer_size=1)  # env not needed for inference

    def predict(obs: object) -> Any:
        action, _ = model.predict(np.asarray(obs), deterministic=True)
        return int(np.asarray(action).flatten()[0])

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


def _q_learning_predict(blob: bytes) -> PredictFn:
    import numpy as np

    data = np.load(BytesIO(blob))
    table = np.asarray(data["qtable"], dtype=np.float64)

    def predict(obs: object) -> Any:
        # The play env one-hot-wraps the discrete obs (shared factory), so decode the integer
        # state with arg-max, then act greedily — the row's best action.
        state = int(np.argmax(np.asarray(obs)))
        return int(np.argmax(table[state]))

    return predict
