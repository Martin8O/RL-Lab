"""verify_env.py - A1 environment smoke test.

Prints interpreter + ML library versions and runs a tiny CartPole PPO training to
prove the whole stack (Gymnasium env -> SB3 PPO -> PyTorch) works end to end.

Hardware note: the current laptop (AMD Ryzen 5 2500U, Vega 8, no NVIDIA GPU) is
CPU-only, so ``CUDA available`` is expected to be **False** here. On the RTX 5070
desktop (PyTorch built for cu128) the same script should print ``True`` and a
device capability of ``(12, 0)``.

Run:
    .\\.venv\\Scripts\\Activate.ps1
    python backend/verify_env.py
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    # --- interpreter -------------------------------------------------------
    print(f"Python        : {platform.python_version()}")
    print(f"Executable    : {sys.executable}")

    # --- core ML stack -----------------------------------------------------
    import torch

    print(f"torch         : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():  # desktop RTX 5070 path
        print(f"CUDA device   : {torch.cuda.get_device_name(0)}")
        print(f"capability    : {torch.cuda.get_device_capability(0)}")

    import gymnasium as gym
    import stable_baselines3 as sb3

    print(f"gymnasium     : {gym.__version__}")
    print(f"SB3           : {sb3.__version__}")

    # --- CartPole PPO smoke test ------------------------------------------
    # Two-line core: build a PPO agent on CartPole and train it briefly. An
    # explicit seed keeps the run reproducible (a project-wide convention).
    from stable_baselines3 import PPO

    print("\nCartPole PPO smoke test (total_timesteps=2000)...")
    model = PPO("MlpPolicy", "CartPole-v1", seed=0, verbose=0)
    model.learn(total_timesteps=2000)

    # Assert the trained policy yields a valid action - proves the full
    # observe -> act loop is wired, not merely that learn() returned.
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)
    assert env.action_space.contains(int(action)), f"invalid action: {action!r}"
    env.close()

    print("OK - PPO trained 2000 steps and produced a valid action.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
