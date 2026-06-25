"""SISL Waterworld (``waterworld_v4``) — vendored from PettingZoo.

**Source:** Farama-Foundation/PettingZoo, tag ``1.24.3`` (the last release that shipped Waterworld;
removed in 1.25.0 with the rest of the ``pymunk`` SISL world). **License:** MIT (Farama Foundation),
unchanged — see the project's ``LICENSE``.

**Local adaptations** (kept to the absolute minimum so the physics/reward/render behaviour is
byte-for-byte the upstream env):
  * ``waterworld.py`` / ``waterworld_base.py`` — the intra-package imports were relativised
    (``from pettingzoo.sisl.waterworld.X`` → ``from .X``) so the package is self-contained here.
  * ``waterworld.py`` — ``agent_selector`` → ``AgentSelector`` (PettingZoo renamed the class; the
    1.26.1 we pin no longer exports the old lowercase name in new code, and the sibling
    ``multiwalker_v9`` uses the new name).
  * ``waterworld_models.py`` — verbatim (it imports only numpy / pygame / pymunk / gymnasium).

Everything else (the ``env`` / ``parallel_env`` / ``raw_env`` factory shape, the ``pymunk`` physics,
the pygame ``rgb_array`` render) is upstream, so it plugs into the cooperative parameter-sharing MA
path (``ma_env`` → SuperSuit) exactly like the stock SISL envs. ``parallel_env`` is what
``ma_env._load_scenario`` calls.
"""

from .waterworld import env, parallel_env, raw_env

__all__ = ["env", "parallel_env", "raw_env"]
