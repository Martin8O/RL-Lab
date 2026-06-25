"""Vendored third-party environments — code we ship in-tree because upstream dropped it.

Some envs we want are no longer packaged by their upstream library, so the only way to keep
offering them is to carry a copy of the source here. Each sub-package documents exactly where it
came from (project, version/tag, license) and what local adaptations were made, so the provenance
stays auditable.

The multi-agent scenario loader (:func:`app.services.ma_env._load_scenario`) probes this package by
name after the real PettingZoo namespaces, so a vendored ``<id>`` resolves as
``app.envs.vendored.<id>`` with the same one-line code path as a stock ``pettingzoo.sisl`` env.

Current contents:
  * ``waterworld_v4`` — SISL Waterworld. **PettingZoo removed Waterworld in 1.25.0** (it depends on
    ``pymunk``, which Farama dropped from the maintained set), so 1.26.1 — the version this project
    pins — ships only ``pursuit_v4`` + ``multiwalker_v9``. Installing ``pymunk`` is necessary but not
    sufficient: the env *code* is gone from the library. Rather than downgrade PettingZoo (which would
    regress the two SISL envs + MPE we already ship), the env is vendored here. See that sub-package's
    ``__init__`` for the source tag + the adaptations.
"""
