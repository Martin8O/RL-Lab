"""Reproducibility provenance (X5) — the citable glue: config-hash, BibTeX, reproduce command, Methods.

A published RL result is only trustworthy if someone else can re-run it. This module turns one run's
:class:`~app.schemas.training.TrainConfig` into the artifacts a methods section needs:

* :func:`config_hash` — a **deterministic sha256** of the canonical config (sorted keys, the ``None``
  algorithm blocks dropped) → a stable citable ID. The same config always hashes to the same 64-hex
  string, on any machine, so it identifies *exactly this experiment*.
* :func:`bibtex` — a ``@software`` entry keyed by env·algo·hash, so the run can be cited.
* :func:`reproduce_command` — a paste-ready ``curl`` that re-launches the run through this app's real
  ``POST /api/train/start`` with the exact config (the CleanRL "one command reproduces it" pattern).
* :func:`methods_facts` — library versions (torch / SB3 / gymnasium / numpy), hardware, and the git
  commit, sampled from the **current** environment (metric frames don't persist HW telemetry, so this
  describes the machine the export runs on — the honest "environment used" statement).

The version/git/hardware probes touch the real environment (imports, ``git``, ``pynvml``) and each
degrades to ``"unknown"`` rather than raising, so an export never fails on a missing tool. Only
:func:`config_hash` / :func:`bibtex` / :func:`reproduce_command` are pure functions of the config —
those are the ones the tests pin for determinism.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from functools import lru_cache
from typing import Any

from app.core.paths import resource_root
from app.schemas.training import TrainConfig

# The public repository the BibTeX + reproduce command point at.
_PROJECT_URL = "https://github.com/Martin8O/RL-Lab"
_PROJECT_TITLE = "RL Lab — an all-in-one reinforcement-learning dashboard"


def _canonical_config(config: TrainConfig) -> dict[str, Any]:
    """The config reduced to its reproducibility-relevant fields, with the ``None`` algorithm blocks
    dropped — so two configs that differ only in an unused (``None``) hyperparameter block hash the
    same, and the surviving block is exactly the one ``algo`` selects."""
    raw = config.model_dump(mode="json")
    return {k: v for k, v in raw.items() if v is not None}


def config_hash(config: TrainConfig) -> str:
    """A stable sha256 (64 hex chars) of the canonical config — the citable experiment ID.

    Deterministic: ``json.dumps(..., sort_keys=True)`` gives one canonical byte string for a given
    config regardless of field insertion order, so the same experiment always yields the same hash.
    """
    canonical = json.dumps(_canonical_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# The per-run fields that vary *within* one experiment (a seed sweep) — dropped from the grouping hash.
_PER_RUN_FIELDS = ("seed", "experiment_id", "experiment_label")


def config_group_hash(config: TrainConfig) -> str:
    """A **seed-independent** sha256 of the config — identifies an *experiment*, not a run (X4).

    Same canonicalization as :func:`config_hash` but with the per-run fields removed (seed + the sweep
    id/label), so every seed of one sweep — and any two runs that differ *only* by seed — hash to the
    same value. This is what "the same experiment, aggregated across seeds" means for auto-grouping runs
    that carry no explicit ``experiment_id``.
    """
    canonical = _canonical_config(config)
    for field in _PER_RUN_FIELDS:
        canonical.pop(field, None)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_year(created_at: str) -> str:
    """The four-digit year from a run's ISO ``created_at`` (used for the BibTeX ``year``); the run's
    own year, not the export year, so the citation is deterministic + historically correct. Falls back
    to the leading 4 chars, then ``"n.d."`` (no date) if the timestamp is unparseable."""
    head = (created_at or "")[:4]
    return head if head.isdigit() else "n.d."


def bibtex(config: TrainConfig, *, created_at: str, label: str) -> str:
    """A ``@software`` BibTeX entry for one run, keyed ``rllab:<env>-<algo>-<hash8>`` (stable)."""
    h = config_hash(config)
    key = f"rllab:{config.env_id}-{config.algo}-{h[:8]}"
    year = _run_year(created_at)
    note = f"Run '{label}', config-hash {h}"
    return "\n".join(
        [
            f"@software{{{key},",
            f"  title  = {{{_PROJECT_TITLE}}},",
            "  author = {Svoboda, Martin},",
            f"  year   = {{{year}}},",
            f"  url    = {{{_PROJECT_URL}}},",
            f"  note   = {{{note}}}",
            "}",
        ]
    )


def reproduce_command(config: TrainConfig) -> str:
    """A paste-ready ``curl`` that re-runs this exact config through ``POST /api/train/start``.

    The whole point of recording the full config (X1/D2) is that the run is reproducible: this app's
    real training endpoint takes a :class:`TrainConfig` body, so replaying the archived config re-launches
    the identical experiment (same env, algo, seed, budget, hyperparameters). CleanRL's "one command" idea,
    made honest for this app.
    """
    body = config.model_dump_json()
    return (
        "curl -X POST http://localhost:8000/api/train/start "
        "-H 'Content-Type: application/json' "
        f"-d '{body}'"
    )


@lru_cache(maxsize=1)
def _library_versions() -> dict[str, str]:
    """Installed versions of the ML stack, each degrading to ``"unknown"`` if the import fails."""
    versions: dict[str, str] = {}
    for name, mod in (("torch", "torch"), ("stable_baselines3", "stable_baselines3"),
                      ("gymnasium", "gymnasium"), ("numpy", "numpy")):
        try:
            versions[name] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 — a missing/broken import must not fail an export
            versions[name] = "unknown"
    return versions


@lru_cache(maxsize=1)
def _git_commit() -> str:
    """The current git commit (short hash + ``-dirty`` when the tree has uncommitted changes), or
    ``"unknown"`` outside a git checkout / when git is unavailable (a packaged build)."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=resource_root(), capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=resource_root(), capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if dirty else rev
    except Exception:  # noqa: BLE001 — no git / not a checkout / timeout → unknown
        return "unknown"


def _gpu_name() -> str:
    """The CUDA device name if torch sees one, else ``"CPU only"``. Never raises."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — no torch / no CUDA → CPU
        pass
    return "CPU only"


def methods_facts() -> dict[str, str]:
    """The environment description for the XLSX ``Methods`` sheet — versions, hardware, git commit.

    Sampled from the machine the export runs on (metric frames don't persist HW telemetry), so it is the
    honest "environment used to produce this export" statement. Every field degrades to ``"unknown"`` /
    ``"CPU only"`` rather than raising.
    """
    libs = _library_versions()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": libs["torch"],
        "stable_baselines3": libs["stable_baselines3"],
        "gymnasium": libs["gymnasium"],
        "numpy": libs["numpy"],
        "gpu": _gpu_name(),
        "git_commit": _git_commit(),
    }
