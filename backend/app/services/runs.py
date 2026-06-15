"""Run history — persist each finished run's config + metric history for comparison.

Each run is a directory under the gitignored ``data/runs/<id>/`` holding:

* ``meta.json``   — :class:`~app.schemas.runs.RunMeta` (listing + provenance)
* ``config.json`` — the full :class:`~app.schemas.training.TrainConfig` (reproducibility)
* ``metrics.json``— the run's per-rollout / per-generation metric frames (overlay source)

The training manager hands a finished run's config + recorded frames here when a run reaches
a terminal state, so past experiments can be listed (``GET /api/runs``) and overlaid on the
chart (``GET /api/runs/{id}``). Like the checkpoint/high-score stores this is deliberately
simple — plain JSON files, no DB — and thread-safe (``save`` runs on the trainer thread while
``list``/``get`` are served from request handlers, so every filesystem access takes a lock).
The root dir is an instance attribute (not a module constant) so tests can use a tmp dir.
"""

import json
import shutil
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.core.paths import data_dir
from app.schemas.runs import RunDetail, RunMeta
from app.schemas.training import TrainConfig, TrainState

logger = get_logger(__name__)

# Runs live under the per-user writable data dir (repo-root data/ in dev, %LOCALAPPDATA% when
# packaged — see app.core.paths), never inside the read-only package tree.
_DEFAULT_ROOT = data_dir() / "runs"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_safe_id(rid: str) -> bool:
    """Guard against path traversal — ids we mint are ``<timestamp>-<hex>`` only."""
    return bool(rid) and all(c.isalnum() or c in "-_" for c in rid)


def _frame_score(algo: str, frame: dict[str, Any]) -> float | None:
    """The per-frame score on the same scale as the env's solved score."""
    return frame.get("best_fitness") if algo == "neuroevolution" else frame.get("ep_rew_mean")


def _frame_x(algo: str, frame: dict[str, Any]) -> float | None:
    """A frame's x-coordinate in this algorithm's chart unit: generation (neuroevolution),
    episode (Q-learning) or timestep (PPO)."""
    if algo == "neuroevolution":
        return frame.get("generation")
    if algo == "q_learning":
        return frame.get("episode")
    return frame.get("timesteps")


def final_score(config: TrainConfig, metrics: list[dict[str, Any]]) -> float | None:
    """The run's final score (last frame) — best_fitness (evolution) / ep_rew_mean (PPO)."""
    return _frame_score(config.algo, metrics[-1]) if metrics else None


def should_archive(state: str, final: float | None, solved_score: float) -> bool:
    """Keep a run only if it finished/stopped *and* reached ≥10% of the solved score.

    Sub-10% runs (random-policy noise, instant stops) just clutter the compare view, so they
    are dropped. When the env's solved score is unknown (0), keep the run rather than lose it.
    """
    if state not in ("finished", "stopped"):
        return False
    if solved_score <= 0:
        return True
    return final is not None and final >= 0.1 * solved_score


def _solved_at(config: TrainConfig, metrics: list[dict[str, Any]], solved_score: float) -> float | None:
    """The x (timestep for PPO, generation for evolution) of the first frame to hit solved."""
    if solved_score <= 0:
        return None
    for f in metrics:
        score = _frame_score(config.algo, f)
        x = _frame_x(config.algo, f)
        if score is not None and x is not None and score >= solved_score:
            return float(x)
    return None


def _derive(
    config: TrainConfig, metrics: list[dict[str, Any]], solved_score: float
) -> dict[str, Any]:
    """Pull the summary fields (final reward, progress, solved-at) out of the recorded frames."""
    last = metrics[-1] if metrics else {}
    solved_at = _solved_at(config, metrics, solved_score)
    if config.algo == "neuroevolution":
        return {
            "final_reward": last.get("best_fitness"),
            "solved_at": solved_at,
            "timesteps": int(last.get("timesteps", 0)),
            "total_timesteps": config.total_timesteps,
            "iteration": None,
            "generation": last.get("generation"),
            "total_generations": last.get("total_generations"),
        }
    if config.algo == "q_learning":
        # Q-learning is episodic — record the episode counter (its chart x-unit) in ``iteration``
        # and the episode budget in ``total_generations`` (the generic "progress total" slot).
        return {
            "final_reward": last.get("ep_rew_mean"),
            "solved_at": solved_at,
            "timesteps": int(last.get("timesteps", 0)),
            "total_timesteps": 0,
            "iteration": last.get("episode"),
            "generation": None,
            "total_generations": last.get("total_episodes"),
        }
    return {
        "final_reward": last.get("ep_rew_mean"),
        "solved_at": solved_at,
        "timesteps": int(last.get("timesteps", 0)),
        "total_timesteps": config.total_timesteps,
        "iteration": last.get("iteration"),
        "generation": None,
        "total_generations": None,
    }


def _default_label(config: TrainConfig, summary: dict[str, Any]) -> str:
    """A readable auto-label: env · algo · progress · final reward."""
    if summary["generation"] is not None:
        progress = f"gen {summary['generation']}"
    elif config.algo == "q_learning":
        progress = f"ep {summary['iteration']}"
    else:
        progress = f"{summary['timesteps'] // 1000}k"
    rew = summary["final_reward"]
    tail = f" · {rew:.0f}" if rew is not None else ""
    return f"{config.env_id} · {config.algo} · {progress}{tail}"


class RunStore:
    """A directory-per-run store of finished training runs under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _run_dir(self, rid: str) -> Path:
        return self.root / rid

    # -- write ------------------------------------------------------------------

    def save(
        self,
        config: TrainConfig,
        metrics: list[dict[str, Any]],
        *,
        state: TrainState,
        started_at: str,
        solved_score: float = 0.0,
        label: str | None = None,
    ) -> RunMeta:
        """Persist a finished run; returns its :class:`RunMeta`."""
        rid = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        summary = _derive(config, metrics, solved_score)
        meta = RunMeta(
            id=rid,
            label=label or _default_label(config, summary),
            env_id=config.env_id,
            algo=config.algo,
            seed=config.seed,
            created_at=started_at,
            finished_at=_utc_now_iso(),
            state=state,
            final_reward=summary["final_reward"],
            solved_at=summary["solved_at"],
            timesteps=summary["timesteps"],
            total_timesteps=summary["total_timesteps"],
            iteration=summary["iteration"],
            generation=summary["generation"],
            total_generations=summary["total_generations"],
            frames=len(metrics),
        )
        with self._lock:
            run = self._run_dir(rid)
            run.mkdir(parents=True, exist_ok=True)
            (run / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
            (run / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
            )
            (run / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Recorded run %s (%s)", rid, meta.label)
        return meta

    # -- read -------------------------------------------------------------------

    def list(self) -> list[RunMeta]:
        """All recorded runs, newest first. Unreadable runs are skipped, not fatal."""
        with self._lock:
            metas: list[RunMeta] = []
            if self.root.exists():
                for run in self.root.iterdir():
                    if not run.is_dir():
                        continue
                    try:
                        metas.append(
                            RunMeta.model_validate_json(
                                (run / "meta.json").read_text(encoding="utf-8")
                            )
                        )
                    except (OSError, ValueError):
                        logger.warning("Skipping unreadable run %s", run.name)
            metas.sort(key=lambda m: m.created_at, reverse=True)
            return metas

    def get(self, rid: str) -> RunDetail | None:
        """Read a run's meta + config + recorded metric frames for the chart overlay."""
        if not _is_safe_id(rid):
            return None
        with self._lock:
            run = self._run_dir(rid)
            try:
                meta = RunMeta.model_validate_json(
                    (run / "meta.json").read_text(encoding="utf-8")
                )
                config = TrainConfig.model_validate_json(
                    (run / "config.json").read_text(encoding="utf-8")
                )
                metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return RunDetail(meta=meta, config=config, metrics=metrics)

    def delete(self, rid: str) -> bool:
        """Remove a run; returns ``True`` if it existed."""
        if not _is_safe_id(rid):
            return False
        with self._lock:
            run = self._run_dir(rid)
            if not run.is_dir():
                return False
            shutil.rmtree(run, ignore_errors=True)
            logger.info("Deleted run %s", rid)
            return True


# Module singleton, pointed at the gitignored data/ dir.
run_store = RunStore(_DEFAULT_ROOT)
