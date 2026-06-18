"""Persistent training checkpoints — save / load / export slots.

Each checkpoint is a directory under the gitignored ``data/checkpoints/<id>/`` holding:

* ``meta.json``   — :class:`~app.schemas.checkpoints.CheckpointMeta` (listing + provenance)
* ``config.json`` — the full :class:`~app.schemas.training.TrainConfig` (resume + reproducibility)
* ``metrics.json``— the run's per-rollout / per-generation metric frames
* the model artifact — ``model.zip`` (PPO SB3 save) or ``population.npz`` (evolution genomes)

This module is deliberately **ML-free**: the trainer serializes the live model to bytes at a
safe point (rollout/generation boundary) and hands them here as a :class:`CheckpointArtifact`;
resume hands the same bytes back to the trainer to deserialize. Keeping torch/numpy out of the
checkpoint store (and the REST layer that imports it) preserves the fast, torch-free boot the
rest of the backend relies on.

Thread-safe: ``save`` runs on a request handler while a run may still be live, so every
filesystem mutation takes a lock. The root dir is an instance attribute (not a module constant)
so tests can point a fresh store at a tmp dir.
"""

import io
import json
import shutil
import threading
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.core.paths import data_dir
from app.schemas.checkpoints import CheckpointMeta
from app.schemas.training import TrainConfig

logger = get_logger(__name__)

# Slots live under the per-user writable data dir (repo-root data/ in dev, %LOCALAPPDATA% when
# packaged — see app.core.paths), never inside the read-only package tree.
_DEFAULT_ROOT = data_dir() / "checkpoints"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _is_safe_id(cid: str) -> bool:
    """Guard against path traversal — ids we mint are ``<timestamp>-<hex>`` only."""
    return bool(cid) and all(c.isalnum() or c in "-_" for c in cid)


@dataclass
class CheckpointArtifact:
    """A serialized snapshot of the live model, captured by the trainer at a safe point.

    ``blob`` is the already-serialized model (``model.zip`` bytes for PPO, ``population.npz``
    bytes for evolution) so this — and the manager that holds it — stay ML-free.
    """

    algo: str
    blob: bytes
    artifact_name: str
    reward: float | None = None
    timesteps: int = 0
    total_timesteps: int = 0
    iteration: int | None = None
    generation: int | None = None
    total_generations: int | None = None


@dataclass
class LoadedCheckpoint:
    """A checkpoint read back from disk, ready to hand to the trainer's resume path."""

    meta: CheckpointMeta
    config: TrainConfig
    blob: bytes


def _default_label(config: TrainConfig, artifact: CheckpointArtifact) -> str:
    """A readable auto-label when the user doesn't name the slot."""
    if artifact.generation is not None:
        progress = f"gen {artifact.generation}"
    elif config.algo == "q_learning":
        progress = f"ep {artifact.iteration}"  # Q-learning stores episodes elapsed in iteration
    elif config.algo == "alphazero":
        # AlphaZero's budget is self-play iterations, not k-steps — `iteration` holds the count, and
        # `timesteps` is games (so `timesteps // 1000` would read a misleading "0k" for a real run).
        progress = f"{artifact.iteration} it"
    else:
        progress = f"{artifact.timesteps // 1000}k"
    return f"{config.env_id} · {config.algo} · {progress}"


class CheckpointStore:
    """A directory-per-slot store of training checkpoints under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()

    def _slot_dir(self, cid: str) -> Path:
        return self.root / cid

    # -- write ------------------------------------------------------------------

    def save(
        self,
        config: TrainConfig,
        artifact: CheckpointArtifact,
        metrics: list[dict],
        label: str | None = None,
    ) -> CheckpointMeta:
        """Persist a new slot from a trainer snapshot; returns its :class:`CheckpointMeta`."""
        cid = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        meta = CheckpointMeta(
            id=cid,
            label=label or _default_label(config, artifact),
            env_id=config.env_id,
            algo=config.algo,
            seed=config.seed,
            created_at=_utc_now_iso(),
            reward=artifact.reward,
            timesteps=artifact.timesteps,
            total_timesteps=artifact.total_timesteps,
            iteration=artifact.iteration,
            generation=artifact.generation,
            total_generations=artifact.total_generations,
            artifact=artifact.artifact_name,
        )
        with self._lock:
            slot = self._slot_dir(cid)
            slot.mkdir(parents=True, exist_ok=True)
            (slot / artifact.artifact_name).write_bytes(artifact.blob)
            (slot / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
            (slot / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
            )
            (slot / "meta.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved checkpoint %s (%s)", cid, meta.label)
        return meta

    # -- read -------------------------------------------------------------------

    def list(self) -> list[CheckpointMeta]:
        """All saved slots, newest first. Unreadable slots are skipped, not fatal."""
        with self._lock:
            metas: list[CheckpointMeta] = []
            if self.root.exists():
                for slot in self.root.iterdir():
                    if not slot.is_dir():
                        continue
                    try:
                        metas.append(
                            CheckpointMeta.model_validate_json(
                                (slot / "meta.json").read_text(encoding="utf-8")
                            )
                        )
                    except (OSError, ValueError):
                        logger.warning("Skipping unreadable checkpoint slot %s", slot.name)
            metas.sort(key=lambda m: m.created_at, reverse=True)
            return metas

    def get(self, cid: str) -> CheckpointMeta | None:
        if not _is_safe_id(cid):
            return None
        with self._lock:
            try:
                return CheckpointMeta.model_validate_json(
                    (self._slot_dir(cid) / "meta.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError):
                return None

    def load(self, cid: str) -> LoadedCheckpoint | None:
        """Read a slot's meta + config + model bytes for the trainer's resume path."""
        if not _is_safe_id(cid):
            return None
        with self._lock:
            slot = self._slot_dir(cid)
            try:
                meta = CheckpointMeta.model_validate_json(
                    (slot / "meta.json").read_text(encoding="utf-8")
                )
                config = TrainConfig.model_validate_json(
                    (slot / "config.json").read_text(encoding="utf-8")
                )
                blob = (slot / meta.artifact).read_bytes()
            except (OSError, ValueError):
                return None
            return LoadedCheckpoint(meta=meta, config=config, blob=blob)

    def export_zip(self, cid: str) -> tuple[bytes, str] | None:
        """Bundle a slot's files into a zip; returns ``(zip_bytes, filename)`` or ``None``."""
        if not _is_safe_id(cid):
            return None
        with self._lock:
            slot = self._slot_dir(cid)
            if not slot.is_dir():
                return None
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(slot.iterdir()):
                    if path.is_file():
                        zf.write(path, arcname=path.name)
            return buf.getvalue(), f"{cid}.zip"

    def delete(self, cid: str) -> bool:
        """Remove a slot; returns ``True`` if it existed."""
        if not _is_safe_id(cid):
            return False
        with self._lock:
            slot = self._slot_dir(cid)
            if not slot.is_dir():
                return False
            shutil.rmtree(slot, ignore_errors=True)
            logger.info("Deleted checkpoint %s", cid)
            return True


# Module singleton, pointed at the gitignored data/ dir.
checkpoint_store = CheckpointStore(_DEFAULT_ROOT)
