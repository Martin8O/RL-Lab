"""Persistent per-environment high scores (the "all-time best" hall of fame).

Stores one best score per env in a gitignored ``data/highscores.json`` so it is per-device
and survives restarts. A new score is recorded only when it strictly beats the stored best;
writes are atomic (temp file + ``os.replace``) so a crash mid-write can't corrupt the file.

Thread-safe: ``record`` is called from the trainer thread (via the training manager) while
``get``/``all`` are served from request handlers, so every access takes a lock. The path is
an instance attribute (not a module constant) so tests can point a fresh store at a tmp dir.
"""

import contextlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.core.paths import data_dir
from app.schemas.highscores import HighScore, HighScoreMeta

logger = get_logger(__name__)

# The file lives under the per-user writable data dir (repo-root data/ in dev, %LOCALAPPDATA%
# when packaged — see app.core.paths), never inside the read-only package tree.
_DEFAULT_PATH = data_dir() / "highscores.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class HighScoreStore:
    """A small JSON-backed map of env_id → best :class:`HighScore`."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._scores: dict[str, HighScore] | None = None  # lazy-loaded cache

    # -- persistence ------------------------------------------------------------

    def _load_locked(self) -> dict[str, HighScore]:
        if self._scores is not None:
            return self._scores
        scores: dict[str, HighScore] = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for env_id, entry in raw.items():
                scores[env_id] = HighScore.model_validate(entry)
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, ValueError):
            # A corrupt/old file should never crash the app — start fresh, keep a copy.
            logger.warning("High-score file unreadable; starting fresh (%s)", self.path)
        self._scores = scores
        return scores

    def _write_locked(self, scores: dict[str, HighScore]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {env_id: hs.model_dump() for env_id, hs in scores.items()}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        # Atomic replace: write a sibling temp file then rename over the target.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    # -- public API -------------------------------------------------------------

    def get(self, env_id: str) -> HighScore | None:
        with self._lock:
            return self._load_locked().get(env_id)

    def all(self) -> list[HighScore]:
        with self._lock:
            return list(self._load_locked().values())

    def record(self, env_id: str, score: float, meta: HighScoreMeta) -> HighScore | None:
        """Persist ``score`` as the new best for ``env_id`` iff it beats the stored best.

        Returns the new :class:`HighScore` when a record was set (so the caller can broadcast
        it), or ``None`` when the existing best stands.
        """
        with self._lock:
            scores = self._load_locked()
            current = scores.get(env_id)
            if current is not None and score <= current.score:
                return None
            record = HighScore(env_id=env_id, score=float(score), meta=meta)
            scores[env_id] = record
            self._write_locked(scores)
            return record


def make_meta(
    algo: str, seed: int, *, generation: int | None = None, iteration: int | None = None
) -> HighScoreMeta:
    """Build a provenance record stamped with the current UTC time."""
    return HighScoreMeta(
        algo=algo,  # type: ignore[arg-type]  # validated against the Algo literal
        seed=seed,
        generation=generation,
        iteration=iteration,
        achieved_at=_utc_now_iso(),
    )


# Module singleton, pointed at the gitignored data/ dir.
highscores = HighScoreStore(_DEFAULT_PATH)
