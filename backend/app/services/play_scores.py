"""Persistent per-environment play leaderboards (named Human + AI halls of fame).

Distinct from the training all-time best (services/highscores.py): this keeps the top
:data:`~app.schemas.play_scores.TOP_N` *named* scores for interactive play, split into a
``human`` and an ``ai`` board, per environment. Stored in a gitignored
``data/play_scores.json`` (per-device, survives restarts); writes are atomic (temp file +
``os.replace``) and every access takes a lock (submits arrive from request handlers).

The AI board keeps one row per distinct model (de-duped by ``model_id``, best score wins) so
replaying the same checkpoint doesn't flood it; the human board appends every qualifying run.
"""

import contextlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.core.logging import get_logger
from app.schemas.play_scores import (
    TOP_N,
    PlayCategory,
    PlayScoreEntry,
    PlayScoreResult,
    PlayScores,
)

logger = get_logger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "play_scores.json"
_MAX_NAME_LEN = 24


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_name(name: str) -> str:
    cleaned = name.strip()[:_MAX_NAME_LEN]
    return cleaned or "—"


class PlayScoreStore:
    """JSON-backed map of env_id → :class:`PlayScores` (two boards per env)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._cache: dict[str, PlayScores] | None = None

    # -- persistence ------------------------------------------------------------

    def _load_locked(self) -> dict[str, PlayScores]:
        if self._cache is not None:
            return self._cache
        cache: dict[str, PlayScores] = {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            for env_id, entry in raw.items():
                scores = PlayScores.model_validate(entry)
                # Trim to the current TOP_N on load, so boards saved under an older (larger) TOP_N
                # immediately present + qualify against the same cutoff the user sees.
                scores.human = scores.human[:TOP_N]
                scores.ai = scores.ai[:TOP_N]
                cache[env_id] = scores
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, ValueError):
            logger.warning("Play-scores file unreadable; starting fresh (%s)", self.path)
        self._cache = cache
        return cache

    def _write_locked(self, cache: dict[str, PlayScores]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {env_id: ps.model_dump() for env_id, ps in cache.items()}
        text = json.dumps(payload, indent=2, ensure_ascii=False)
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

    def get(self, env_id: str) -> PlayScores:
        """The two boards for ``env_id`` (empty boards if the env has no scores yet)."""
        with self._lock:
            existing = self._load_locked().get(env_id)
            return existing.model_copy(deep=True) if existing else PlayScores(env_id=env_id)

    def submit(
        self,
        env_id: str,
        category: PlayCategory,
        name: str,
        score: float,
        *,
        steps: int = 0,
        model_id: str | None = None,
        algo: str | None = None,
    ) -> PlayScoreResult:
        """Try to place a finished session on a board; persist iff it makes the top-N.

        Returns the (possibly unchanged) boards plus whether the entry qualified and its
        1-based rank. The AI board de-dupes by ``model_id`` (keeping the better score).
        """
        entry = PlayScoreEntry(
            name=_clean_name(name),
            score=float(score),
            steps=int(steps),
            achieved_at=_utc_now_iso(),
            model_id=model_id,
            algo=algo,
        )
        with self._lock:
            cache = self._load_locked()
            scores = cache.get(env_id) or PlayScores(env_id=env_id)
            board = list(getattr(scores, category))

            added = True
            if category == "ai" and entry.model_id is not None:
                prev = [e for e in board if e.model_id == entry.model_id]
                if prev and entry.score <= max(p.score for p in prev):
                    added = False  # an equal/better run of this model already stands
                else:
                    board = [e for e in board if e.model_id != entry.model_id]
                    board.append(entry)
            else:
                board.append(entry)

            board.sort(key=lambda e: e.score, reverse=True)
            trimmed = board[:TOP_N]
            qualified = added and entry in trimmed
            rank = trimmed.index(entry) + 1 if qualified else None

            if trimmed != list(getattr(scores, category)):
                setattr(scores, category, trimmed)
                cache[env_id] = scores
                self._write_locked(cache)

            return PlayScoreResult(
                scores=scores.model_copy(deep=True), qualified=qualified, rank=rank
            )


# Module singleton, pointed at the gitignored data/ dir.
play_scores = PlayScoreStore(_DEFAULT_PATH)
