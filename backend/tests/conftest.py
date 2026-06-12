"""Shared test fixtures.

Every test gets the high-score singleton redirected at a fresh tmp file, so trainer/manager
tests (which run real CartPole runs and therefore *record* scores) never touch the real
gitignored ``data/highscores.json`` and never leak state between tests.
"""

import pytest


@pytest.fixture(autouse=True)
def _isolate_highscores(tmp_path, monkeypatch):
    from app.services import highscores as hs

    monkeypatch.setattr(hs.highscores, "path", tmp_path / "highscores.json")
    monkeypatch.setattr(hs.highscores, "_scores", None)  # drop any cached map
    yield
