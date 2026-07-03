"""Analysis engine (Phase X / DataLab) — pure, unit-tested reductions over finished runs.

The modules here take *already-loaded* run data (frames + the env's skill range) and return plain
data — **no I/O, no global state** — so they are trivially testable and reusable by both the REST
endpoints (``app.api.analysis``) and, later, the export pipeline (X5). The caller (the API route)
owns loading a run from the ``data/runs/`` store and passing its frames in.
"""
