"""Parallel AlphaZero self-play across independent GPU actor processes (G6i, ADR-062).

The G6h actor–learner split (:mod:`app.services.trainer_az`) runs self-play on a background *thread*
sharing this process with the learner. On chess that thread shares the **GIL** with the learner and
starves during each training burst, so the GPU sat ~49 % busy. This module replaces that single
thread with ``actor_processes`` *independent* worker **processes**, each holding its OWN CUDA net and
self-playing locally via the **unchanged** :func:`app.services.az_batch.self_play_rolling` — **no
central inference server and no per-round IPC** (the original G6i design, measured and discarded: per
MCTS-round Queue traffic is latency-bound and slower than one process). Finished games flow back over a
single result :class:`multiprocessing.Queue`; a collector thread drains them into the learner's shared
replay buffer + live games counter — the *same* counters the G6h ticker, eval and snapshot already read.

**Risk-gate verdict (2026-06-22, RTX 5070, probes in gitignored ``Local/_g6i_*.py``):** 2 worker
processes give **~1.6× chess self-play** (in-proc actor 3.3–3.8 → 2 procs 5.3–5.9 games/s) at **GPU
49 → 94 %**, peak VRAM ~4.5 GB. Separate processes escape both the GIL contention and add a CPU core for
the pure-Python MCTS tree-glue (the real bottleneck). **2 is the Windows sweet spot** — 3 ≈ worse, 4
collapses (Windows WDDM has no MPS to share the GPU across many contexts).

**Decoupled net sync (ADR-008/019 preserved):** the learner publishes its freshly trained weights as a
shared file (an atomic ``torch.save`` of the CPU ``state_dict``) once per iteration; workers
``load_state_dict`` from it **between games** when its mtime changes. Workers never touch the learner's
GPU net — they read a snapshot, exactly like the in-process actor. A slightly-stale actor net is fine.

**Reproducibility:** multiple processes ⇒ **policy-level** reproducibility only (each worker seeded), like
the SuperSuit multi-agent path (ADR-038) and the G6h threaded actor (ADR-059).

``torch`` is imported lazily inside the worker / publish helpers so importing this module stays cheap.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import tempfile
import threading
import time
from collections import deque
from typing import Any

import numpy as np

# Brief window (s) to catch finished games after stop while the spawned workers exit, before
# force-terminating any straggler. self_play_rolling checks should_stop every ply (one short forward),
# so a worker leaves its loop within a ply of seeing the stop event; in-flight games are disposable (the
# buffer is already full), so this stays short — Stop responsiveness matters more than the last few games.
_DRAIN_AFTER_STOP_S = 1.5


def _publish_net_file(state_dict_cpu: dict[str, Any], path: str) -> None:
    """Atomically write a CPU ``state_dict`` to ``path`` (write a temp file, then ``os.replace``).

    The atomic replace means a worker reading the file always sees a complete net — never a half-written
    one — and its mtime jumps so the workers know to reload. Called by the learner once per iteration
    (the decoupled snapshot — NOT a per-round 160 MB Queue pickle)."""
    import torch

    tmp = f"{path}.tmp{os.getpid()}"
    torch.save(state_dict_cpu, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem


def _load_net_file(net: Any, path: str, device: str) -> None:
    """Load a published ``state_dict`` file into ``net`` (eval mode), on ``device``."""
    import torch

    sd = torch.load(path, map_location=device, weights_only=True)
    net.load_state_dict(sd)
    net.eval()


def _actor_worker(
    gym_id: str,
    build: dict[str, Any],
    search: dict[str, Any],
    seed: int,
    per_worker_parallel: int,
    net_path: str,
    result_q: Any,
    stop_event: Any,
    pause_event: Any,
) -> None:
    """One self-play worker process (module-level ⇒ picklable for Windows ``spawn``).

    Builds its OWN :class:`az_net.AZModel` on CUDA, loads the initial published net, then runs the
    unchanged :func:`az_batch.self_play_rolling`, putting each finished game's ``(examples, returns)`` on
    ``result_q``. Between games it reloads the learner's latest net (when the shared file's mtime changed)
    and idles while ``pause_event`` is set. Exits when ``stop_event`` is set (checked between plies via
    ``self_play_rolling``'s ``should_stop`` and again between games)."""
    import torch

    from app.services import az_batch, az_net, board_engine

    game = board_engine.load_game(gym_id)
    model = az_net.AZModel(
        game, channels=build["channels"], blocks=build["blocks"], device="cuda", norm=build["norm"]
    )
    _load_net_file(model.net, net_path, "cuda")
    last_mtime = os.path.getmtime(net_path)
    rng = np.random.default_rng(seed)
    try:
        for game_examples, returns in az_batch.self_play_rolling(
            model, per_worker_parallel, search["sims"], search["c_puct"], search["dir_alpha"],
            search["dir_frac"], search["temp_moves"], rng, gumbel=search["gumbel"],
            gumbel_considered=search["gumbel_considered"], max_game_plies=search["ply_cap"],
            should_stop=lambda: stop_event.is_set(),
        ):
            result_q.put((game_examples, returns))
            # Adopt the learner's latest net between games (decoupled snapshot — same as the in-proc actor).
            try:
                mtime = os.path.getmtime(net_path)
                if mtime > last_mtime:
                    _load_net_file(model.net, net_path, "cuda")
                    last_mtime = mtime
            except (OSError, FileNotFoundError):
                pass  # mid-replace or torn down — keep the current net
            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.05)
    except (EOFError, OSError, BrokenPipeError, KeyboardInterrupt):
        pass  # parent tore the queue down / process is being terminated
    finally:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()  # release this worker's CUDA context promptly


class ParallelActor:
    """Controls ``n_workers`` self-play worker processes + a collector thread feeding the shared buffer.

    A drop-in replacement for the trainer's in-process ``_actor`` thread when ``actor_processes > 1`` on
    CUDA: the learner publishes its net via :meth:`publish_net` each iteration, the workers self-play and
    return finished games, and the collector drains them into the *same* ``buffer`` / ``live_games``
    counter the rest of the trainer already reads. Lifecycle is Windows-hardened (spawn context; on stop,
    drain then ``cancel_join_thread`` *before* terminate — a terminated spawn-worker's Queue feeder
    otherwise deadlocks the parent, hit live in the probe)."""

    def __init__(
        self,
        *,
        gym_id: str,
        n_workers: int,
        per_worker_parallel: int,
        build: dict[str, Any],
        search: dict[str, Any],
        base_seed: int,
        initial_state_dict: dict[str, Any],
        buffer: deque[Any],
        buffer_lock: threading.Lock,
        live_games: list[int],
        live_plies: list[int],
        control: Any,
    ) -> None:
        self._gym_id = gym_id
        self._n_workers = n_workers
        self._per_worker_parallel = per_worker_parallel
        self._build = build
        self._search = search
        self._base_seed = base_seed
        self._buffer = buffer
        self._buffer_lock = buffer_lock
        self._live_games = live_games
        self._live_plies = live_plies
        self._control = control

        self._ctx = mp.get_context("spawn")
        self._result_q: Any = self._ctx.Queue()
        self._stop_event: Any = self._ctx.Event()
        self._pause_event: Any = self._ctx.Event()
        self._procs: list[Any] = []
        self._collector_stop = threading.Event()
        self._collector: threading.Thread | None = None
        # Decoupled net-sync file: written atomically by the learner, read by every worker between games.
        fd, self._net_path = tempfile.mkstemp(prefix="az_net_", suffix=".pt")
        os.close(fd)
        _publish_net_file(initial_state_dict, self._net_path)

    def publish_net(self, state_dict_cpu: dict[str, Any]) -> None:
        """Publish the learner's freshly trained weights for the workers (atomic file write)."""
        _publish_net_file(state_dict_cpu, self._net_path)

    def is_alive(self) -> bool:
        """True while at least one worker is still self-playing (the learner waits on this)."""
        return any(p.is_alive() for p in self._procs)

    def start(self) -> None:
        self._procs = [
            self._ctx.Process(
                target=_actor_worker,
                name=f"az-actor-{w}",
                args=(
                    self._gym_id, self._build, self._search, self._base_seed + 1 + w * 7919,
                    self._per_worker_parallel, self._net_path, self._result_q,
                    self._stop_event, self._pause_event,
                ),
                daemon=True,
            )
            for w in range(self._n_workers)
        ]
        for p in self._procs:
            p.start()
        self._collector = threading.Thread(target=self._collect, name="az-collector", daemon=True)
        self._collector.start()

    def _ingest(self, item: tuple[list[Any], Any]) -> None:
        game_examples, _returns = item
        with self._buffer_lock:
            self._buffer.extend(game_examples)
        self._live_games[0] += 1
        self._live_plies[0] += len(game_examples)  # each example is one ply → the env-steps axis (X1)

    def _collect(self) -> None:
        # Drain finished games into the shared buffer + counter, and mirror the trainer's pause/stop to the
        # workers' mp events. Runs until told to stop (the controller then does the bounded shutdown drain).
        while not self._collector_stop.is_set():
            if self._control.paused:
                self._pause_event.set()
            else:
                self._pause_event.clear()
            if self._control.stop_requested:
                self._stop_event.set()
            try:
                self._ingest(self._result_q.get(timeout=0.1))
            except queue.Empty:
                continue

    def stop(self) -> None:
        """Retire the workers + collector (Windows-hardened): stop the collector, signal the workers,
        drain remaining finished games, ``cancel_join_thread`` to unblock the parent, then terminate."""
        self._collector_stop.set()
        if self._collector is not None:
            self._collector.join(timeout=2.0)
        self._stop_event.set()
        # Single-threaded shutdown drain (the collector has exited, so no double-drain race): collect any
        # games the workers queued before observing stop, for a bounded window while they wind down.
        drain_until = time.monotonic() + _DRAIN_AFTER_STOP_S
        while time.monotonic() < drain_until and self.is_alive():
            with contextlib.suppress(queue.Empty):
                self._ingest(self._result_q.get(timeout=0.1))
        # A terminated spawn-worker's Queue feeder thread would otherwise deadlock the parent at join —
        # cancel it before terminating (the fix proven in the probe's run_workers). Terminate any worker
        # still alive after the grace window immediately (don't wait it out — its in-flight game is
        # disposable), then join briefly to reap the process.
        self._result_q.cancel_join_thread()
        for p in self._procs:
            if p.is_alive():
                p.terminate()
        for p in self._procs:
            p.join(timeout=2.0)
        with contextlib.suppress(OSError):
            os.remove(self._net_path)


def build_search(
    *,
    sims: int,
    c_puct: float,
    dir_alpha: float,
    dir_frac: float,
    temp_moves: int,
    gumbel: bool,
    gumbel_considered: int,
    ply_cap: int | None,
) -> dict[str, Any]:
    """Pack the self-play search hyperparameters a worker needs (must be picklable for spawn)."""
    return {
        "sims": sims, "c_puct": c_puct, "dir_alpha": dir_alpha, "dir_frac": dir_frac,
        "temp_moves": temp_moves, "gumbel": gumbel, "gumbel_considered": gumbel_considered,
        "ply_cap": ply_cap,
    }


def cpu_state_dict(net: Any) -> dict[str, Any]:
    """A detached CPU copy of ``net``'s ``state_dict`` (the decoupled snapshot the workers reload)."""
    return {k: v.detach().cpu() for k, v in net.state_dict().items()}


__all__ = [
    "ParallelActor",
    "build_search",
    "cpu_state_dict",
]
