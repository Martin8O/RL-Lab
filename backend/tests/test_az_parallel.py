"""G6i — parallel self-play across independent GPU actor processes (``az_parallel``, ADR-062).

Two cheap CPU-only checks of the decoupled net-sync file + the picklable helpers (these carry the
correctness the workers depend on), plus one guarded end-to-end smoke that actually spawns two CUDA
worker processes — skipped where there is no GPU (the parallel path is CUDA-only by design) and kept
short (a tiny net + tiny game) so it adds little to the gate.
"""

import time

import numpy as np
import pytest
from app.services import az_net, az_parallel, board_engine


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _tiny_model(game_id: str = "tic_tac_toe"):
    game = board_engine.load_game(game_id)
    model = az_net.AZModel(game, channels=8, blocks=1, device="cpu", norm="group")
    model.net.eval()
    return game, model


def test_publish_and_load_net_file_roundtrips(tmp_path) -> None:
    """The decoupled snapshot a worker reads is byte-faithful: publish a CPU state_dict atomically, load
    it into a fresh net of the same architecture, and the weights match exactly."""
    import torch

    _game, model = _tiny_model()
    path = str(tmp_path / "net.pt")
    az_parallel._publish_net_file(az_parallel.cpu_state_dict(model.net), path)

    _g2, model2 = _tiny_model()
    az_parallel._load_net_file(model2.net, path, "cpu")
    for a, b in zip(model.net.state_dict().values(), model2.net.state_dict().values(), strict=True):
        assert torch.equal(a.cpu(), b.cpu())


def test_build_search_packs_picklable_dict() -> None:
    """The search bundle handed to a spawned worker carries every knob ``self_play_rolling`` needs."""
    s = az_parallel.build_search(
        sims=16, c_puct=2.0, dir_alpha=0.3, dir_frac=0.25, temp_moves=6,
        gumbel=True, gumbel_considered=16, ply_cap=160,
    )
    assert s["sims"] == 16 and s["gumbel"] is True and s["ply_cap"] == 160
    import pickle

    assert pickle.loads(pickle.dumps(s)) == s  # picklable for Windows spawn


@pytest.mark.skipif(not _has_cuda(), reason="parallel self-play is CUDA-only (G6i)")
def test_parallel_actor_two_workers_smoke() -> None:
    """End-to-end: two spawned GPU workers self-play a tiny net into the shared buffer + counter, then
    stop cleanly. Asserts ≥1 finished game with valid ``ValuedExample``s and no live worker after stop."""
    import threading

    from app.services.train_control import TrainControl

    _game, model = _tiny_model()
    buffer: list = []  # a list works for the smoke (the trainer uses a bounded deque)
    actor = az_parallel.ParallelActor(
        gym_id="tic_tac_toe", n_workers=2, per_worker_parallel=2,
        build={"channels": model.channels, "blocks": model.blocks, "norm": model.norm},
        search=az_parallel.build_search(
            sims=8, c_puct=2.0, dir_alpha=1.0, dir_frac=0.25, temp_moves=2,
            gumbel=True, gumbel_considered=8, ply_cap=None,
        ),
        base_seed=0, initial_state_dict=az_parallel.cpu_state_dict(model.net),
        buffer=buffer, buffer_lock=threading.Lock(), live_games=[0], live_plies=[0],
        control=TrainControl(),
    )
    actor.start()
    try:
        deadline = time.monotonic() + 90.0  # generous: spawn + CUDA init per worker, then play
        while actor._live_games[0] < 1 and time.monotonic() < deadline:
            time.sleep(0.2)
        assert actor._live_games[0] >= 1, "no self-play game arrived from the worker processes"
        assert actor._live_plies[0] >= actor._live_games[0]  # ≥1 ply per game — the env-steps axis (X1)
    finally:
        actor.stop()

    assert not actor.is_alive()  # clean stop: no worker left running
    obs, target, value = buffer[0]
    assert obs.shape == (model.planes, model.rows, model.cols)
    assert abs(float(np.asarray(target).sum()) - 1.0) < 1e-4  # a simplex policy target
    assert all(-1.0 <= float(v) <= 1.0 for _o, _t, v in buffer)  # outcome-valued in [−1, 1]
