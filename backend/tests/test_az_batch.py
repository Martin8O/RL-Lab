"""G6g — batched-GPU AlphaZero engine (parallel self-play + a batch-aware MCTS, ``az_batch``).

Unit coverage of the batch-aware MCTS primitive and its two drivers (parallel self-play + batched eval),
plus the engine invariants the GPU build relies on. CPU-only and tiny so it runs fast in the gate — the
batching is correct regardless of device (the GPU only makes the per-step forward wider).
"""

import numpy as np
from app.services import az_batch, az_net, board_engine


def _model(game_id: str = "connect_four", channels: int = 8, blocks: int = 1):
    game = board_engine.load_game(game_id)
    model = az_net.AZModel(game, channels=channels, blocks=blocks, device="cpu", norm="group")
    model.net.eval()
    return game, model


def test_batched_search_returns_legal_visit_distributions() -> None:
    """A cohort of independent positions → one normalized visit distribution each: mass only on legal
    moves (illegal actions stay 0), and the argmax is a legal action."""
    game, model = _model()
    states = [game.new_initial_state() for _ in range(3)]
    dists = az_batch.batched_search(
        model, states, sims=16, c_puct=2.0, dir_alpha=1.0, dir_frac=0.25,
        add_noise=True, rng=np.random.default_rng(0),
    )
    assert len(dists) == 3
    for st, d in zip(states, dists, strict=True):
        assert d.shape == (game.num_distinct_actions(),)
        assert abs(float(d.sum()) - 1.0) < 1e-6
        legal = set(st.legal_actions())
        assert all(d[a] == 0.0 for a in range(len(d)) if a not in legal)  # no mass on illegal moves
        assert int(d.argmax()) in legal


def test_self_play_parallel_yields_valued_examples_and_exact_game_count() -> None:
    """Parallel self-play produces exactly ``num_games`` games (across cohorts when parallel < num_games)
    and per-move (planes, simplex policy target, value ∈ [−1, 1]) examples paired with the outcome."""
    game, model = _model("tic_tac_toe")
    examples, returns = az_batch.self_play_parallel(
        model, num_games=5, parallel=2, sims=12, c_puct=2.0, dir_alpha=1.0, dir_frac=0.25,
        temp_moves=2, rng=np.random.default_rng(1), base_seed=1,
    )
    assert len(returns) == 5  # exact count even though parallel(2) < num_games(5) → multiple cohorts
    obs, target, value = examples[0]
    assert obs.shape == (model.planes, model.rows, model.cols)
    assert abs(float(target.sum()) - 1.0) < 1e-5
    assert all(-1.0 <= v <= 1.0 for _, _, v in examples)


def test_self_play_parallel_on_game_done_and_stop() -> None:
    """``on_game_done`` fires once per finished game (cumulative count); ``should_stop`` aborts promptly."""
    _game, model = _model("tic_tac_toe")
    seen: list[int] = []
    az_batch.self_play_parallel(
        model, 4, 4, 8, 2.0, 1.0, 0.25, 2, np.random.default_rng(2), base_seed=2,
        on_game_done=seen.append,
    )
    assert seen == [1, 2, 3, 4]  # cumulative finished-game count, one call per game

    examples, returns = az_batch.self_play_parallel(
        model, 4, 4, 8, 2.0, 1.0, 0.25, 2, np.random.default_rng(3), base_seed=3,
        should_stop=lambda: True,
    )
    assert returns == [] and examples == []  # stop before any work → nothing generated


def test_eval_vs_mcts_parallel_is_bounded() -> None:
    """The batched eval (net's moves batched across games vs a reference MCTS) returns a mean game
    result in [−1, 1] — the same skill-curve unit as ``board_engine.eval_vs_mcts``."""
    game, model = _model("connect_four")
    score = az_batch.eval_vs_mcts_parallel(
        model, game, ref_sims=10, n_games=4, eval_sims=8, c_puct=2.0, base_seed=0,
    )
    assert -1.0 <= score <= 1.0


def test_norm_checkpoint_roundtrips_through_blob() -> None:
    """A GroupNorm net (the G6g default) serializes + reloads byte-faithfully — the ``norm`` field in the
    blob rebuilds the right architecture, so Save/Load + Play-vs-net work for the bigger batched net."""
    game, model = _model("connect_four", channels=16, blocks=2)
    blob = model.state_blob()
    rebuilt, _games = az_net.build_model_from_blob(blob, game, device="cpu")
    assert rebuilt.norm == "group"
    # Same forward on a zero board (weights round-tripped exactly).
    obs = np.zeros((1, model.planes, model.rows, model.cols), dtype=np.float32)
    a_logits, a_val = model.infer_batch(obs)
    b_logits, b_val = rebuilt.infer_batch(obs)
    assert np.allclose(a_logits, b_logits) and np.allclose(a_val, b_val)
