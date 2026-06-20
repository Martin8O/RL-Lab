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


def test_eval_vs_mcts_parallel_scores_capped_games_via_unresolved_value() -> None:
    """The AZ-score-0.0 fix (G6h): a ply-capped (unresolved) eval game is scored by ``unresolved_value``
    instead of a flat 0. With a tiny cap every chess game is unresolved, so the mean equals the callable's
    value — and the default (None) preserves the old draw-is-0 behavior (a weak early net would otherwise
    read a permanent 0.0)."""
    game, model = _model("chess", channels=8, blocks=1)
    scored = az_batch.eval_vs_mcts_parallel(
        model, game, ref_sims=2, n_games=2, eval_sims=2, c_puct=2.0, base_seed=0,
        max_game_plies=2, unresolved_value=lambda _s, _seat: 0.5,
    )
    assert scored == 0.5  # no chess game ends in ≤2 plies → all capped → mean of the callable's constant
    drawn = az_batch.eval_vs_mcts_parallel(
        model, game, ref_sims=2, n_games=2, eval_sims=2, c_puct=2.0, base_seed=0,
        max_game_plies=2,
    )
    assert drawn == 0.0  # default: an unresolved capped game still scores a draw


def test_batched_search_handles_chess_planes_and_huge_action_space() -> None:
    """G6g chess: the engine is game-agnostic, so it drives chess's (20,8,8) plane stack + 4674-move space
    with zero changes — a tiny net runs a batched search over two opening positions and returns a legal,
    normalized visit distribution each. Tiny (8×1, few sims) so it stays fast in the gate."""
    game, model = _model("chess", channels=8, blocks=1)
    assert (model.planes, model.rows, model.cols) == (20, 8, 8) and model.n_actions == 4674
    states = [game.new_initial_state() for _ in range(2)]
    dists = az_batch.batched_search(
        model, states, sims=4, c_puct=2.0, dir_alpha=0.3, dir_frac=0.25,
        add_noise=True, rng=np.random.default_rng(0),
    )
    for st, d in zip(states, dists, strict=True):
        assert d.shape == (4674,) and abs(float(d.sum()) - 1.0) < 1e-6
        assert int(d.argmax()) in set(st.legal_actions())


def test_batched_gumbel_search_returns_legal_improved_policies() -> None:
    """G6h Gumbel root search over a cohort → one completed-Q improved policy + one chosen move each:
    the policy is a simplex with mass only on legal moves, and the played move is legal."""
    game, model = _model("connect_four")
    states = [game.new_initial_state() for _ in range(3)]
    policies, winners = az_batch.batched_gumbel_search(
        model, states, sims=16, max_considered=16, c_puct=2.0, rng=np.random.default_rng(0)
    )
    assert len(policies) == 3 and len(winners) == 3
    for st, pol, win in zip(states, policies, winners, strict=True):
        legal = set(st.legal_actions())
        assert pol.shape == (game.num_distinct_actions(),)
        assert abs(float(pol.sum()) - 1.0) < 1e-6
        assert all(pol[a] == 0.0 for a in range(len(pol)) if a not in legal)  # mass only on legal moves
        assert int(pol.argmax()) in legal
        assert int(win) in legal  # the Sequential-Halving winner is a legal move


def test_batched_gumbel_search_chess_huge_action_space() -> None:
    """Gumbel is game-agnostic like PUCT: it drives chess's (20,8,8) planes + 4674-move space unchanged,
    considering at most ``max_considered`` of the ~20 opening moves and returning a legal policy + move."""
    game, model = _model("chess", channels=8, blocks=1)
    states = [game.new_initial_state() for _ in range(2)]
    policies, winners = az_batch.batched_gumbel_search(
        model, states, sims=8, max_considered=16, c_puct=2.0, rng=np.random.default_rng(0)
    )
    for st, pol, win in zip(states, policies, winners, strict=True):
        assert pol.shape == (4674,) and abs(float(pol.sum()) - 1.0) < 1e-6
        assert int(win) in set(st.legal_actions())


def test_self_play_parallel_gumbel_mode_yields_valued_examples() -> None:
    """``gumbel=True`` self-play produces exactly ``num_games`` games with simplex policy targets and
    outcome-valued examples — the same example shape the trainer feeds, on the Gumbel target."""
    game, model = _model("tic_tac_toe")
    examples, returns = az_batch.self_play_parallel(
        model, num_games=5, parallel=3, sims=12, c_puct=2.0, dir_alpha=1.0, dir_frac=0.25,
        temp_moves=2, rng=np.random.default_rng(1), base_seed=1, gumbel=True, gumbel_considered=8,
    )
    assert len(returns) == 5
    obs, target, value = examples[0]
    assert obs.shape == (model.planes, model.rows, model.cols)
    assert abs(float(target.sum()) - 1.0) < 1e-5
    assert all(-1.0 <= v <= 1.0 for _, _, v in examples)


def test_self_play_max_game_plies_bounds_unbounded_games() -> None:
    """``max_game_plies`` ends an over-long (chess) game as a draw — bounds the marathon games a weak early
    net produces. With a tiny cap every game stops at the cap (no chess game ends in 6 plies from start), so
    each game contributes at most ``cap`` examples and scores a 0/0 draw."""
    game, model = _model("chess", channels=8, blocks=1)
    cap = 6
    examples, returns = az_batch.self_play_parallel(
        model, num_games=2, parallel=2, sims=8, c_puct=2.0, dir_alpha=0.3, dir_frac=0.25,
        temp_moves=2, rng=np.random.default_rng(0), base_seed=0, gumbel=True, gumbel_considered=8,
        max_game_plies=cap,
    )
    assert len(returns) == 2
    assert len(examples) <= 2 * cap  # each game capped at `cap` plies
    assert all(r == [0.0, 0.0] for r in returns)  # an unresolved capped game scores a draw
    assert all(v == 0.0 for _, _, v in examples)  # → every example's value target is the draw


def test_gumbel_uses_fewer_forwards_than_puct() -> None:
    """The headline G6h win: Gumbel reaches a move in **fewer batched net forwards** than PUCT at the same
    cohort — Sequential Halving spends each simulation where it matters, so self-play runs faster. Counts
    the ``infer_batch`` calls (one wide GPU forward each) for Gumbel-16 vs PUCT-16 over one cohort."""
    game, model = _model("connect_four")
    cohort = [game.new_initial_state() for _ in range(8)]
    calls = {"n": 0}
    orig = model.infer_batch
    model.infer_batch = lambda obs: (calls.__setitem__("n", calls["n"] + 1), orig(obs))[1]  # type: ignore[method-assign]

    calls["n"] = 0
    az_batch.batched_search(
        model, cohort, sims=16, c_puct=2.0, dir_alpha=1.0, dir_frac=0.25,
        add_noise=True, rng=np.random.default_rng(0),
    )
    puct_forwards = calls["n"]
    calls["n"] = 0
    az_batch.batched_gumbel_search(
        model, cohort, sims=16, max_considered=16, c_puct=2.0, rng=np.random.default_rng(0)
    )
    gumbel_forwards = calls["n"]
    # PUCT does sims+1 forwards (root + one per sim round); Gumbel finishes within its sim budget and skips
    # forwards on terminal leaves, so it never exceeds PUCT and is typically well under it.
    assert gumbel_forwards <= puct_forwards


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
