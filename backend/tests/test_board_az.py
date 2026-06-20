"""G6f — AlphaZero-lite board trainer (CNN policy+value + neural-guided MCTS self-play, ADR-055).

Covers the engine pieces (the CNN over observation planes, a self-play game producing visit-count
targets, a net update, the neural-MCTS move fn, and the decoupled CPU snapshot), plus a short
end-to-end ``train_az`` run (metrics emitted, an AlphaZero ``board.zip`` packed + reloadable as a legal
neural-MCTS opponent). CPU-only and tiny (a few sims / a couple of games) so it runs in a few seconds.
"""

import numpy as np
import pyspiel
from app.schemas.training import AlphaZeroHyperparams, TrainConfig
from app.services import az_net, board_engine
from app.services.train_control import TrainControl
from app.services.trainer_az import train_az


def _tiny() -> AlphaZeroHyperparams:
    """A minimal AZ config — tiny net, few sims, 2 games/iter — so the test is a couple of seconds.

    Self-play defaults to Gumbel (G6h); a low ``gumbel_sims`` keeps it quick. ``simulations`` is the
    PUCT-fallback budget exercised by :func:`test_train_az_puct_fallback_path`."""
    return AlphaZeroHyperparams(
        iterations=1, games_per_iter=2, gumbel_sims=6, gumbel_considered=6, simulations=8,
        eval_simulations=6, play_simulations=8, channels=8, blocks=1, batch_size=16, buffer_size=500,
    )


def test_az_model_reshapes_observation_planes() -> None:
    """The CNN reads the OpenSpiel observation tensor as (planes, rows, cols) — game-agnostic geometry."""
    game = board_engine.load_game("connect_four")
    model = az_net.AZModel(game, channels=8, blocks=1, device="cpu")
    assert (model.planes, model.rows, model.cols) == (3, 6, 7)
    assert model.n_actions == 7


def test_self_play_game_yields_visit_count_targets() -> None:
    """One self-play game returns per-move (planes, policy-target-over-all-actions, player) + returns."""
    game = board_engine.load_game("tic_tac_toe")
    model = az_net.AZModel(game, channels=8, blocks=1, device="cpu")
    rng = np.random.default_rng(0)
    examples, returns = az_net.self_play_game(
        model, sims=8, c_puct=2.0, dirichlet_alpha=1.0, dirichlet_frac=0.25,
        temp_moves=2, rng=rng, seed=0,
    )
    assert examples and len(returns) == 2
    obs, target, player = examples[0]
    assert obs.shape == (3, 3, 3)  # planes × rows × cols
    assert target.shape == (game.num_distinct_actions(),)
    assert abs(float(target.sum()) - 1.0) < 1e-5  # a probability distribution over moves
    assert player in (0, 1)


def test_train_on_buffer_reduces_loss_and_updates_net() -> None:
    """A few SGD steps on a self-play buffer return a finite loss and change the net's weights."""
    import torch

    game = board_engine.load_game("tic_tac_toe")
    model = az_net.AZModel(game, channels=8, blocks=1, device="cpu")
    rng = np.random.default_rng(0)
    buffer = []
    for g in range(3):
        ex, ret = az_net.self_play_game(model, 8, 2.0, 1.0, 0.25, 2, rng, seed=g)
        buffer += [(o, t, ret[p]) for o, t, p in ex]
    before = model.net.policy.weight.detach().clone()
    opt = torch.optim.Adam(model.net.parameters(), lr=1e-3)
    loss = az_net.train_on_buffer(model, opt, buffer, batch_size=16, steps=5, rng=rng)
    assert np.isfinite(loss)
    assert not torch.equal(before, model.net.policy.weight.detach())  # the net actually trained


def test_material_score_balanced_start_and_sign_convention() -> None:
    """The AZ-score-0.0 fix (G6h): a capped chess eval game is scored by material balance, not a flat 0.

    The opening is balanced (0 either side); the pure ``_material_balance`` helper reads positive for the
    side that's up material and negative for the other (uppercase FEN = white = player 0), clamped to <1
    (±1 stays reserved for a real checkmate). ``unresolved_value_fn`` wires it to chess only."""
    game = board_engine.load_game("chess")
    state = game.new_initial_state()
    assert board_engine.material_score(state, 0) == 0.0  # the starting position is materially balanced
    assert board_engine.material_score(state, 1) == 0.0

    up_white = ["Q", "P", "K", "k"]  # white has an extra queen + pawn vs a lone black king
    assert board_engine._material_balance(up_white, 0) > 0  # good for white (player 0)
    assert board_engine._material_balance(up_white, 1) < 0  # bad for black (player 1)
    assert board_engine._material_balance(["Q", "Q", "Q", "K", "k"], 0) == 0.99  # clamped below a real win

    assert board_engine.unresolved_value_fn("chess") is board_engine.material_score
    assert board_engine.unresolved_value_fn("connect_four") is None  # bounded boards never hit a cap


def test_az_move_fn_returns_legal_move() -> None:
    """The neural-MCTS move fn (Play/eval inference) always returns a legal action."""
    game = board_engine.load_game("connect_four")
    model = az_net.AZModel(game, channels=8, blocks=1, device="cpu")
    move = az_net.az_move_fn(model, sims=8, seed=0)
    state = game.new_initial_state()
    assert int(move(state)) in state.legal_actions()


def test_train_az_emits_metrics_and_packs_reloadable_checkpoint() -> None:
    """A short end-to-end run emits metrics/progress, publishes a preview policy, and packs an
    AlphaZero ``board.zip`` that reloads as a legal neural-MCTS opponent (the Play/Watch load path)."""
    seen = {"metrics": 0, "progress": 0, "policy": 0, "snap": None}
    cfg = TrainConfig(env_id="connect_four", algo="alphazero", seed=3, total_timesteps=2, alphazero=_tiny())
    state = train_az(
        cfg, "connect_four", TrainControl(),
        lambda m: seen.__setitem__("metrics", seen["metrics"] + 1),
        lambda p: seen.__setitem__("progress", seen["progress"] + 1),
        lambda f: seen.__setitem__("policy", seen["policy"] + 1),
        lambda a: seen.__setitem__("snap", a),
        None,
    )
    assert state == "finished"
    # metrics: one per learner round (iterations=1 → ≥1); progress: ticker bookends a run with ≥2 frames;
    # policy: published initially + per round.
    assert seen["metrics"] >= 1 and seen["progress"] >= 2 and seen["policy"] >= 1
    snap = seen["snap"]
    assert snap is not None and snap.algo == "alphazero" and snap.artifact_name == "board.zip"

    # The packed blob reloads (Play-vs-net path) → a legal neural-MCTS move.
    game = board_engine.load_game("connect_four")
    model, games_played = az_net.build_model_from_blob(snap.blob, game, device="cpu")
    # The background actor plays continuously through the round's train, so it produces AT LEAST the
    # round's games_per_iter budget (often more — it doesn't stop while the learner trains, G6h/ADR-059).
    assert games_played >= cfg.alphazero.games_per_iter
    move = az_net.az_move_fn(model, sims=8, seed=None)
    assert int(move(game.new_initial_state())) in game.new_initial_state().legal_actions()
    # The raw-policy predict (the cosmetic preview path) is also legal.
    predict = az_net.load_az_predict(snap.blob)
    st = game.new_initial_state()
    mask = np.zeros(game.num_distinct_actions(), dtype=bool)
    for a in st.legal_actions():
        mask[a] = True
    assert int(predict(np.asarray(st.observation_tensor(st.current_player()), dtype=np.float32), mask)) in st.legal_actions()


def test_train_az_score_placeholder_then_real_value() -> None:
    """The displayed score reads None (→ frontend "—", not a fake 0.0 = 50%) until the first eval lands
    (G6h score-0.0 follow-up); the metrics frame is emitted only after an eval, so it is always a valid ±1
    score. Captured on connect_four (fast) — the placeholder + recent-eval smoothing are game-agnostic."""
    prog: list[float | None] = []
    metrics: list[float | None] = []
    cfg = TrainConfig(env_id="connect_four", algo="alphazero", seed=3, total_timesteps=2, alphazero=_tiny())
    train_az(
        cfg, "connect_four", TrainControl(),
        lambda m: metrics.append(m.ep_rew_mean),
        lambda p: prog.append(p.ep_rew_mean),
    )
    assert prog[0] is None  # the immediate t=0 progress frame — before any game is scored (placeholder)
    assert metrics and all(r is not None and -1.0 <= r <= 1.0 for r in metrics)  # emit() runs post-eval


def test_train_az_puct_fallback_path() -> None:
    """``use_gumbel=False`` keeps the original PUCT self-play search working end-to-end (G6h kept it as a
    fallback). A short run still emits metrics and finishes — the non-default trainer branch is covered."""
    hp = AlphaZeroHyperparams(
        iterations=1, games_per_iter=2, simulations=6, eval_simulations=6, play_simulations=8,
        channels=8, blocks=1, batch_size=16, buffer_size=500, use_gumbel=False,
    )
    cfg = TrainConfig(env_id="connect_four", algo="alphazero", seed=3, total_timesteps=2, alphazero=hp)
    seen = {"metrics": 0}
    state = train_az(
        cfg, "connect_four", TrainControl(),
        lambda m: seen.__setitem__("metrics", seen["metrics"] + 1),
    )
    assert state == "finished" and seen["metrics"] >= 1


def test_az_supported_on_small_board_games() -> None:
    """alphazero is opted in via supported_algos on the small boards where it's validated (TTT, Connect
    Four), not on the big 8×8 games (Othello/Breakthrough — too noisy at a tolerable budget) or others."""
    from app.envs.registry import get_env

    for board in ("tictactoe", "connect_four"):
        assert "alphazero" in get_env(board).supported_algos, board
    for not_az in ("othello", "breakthrough", "cartpole"):
        assert "alphazero" not in get_env(not_az).supported_algos, not_az


def test_az_only_for_three_plane_board_games() -> None:
    """AZModel rejects a game whose observation tensor isn't a (planes, rows, cols) stack."""
    import pytest

    # kuhn_poker has a flat 1-D information-state/obs tensor → not a CNN board.
    game = pyspiel.load_game("kuhn_poker")
    with pytest.raises(ValueError, match="3D observation"):
        az_net.AZModel(game, channels=8, blocks=1, device="cpu")
