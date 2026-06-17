"""G6a — OpenSpiel board-game subsystem (ADR-050): the registry row + routing, the game-agnostic
board engine, the play-session board branch, training gating, and the W/D/L contract.

A board game is the 7th seam: a 2-player, turn-based, perfect-info, zero-sum game with legal-move
masking and self-play — OpenSpiel's pyspiel.State API, NOT a gym.Env — so it is routed via
is_board_game (like is_multi_agent) and never goes through app.envs.factory.make_env. G6a ships ONE
game (Tic-Tac-Toe) human-playable vs a training-free MCTS opponent; the engine/session/contract are
game-agnostic, which a Connect Four test exercises through the same functions. The neural self-play
trainer is G6b, so training stays gated here.
"""

from app.envs.registry import get_env
from app.main import app
from app.schemas.play import PlayConfig
from app.services import board_engine
from app.services.connection_manager import manager
from app.services.play_session import PlaySession
from fastapi.testclient import TestClient

client = TestClient(app)


# -- registry + routing -----------------------------------------------------


def test_tictactoe_registered() -> None:
    spec = get_env("tictactoe")
    assert spec is not None, "tictactoe not registered"
    assert spec.gym_id == "tic_tac_toe"  # the OpenSpiel short name
    assert spec.family == "board"
    assert spec.action_space == "discrete"
    assert spec.supported_algos == ["ppo"]  # self-play surfaced as ppo (simple_tag precedent)
    assert spec.human_playable is True
    assert spec.competitive is True
    assert spec.turn_based is True
    assert spec.hw_requirement == "cpu"  # MCTS needs no GPU
    assert spec.train_implemented is False  # neural self-play trainer = G6b
    assert spec.min_score == -1.0 and spec.solved_score == 1.0  # zero-sum loss / win


def test_is_board_game_flag() -> None:
    assert board_engine.is_board_game(get_env("tictactoe")) is True
    assert board_engine.is_board_game(get_env("cartpole")) is False
    assert board_engine.is_board_game(get_env("mpe_tag")) is False  # competitive but petting_zoo
    assert board_engine.is_board_game(None) is False


def test_tictactoe_listed_in_envs_api() -> None:
    rows = client.get("/api/envs").json()
    board = [r for r in rows if r["id"] == "tictactoe"]
    assert len(board) == 1 and board[0]["family"] == "board"


# -- the game-agnostic board engine -----------------------------------------


def test_board_payload_initial_and_after_move() -> None:
    game = board_engine.load_game("tic_tac_toe")
    state = game.new_initial_state()
    p = board_engine.board_payload(state, None)
    assert p["rows"] == 3 and p["cols"] == 3 and len(p["cells"]) == 9
    assert p["cells"] == ["."] * 9
    assert p["legal_actions"] == list(range(9))
    assert p["current_player"] == 0 and p["last_action"] is None
    assert p["is_terminal"] is False and p["winner"] is None

    state.apply_action(4)  # centre
    p = board_engine.board_payload(state, 4)
    assert p["cells"][4] != "."  # the centre cell is now occupied
    assert 4 not in p["legal_actions"]  # no longer legal
    assert p["current_player"] == 1 and p["last_action"] == 4


def test_mcts_opponent_picks_legal_moves_and_is_seed_reproducible() -> None:
    game = board_engine.load_game("tic_tac_toe")
    state = game.new_initial_state()
    a1 = board_engine.MctsOpponent(game, max_simulations=40, seed=7).step(state)
    a2 = board_engine.MctsOpponent(game, max_simulations=40, seed=7).step(state)
    assert a1 in state.legal_actions()
    assert a1 == a2  # same seed → same move (reproducible demo)


def test_strong_mcts_pair_draws_tic_tac_toe() -> None:
    """The TTT correctness invariant: with enough look-ahead on both sides, every game is a draw."""
    game = board_engine.load_game("tic_tac_toe")
    sims = board_engine.STRENGTH_SIMS["hard"]
    for trial in range(3):
        state = game.new_initial_state()
        bots = [board_engine.MctsOpponent(game, sims, seed=trial * 2 + p) for p in range(2)]
        while not state.is_terminal():
            state.apply_action(bots[state.current_player()].step(state))
        assert board_engine.board_payload(state, None)["winner"] is None  # a draw
        assert board_engine.outcome(state, 0) == (0.0, "draw")


def test_outcome_labels() -> None:
    game = board_engine.load_game("tic_tac_toe")
    # X (player 0) plays a winning column 0,1,2 vs O wasting the bottom row.
    state = game.new_initial_state()
    for move in (0, 3, 1, 4, 2):  # X: 0,1,2 (top row); O: 3,4
        state.apply_action(move)
    assert state.is_terminal()
    assert board_engine.outcome(state, 0) == (1.0, "win")
    assert board_engine.outcome(state, 1) == (-1.0, "loss")
    assert board_engine.board_payload(state, 2)["winner"] == 0


def test_engine_is_game_agnostic_connect_four() -> None:
    """A SECOND OpenSpiel game flows through the SAME engine functions — proves no TTT hardcoding."""
    game = board_engine.load_game("connect_four")
    state = game.new_initial_state()
    p = board_engine.board_payload(state, None)
    assert p["rows"] == 6 and p["cols"] == 7 and len(p["cells"]) == 42
    assert p["legal_actions"] == list(range(7))  # seven columns
    move = board_engine.MctsOpponent(game, max_simulations=20, seed=0).step(state)
    assert move in state.legal_actions()


# -- play-session board branch (the watch loop, end to end) ------------------


def test_board_watch_session_plays_to_a_draw() -> None:
    """An AI-vs-AI watch on Tic-Tac-Toe runs _run_board → emits board play_frames → finalises with a
    win/draw/loss outcome (a hard pair draws). Exercises the whole board play path without WS by
    capturing the broadcast frames directly (no event loop bound → the real broadcast would no-op)."""
    sess = PlaySession(manager)
    frames: list[dict] = []
    sess._broadcast = lambda frame: frames.append(frame)  # type: ignore[method-assign]

    sess.start(PlayConfig(env_id="tictactoe", mode="ai", ai_strength="hard", speed=20.0, seed=1))
    sess.join(timeout=30.0)

    status = sess.status()
    assert status.state == "finished"
    assert status.result is not None
    assert status.result.rating is None  # board games carry an outcome, not a continuous rating
    assert status.result.outcome == "draw"  # hard-vs-hard TTT is always a draw

    play_frames = [f for f in frames if f.get("type") == "play_frame"]
    assert play_frames, "no board play_frame was emitted"
    assert all("board" in f for f in play_frames)
    assert len(play_frames[-1]["board"]["cells"]) == 9
    assert play_frames[-1]["board"]["is_terminal"] is True
    results = [f for f in frames if f.get("type") == "play_result"]
    assert results and results[-1]["outcome"] == "draw"


def test_board_human_vs_mcts_reaches_a_valid_outcome() -> None:
    """The human-vs-MCTS *logic*: a human (here, always the first legal cell) vs a hard MCTS reaches a
    terminal state with a valid {win, draw, loss} outcome — and a perfect MCTS never loses, so the
    human side can only draw or lose, never win."""
    game = board_engine.load_game("tic_tac_toe")
    human_side = 0
    ai = board_engine.MctsOpponent(game, board_engine.STRENGTH_SIMS["hard"], seed=3)
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.current_player() == human_side:
            state.apply_action(state.legal_actions()[0])  # the human's (naive) click
        else:
            state.apply_action(ai.step(state))
    _, label = board_engine.outcome(state, human_side)
    assert label in {"win", "draw", "loss"}
    assert label != "win"  # a perfect opponent never lets a naive human win


# -- training gate ----------------------------------------------------------


def test_board_training_is_gated() -> None:
    """Starting training for a board game is rejected (the neural self-play trainer is G6b); the env
    stays human-playable via Play. train_implemented=False is enforced by the manager backstop."""
    resp = client.post(
        "/api/train/start",
        json={
            "env_id": "tictactoe", "algo": "ppo", "seed": 1, "total_timesteps": 1000,
            "hyperparams": {
                "learning_rate": 3e-4, "gamma": 0.99, "clip_range": 0.2, "ent_coef": 0.0,
                "n_steps": 128, "batch_size": 64, "n_hidden_layers": 2,
                "neurons_per_layer": 64, "activation": "tanh",
            },
        },
    )
    assert resp.status_code == 400
