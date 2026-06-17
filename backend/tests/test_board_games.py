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
    assert spec.hw_requirement == "cpu"  # MCTS + the MaskablePPO board trainer both run on CPU
    assert spec.train_implemented is True  # neural trainer landed in G6b (MaskablePPO vs MCTS teacher)
    assert spec.min_score == -1.0 and spec.solved_score == 1.0  # zero-sum loss / win


def test_is_board_game_flag() -> None:
    assert board_engine.is_board_game(get_env("tictactoe")) is True
    assert board_engine.is_board_game(get_env("connect_four")) is True  # G6c — the 2nd board game
    assert board_engine.is_board_game(get_env("cartpole")) is False
    assert board_engine.is_board_game(get_env("mpe_tag")) is False  # competitive but petting_zoo
    assert board_engine.is_board_game(None) is False


def test_connect_four_registered() -> None:
    """G6c — the SECOND board game ships as data + a renderer glyph map (no engine code): same family /
    routing / trainability as Tic-Tac-Toe, only bigger (7 column actions over a 6×7 board)."""
    spec = get_env("connect_four")
    assert spec is not None, "connect_four not registered"
    assert spec.gym_id == "connect_four"  # the OpenSpiel short name (resolved by board_engine)
    assert spec.family == "board" and spec.action_space == "discrete"
    assert spec.supported_algos == ["ppo"]  # routed to the board trainer by is_board_game
    assert spec.human_playable is True and spec.competitive is True and spec.turn_based is True
    assert spec.hw_requirement == "cpu" and spec.train_implemented is True  # CPU, trains like TTT
    assert spec.min_score == -1.0 and spec.solved_score == 1.0  # same zero-sum chart scale


def test_connect_four_listed_in_envs_api() -> None:
    rows = client.get("/api/envs").json()
    board = [r for r in rows if r["id"] == "connect_four"]
    assert len(board) == 1 and board[0]["family"] == "board"


def test_board_profile_is_game_tuned() -> None:
    """G6c — the per-game training/eval profile: Tic-Tac-Toe trains/scores vs the medium MCTS, while the
    far bigger Connect Four uses the beatable easy MCTS so its honest skill curve climbs (not flat at the
    loss floor). Unlisted games fall back to the Tic-Tac-Toe profile."""
    assert board_engine.board_profile("tic_tac_toe").eval_strength == "medium"
    assert board_engine.board_profile("connect_four").eval_strength == "easy"
    assert board_engine.board_profile("connect_four").teacher_end == "easy"
    assert board_engine.board_profile("some_unlisted_game") == board_engine.board_profile("tic_tac_toe")


def test_tictactoe_listed_in_envs_api() -> None:
    rows = client.get("/api/envs").json()
    board = [r for r in rows if r["id"] == "tictactoe"]
    assert len(board) == 1 and board[0]["family"] == "board"


# -- Othello (G6d) — the 3rd board game: decorated board string + a pass move -----------------------


def test_othello_registered() -> None:
    """G6d — the THIRD board game ships as data only (the engine resolves it from gym_id): same family /
    routing / trainability as the others, but bigger (65 actions = 64 cells + a pass) and 8×8."""
    spec = get_env("othello")
    assert spec is not None, "othello not registered"
    assert spec.gym_id == "othello"  # the OpenSpiel short name (resolved by board_engine)
    assert spec.family == "board" and spec.action_space == "discrete"
    assert spec.supported_algos == ["ppo"]  # routed to the board trainer by is_board_game
    assert spec.human_playable is True and spec.competitive is True and spec.turn_based is True
    assert spec.hw_requirement == "cpu" and spec.train_implemented is True  # CPU, trains like the others
    assert spec.min_score == -1.0 and spec.solved_score == 1.0  # same zero-sum chart scale


def test_othello_listed_in_envs_api() -> None:
    rows = client.get("/api/envs").json()
    board = [r for r in rows if r["id"] == "othello"]
    assert len(board) == 1 and board[0]["family"] == "board"


def test_othello_profile_uses_novice_teacher() -> None:
    """G6d — Othello is much bigger than Connect Four, so it trains against the near-random NOVICE
    teacher (ramping to easy) and is scored vs easy, so its honest curve climbs instead of sitting at
    the loss floor. ``novice`` is a sub-easy teacher tier, weaker than the weakest play difficulty."""
    prof = board_engine.board_profile("othello")
    assert prof.eval_strength == "easy"
    assert prof.teacher_start == "novice" and prof.teacher_end == "easy"
    assert board_engine.STRENGTH_SIMS["novice"] < board_engine.STRENGTH_SIMS["easy"]


def test_othello_decorated_board_parses_to_clean_8x8() -> None:
    """Othello's ``str(state)`` is DECORATED (a header, ``a b c`` / ``1 2 3`` labels, ``-`` for empty,
    spaces between cells), unlike TTT/Connect Four's clean grid — yet ``board_payload`` must still yield
    a clean 8×8 grid of only ``.``/``x``/``o`` with no stray label chars leaking in (G6d)."""
    game = board_engine.load_game("othello")
    state = game.new_initial_state()
    p = board_engine.board_payload(state, None)
    assert p["rows"] == 8 and p["cols"] == 8 and len(p["cells"]) == 64
    assert set(p["cells"]) <= {".", "x", "o"}  # decoration (labels, headers) must not leak through
    assert p["cells"].count("x") == 2 and p["cells"].count("o") == 2  # the four central opening discs
    assert p["pass_action"] is None  # no forced pass at the opening
    state.apply_action(19)  # a known legal opening move (d3)
    p = board_engine.board_payload(state, 19)
    assert p["cells"].count("x") + p["cells"].count("o") == 5  # the four openers + the placed disc (≥)


def test_othello_pass_move_detected_generically() -> None:
    """A forced pass surfaces as ``BoardState.pass_action`` — the legal action whose ``action_to_string``
    reads ``"pass"`` (game-agnostic; also Go). Deterministic: a seeded random rollout reaches a pass."""
    import numpy as np

    game = board_engine.load_game("othello")
    rng = np.random.default_rng(5)  # this seed's rollout hits a forced pass near the endgame
    state = game.new_initial_state()
    saw_pass = False
    while not state.is_terminal():
        p = board_engine.board_payload(state, None)
        if p["pass_action"] is not None:
            saw_pass = True
            assert p["pass_action"] in state.legal_actions()
            assert state.action_to_string(state.current_player(), p["pass_action"]).lower() == "pass"
        state.apply_action(int(rng.choice(state.legal_actions())))
    assert saw_pass, "the seeded Othello rollout never reached the pass move"


def test_clean_grid_games_have_no_pass_action() -> None:
    """The pass field stays None for games that never pass (TTT/Connect Four) — additive + inert."""
    for gym_id in ("tic_tac_toe", "connect_four"):
        game = board_engine.load_game(gym_id)
        assert board_engine.board_payload(game.new_initial_state(), None)["pass_action"] is None


# -- G6e: move-based board interaction (Breakthrough, ADR-054) --------------


def test_breakthrough_registered() -> None:
    spec = get_env("breakthrough")
    assert spec is not None and spec.gym_id == "breakthrough"
    assert spec.family == "board" and spec.action_space == "discrete"
    assert spec.train_implemented is True and spec.hw_requirement == "cpu"
    assert spec.min_score == -1.0 and spec.solved_score == 1.0


def test_breakthrough_compact_board_parses_to_clean_8x8() -> None:
    """OpenSpiel prints Breakthrough's 8×8 board *compactly* (a row-label digit glued to packed cells,
    plus a trailing ``abcdefgh`` column-label line), unlike Othello's spaced grid — yet ``board_payload``
    must still yield a clean 8×8 grid of only ``.``/``b``/``w`` with no label chars leaking in (G6e)."""
    game = board_engine.load_game("breakthrough")
    p = board_engine.board_payload(game.new_initial_state(), None)
    assert p["rows"] == 8 and p["cols"] == 8 and len(p["cells"]) == 64
    assert set(p["cells"]) <= {".", "b", "w"}  # the compact labels/headers must not leak through
    assert p["cells"].count("b") == 16 and p["cells"].count("w") == 16  # two full rows each


def test_breakthrough_streams_from_to_move_map() -> None:
    """A move game streams a per-legal-action ``{action, from_cell, to_cell}`` map (G6e): every legal
    action decodes, starts on a current-player piece and lands on an empty or capturable enemy cell."""
    game = board_engine.load_game("breakthrough")
    state = game.new_initial_state()
    p = board_engine.board_payload(state, None)
    moves = p["moves"]
    assert moves is not None and {m["action"] for m in moves} == set(p["legal_actions"])  # all decoded
    cells = p["cells"]
    for m in moves:
        assert 0 <= m["from_cell"] < 64 and 0 <= m["to_cell"] < 64
        assert cells[m["from_cell"]] == "b"  # player 0 ('b') moves first
        assert cells[m["to_cell"]] in (".", "w")  # step into empty or capture an enemy diagonally
    assert p["last_from"] is None and p["last_to"] is None  # no move played yet


def test_breakthrough_clicked_action_applies_and_reports_last_move() -> None:
    """The streamed action int round-trips: applying it moves the piece, and the next payload reports
    that move's from/to cells (for the move highlight) decoded from the *current* state."""
    game = board_engine.load_game("breakthrough")
    state = game.new_initial_state()
    move = board_engine.board_payload(state, None)["moves"][0]
    state.apply_action(move["action"])
    p = board_engine.board_payload(state, move["action"])
    assert p["last_from"] == move["from_cell"] and p["last_to"] == move["to_cell"]
    assert p["cells"][move["from_cell"]] == "."  # the piece vacated its square
    assert p["cells"][move["to_cell"]] == "b"  # …and now stands on the destination


def test_placement_games_omit_the_move_fields() -> None:
    """The move map is additive: placement games (TTT/Connect Four/Othello) carry no ``moves``/
    ``last_from``/``last_to`` keys at all, so their payloads stay byte-identical to before G6e."""
    for gym_id in ("tic_tac_toe", "connect_four", "othello"):
        game = board_engine.load_game(gym_id)
        p = board_engine.board_payload(game.new_initial_state(), None)
        assert "moves" not in p and "last_from" not in p and "last_to" not in p


def test_breakthrough_profile_trains_cheap_scores_vs_medium() -> None:
    """Breakthrough is taught novice→easy (cheap, fast self-play) but scored vs the MEDIUM reference:
    eval-vs-easy saturates at +1 almost at once, so medium is the honest, non-saturating yardstick."""
    prof = board_engine.board_profile("breakthrough")
    assert prof.eval_strength == "medium"
    assert prof.teacher_start == "novice" and prof.teacher_end == "easy"


def test_eval_vs_mcts_aborts_immediately_when_should_stop() -> None:
    """A long reference eval must yield the moment training Stop is requested (G6e review fix): with
    ``should_stop`` already True it returns at once without playing a game, so Stop isn't held up."""
    calls = {"predict": 0}

    def predict(_obs: object, _mask: object) -> int:
        calls["predict"] += 1  # must never be reached — the abort fires before any move
        return 0

    game = board_engine.load_game("tic_tac_toe")
    score = board_engine.eval_vs_mcts(predict, game, sims=10, n_games=20, should_stop=lambda: True)
    assert score == 0.0  # no games scored
    assert calls["predict"] == 0  # aborted before playing a single move


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


# -- training un-gated (G6b) -------------------------------------------------


def test_board_training_no_longer_gated() -> None:
    """G6b flipped the board trainer on: a board env is now trainable (``train_implemented=True``) on
    CPU, so the manager's gate (train_implemented + GPU checks) no longer rejects it. The trainer
    itself is exercised end-to-end in ``test_board_train.py`` (a real short MaskablePPO run)."""
    spec = get_env("tictactoe")
    assert spec is not None
    assert spec.train_implemented is True  # no longer rejected by the train_implemented backstop
    assert spec.hw_requirement == "cpu"  # and not GPU-gated either → the manager accepts the run
    assert "ppo" in spec.supported_algos  # routed to trainer_board by is_board_game (still algo=="ppo")
