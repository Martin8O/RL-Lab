"""G6b — neural board trainer (MaskablePPO vs the MCTS teacher, ADR-051).

Covers the game-agnostic engine pieces (self-play env + masking, masked snapshot, eval-vs-MCTS), a
short end-to-end ``train_board`` run (metrics emitted, ``board.zip`` packed + reloadable as a legal
masked opponent), and game-agnosticism via a *second* OpenSpiel game (connect_four) through the same
self-play env. CPU-only; the short run is a couple of seconds.
"""

import numpy as np
from app.schemas.training import PPOHyperparams, SelfPlayHyperparams, TrainConfig
from app.services import board_engine
from app.services.train_control import TrainControl
from app.services.trainer_board import train_board


def _mask(state, n_actions: int) -> np.ndarray:
    m = np.zeros(n_actions, dtype=bool)
    for a in state.legal_actions(state.current_player()):
        m[a] = True
    return m


def test_maskable_ppo_imports() -> None:
    """The sb3-contrib dependency (the board trainer's masked PPO) is installed + importable."""
    from sb3_contrib import MaskablePPO  # noqa: F401


def test_self_play_env_masks_and_steps_tictactoe() -> None:
    game = board_engine.load_game("tic_tac_toe")
    env = board_engine.make_self_play_env(game, None, seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (game.observation_tensor_size(),)
    mask = env.action_masks()
    assert mask.shape == (game.num_distinct_actions(),) and bool(mask.any())
    # Playing only masked-legal moves advances to a terminal with a zero-sum reward.
    done = False
    reward = 0.0
    for _ in range(50):
        legal = np.flatnonzero(env.action_masks())
        obs, reward, done, _, _ = env.step(int(legal[0]))
        if done:
            break
    assert done and reward in (-1.0, 0.0, 1.0)


def test_self_play_env_game_agnostic_connect_four() -> None:
    """The SAME self-play env wraps a second OpenSpiel game with no TTT hardcoding (126-vec obs, 7 cols)."""
    game = board_engine.load_game("connect_four")
    env = board_engine.make_self_play_env(game, None, seed=1)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (game.observation_tensor_size(),)
    assert env.action_masks().shape == (game.num_distinct_actions(),)
    assert int(env.action_masks().sum()) == 7  # seven open columns at the start


def test_eval_vs_mcts_returns_score_in_range() -> None:
    game = board_engine.load_game("tic_tac_toe")
    rng = np.random.default_rng(0)

    def random_predict(obs, mask):  # ignores obs; picks a random legal move
        return int(rng.choice(np.flatnonzero(np.asarray(mask))))

    score = board_engine.eval_vs_mcts(random_predict, game, sims=10, n_games=6, seed=0)
    assert -1.0 <= score <= 1.0


def test_train_board_short_run_emits_metrics_and_packs_reloadable_checkpoint() -> None:
    cfg = TrainConfig(
        env_id="tictactoe",
        algo="ppo",
        seed=0,
        total_timesteps=1500,
        hyperparams=PPOHyperparams(ent_coef=0.01, n_steps=256, batch_size=64),
        self_play=SelfPlayHyperparams(rounds=2),
    )
    metrics: list = []
    progress: list = []
    snaps: list = []
    policies: list = []
    terminal = train_board(
        cfg,
        "tic_tac_toe",
        TrainControl(),
        on_metrics=metrics.append,
        on_progress=progress.append,
        on_policy=policies.append,
        on_snapshot=snaps.append,
    )
    assert terminal == "finished"
    # The learning curve is eval-vs-reference-MCTS ∈ [−1, 1] (reported as ep_rew_mean).
    assert metrics and all(-1.0 <= m.ep_rew_mean <= 1.0 for m in metrics)
    # Progress frames are emitted too (the Reward chart tab reads progressHistory, not metricsHistory),
    # carrying the same eval score — without them the board reward line would be blank.
    assert progress and all(-1.0 <= p.ep_rew_mean <= 1.0 for p in progress)
    # A decoupled preview policy was published, and a board.zip checkpoint was packed.
    assert policies
    assert snaps and snaps[-1].artifact_name == "board.zip" and snaps[-1].algo == "ppo"
    # Save must work from the moment the run starts — not only after the first round boundary (G6e fix):
    # an immediate snapshot is published at 0 steps, with more following at each rollout boundary.
    assert snaps[0].timesteps == 0
    # The packed checkpoint reloads as a masked predict that always returns a LEGAL move.
    predict = board_engine.load_board_predict(snaps[-1].blob)
    game = board_engine.load_game("tic_tac_toe")
    state = game.new_initial_state()
    state.apply_action(4)  # centre taken → 8 legal moves remain
    n = game.num_distinct_actions()
    move = predict(
        np.asarray(state.observation_tensor(state.current_player()), dtype=np.float32),
        _mask(state, n),
    )
    assert move in state.legal_actions(state.current_player())


def test_board_move_fn_picks_legal_moves() -> None:
    """board_move_fn adapts a masked predict into a (state)->action move that is always legal."""
    game = board_engine.load_game("tic_tac_toe")
    rng = np.random.default_rng(2)

    def random_predict(obs, mask):
        return int(rng.choice(np.flatnonzero(np.asarray(mask))))

    move = board_engine.board_move_fn(game, random_predict)
    state = game.new_initial_state()
    while not state.is_terminal():
        a = move(state)
        assert a in state.legal_actions(state.current_player())
        state.apply_action(a)
