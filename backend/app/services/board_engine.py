"""Board-game subsystem core (G6a) — a game-agnostic wrapper over DeepMind **OpenSpiel**.

The 7th seam (ADR-050): a board game is a third shape next to the single-agent Gymnasium
``step()`` loop and the PettingZoo *parallel* multi-agent loop — a 2-player, **turn-based**,
perfect-information, zero-sum game with **legal-move masking** and **self-play**. That is
OpenSpiel's ``pyspiel.State`` API (``current_player`` / ``legal_actions`` / ``apply_action`` /
``is_terminal`` / ``returns``), **not** a ``gym.Env``, so board games are routed here via
:func:`is_board_game` (mirroring :func:`app.services.ma_env.is_multi_agent`) instead of through
``app.envs.factory.make_env``.

This module is **pure ``pyspiel`` + numpy, no torch** — the G6a opponent is a *training-free*
MCTS (``open_spiel.python.algorithms.mcts``). It is deliberately **game-agnostic**: every
function keys off the generic OpenSpiel API, so Connect Four / chess / go drop in as data +
a renderer with zero engine changes (a test loads ``connect_four`` through these same
functions to prove it). The neural self-play trainer is G6b.

Kept import-light: ``pyspiel`` is imported lazily inside the functions that need it, so
:func:`is_board_game` and module import stay cheap on the hot paths (play_session,
training_manager) — the same discipline as ``ma_env``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

if TYPE_CHECKING:  # only for type hints — never imported at runtime on the light paths
    from app.envs.registry import EnvSpec

# A masked inference fn over a trained board net: (flat obs, legal-action bool mask) -> a legal action
# index. Used as the self-play env opponent, the live-preview policy, and the human's play opponent.
MaskedPredictFn = Callable[[Any, Any], int]

# MCTS opponent strength → number of Monte-Carlo simulations per move. More simulations = stronger
# (and slower) play. Tic-Tac-Toe is tiny, so even "hard" is sub-millisecond; the spread is tuned so
# "easy" is genuinely beatable by a beginner while "hard" plays optimally (a hard-vs-hard pair draws
# — the TTT correctness invariant). Keyed by the play-config ``ai_strength``.
STRENGTH_SIMS: dict[str, int] = {"easy": 10, "medium": 80, "hard": 400}
_DEFAULT_STRENGTH = "medium"


class BoardProfile(NamedTuple):
    """Per-game training/eval strength (G6c) — the one number that has to vary with game size.

    ``eval_strength`` is the *fixed reference MCTS* the live skill curve (``ep_rew_mean``) is measured
    against; ``teacher_start``/``teacher_end`` bound the easy→… curriculum the net trains against.
    """

    eval_strength: str  # the beatable yardstick the learning chart is scored against
    teacher_start: str  # MCTS strength on the first round …
    teacher_end: str  # … ramping to this on the last round (a gentle curriculum)


# Tic-Tac-Toe is tiny (5 478 states): a net reaches the *medium* MCTS's level (drawing) in ~80–100k
# steps, so it both trains against and is scored against medium. Connect Four is a vastly bigger game
# (~10^13 states): on a CPU budget the net learns to beat the *easy* search bot but not the medium one
# (verified — eval-vs-easy climbs −0.6→+0.6 over ~90k steps, while eval-vs-medium stays ≈−0.9). So it
# is both taught by and scored against EASY — scoring it vs medium would pin the honest skill curve at
# the loss floor while the net is genuinely improving. Unlisted games fall back to the TTT profile.
# Keyed by the OpenSpiel short name (``gym_id``); the renderer/contract stay fully game-agnostic.
BOARD_PROFILES: dict[str, BoardProfile] = {
    "tic_tac_toe": BoardProfile(eval_strength="medium", teacher_start="easy", teacher_end="medium"),
    "connect_four": BoardProfile(eval_strength="easy", teacher_start="easy", teacher_end="easy"),
}
_DEFAULT_PROFILE = BOARD_PROFILES["tic_tac_toe"]


def board_profile(gym_id: str) -> BoardProfile:
    """The per-game training/eval strength profile (defaults to the Tic-Tac-Toe profile)."""
    return BOARD_PROFILES.get(gym_id, _DEFAULT_PROFILE)


def is_board_game(spec: EnvSpec | None) -> bool:
    """Whether ``spec`` is an OpenSpiel board game (routed to this subsystem, not ``make_env``).

    Light by design (no ``pyspiel`` import) so the hot paths can branch on it cheaply — the
    board-family parallel to :func:`app.services.ma_env.is_multi_agent`.
    """
    return spec is not None and spec.family == "board"


def strength_sims(ai_strength: str | None) -> int:
    """Map an ``ai_strength`` id (easy/medium/hard) to its MCTS simulation count."""
    return STRENGTH_SIMS.get(ai_strength or _DEFAULT_STRENGTH, STRENGTH_SIMS[_DEFAULT_STRENGTH])


def load_game(short_name: str) -> Any:
    """Load an OpenSpiel game by its short name (``"tic_tac_toe"``, ``"connect_four"``, …)."""
    import pyspiel  # lazy — keeps module import cheap on the hot paths

    return pyspiel.load_game(short_name)


class MctsOpponent:
    """A training-free MCTS bot — the G6a board AI (no neural net, no GPU).

    Wraps ``open_spiel.python.algorithms.mcts.MCTSBot`` with a ``RandomRolloutEvaluator``
    (the pure-Python MCTS — cleanly seedable via a numpy ``RandomState``, and the through-line
    to a G6b AlphaZero-lite trainer that swaps the random rollout for a neural evaluator).
    ``max_simulations`` is the strength knob; ``seed`` makes the bot reproducible for a demo
    (``None`` ⇒ a fresh, varied opponent each game, per the play convention for human play).
    """

    def __init__(self, game: Any, max_simulations: int, seed: int | None) -> None:
        import numpy as np
        from open_spiel.python.algorithms import mcts

        evaluator = mcts.RandomRolloutEvaluator(
            n_rollouts=1, random_state=np.random.RandomState(seed)
        )
        self._bot = mcts.MCTSBot(
            game,
            uct_c=2.0,
            max_simulations=max_simulations,
            evaluator=evaluator,
            random_state=np.random.RandomState(seed),
        )

    def step(self, state: Any) -> int:
        """Pick a (legal) action for the player to move in ``state``."""
        return int(self._bot.step(state))


def board_payload(state: Any, last_action: int | None) -> dict[str, Any]:
    """The streamed ``BoardState`` for one ply — built from the **generic** ``pyspiel.State`` API.

    ``cells`` is the board as a row-major list of single glyph characters parsed from
    ``str(state)`` (``"."`` empty, ``"x"``/``"o"`` for Tic-Tac-Toe, etc.). Reading the board as
    characters is the one game-agnostic extraction that works for any ASCII-grid OpenSpiel game;
    the only game-specific bit — mapping a glyph to a piece/player for rendering — lives in the
    **renderer** (``frontend/src/content/boardGames.ts``), where the prompt permits TTT specifics.
    Everything else (legal moves, whose turn, terminality, the winner) is the generic API.
    """
    text = str(state).strip("\n")
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip() != ""]
    rows = len(lines)
    cols = max((len(ln) for ln in lines), default=0)
    cells: list[str] = []
    for ln in lines:
        for c in range(cols):
            cells.append(ln[c] if c < len(ln) else " ")

    terminal = bool(state.is_terminal())
    winner: int | None = None
    if terminal:
        returns = list(state.returns())
        top = max(returns)
        # A single strictly-positive return = that player won; all-equal (e.g. [0, 0]) = a draw.
        if top > 0 and returns.count(top) == 1:
            winner = returns.index(top)

    return {
        "cells": cells,
        "rows": rows,
        "cols": cols,
        # No legal moves once the game is over (it's nobody's turn).
        "legal_actions": [] if terminal else [int(a) for a in state.legal_actions()],
        "current_player": int(state.current_player()),
        "last_action": int(last_action) if last_action is not None else None,
        "is_terminal": terminal,
        "winner": winner,
    }


def outcome(state: Any, player: int) -> tuple[float, Literal["win", "draw", "loss"]]:
    """The zero-sum result of a finished game for ``player``: ``(value, label)``.

    ``value`` is ``returns()[player]`` ∈ {−1, 0, 1}; ``label`` is ``"win"`` / ``"draw"`` /
    ``"loss"`` — the honest 3-valued readout (board games have no continuous skill %).
    """
    value = float(state.returns()[player])
    label: Literal["win", "draw", "loss"] = "win" if value > 0 else "loss" if value < 0 else "draw"
    return value, label


# ---------------------------------------------------------------------------
# Neural training (G6b, ADR-051) — MaskablePPO learns by playing the G6a MCTS *teacher*.
#
# A risk-gate (Local/_probe_maskable_selfplay.py) showed pure frozen self-play barely learns
# Tic-Tac-Toe in a CPU budget, while masked PPO trained *against the MCTS opponent* reaches a
# near-optimal (drawing) policy in ~80k steps / ~2 min on CPU. So the trainer's opponent is the
# G6a MctsOpponent (with an action mask from legal_actions()); pure self-play is parked.
#
# The pieces below are the game-agnostic glue between the pyspiel game and SB3:
#   * make_self_play_env  — a single-agent gym view of the 2-player game (ActionMasker-wrapped),
#   * build_board_predict / load_board_predict — a decoupled masked CPU snapshot (ADR-019-safe),
#   * board_move_fn       — adapt a masked predict into a (state)->action move (env/preview/play),
#   * eval_vs_mcts        — the eval-score-vs-reference-MCTS that drives the live learning chart.
# gymnasium / sb3-contrib are imported lazily so is_board_game + module import stay torch/gym-free.
# ---------------------------------------------------------------------------


def _legal_mask(state: Any, player: int, n_actions: int) -> Any:
    """A boolean ``[n_actions]`` mask of the moves legal for ``player`` (for MaskablePPO)."""
    import numpy as np

    mask = np.zeros(n_actions, dtype=bool)
    for a in state.legal_actions(player):
        mask[int(a)] = True
    return mask


def _obs_vec(state: Any, player: int) -> Any:
    """The flat ``observation_tensor`` from ``player``'s perspective (the policy's input)."""
    import numpy as np

    return np.asarray(state.observation_tensor(player), dtype=np.float32)


_SELF_PLAY_ENV_CLASS: Any = None


def _self_play_env_class() -> Any:
    """Lazily define + cache the single-agent self-play ``gym.Env`` (keeps gymnasium off the light path).

    Mirrors ``ma_env._species_env_class`` — the class is defined inside the function so importing
    :mod:`board_engine` (and the hot ``is_board_game`` check) never imports gymnasium.
    """
    global _SELF_PLAY_ENV_CLASS
    if _SELF_PLAY_ENV_CLASS is not None:
        return _SELF_PLAY_ENV_CLASS

    import numpy as np
    from gymnasium import Env, spaces

    class BoardSelfPlayEnv(Env):  # type: ignore[misc]  # gymnasium.Env is an untyped generic base
        """A single-agent view of a 2-player, turn-based, perfect-info game (G6b).

        The learner takes a **randomised seat** each episode (so one policy learns to play both
        sides); the opponent — a ``(state) -> action`` callable (the G6a MCTS, a frozen net snapshot,
        or random when ``None``) — acts inside ``reset``/``step`` until it is the learner's turn or the
        game ends. The observation is the flat ``observation_tensor(seat)``; the reward is the zero-sum
        ``returns()[seat] ∈ {−1,0,1}`` at terminal (0 in between). ``action_masks()`` exposes the legal
        moves so MaskablePPO never proposes an illegal one. Game-agnostic — keys only off the generic
        pyspiel API (chance nodes are sampled, so dice games would slot in too; TTT has none).
        """

        metadata = {"render_modes": []}

        def __init__(self, game: Any, opponent_move: Any = None, seed: int = 0) -> None:
            super().__init__()
            self.game = game
            self.n_actions = int(game.num_distinct_actions())
            self.observation_space = spaces.Box(
                low=-1.0, high=1.0, shape=(int(game.observation_tensor_size()),), dtype=np.float32
            )
            self.action_space = spaces.Discrete(self.n_actions)
            self._opp = opponent_move  # (state) -> action, or None for a random opponent
            self._rng = np.random.default_rng(seed)
            self._state: Any = None
            self._seat = 0

        def action_masks(self) -> Any:
            return _legal_mask(self._state, self._seat, self.n_actions)

        def _obs(self) -> Any:
            return _obs_vec(self._state, self._seat)

        def _advance_opponent(self) -> None:
            """Play chance + opponent turns until it is the learner's turn or the game is over."""
            st = self._state
            while not st.is_terminal() and st.current_player() != self._seat:
                if st.is_chance_node():
                    outcomes = st.chance_outcomes()
                    a = int(self._rng.choice([a for a, _ in outcomes]))
                elif self._opp is None:
                    a = int(self._rng.choice(st.legal_actions()))
                else:
                    a = int(self._opp(st))
                    if a not in st.legal_actions():  # a flaky opponent must not desync the game
                        a = int(self._rng.choice(st.legal_actions()))
                st.apply_action(a)

        def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, dict]:
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._state = self.game.new_initial_state()
            self._seat = int(self._rng.integers(self.game.num_players()))
            self._advance_opponent()
            return self._obs(), {}

        def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
            st = self._state
            legal = st.legal_actions(self._seat)
            a = int(action)
            if a not in legal:  # the action mask should prevent this; stay safe regardless
                a = int(legal[0])
            st.apply_action(a)
            self._advance_opponent()
            done = bool(st.is_terminal())
            reward = float(st.returns()[self._seat]) if done else 0.0
            return self._obs(), reward, done, False, {}

    _SELF_PLAY_ENV_CLASS = BoardSelfPlayEnv
    return _SELF_PLAY_ENV_CLASS


def _env_action_masks(env: Any) -> Any:
    """The mask fn ``ActionMasker`` calls as ``fn(env)`` (so it must take the env, not be a bound,
    no-arg method — the ``str`` form of ActionMasker would mis-call it). ``Any`` env keeps mypy happy."""
    return env.action_masks()


def make_self_play_env(game: Any, opponent_move: Any = None, seed: int = 0) -> Any:
    """The ``ActionMasker``-wrapped single-agent self-play env, so MaskablePPO finds the legal-move
    mask (``get_action_masks`` calls ``env.action_masks()`` through the wrapper)."""
    from sb3_contrib.common.wrappers import ActionMasker

    env = _self_play_env_class()(game, opponent_move, seed)
    return ActionMasker(env, _env_action_masks)


def _wrap_masked(snap: Any, deterministic: bool) -> MaskedPredictFn:
    """Wrap a loaded MaskablePPO into ``predict(obs, mask) -> legal action`` (its own masked inference)."""
    import numpy as np

    def predict(obs: Any, mask: Any) -> int:
        action, _ = snap.predict(
            np.asarray(obs), action_masks=np.asarray(mask), deterministic=deterministic
        )
        return int(np.asarray(action).flatten()[0])

    return predict


def build_board_predict(model: Any, deterministic: bool = True) -> MaskedPredictFn:
    """A decoupled masked predict fn over a CPU snapshot of a *live* MaskablePPO (ADR-019-safe).

    Round-trips the model through ``save``/``load`` into an independent CPU model — no shared tensor
    storage with the trainer (the same isolation as the Atari CNN snapshot, ADR-044), so forwarding it
    cannot perturb training. Built at a round boundary (a quiescent point) on the trainer thread."""
    import io

    from sb3_contrib import MaskablePPO

    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)
    return _wrap_masked(MaskablePPO.load(buf, env=None, device="cpu"), deterministic)


def load_board_predict(blob: bytes, deterministic: bool = True) -> MaskedPredictFn:
    """Load a saved ``board.zip`` (MaskablePPO) for inference only → ``predict(obs, mask) -> action``.

    ``env=None`` — the saved zip carries the obs/action spaces, so no env (and no pyspiel game) is
    built here; the caller supplies the obs + legal mask per move. For Play-vs-net + Watch-AI."""
    import io

    from sb3_contrib import MaskablePPO

    return _wrap_masked(MaskablePPO.load(io.BytesIO(blob), env=None, device="cpu"), deterministic)


def board_move_fn(game: Any, predict: MaskedPredictFn) -> Callable[[Any], int]:
    """Adapt a masked ``predict(obs, mask)`` into a ``(state) -> action`` move — for the self-play env
    opponent, the live-preview self-play loop and the human's play opponent. Game-agnostic."""
    n_actions = int(game.num_distinct_actions())

    def move(state: Any) -> int:
        player = state.current_player()
        return int(predict(_obs_vec(state, player), _legal_mask(state, player, n_actions)))

    return move


def eval_vs_mcts(
    predict: MaskedPredictFn, game: Any, sims: int, n_games: int = 20, seed: int = 0
) -> float:
    """Mean game result ∈ [−1, 1] for the net vs a fixed reference MCTS — the chart's honest skill curve.

    The net's seat alternates each game so both sides are measured; the per-game value is
    ``returns()[net_seat]`` (+1 win / 0 draw / −1 loss). For Tic-Tac-Toe a well-trained net converges
    toward 0 (it draws — the game's ceiling against strong play). Reuses the G6a :class:`MctsOpponent`.
    """
    import numpy as np

    n_actions = int(game.num_distinct_actions())
    rng = np.random.default_rng(seed)
    total = 0.0
    for g in range(n_games):
        state = game.new_initial_state()
        net_seat = g % game.num_players()
        bot = MctsOpponent(game, sims, seed + g)
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                state.apply_action(int(rng.choice([a for a, _ in outcomes])))
                continue
            player = state.current_player()
            if player == net_seat:
                action = int(predict(_obs_vec(state, player), _legal_mask(state, player, n_actions)))
            else:
                action = bot.step(state)
            state.apply_action(action)
        total += float(state.returns()[net_seat])
    return total / max(1, n_games)
