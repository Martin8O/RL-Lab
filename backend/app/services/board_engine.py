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

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # only for type hints — never imported at runtime on the light paths
    from app.envs.registry import EnvSpec

# MCTS opponent strength → number of Monte-Carlo simulations per move. More simulations = stronger
# (and slower) play. Tic-Tac-Toe is tiny, so even "hard" is sub-millisecond; the spread is tuned so
# "easy" is genuinely beatable by a beginner while "hard" plays optimally (a hard-vs-hard pair draws
# — the TTT correctness invariant). Keyed by the play-config ``ai_strength``.
STRENGTH_SIMS: dict[str, int] = {"easy": 10, "medium": 80, "hard": 400}
_DEFAULT_STRENGTH = "medium"


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
