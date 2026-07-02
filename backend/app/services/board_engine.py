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

import re
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
#
# ``novice`` (G6d) is a sub-easy *teacher* tier, NOT a play difficulty: the play picker only offers
# easy/medium/hard, so it never surfaces as an opponent the user faces. It exists because a big game
# (Othello, ~10^28 states) trained from scratch against even the easy bot rarely wins → almost no
# learning signal; a near-random novice teacher hands the fresh net enough wins to start climbing
# (verified — eval-vs-easy goes −0.7→+0.2 with a novice teacher, vs barely moving against easy). Used
# only via :data:`BOARD_PROFILES`.
STRENGTH_SIMS: dict[str, int] = {"novice": 3, "easy": 10, "medium": 80, "hard": 400}
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
# Othello (G6d, ~10^28 states) is far bigger again. Against the easy bot a fresh net almost never wins,
# so it barely learns; trained against the near-random NOVICE teacher (ramping up to easy) it gets
# enough early wins to climb, and scored against the easy reference its honest curve rises from ≈−0.7
# to ≈+0.2 over a CPU budget (verified). So it is taught novice→easy and scored vs easy.
# Breakthrough (G6e) learns *fast*: trained against the near-random NOVICE→easy teacher it crushes the
# easy reference (eval-vs-easy hits +1 by ~20k steps, verified). Easy is therefore a saturating, useless
# yardstick, so it is scored against the tougher MEDIUM reference — the honest curve then climbs from
# ≈−0.9 to ≈+0.6/+0.9 over a CPU budget (verified) instead of pinning at the top. The cheap teacher keeps
# self-play fast (~250–530 steps/s) even though the medium eval is the stronger bar.
BOARD_PROFILES: dict[str, BoardProfile] = {
    "tic_tac_toe": BoardProfile(eval_strength="medium", teacher_start="easy", teacher_end="medium"),
    "connect_four": BoardProfile(eval_strength="easy", teacher_start="easy", teacher_end="easy"),
    "othello": BoardProfile(eval_strength="easy", teacher_start="novice", teacher_end="easy"),
    "breakthrough": BoardProfile(eval_strength="medium", teacher_start="novice", teacher_end="easy"),
    # Chess (G6g) is AlphaZero-only (no MCTS teacher → teacher fields unused), and far too deep to beat a
    # real searcher on any tolerable budget. Scored against the **novice** (3-sim) reference: a fresh net
    # already draws it (≈0.00, both near-random — measured), then the curve climbs as self-play teaches it
    # to beat the weak searcher. A stronger reference (easy) pins a fresh net at ≈−0.7 AND makes the
    # random-rollout eval much slower (chess rollouts play full games), so novice is both honest and cheap.
    "chess": BoardProfile(eval_strength="novice", teacher_start="novice", teacher_end="novice"),
    # Checkers / Dáma (G6-Dáma) — an 8×8 move game with the LOWEST branching of any board game here (~5
    # legal moves/pos: mandatory captures prune the tree, min 1), so — like Connect Four, NOT the noisy
    # high-branching Othello/Breakthrough — the value targets are sharp and BOTH trainers learn it.
    # Scored against the EASY reference: an AZ risk-gate (Local/_probe_checkers_az.py) measured a fresh net
    # at ≈−0.42 vs easy climbing to ≈+0.17 after ~600 self-play games (NO regression below fresh — the pass
    # signature), while vs the MEDIUM (80-sim) MCTS both fresh and trained sit pinned at ≈−1.0 (too strong
    # for a tolerable budget). So easy is the honest, climbing yardstick (medium would pin the curve at the
    # loss floor). The PPO baseline trains novice→easy (a fresh net rarely beats easy, so the near-random
    # novice teacher hands it early wins to start climbing) and is scored vs the same easy reference.
    "checkers": BoardProfile(eval_strength="easy", teacher_start="novice", teacher_end="easy"),
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


class BoardStrFormat(NamedTuple):
    """How to read the board grid out of ``str(state)`` for a game (G6d/G6e).

    OpenSpiel prints its boards in a few different ASCII layouts; this descriptor captures the variation
    so :func:`_board_grid` can extract a clean ``rows × cols`` cell list while the rest of the subsystem
    stays game-agnostic. ``kind`` selects the layout, ``empty`` is the empty-cell marker:

    * ``"clean"`` (Tic-Tac-Toe, Connect Four) — one row per line, one char per cell, no labels; read
      directly (byte-identical to G6a/G6c).
    * ``"spaced"`` (Othello) — a header + ``a b c`` / ``1 … 8`` labels, cells **space-separated**; each
      board row is the line whose first whitespace token is a digit (the row label).
    * ``"compact"`` (Breakthrough) — ``a … h`` / ``1 … 8`` labels, cells **packed with no separators**
      and a single-char row label prefixed directly (``"8bbbbbbbb"``); strip the leading digit label,
      take ``cols`` chars. A trailing column-label line (``" abcdefgh"``) has no leading digit → skipped.
    * ``"fen"`` (chess, G6g) — ``str(state)`` is a single **FEN** line (``"rnbqkbnr/pppppppp/…/RNBQKBNR
      w KQkq - 0 1"``); the first space-separated field is the piece placement, ranks ``8 → 1`` split by
      ``/`` with digits = runs of empties. **Case is preserved** (uppercase = white, lowercase = black),
      unlike the other kinds, since chess distinguishes the two sides by piece-letter case, not glyph.
    * ``"checkers"`` (Dáma, G6-Dáma) — a ``"compact"``-like packed grid (``"8.8.....8"`` = rank-label ``8``
      glued to ``cols`` cells), BUT a **king glyph is a DIGIT** (``"8"`` = white king, ``"*"`` = black king,
      ``"o"``/``"+"`` = men), so the ``"compact"`` strip-*all*-leading-digits parser would eat a king in
      column ``a``. Here the rank label is a fixed-width **prefix** and the cells are the **suffix**, so we
      take the **last ``cols`` chars** of each rank-digit-led line — digit-safe and **case-preserved** (the
      renderer needs ``"8"``/``"*"`` to draw kings). The trailing coord line (``" abcdefgh"``) starts with a
      space, not a rank digit, so it is skipped.

    The ``"spaced"``/``"compact"``/``"fen"``/``"checkers"`` paths take the authoritative ``rows × cols`` from
    the ``observation_tensor`` shape so header/label lines never leak into the grid. Keyed by the OpenSpiel
    short name; defaults to the clean format.
    """

    empty: str  # the empty-cell marker in str(state) ("." for most games, "-" for Othello)
    kind: Literal["clean", "spaced", "compact", "fen", "checkers"]


_DEFAULT_STR_FORMAT = BoardStrFormat(empty=".", kind="clean")
_BOARD_STR_FORMATS: dict[str, BoardStrFormat] = {
    "othello": BoardStrFormat(empty="-", kind="spaced"),
    "breakthrough": BoardStrFormat(empty=".", kind="compact"),
    "chess": BoardStrFormat(empty=".", kind="fen"),
    "checkers": BoardStrFormat(empty=".", kind="checkers"),
}


def _board_grid(state: Any) -> tuple[int, int, list[str]]:
    """Extract the ``(rows, cols, cells)`` grid from ``str(state)``, normalising empties to ``"."``.

    Cells are row-major single glyph chars (``"."`` empty, ``"x"``/``"o"``/``"b"``/``"w"`` pieces).
    Dispatches on the per-game :class:`BoardStrFormat`: clean grids (TTT/Connect Four) parse
    byte-identically to G6a, while decorated grids (Othello spaced, Breakthrough compact) are
    detokenised against the true ``rows × cols`` from the ``observation_tensor`` shape.
    """
    game = state.get_game()
    fmt = _BOARD_STR_FORMATS.get(game.get_type().short_name, _DEFAULT_STR_FORMAT)
    text = str(state).strip("\n")

    if fmt.kind == "clean":
        lines = [ln.rstrip() for ln in text.split("\n") if ln.strip() != ""]
        rows = len(lines)
        cols = max((len(ln) for ln in lines), default=0)
        cells: list[str] = []
        for ln in lines:
            for c in range(cols):
                ch = ln[c] if c < len(ln) else " "
                cells.append("." if ch == fmt.empty else ch)
        return rows, cols, cells

    if fmt.kind == "fen":
        # Chess: str(state) is one FEN line; its first field is the piece placement (ranks 8→1 split by
        # '/', digits = empty runs). Case is preserved so the renderer can tell white (UPPER) from black.
        placement = text.split(" ", 1)[0]
        shape = game.observation_tensor_shape()
        rows, cols = int(shape[-2]), int(shape[-1])
        grid_fen: list[str] = []
        for rank in placement.split("/"):
            for ch in rank:
                if ch.isdigit():
                    grid_fen.extend(["."] * int(ch))
                else:
                    grid_fen.append(ch)
        return rows, cols, grid_fen[: rows * cols]

    if fmt.kind == "checkers":
        # Checkers (Dáma): "compact"-like packed rows, but the king glyph "8" is a DIGIT, so strip the
        # rank label by taking the SUFFIX (the last `cols` chars) instead of stripping leading digits —
        # digit-safe and case-preserved. Skip the trailing coord line (" abcdefgh", starts with a space).
        shape = game.observation_tensor_shape()
        rows, cols = int(shape[-2]), int(shape[-1])
        grid_chk: list[str] = []
        for ln in text.split("\n"):
            if len(grid_chk) >= rows * cols:
                break
            row = ln.rstrip()
            if row.lstrip()[:1].isdigit() and len(row) >= cols:
                grid_chk.extend("." if ch == fmt.empty else ch for ch in row[-cols:])
        return rows, cols, grid_chk

    # Decorated grids: the obs-tensor shape gives the authoritative rows × cols.
    shape = game.observation_tensor_shape()
    rows, cols = int(shape[-2]), int(shape[-1])
    grid: list[str] = []
    for ln in text.split("\n"):
        if len(grid) >= rows * cols:
            break
        if fmt.kind == "spaced":
            # Each board row: a digit row-label token followed by `cols` single-char cell tokens.
            toks = ln.split()
            if len(toks) >= cols + 1 and toks[0].isdigit():
                grid.extend("." if ch == fmt.empty else ch for ch in toks[1 : 1 + cols])
        else:  # "compact": a leading digit label glued to `cols` packed cell chars.
            s = ln.strip()
            if s and s[0].isdigit():
                k = 0
                while k < len(s) and s[k].isdigit():
                    k += 1  # consume the (1+ digit) row label
                row_cells = s[k : k + cols]
                if len(row_cells) == cols:
                    grid.extend("." if ch == fmt.empty else ch for ch in row_cells)
    return rows, cols, grid


# Move-based board games (G6e, ADR-054) — a move is a (from-square → to-square), not a placement, so
# the renderer needs the board cells each legal action moves between. Breakthrough's ``action_to_string``
# is a clean coordinate pair (``"a7a6"`` = file a, rank 7 → file a, rank 6), which :func:`_parse_move`
# decodes generically: a square token is ``<file letter><rank number>`` (file ``a`` = column 0, rank 1 =
# the bottom row), so the convention carries to **checkers** — whose moves are the same clean coordinate
# pairs (a step ``"a3b4"`` or a capture *jump* ``"d6f4"``, distance-2 but still exactly two squares) and
# whose **multi-jumps are sequential single actions by the same player** (each continuation jump is its own
# legal move, so the two-step select→destination click flow handles a jump chain one hop at a time) — and,
# with a richer decoder, chess (whose SAN strings — ``"Nc3"``, ``"O-O"`` — *don't* parse to two squares
# here, so it uses the board-diff decoder in ``_DIFF_MOVE_GAMES``). Listed games stream a per-legal-action
# ``{from,to}`` map; every other game leaves ``BoardState.moves`` absent (byte-identical).
_MOVE_GAMES: frozenset[str] = frozenset({"breakthrough", "checkers"})
_SQUARE_RE = re.compile(r"([a-z])([0-9]+)")


def _square_to_cell(file_ch: str, rank: int, rows: int, cols: int) -> int | None:
    """A board square (``file_ch`` ``a``-based column, ``rank`` 1-based from the **bottom**) → a row-major
    cell index matching :func:`_board_grid` (rank ``rows`` is the top line). ``None`` if out of bounds."""
    col = ord(file_ch) - ord("a")
    if not (0 <= col < cols and 1 <= rank <= rows):
        return None
    return (rows - rank) * cols + col


def _parse_move(move_str: str, rows: int, cols: int) -> tuple[int, int] | None:
    """Decode an ``action_to_string`` move into ``(from_cell, to_cell)`` if it is a clean coordinate
    pair (exactly two valid algebraic squares), else ``None`` (a non-coordinate move — pass, chess SAN)."""
    squares = _SQUARE_RE.findall(move_str.strip())
    if len(squares) != 2:
        return None
    cells = [_square_to_cell(f, int(r), rows, cols) for f, r in squares]
    if cells[0] is None or cells[1] is None:
        return None
    return cells[0], cells[1]


def _legal_moves(state: Any, rows: int, cols: int) -> list[dict[str, int]]:
    """For the current player, each legal action's ``{action, from_cell, to_cell}`` (move games, G6e).

    The client maps a clicked (from, to) pair back to the action int; actions that don't decode to a
    coordinate pair are omitted (handled elsewhere — a pass rides ``pass_action``)."""
    player = int(state.current_player())
    out: list[dict[str, int]] = []
    for a in state.legal_actions():
        parsed = _parse_move(state.action_to_string(player, int(a)), rows, cols)
        if parsed is not None:
            out.append({"action": int(a), "from_cell": parsed[0], "to_cell": parsed[1]})
    return out


def _last_move_cells(state: Any, last_action: int | None, rows: int, cols: int) -> tuple[int | None, int | None]:
    """The ``(from_cell, to_cell)`` of the move just played, for a last-move highlight (move games).

    ``action_to_string`` is position-independent for these games (verified), so the *current* state can
    decode an action played by either player; we pass a valid player index (the current one, or 0 at a
    terminal node where ``current_player()`` is the sentinel). ``(None, None)`` if it doesn't decode."""
    if last_action is None:
        return None, None
    cp = int(state.current_player())
    player = cp if 0 <= cp < int(state.get_game().num_players()) else 0
    try:
        parsed = _parse_move(state.action_to_string(player, int(last_action)), rows, cols)
    except Exception:  # noqa: BLE001 — a non-decodable action must not break the frame
        return None, None
    return parsed if parsed is not None else (None, None)


# Diff-decoded move games (chess, G6g) — unlike Breakthrough's clean ``"a7a6"`` coordinate pairs, chess
# ``action_to_string`` is SAN (``"Nc3"``, ``"O-O"``, ``"exd6"``, ``"e8=Q+"``), which can't be parsed as two
# squares. The robust, game-agnostic decoder instead **applies the action on a clone and diffs the board**:
# the vacated square is the ``from``, the square that gained the mover is the ``to``. This handles every
# special move with no chess-specific rules — castling (the king's e1→g1 squares), en-passant (the pawn's
# e5→d6), and **promotion** (e7→e8 with the new piece letter), where the same (from,to) has up to four
# legal actions (=Q/=R/=B/=N) that the streamed ``promotion`` letter disambiguates so the client can show a
# piece picker. Verified on every move type (``Local/_probe_chess2.py``); ~0.3 ms per move (clone is free).
_DIFF_MOVE_GAMES: frozenset[str] = frozenset({"chess"})
# FEN piece letters by case (uppercase = white = player 0, lowercase = black = player 1); promotable targets.
_PAWNS = frozenset({"P", "p"})
_PROMO_PIECES = frozenset({"Q", "R", "B", "N", "q", "r", "b", "n"})

# Standard piece values (FEN letters, case-insensitive) for scoring an UNRESOLVED chess game by material.
# King omitted — it is always present, so it never shifts the balance. ``_MATERIAL_NORM`` maps a typical
# decisive edge into (−1, 1); the clamp keeps |score| < 1 so a material lead never reads as a real
# (checkmate) win, which stays reserved for ±1.
_PIECE_VALUE: dict[str, float] = {"p": 1.0, "n": 3.0, "b": 3.0, "r": 5.0, "q": 9.0}
_MATERIAL_NORM = 10.0


def _material_balance(cells: list[str], seat: int) -> float:
    """Signed, normalized material balance ∈ (−1, 1) from ``seat``'s perspective over ``_board_grid`` cells
    (case-preserved FEN letters; uppercase = white = player 0). Pure helper so the sign convention is unit
    testable without a live pyspiel state."""
    white = sum(_PIECE_VALUE.get(c.lower(), 0.0) for c in cells if c.isupper())
    black = sum(_PIECE_VALUE.get(c.lower(), 0.0) for c in cells if c.islower())
    diff = (white - black) if seat == 0 else (black - white)
    return float(max(-0.99, min(0.99, diff / _MATERIAL_NORM)))


def material_score(state: Any, seat: int) -> float:
    """Material-balance score ∈ (−1, 1) for an unresolved chess position, from ``seat``'s perspective."""
    return _material_balance(_board_grid(state)[2], seat)


def unresolved_value_fn(gym_id: str) -> Callable[[Any, int], float] | None:
    """A scorer for a ply-capped (unresolved) eval game, or ``None`` to score it a flat draw (0).

    A near-random early chess net almost never delivers mate inside the eval ply cap, so without this every
    eval game hits the cap and scores 0 — pinning the reported AZ score at **0.0** even while the net is
    learning to win material (the user-flagged "AZ score 0.0", G6h). Material gives a continuous, *objective*
    signal (unlike the net's own value head, which would be circular). Only chess defines one; the small
    bounded boards never reach a cap, so they pass ``None`` and keep the exact prior draw-is-0 behavior."""
    return material_score if gym_id == "chess" else None


def _diff_from_to(before: list[str], after: list[str]) -> tuple[int, int, str | None] | None:
    """Decode ``(from_cell, to_cell, promotion)`` by diffing two board-cell grids (chess, G6g).

    ``before``/``after`` are row-major ``_board_grid`` cells (case-preserved FEN letters, ``"."`` empty).
    The ``to`` square is the one that gained the moving piece (preferring the **king** so castling reports
    the king's move, not the rook's); the ``from`` is the vacated square holding that same piece before
    (or a pawn, for a promotion). ``promotion`` is the lower-cased new piece letter when a pawn changed
    type on the move (``"q"``/``"r"``/``"b"``/``"n"``), else ``None``. ``None`` if it can't be decoded."""
    if len(before) != len(after):
        return None
    emptied = [i for i in range(len(before)) if before[i] != "." and after[i] == "."]
    appeared = [i for i in range(len(after)) if after[i] != "." and after[i] != before[i]]
    if not appeared:
        return None
    to_cell = next((i for i in appeared if after[i] in ("k", "K")), appeared[0])
    mover = after[to_cell]
    cands = [i for i in emptied if before[i] == mover]
    if not cands and mover in _PROMO_PIECES:  # promotion: the pawn vacated, the piece type changed
        cands = [i for i in emptied if before[i] in _PAWNS]
    if not cands:
        cands = emptied
    if not cands:
        return None
    from_cell = cands[0]
    promotion = mover.lower() if (before[from_cell] in _PAWNS and after[to_cell] not in _PAWNS) else None
    return from_cell, to_cell, promotion


def _diff_legal_moves(state: Any) -> list[dict[str, Any]]:
    """Each legal action's ``{action, from_cell, to_cell, promotion?}`` via the board-diff decoder (chess).

    Clones + applies each legal action and diffs the board, so a promotion yields four entries sharing
    (from,to) with distinct ``promotion`` letters. Undecodable actions are skipped (none in practice)."""
    before = _board_grid(state)[2]
    out: list[dict[str, Any]] = []
    for a in state.legal_actions():
        nxt = state.clone()
        nxt.apply_action(int(a))
        decoded = _diff_from_to(before, _board_grid(nxt)[2])
        if decoded is None:
            continue
        from_cell, to_cell, promo = decoded
        move: dict[str, Any] = {"action": int(a), "from_cell": from_cell, "to_cell": to_cell}
        if promo is not None:
            move["promotion"] = promo
        out.append(move)
    return out


def _pass_action(state: Any) -> int | None:
    """The index of a legal **pass** move (Othello when a player has no placement; Go), else ``None``.

    Game-agnostic: a pass is the legal action whose ``action_to_string`` reads ``"pass"`` — it maps to
    no board cell, so the renderer shows a dedicated Pass button rather than a cell click. ``None`` for
    games without one (TTT/Connect Four never pass) and at terminal/chance nodes (nobody to move)."""
    if state.is_terminal() or state.is_chance_node():
        return None
    player = int(state.current_player())
    for a in state.legal_actions():
        if state.action_to_string(player, int(a)).strip().lower() == "pass":
            return int(a)
    return None


def board_cells(state: Any) -> list[str]:
    """The board as a row-major glyph list (the renderer's ``BoardState.cells``) — exposed so a play/preview
    loop can snapshot the *previous* board before applying a move and hand it to :func:`board_payload` as
    ``prev_cells`` (the chess last-move highlight is decoded by diffing prev → current, G6g)."""
    return _board_grid(state)[2]


def board_payload(state: Any, last_action: int | None, prev_cells: list[str] | None = None) -> dict[str, Any]:
    """The streamed ``BoardState`` for one ply — built from the **generic** ``pyspiel.State`` API.

    ``cells`` is the board as a row-major list of single glyph characters (``"."`` empty, ``"x"``/``"o"``
    pieces) extracted by :func:`_board_grid`, which handles clean (TTT/Connect Four), decorated (Othello)
    and FEN (chess) OpenSpiel board strings. The only game-specific bit — mapping a glyph to a piece/player
    for rendering — lives in the **renderer** (``frontend/src/content/boardGames.ts``). Everything else
    (legal moves, whose turn, terminality, the winner, a forced pass) is the generic API.

    ``prev_cells`` (the board *before* ``last_action``, from :func:`board_cells`) lets a diff-decoded move
    game (chess) report the last move's from/to for the highlight — chess SAN can't be re-decoded from the
    post-move state, so the caller passes the previous board. Optional + ignored by every other game.
    """
    rows, cols, cells = _board_grid(state)

    terminal = bool(state.is_terminal())
    winner: int | None = None
    if terminal:
        returns = list(state.returns())
        top = max(returns)
        # A single strictly-positive return = that player won; all-equal (e.g. [0, 0]) = a draw.
        if top > 0 and returns.count(top) == 1:
            winner = returns.index(top)

    payload: dict[str, Any] = {
        "cells": cells,
        "rows": rows,
        "cols": cols,
        # No legal moves once the game is over (it's nobody's turn).
        "legal_actions": [] if terminal else [int(a) for a in state.legal_actions()],
        "current_player": int(state.current_player()),
        "last_action": int(last_action) if last_action is not None else None,
        "is_terminal": terminal,
        "winner": winner,
        # A legal "pass" move (Othello when stuck), or None — the renderer shows a Pass button for it.
        "pass_action": _pass_action(state),
    }

    # Move-based games: the move isn't a placement, so add the per-legal-action (from→to) cell map the
    # client needs to turn a clicked pair into an action + the last move's cells for the move highlight.
    # Absent for placement games (TTT/Connect Four/Othello) — byte-identical. Two decoders: Breakthrough
    # parses coordinate strings (position-independent), chess diffs the board (handles promotion/castling).
    short_name = state.get_game().get_type().short_name
    if short_name in _MOVE_GAMES:
        payload["moves"] = [] if terminal else _legal_moves(state, rows, cols)
        payload["last_from"], payload["last_to"] = _last_move_cells(state, last_action, rows, cols)
    elif short_name in _DIFF_MOVE_GAMES:
        payload["moves"] = [] if terminal else _diff_legal_moves(state)
        # The last move can't be re-decoded from the post-move state (SAN), so diff prev → current cells.
        if prev_cells is not None and (decoded := _diff_from_to(prev_cells, cells)) is not None:
            payload["last_from"], payload["last_to"] = decoded[0], decoded[1]
        else:
            payload["last_from"], payload["last_to"] = None, None

    return payload


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
    move: Callable[[Any], int],
    game: Any,
    sims: int,
    n_games: int = 20,
    seed: int = 0,
    should_stop: Callable[[], bool] | None = None,
) -> float:
    """Mean game result ∈ [−1, 1] for a net player vs a fixed reference MCTS — the chart's skill curve.

    ``move`` is a ``(state) -> action`` player: the G6b MaskablePPO net (wrapped via
    :func:`board_move_fn`) or the G6f AlphaZero **neural-MCTS** player (``az_net.az_move_fn``). Taking a
    ``(state) -> action`` (not the bare masked predict) lets AlphaZero be measured at its real strength
    — with search — so its curve reflects the move it would actually play. The net's seat alternates
    each game so both sides are measured; the per-game value is ``returns()[net_seat]`` (+1 win / 0 draw
    / −1 loss). For Tic-Tac-Toe a well-trained player converges toward 0 (it draws — the game's ceiling
    against strong play). Reuses the G6a :class:`MctsOpponent` as the reference.

    ``should_stop`` lets a caller abort a long eval the moment a training Stop is requested (a strong
    reference like Breakthrough's medium MCTS is ~9 s for 20 games — without this, Stop waits it out);
    the partial mean is returned and discarded by the stopping trainer, so correctness is unaffected.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    total = 0.0
    for g in range(n_games):
        if should_stop is not None and should_stop():
            return total / max(1, g)  # abort promptly on Stop; the result is unused
        state = game.new_initial_state()
        net_seat = g % game.num_players()
        bot = MctsOpponent(game, sims, seed + g)
        while not state.is_terminal():
            # Per-move stop check too: a single Breakthrough game vs a strong MCTS can take a few
            # seconds, so checking only per game still lets one long game finish before Stop bites.
            if should_stop is not None and should_stop():
                return total / max(1, g)
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                state.apply_action(int(rng.choice([a for a, _ in outcomes])))
                continue
            action = int(move(state)) if state.current_player() == net_seat else bot.step(state)
            state.apply_action(action)
        total += float(state.returns()[net_seat])
    return total / max(1, n_games)
