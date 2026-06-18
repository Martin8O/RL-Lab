"""Batched-GPU AlphaZero engine (G6g, first half) — parallel self-play + a batch-aware MCTS.

G6f's AlphaZero-lite ran **single-position** MCTS forwards (batch 1), the one regime where a GPU sits
idle or loses to the CPU (measured on Connect Four, ``Local/_probe_g6f_batch.py``: the net forward is
GPU **0.8× at batch 1** but **6× at batch 64** and **11–18× at batch 256**, wider with a bigger net).
This module rebuilds the self-play **data generator** to be GPU-bound: **B games run concurrently** and
every MCTS simulation step batches all B in-flight leaf evaluations into **one** net forward — the
AlphaZero "inference-server" pattern. OpenSpiel's own AlphaZero is TensorFlow-only (TF-GPU is dead on
native Windows), so the tree search here is a **custom, batch-aware PyTorch MCTS**, not OpenSpiel's
synchronous :class:`mcts.MCTSBot`.

The search is the standard PUCT AlphaZero MCTS, but driven over a *cohort* of independent games at once:
each simulation round descends exactly one leaf per active game, the leaves are evaluated in a single
batched forward (:meth:`app.services.az_net.AZModel.infer_batch`), then expanded + backed up. So ``sims``
rounds = ``sims`` batched forwards for the **whole** cohort, instead of B×sims batch-1 forwards. The tree
itself is pure ``pyspiel`` + numpy (only the leaf evaluation touches torch, inside ``infer_batch``); it is
**game-agnostic**, keying only off the generic ``pyspiel.State`` API like :mod:`app.services.board_engine`.

Two drivers sit on the shared :func:`batched_search` primitive:

* :func:`self_play_parallel` — the training data generator (Dirichlet root noise + temperature sampling),
* :func:`eval_vs_mcts_parallel` — a GPU-batched eval of the net (noise-free, argmax) vs a fixed reference
  MCTS, the batched counterpart of :func:`app.services.board_engine.eval_vs_mcts`.

Inference for a *single* human game (Play-vs-net) stays on G6f's :func:`app.services.az_net.az_move_fn`
(one move at a time is inherently batch-1, so OpenSpiel's MCTSBot is fine there).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from app.services import board_engine

# A finished-and-valued self-play example: (canonical obs planes, MCTS visit-count policy target over all
# actions, game value ∈ [−1, 1] from the moving player's perspective). The trainer feeds these to the net.
ValuedExample = tuple[np.ndarray, np.ndarray, float]


class _Node:
    """One MCTS tree node = a (decision-or-terminal) game position + its child edge statistics.

    Edges are keyed by action: ``child_p`` priors (from the net policy head, softmaxed over legal moves),
    ``child_n`` visit counts, ``child_w`` summed values (each from the perspective of *this* node's
    ``to_play``). ``children`` are created lazily on first descent. Chance nodes never become tree nodes —
    :func:`_advance_chance` samples through them when a child state is built — so every node here is a
    decision node (or terminal), and ``to_play`` is a real player index for the backup sign."""

    __slots__ = ("state", "to_play", "expanded", "child_p", "child_n", "child_w", "children")

    def __init__(self, state: Any) -> None:
        self.state = state
        self.to_play = int(state.current_player())
        self.expanded = False
        self.child_p: dict[int, float] = {}
        self.child_n: dict[int, int] = {}
        self.child_w: dict[int, float] = {}
        self.children: dict[int, _Node] = {}


def _advance_chance(state: Any, rng: np.random.Generator) -> None:
    """Sample through any chance nodes so the position becomes a decision (or terminal) node.

    Board games here (TTT/Connect Four/Othello/Breakthrough) have no chance nodes, so this is a no-op for
    them; it keeps the search game-agnostic for a future stochastic game (and matches G6f's chance handling).
    """
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions = [a for a, _ in outcomes]
        probs = np.array([p for _, p in outcomes], dtype=np.float64)
        state.apply_action(int(rng.choice(actions, p=probs / probs.sum())))


def _child_state(state: Any, action: int, rng: np.random.Generator) -> Any:
    """Clone ``state``, apply ``action``, and advance past any resulting chance nodes."""
    child = state.clone()
    child.apply_action(int(action))
    _advance_chance(child, rng)
    return child


def _select_action(node: _Node, c_puct: float) -> int:
    """PUCT child selection: argmax over legal edges of ``Q + c_puct · P · √(ΣN + 1) / (1 + N)``.

    ``Q = W/N`` (0 for an unvisited edge) is the mean value for ``node.to_play`` of that move; the ``√(ΣN
    + 1)`` (rather than ``√ΣN``) makes the *first* selection at a freshly expanded node follow the prior
    (otherwise every edge scores 0). The player to move maximises their own value, so this is correct for
    both seats without any seat-specific bookkeeping."""
    total = 0
    for n in node.child_n.values():
        total += n
    sqrt_total = math.sqrt(total + 1)
    best_a, best_score = -1, -1e30
    for a, p in node.child_p.items():
        n = node.child_n[a]
        q = node.child_w[a] / n if n > 0 else 0.0
        score = q + c_puct * p * sqrt_total / (1 + n)
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def _select_to_leaf(root: _Node, c_puct: float, rng: np.random.Generator) -> tuple[_Node, list[tuple[_Node, int]]]:
    """Descend by PUCT from ``root`` to an unexpanded (or terminal) leaf → ``(leaf, path)``.

    ``path`` is the list of traversed ``(node, action)`` edges, used by :func:`_backup`."""
    node = root
    path: list[tuple[_Node, int]] = []
    while node.expanded and not node.state.is_terminal():
        a = _select_action(node, c_puct)
        path.append((node, a))
        child = node.children.get(a)
        if child is None:
            child = _Node(_child_state(node.state, a, rng))
            node.children[a] = child
        node = child
    return node, path


def _expand(node: _Node, policy_logits: np.ndarray) -> None:
    """Attach legal-move priors (softmax of the policy head over the legal actions) and open the edges."""
    legal = node.state.legal_actions(node.to_play)
    sub = policy_logits[legal]
    sub = np.exp(sub - sub.max())
    probs = sub / sub.sum()
    for a, p in zip(legal, probs, strict=True):
        ia = int(a)
        node.child_p[ia] = float(p)
        node.child_n[ia] = 0
        node.child_w[ia] = 0.0
    node.expanded = True


def _backup(
    path: list[tuple[_Node, int]],
    leaf_value: float,
    leaf_to_play: int,
    terminal_returns: list[float] | None,
) -> None:
    """Propagate a leaf evaluation up its path, adding the value *for each node's own mover*.

    For a net-evaluated leaf, ``leaf_value`` is the estimated return for ``leaf_to_play`` (2-player
    zero-sum → negate it for the other seat). For a terminal leaf, ``terminal_returns`` is the true
    per-player outcome. Comparing actual ``to_play`` values (instead of assuming strict alternation)
    keeps it correct for games where the same player can move twice (an Othello pass)."""
    for node, a in path:
        node.child_n[a] += 1
        if terminal_returns is not None:
            node.child_w[a] += terminal_returns[node.to_play]
        else:
            node.child_w[a] += leaf_value if node.to_play == leaf_to_play else -leaf_value


def _add_dirichlet(root: _Node, alpha: float, frac: float, rng: np.random.Generator) -> None:
    """Mix Dirichlet noise into the root priors (self-play exploration, the AlphaZero recipe)."""
    actions = list(root.child_p.keys())
    if not actions:
        return
    noise = rng.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise, strict=True):
        root.child_p[a] = (1 - frac) * root.child_p[a] + frac * float(n)


def _expand_roots_batch(model: Any, roots: list[_Node]) -> None:
    """Expand every (non-terminal) root in ONE batched forward — the first inference of a search."""
    obs = [np.asarray(r.state.observation_tensor(r.to_play), dtype=np.float32) for r in roots]
    logits, _values = model.infer_batch(np.stack(obs))
    for r, lg in zip(roots, logits, strict=True):
        _expand(r, lg)


def batched_search(
    model: Any,
    states: list[Any],
    sims: int,
    c_puct: float,
    dir_alpha: float,
    dir_frac: float,
    add_noise: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Run ``sims`` PUCT simulations for **each** non-terminal state, batching all leaf forwards per round.

    Returns one normalized **visit-count distribution** (``np.ndarray[n_actions]``) per input state — the
    AlphaZero improved policy (sampled for self-play, argmaxed for eval/play). The input ``states`` are not
    mutated (each tree roots on a clone). The whole cohort shares one forward per simulation round, so this
    is ``sims + 1`` batched forwards total regardless of the cohort size B — the GPU-bound win over G6f."""
    n_actions = model.n_actions
    roots = [_Node(s.clone()) for s in states]

    _expand_roots_batch(model, roots)
    if add_noise:
        for r in roots:
            _add_dirichlet(r, dir_alpha, dir_frac, rng)

    for _ in range(sims):
        leaves: list[tuple[_Node, list[tuple[_Node, int]]]] = []
        eval_obs: list[np.ndarray] = []
        eval_slots: list[int] = []
        for i, root in enumerate(roots):
            leaf, path = _select_to_leaf(root, c_puct, rng)
            leaves.append((leaf, path))
            if leaf.state.is_terminal():
                _backup(path, 0.0, 0, list(leaf.state.returns()))
            else:
                eval_slots.append(i)
                eval_obs.append(np.asarray(leaf.state.observation_tensor(leaf.to_play), dtype=np.float32))
        if eval_obs:
            logits, values = model.infer_batch(np.stack(eval_obs))
            for k, i in enumerate(eval_slots):
                leaf, path = leaves[i]
                _expand(leaf, logits[k])
                _backup(path, float(values[k]), leaf.to_play, None)

    dists: list[np.ndarray] = []
    for root in roots:
        d = np.zeros(n_actions, dtype=np.float64)
        for a, n in root.child_n.items():
            d[a] = n
        s = d.sum()
        dists.append(d / s if s > 0 else d)
    return dists


def self_play_parallel(
    model: Any,
    num_games: int,
    parallel: int,
    sims: int,
    c_puct: float,
    dir_alpha: float,
    dir_frac: float,
    temp_moves: int,
    rng: np.random.Generator,
    base_seed: int,
    on_game_done: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[ValuedExample], list[list[float]]]:
    """Generate ``num_games`` neural-guided-MCTS self-play games with **batched** search → (examples, returns).

    Games run in cohorts of ``parallel``; within a cohort every live game steps in lockstep (one batched
    MCTS search per ply over the live games) and a finished game drops out, so each ply is ONE batched net
    forward over the games still running. Each move records the visit-count policy target; the first
    ``temp_moves`` plies sample that distribution (opening diversity), the rest play greedily. ``examples``
    is the flat list of ``(obs, policy_target, value)`` already paired with the game outcome; ``returns`` is
    the per-game ``returns()`` array (for the games counter / logging).

    ``on_game_done(finished_so_far)`` fires as each game terminates (the trainer uses it to advance the live
    chart smoothly); ``should_stop`` aborts between plies, returning whatever finished so far (the caller
    discards a stopped iteration)."""
    examples: list[ValuedExample] = []
    all_returns: list[list[float]] = []
    finished = 0
    remaining = num_games
    cohort = 0
    while remaining > 0:
        if should_stop is not None and should_stop():
            return examples, all_returns
        size = min(parallel, remaining)
        games = [model.game.new_initial_state() for _ in range(size)]
        for g in games:
            _advance_chance(g, rng)
        # Per-game (obs, target, player) history, valued by the outcome once the game ends.
        history: list[list[tuple[np.ndarray, np.ndarray, int]]] = [[] for _ in range(size)]
        move_idx = [0] * size
        # A distinct rng per cohort keeps games reproducible from base_seed regardless of cohort order.
        crng = np.random.default_rng(base_seed + cohort)
        active = list(range(size))
        while active:
            if should_stop is not None and should_stop():
                return examples, all_returns
            dists = batched_search(
                model, [games[i] for i in active], sims, c_puct, dir_alpha, dir_frac, True, crng
            )
            still: list[int] = []
            for slot, i in enumerate(active):
                st = games[i]
                player = int(st.current_player())
                target = dists[slot]
                obs = np.asarray(st.observation_tensor(player), dtype=np.float32).reshape(
                    model.planes, model.rows, model.cols
                )
                history[i].append((obs, target.astype(np.float32), player))
                total = target.sum()
                if move_idx[i] < temp_moves and total > 0:
                    st.apply_action(int(crng.choice(model.n_actions, p=target / total)))
                else:
                    st.apply_action(int(target.argmax()))
                _advance_chance(st, crng)
                move_idx[i] += 1
                if st.is_terminal():
                    ret = list(st.returns())
                    all_returns.append(ret)
                    for obs_h, tgt_h, pl_h in history[i]:
                        examples.append((obs_h, tgt_h, float(ret[pl_h])))
                    finished += 1
                    if on_game_done is not None:
                        on_game_done(finished)
                else:
                    still.append(i)
            active = still
        remaining -= size
        cohort += 1
    return examples, all_returns


def eval_vs_mcts_parallel(
    model: Any,
    game: Any,
    ref_sims: int,
    n_games: int,
    eval_sims: int,
    c_puct: float,
    base_seed: int,
    should_stop: Callable[[], bool] | None = None,
) -> float:
    """Mean game result ∈ [−1, 1] for the AZ net (neural-MCTS, argmax) vs a fixed reference MCTS, with the
    net's moves **batched** across all eval games — the GPU-bound counterpart of ``board_engine.eval_vs_mcts``.

    The net's seat alternates per game (so both sides are measured); each ply, every game whose turn is the
    net's is searched in ONE batched, noise-free call (then plays the most-visited move), while the
    reference's turns are taken by a per-game ``RandomRollout`` MCTS on the CPU (the same yardstick the PPO
    baseline is scored against, so the curves stay comparable). ``should_stop`` aborts early; the partial
    mean is returned (and discarded by a stopping trainer), so correctness is unaffected."""
    rng = np.random.default_rng(base_seed)
    states = [game.new_initial_state() for _ in range(n_games)]
    for s in states:
        _advance_chance(s, rng)
    net_seat = [g % game.num_players() for g in range(n_games)]
    bots = [board_engine.MctsOpponent(game, ref_sims, base_seed + g) for g in range(n_games)]

    def _mean_done() -> float:
        done = [i for i in range(n_games) if states[i].is_terminal()]
        return float(np.mean([states[i].returns()[net_seat[i]] for i in done])) if done else 0.0

    active = [i for i in range(n_games) if not states[i].is_terminal()]
    while active:
        if should_stop is not None and should_stop():
            return _mean_done()
        net_games = [i for i in active if int(states[i].current_player()) == net_seat[i]]
        ref_games = [i for i in active if int(states[i].current_player()) != net_seat[i]]
        if net_games:
            dists = batched_search(
                model, [states[i] for i in net_games], eval_sims, c_puct, 0.0, 0.0, False, rng
            )
            for slot, i in enumerate(net_games):
                states[i].apply_action(int(dists[slot].argmax()))
                _advance_chance(states[i], rng)
        for i in ref_games:
            states[i].apply_action(int(bots[i].step(states[i])))
            _advance_chance(states[i], rng)
        active = [i for i in active if not states[i].is_terminal()]
    return float(np.mean([states[i].returns()[net_seat[i]] for i in range(n_games)]))


__all__ = ["ValuedExample", "batched_search", "eval_vs_mcts_parallel", "self_play_parallel"]
