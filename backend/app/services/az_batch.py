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
from collections.abc import Callable, Iterator
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


def _expand_roots_batch(model: Any, roots: list[_Node]) -> np.ndarray:
    """Expand every (non-terminal) root in ONE batched forward → the per-root value-head estimates.

    The first inference of a search. PUCT discards the returned values; Gumbel keeps them as each root's
    ``v_hat`` for the value-completion of unsearched moves (:meth:`_GumbelRoot._v_mix`)."""
    obs = [np.asarray(r.state.observation_tensor(r.to_play), dtype=np.float32) for r in roots]
    logits, values = model.infer_batch(np.stack(obs))
    for r, lg in zip(roots, logits, strict=True):
        _expand(r, lg)
    return values


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


# ---------------------------------------------------------------------------------------------------
# Gumbel AlphaZero (G6h) — a low-simulation root search that provably improves the policy.
#
# Standard PUCT MCTS needs *hundreds* of sims for a reliable visit-count target; with the few sims a
# desktop budget allows, that target is noisy. Gumbel AlphaZero (Danihelka et al., 2022) replaces the
# **root** action choice with **Gumbel-Top-k sampling + Sequential Halving** — a best-arm-identification
# bandit that improves the policy even at 8–32 sims — and trains on the **completed-Q improved policy**
# instead of raw visit counts. Only the root changes; inner nodes keep PUCT (:func:`_select_action`), and
# the batched inference-server pattern is preserved (one forward per simulation round over the cohort), so
# Gumbel self-play costs ~``sims`` forwards/move — *fewer* than PUCT — for an equal-or-better target.
#
# σ and the value-mixing follow the paper: ``σ(q) = (c_visit + max_b N(b))·c_scale·q`` monotonically maps
# a completed Q into a logit bump, and an unsearched action's Q is completed by ``v_mix`` (the visit-count-
# weighted blend of the root value estimate and the searched children's mean values). The persistent
# Gumbel sample ``g(a)`` injects the self-play exploration, so Gumbel mode needs no Dirichlet noise.
_C_VISIT = 50.0  # σ visit offset (paper default) — keeps σ(q) small until a node is well visited
_C_SCALE = 1.0  # σ scale (paper default)


def _sigma(q: float, max_visit: int) -> float:
    """The monotonic Q→logit transform ``(c_visit + max_b N(b))·c_scale·q`` (Gumbel AZ, eq. 8)."""
    return (_C_VISIT + max_visit) * _C_SCALE * q


def _descend_forced(
    root: _Node, root_action: int, c_puct: float, rng: np.random.Generator
) -> tuple[_Node, list[tuple[_Node, int]]]:
    """Descend a simulation that is **committed** to ``root_action`` at the root, then PUCT below it.

    Gumbel chooses the *root* edge (via Sequential Halving); everything beneath the root is the ordinary
    PUCT tree policy, so this forces the first edge and reuses :func:`_select_to_leaf` for the rest. The
    root child is created lazily on its first visit (so a root move's child is evaluated the first time
    Sequential Halving spends a simulation on it)."""
    child = root.children.get(root_action)
    if child is None:
        child = _Node(_child_state(root.state, root_action, rng))
        root.children[root_action] = child
    leaf, sub = _select_to_leaf(child, c_puct, rng)
    return leaf, [(root, root_action), *sub]


class _GumbelRoot:
    """One game's Gumbel-AlphaZero root planner, driven in lockstep across the cohort by a shared search.

    Holds the (already-expanded) root, its value-head estimate ``v_hat``, the persistent per-action Gumbel
    samples ``g(a)``, and the Sequential-Halving state: a ``considered`` set (the Gumbel-Top-``m`` actions),
    visited one-per-sweep and halved by the Gumbel score after each sweep until one move or the simulation
    budget remains. :meth:`next_action` yields the root action to simulate this round (or ``None`` when
    done); once the cohort's batched backups have landed, :meth:`policy_target` is the training target (the
    completed-Q improved policy) and :meth:`winner` the move actually played in self-play."""

    __slots__ = ("root", "v_hat", "legal", "logit", "gumbel", "considered", "sweep", "sims_left", "done")

    def __init__(
        self, root: _Node, v_hat: float, sims: int, max_considered: int, rng: np.random.Generator
    ) -> None:
        self.root = root
        self.v_hat = v_hat
        self.legal: list[int] = list(root.child_p.keys())
        # Effective logits = log(prior); softmax is shift-invariant so the missing log-partition constant
        # cancels in both the Gumbel-Top-k argmax and the completed-policy softmax. child_p > 0 (softmax).
        self.logit: dict[int, float] = {a: math.log(root.child_p[a]) for a in self.legal}
        g = rng.gumbel(size=len(self.legal))  # one persistent Gumbel sample per legal action
        self.gumbel: dict[int, float] = {a: float(g[k]) for k, a in enumerate(self.legal)}
        m = min(max_considered, len(self.legal))
        self.sims_left = sims
        if m <= 1:  # a forced (or no) move — no search needed; the target is a one-hot
            self.considered = list(self.legal)
            self.sweep: list[int] = []
            self.done = True
        else:  # the Gumbel-Top-m considered set, by g(a) + logit(a)
            order = sorted(self.legal, key=lambda a: self.gumbel[a] + self.logit[a], reverse=True)
            self.considered = order[:m]
            self.sweep = list(self.considered)
            self.done = False

    def _max_visit(self) -> int:
        return max((self.root.child_n.get(a, 0) for a in self.legal), default=0)

    def _q(self, a: int) -> float:
        n = self.root.child_n.get(a, 0)
        return self.root.child_w[a] / n if n > 0 else 0.0

    def _v_mix(self) -> float:
        """Value-completion for unsearched moves: the visit-count-weighted blend of ``v_hat`` and the
        searched children's mean values (Gumbel AZ, eq. 9). Falls back to ``v_hat`` before any visit."""
        visited = [a for a in self.legal if self.root.child_n.get(a, 0) > 0]
        sum_n = sum(self.root.child_n.get(a, 0) for a in self.legal)
        if not visited or sum_n == 0:
            return self.v_hat
        sum_pi = sum(self.root.child_p[a] for a in visited)
        weighted_q = sum(self.root.child_p[a] * self._q(a) for a in visited) / sum_pi
        return (self.v_hat + sum_n * weighted_q) / (1 + sum_n)

    def _completed_q(self, a: int, v_mix: float) -> float:
        return self._q(a) if self.root.child_n.get(a, 0) > 0 else v_mix

    def _score(self, a: int, v_mix: float, max_visit: int) -> float:
        """The Gumbel selection score ``g(a) + logit(a) + σ(completedQ(a))`` (root argmax + halving key)."""
        return self.gumbel[a] + self.logit[a] + _sigma(self._completed_q(a, v_mix), max_visit)

    def _halve(self) -> None:
        v_mix, max_visit = self._v_mix(), self._max_visit()
        self.considered.sort(key=lambda a: self._score(a, v_mix, max_visit), reverse=True)
        self.considered = self.considered[: max(1, (len(self.considered) + 1) // 2)]

    def next_action(self) -> int | None:
        """The root action to commit this round's simulation to, or ``None`` once this game is done."""
        if self.done:
            return None
        if not self.sweep:  # a full sweep finished → Sequential-Halving step, then the next sweep
            if len(self.considered) <= 1:
                self.done = True
                return None
            self._halve()
            if len(self.considered) <= 1:
                self.done = True
                return None
            self.sweep = list(self.considered)
        if self.sims_left <= 0:
            self.done = True
            return None
        self.sims_left -= 1
        return self.sweep.pop(0)

    def winner(self) -> int:
        """The move played in self-play = the highest-scoring survivor (the Sequential-Halving winner)."""
        pool = self.considered or self.legal
        if not pool:
            return 0
        v_mix, max_visit = self._v_mix(), self._max_visit()
        return max(pool, key=lambda a: self._score(a, v_mix, max_visit))

    def policy_target(self, n_actions: int) -> np.ndarray:
        """The training target = the completed-Q improved policy ``softmax_a(logit(a) + σ(completedQ(a)))``
        over **all** legal moves (unsearched ones completed by ``v_mix``), placed in an ``n_actions`` vector."""
        out = np.zeros(n_actions, dtype=np.float64)
        if not self.legal:
            return out
        v_mix, max_visit = self._v_mix(), self._max_visit()
        s = np.array(
            [self.logit[a] + _sigma(self._completed_q(a, v_mix), max_visit) for a in self.legal]
        )
        s -= s.max()
        ex = np.exp(s)
        probs = ex / ex.sum()
        for a, p in zip(self.legal, probs, strict=True):
            out[a] = p
        return out


def batched_gumbel_search(
    model: Any,
    states: list[Any],
    sims: int,
    max_considered: int,
    c_puct: float,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[int]]:
    """Gumbel-AlphaZero root search over a cohort → ``(improved_policy_targets, chosen_moves)`` per state.

    Each non-terminal ``state`` roots its own :class:`_GumbelRoot` planner; the cohort runs in lockstep
    simulation rounds, each gathering one committed-root leaf per still-searching game into ONE batched
    forward (the same inference-server pattern as :func:`batched_search`). ``sims`` is the per-move
    simulation budget (Gumbel reaches a good move at far fewer than PUCT), ``max_considered`` the number of
    root moves Sequential Halving picks among (capped by the legal-move count). Returns the completed-Q
    improved policy (the training target, sums to 1, mass on legal moves) and the Sequential-Halving winner
    (the self-play move) for each input state; the inputs are not mutated (each tree roots on a clone)."""
    roots = [_Node(s.clone()) for s in states]
    values = _expand_roots_batch(model, roots)
    planners = [
        _GumbelRoot(root, float(v), sims, max_considered, rng)
        for root, v in zip(roots, values, strict=True)
    ]
    while True:
        pending: list[tuple[_Node, list[tuple[_Node, int]]]] = []
        active = False
        for planner in planners:
            a = planner.next_action()
            if a is None:
                continue
            active = True
            leaf, path = _descend_forced(planner.root, a, c_puct, rng)
            if leaf.state.is_terminal():
                _backup(path, 0.0, 0, list(leaf.state.returns()))
            else:
                pending.append((leaf, path))
        if not active:
            break
        if pending:
            obs = np.stack(
                [np.asarray(lf.state.observation_tensor(lf.to_play), dtype=np.float32) for lf, _ in pending]
            )
            logits, vals = model.infer_batch(obs)
            for k, (leaf, path) in enumerate(pending):
                _expand(leaf, logits[k])
                _backup(path, float(vals[k]), leaf.to_play, None)
    policies = [p.policy_target(model.n_actions) for p in planners]
    winners = [p.winner() for p in planners]
    return policies, winners


def self_play_rolling(
    model: Any,
    parallel: int,
    sims: int,
    c_puct: float,
    dir_alpha: float,
    dir_frac: float,
    temp_moves: int,
    rng: np.random.Generator,
    gumbel: bool = False,
    gumbel_considered: int = 16,
    max_game_plies: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    max_started: int | None = None,
) -> Iterator[tuple[list[ValuedExample], list[float]]]:
    """Yield finished self-play games one-by-one from a **rolling** cohort — the continuous-flow generator
    behind both the bounded :func:`self_play_parallel` and the background self-play actor (G6h).

    A synchronous cohort starts all games together and they finish together, so the games counter lurches
    in bursts (and the GPU batch shrinks to nothing at the end of each cohort). This instead keeps the batch
    topped up to ``parallel``: each finished game is yielded immediately and replaced by a fresh one, so
    games complete **continuously** (a smooth counter) and the GPU batch stays full. The (re)fill is
    staggered a few games per ply so replacements — and the very first cohort — don't re-synchronise.

    Runs until ``should_stop`` (or, with ``max_started`` set, until that many games have started and the
    in-flight ones drained — the bounded mode). ``model`` is read fresh each ply, so the caller may hot-swap
    its weights between yields (the actor picks up the learner's latest net this way). Each ply is ONE
    batched forward; search mode (Gumbel vs PUCT) and ``max_game_plies`` match :func:`self_play_parallel`."""
    slots: list[dict[str, Any]] = []  # live games: {"state", "history": [...], "ply", "cap"}
    started = 0
    per_ply_fill = max(1, parallel // 8)  # stagger the (re)fill so completions don't re-synchronise

    def _new_game() -> dict[str, Any]:
        st = model.game.new_initial_state()
        _advance_chance(st, rng)
        # Per-game ply cap with jitter: a weak early net rarely converts an unbounded game (chess), so
        # without this they'd all hit the SAME cap on (nearly) the same ply → completions re-synchronise
        # into a burst (the lockstep problem the rolling cohort exists to avoid). Jittering the cap ±25 %
        # spreads those forced draws out, keeping the counter smooth. None ⇒ uncapped (small bounded games).
        cap = max_game_plies - int(rng.integers(0, max(1, max_game_plies // 4))) if max_game_plies else None
        return {"state": st, "history": [], "ply": 0, "cap": cap}

    while should_stop is None or not should_stop():
        room = parallel - len(slots)
        if max_started is not None:
            room = min(room, max_started - started)
        for _ in range(min(per_ply_fill, max(0, room))):
            slots.append(_new_game())
            started += 1
        if not slots:
            if max_started is not None and started >= max_started:
                return  # bounded mode: every game started and drained
            continue
        live = [s["state"] for s in slots]
        if gumbel:
            dists, winners = batched_gumbel_search(model, live, sims, gumbel_considered, c_puct, rng)
        else:
            dists = batched_search(model, live, sims, c_puct, dir_alpha, dir_frac, True, rng)
        survivors: list[dict[str, Any]] = []
        for k, s in enumerate(slots):
            st = s["state"]
            player = int(st.current_player())
            target = dists[k]
            obs = np.asarray(st.observation_tensor(player), dtype=np.float32).reshape(
                model.planes, model.rows, model.cols
            )
            s["history"].append((obs, target.astype(np.float32), player))
            if gumbel:
                st.apply_action(int(winners[k]))  # Sequential-Halving winner (Gumbel noise = exploration)
            elif s["ply"] < temp_moves and (total := target.sum()) > 0:
                st.apply_action(int(rng.choice(model.n_actions, p=target / total)))
            else:
                st.apply_action(int(target.argmax()))
            _advance_chance(st, rng)
            s["ply"] += 1
            # End on a natural terminal OR the ply cap. The cap bounds the marathon games an unbounded game
            # (chess) produces while the net is weak: a near-random early policy rarely converts, so games
            # otherwise drag to the 75-move forced-draw ceiling. A capped game scores 0 (a draw — the honest
            # label for an unresolved position heading there anyway). No effect on bounded games (TTT/C4).
            capped = s["cap"] is not None and s["ply"] >= s["cap"]
            if st.is_terminal() or capped:
                ret = list(st.returns()) if st.is_terminal() else [0.0] * model.num_players
                yield [(o, t, float(ret[p])) for o, t, p in s["history"]], ret
            else:
                survivors.append(s)
        slots = survivors


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
    gumbel: bool = False,
    gumbel_considered: int = 16,
    max_game_plies: int | None = None,
) -> tuple[list[ValuedExample], list[list[float]]]:
    """Generate exactly ``num_games`` self-play games (the bounded :func:`self_play_rolling`) → (examples,
    returns).

    The rolling cohort completes games one-by-one and replaces them (no lockstep bursts, full GPU batch),
    bounded to ``num_games``. ``examples`` is the flat outcome-valued ``(obs, policy_target, value)`` list;
    ``returns`` the per-game ``returns()`` array. ``on_game_done(finished_so_far)`` fires per completion;
    ``should_stop`` aborts between plies (returning what finished so far). Search mode (Gumbel/PUCT),
    ``temp_moves`` and ``max_game_plies`` are documented on :func:`self_play_rolling`. The per-call rng is
    seeded from ``base_seed`` (deterministic regardless of the shared ``rng``'s state)."""
    crng = np.random.default_rng(base_seed)
    examples: list[ValuedExample] = []
    all_returns: list[list[float]] = []
    for finished, (game_examples, ret) in enumerate(
        self_play_rolling(
            model, min(parallel, num_games), sims, c_puct, dir_alpha, dir_frac, temp_moves, crng,
            gumbel=gumbel, gumbel_considered=gumbel_considered, max_game_plies=max_game_plies,
            should_stop=should_stop, max_started=num_games,
        ),
        start=1,
    ):
        examples.extend(game_examples)
        all_returns.append(ret)
        if on_game_done is not None:
            on_game_done(finished)
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
    max_game_plies: int | None = None,
    unresolved_value: Callable[[Any, int], float] | None = None,
) -> float:
    """Mean game result ∈ [−1, 1] for the AZ net (neural-MCTS, argmax) vs a fixed reference MCTS, with the
    net's moves **batched** across all eval games — the GPU-bound counterpart of ``board_engine.eval_vs_mcts``.

    The net's seat alternates per game (so both sides are measured); each ply, every game whose turn is the
    net's is searched in ONE batched, noise-free call (then plays the most-visited move), while the
    reference's turns are taken by a per-game ``RandomRollout`` MCTS on the CPU (the same yardstick the PPO
    baseline is scored against, so the curves stay comparable). ``should_stop`` aborts early; the partial
    mean is returned (and discarded by a stopping trainer), so correctness is unaffected. ``max_game_plies``
    (chess) ends an over-long eval game without a result so one marathon game can't stall the whole eval;
    such a capped game is scored by ``unresolved_value(state, net_seat)`` (chess → material balance) instead
    of a flat 0 — otherwise a weak early net, which rarely mates inside the cap, reads a permanent 0.0 (G6h).
    ``unresolved_value=None`` keeps the prior behavior (a capped game is a 0 draw)."""
    rng = np.random.default_rng(base_seed)
    states = [game.new_initial_state() for _ in range(n_games)]
    for s in states:
        _advance_chance(s, rng)
    net_seat = [g % game.num_players() for g in range(n_games)]
    bots = [board_engine.MctsOpponent(game, ref_sims, base_seed + g) for g in range(n_games)]
    plies = [0] * n_games

    def _done(i: int) -> bool:  # terminal, or capped (an over-long game counts as an unresolved draw)
        return states[i].is_terminal() or (max_game_plies is not None and plies[i] >= max_game_plies)

    def _result(i: int) -> float:  # net's score: real outcome if terminal, else the capped-game value
        if states[i].is_terminal():
            return float(states[i].returns()[net_seat[i]])
        return unresolved_value(states[i], net_seat[i]) if unresolved_value is not None else 0.0

    def _mean_done() -> float:
        done = [i for i in range(n_games) if _done(i)]
        return float(np.mean([_result(i) for i in done])) if done else 0.0

    active = [i for i in range(n_games) if not _done(i)]
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
                plies[i] += 1
        for i in ref_games:
            states[i].apply_action(int(bots[i].step(states[i])))
            _advance_chance(states[i], rng)
            plies[i] += 1
        active = [i for i in active if not _done(i)]
    return float(np.mean([_result(i) for i in range(n_games)]))


__all__ = [
    "ValuedExample",
    "batched_gumbel_search",
    "batched_search",
    "eval_vs_mcts_parallel",
    "self_play_parallel",
    "self_play_rolling",
]
