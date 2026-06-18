"""AlphaZero-lite engine (G6f, ADR-055) — a PyTorch CNN policy+value net guided by OpenSpiel MCTS.

The 4th learning algorithm (``alphazero``) and the *algorithm jump* of the board branch. G6b's
MaskablePPO-vs-MCTS learns *small* games but plateaus; the AlphaZero recipe reaches stronger play with
a **CNN policy+value net** that **guides Monte-Carlo Tree Search** (replacing G6a's random rollouts),
trained on **self-play** data (MCTS visit counts → the policy target, the game outcome → the value
target).

The risk-gate (``Local/_probe_g6f.py``) settled the build: rather than OpenSpiel's TensorFlow AlphaZero
reference (a heavyweight, Windows-hostile dependency), this plugs a **PyTorch** net into OpenSpiel's
**already-working** :class:`mcts.MCTSBot` as a neural :class:`mcts.Evaluator` — PUCT child selection +
Dirichlet root noise + visit-count policy targets. So the proven G6a MCTS is reused; only the random
rollout becomes a neural forward. GPU/CUDA when available (the G4b device path), CPU otherwise.

Game-agnostic: every board game whose ``observation_tensor`` is a ``(planes, rows, cols)`` stack (every
2-player OpenSpiel board game — Tic-Tac-Toe ``[3,3,3]``, Connect Four ``[3,6,7]``, Breakthrough/Othello
``[3,8,8]``) is a CNN input with zero engine changes. The trained net exposes the **same**
``(flat_obs, legal_mask) -> action`` predict shape as the G6b MaskablePPO net
(:data:`app.services.board_engine.MaskedPredictFn`), so it reuses ``board_engine.eval_vs_mcts`` /
``board_move_fn`` and the live-preview + play-vs-net + Save/Load lanes **unchanged** — the AZ-specific
work is confined to this module + :mod:`app.services.trainer_az`.

Kept import-light: ``torch`` is imported lazily inside the factories/methods (the net class is defined
inside a cached factory, mirroring ``board_engine._self_play_env_class``), so importing this module —
and the hot ``is_board_game`` check that may pull it — stays cheap until a run actually starts.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from typing import Any

import numpy as np

# A move over a finished self-play game: the canonical (player-perspective) observation planes, the MCTS
# visit-count policy target over all actions, and the player to move — paired with the game outcome later.
SelfPlayExample = tuple[np.ndarray, np.ndarray, int]

_NET_CLASS: Any = None


def _net_class() -> Any:
    """Lazily define + cache the AlphaZero CNN (keeps ``torch`` off this module's import path).

    A small **residual conv tower** (BatchNorm-free, so single-position self-play forwards and batched
    training share one mode — none of the BN train/eval running-stat pitfalls) feeding two heads: a
    **policy** head (logits over every action) and a **value** head (``tanh`` ∈ [−1, 1]). Sized for
    small boards; ``channels``/``blocks`` are tunable. Mirrors ``board_engine._self_play_env_class``.
    """
    global _NET_CLASS
    if _NET_CLASS is not None:
        return _NET_CLASS

    import torch
    from torch import nn

    class _ResBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, c: int) -> None:
            super().__init__()
            self.c1 = nn.Conv2d(c, c, 3, padding=1)
            self.c2 = nn.Conv2d(c, c, 3, padding=1)

        def forward(self, x: Any) -> Any:
            z = torch.relu(self.c1(x))
            z = self.c2(z)
            return torch.relu(x + z)

    class AZNet(nn.Module):  # type: ignore[misc]
        def __init__(
            self, planes: int, h: int, w: int, n_actions: int, channels: int, blocks: int
        ) -> None:
            super().__init__()
            self.stem = nn.Conv2d(planes, channels, 3, padding=1)
            self.tower = nn.ModuleList([_ResBlock(channels) for _ in range(blocks)])
            self.policy = nn.Linear(channels * h * w, n_actions)
            self.value_hidden = nn.Linear(channels * h * w, channels)
            self.value_out = nn.Linear(channels, 1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            z = torch.relu(self.stem(x))
            for block in self.tower:
                z = block(z)
            z = z.flatten(1)
            policy_logits = self.policy(z)
            value = torch.tanh(self.value_out(torch.relu(self.value_hidden(z))))
            return policy_logits, value

    _NET_CLASS = AZNet
    return _NET_CLASS


class AZModel:
    """The CNN + its board geometry — builds the MCTS evaluator, self-play games, and predict fns.

    Owns one net on ``device`` (``"cuda"`` when available). The geometry (``planes × rows × cols``,
    ``n_actions``) is read from the OpenSpiel game once, so reshaping a flat ``observation_tensor`` into
    CNN planes and serializing/rebuilding the net for a checkpoint are both game-agnostic.
    """

    def __init__(
        self, game: Any, channels: int = 64, blocks: int = 4, device: str = "cpu"
    ) -> None:
        shape = game.observation_tensor_shape()
        if len(shape) != 3:  # AZ needs a (planes, rows, cols) board stack — every 2-player board game has one
            raise ValueError(f"AlphaZero needs a 3D observation tensor, got shape {list(shape)}")
        self.game = game
        self.planes, self.rows, self.cols = int(shape[0]), int(shape[1]), int(shape[2])
        self.n_actions = int(game.num_distinct_actions())
        self.num_players = int(game.num_players())
        self.channels = channels
        self.blocks = blocks
        self.device = device
        self.net = _net_class()(
            self.planes, self.rows, self.cols, self.n_actions, channels, blocks
        ).to(device)

    # -- serialization (the decoupled CNN snapshot — ADR-019/044) ----------------------------------

    def state_blob(self) -> bytes:
        """Serialize the net **with its architecture** so :meth:`load_blob` can rebuild it standalone
        (a saved ``board.zip``-equivalent for AZ). Carries the geometry the checkpoint loader needs."""
        import torch

        buf = io.BytesIO()
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
                "planes": self.planes, "rows": self.rows, "cols": self.cols,
                "n_actions": self.n_actions, "num_players": self.num_players,
                "channels": self.channels, "blocks": self.blocks,
            },
            buf,
        )
        return buf.getvalue()

    def cpu_copy_net(self) -> Any:
        """An independent **CPU** copy of the net (no shared tensor storage with the live GPU model),
        for the decoupled preview/play snapshot — forwarding it can never perturb training (ADR-019)."""
        clone = _net_class()(
            self.planes, self.rows, self.cols, self.n_actions, self.channels, self.blocks
        )
        clone.load_state_dict({k: v.cpu() for k, v in self.net.state_dict().items()})
        clone.eval()
        return clone


def load_az_model(blob: bytes, device: str = "cpu") -> Any:
    """Load the raw AZ blob dict (state_dict + geometry + games_played) onto ``device``."""
    import torch

    return torch.load(io.BytesIO(blob), map_location=device, weights_only=False)


def build_model_from_blob(blob: bytes, game: Any, device: str = "cpu") -> tuple[AZModel, int]:
    """Rebuild a full :class:`AZModel` (net weights loaded, eval mode) from a saved AZ blob + its game.

    Returns ``(model, games_played)`` — used by the trainer's resume path and by Play/Watch to build a
    neural-MCTS opponent. The blob carries the architecture, so the rebuilt net matches the saved one."""
    data = load_az_model(blob, device=device)
    model = AZModel(game, channels=data["channels"], blocks=data["blocks"], device=device)
    model.net.load_state_dict(data["state_dict"])
    model.net.eval()
    return model, int(data.get("games_played", 0))


# ---------------------------------------------------------------------------------------------------
# Inference predict fns — the same (flat_obs, legal_mask) -> action shape as board_engine.MaskedPredictFn,
# so the trained AZ net drops into board_engine.eval_vs_mcts / board_move_fn + the preview/play lanes.
# ---------------------------------------------------------------------------------------------------


def _wrap_net_predict(
    net: Any, planes: int, rows: int, cols: int, deterministic: bool
) -> Callable[[Any, Any], int]:
    """A masked ``predict(flat_obs, mask) -> legal action`` over a (CPU) net's **policy head**.

    Reshapes the flat ``observation_tensor`` into CNN planes, masks illegal moves to ``-inf``, then
    argmaxes (deterministic, for play/eval) or samples the softmax (non-deterministic, so the live
    preview's self-play games vary). The value head is unused at inference — a well-trained AZ policy
    head already approximates the MCTS-improved move, which is strong enough to play against directly
    (running MCTS at inference would be stronger still; left as a future refinement)."""
    import torch

    def predict(obs: Any, mask: Any) -> int:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).reshape(1, planes, rows, cols)
        with torch.no_grad():
            logits, _ = net(x)
        logits = logits[0].numpy().astype(np.float64)
        mask_arr = np.asarray(mask, dtype=bool)
        logits[~mask_arr] = -np.inf
        if deterministic:
            return int(np.argmax(logits))
        # Softmax sample over the legal moves (numerically stable).
        z = logits - np.max(logits)
        probs = np.exp(z)
        probs[~mask_arr] = 0.0
        total = probs.sum()
        if total <= 0:  # degenerate (shouldn't happen) — fall back to a uniform legal pick
            legal = np.flatnonzero(mask_arr)
            return int(np.random.choice(legal))
        return int(np.random.choice(len(probs), p=probs / total))

    return predict


def build_az_predict(model: AZModel, deterministic: bool = True) -> Callable[[Any, Any], int]:
    """A decoupled masked predict fn over a **CPU snapshot** of the live AZ net (ADR-019-safe)."""
    return _wrap_net_predict(model.cpu_copy_net(), model.planes, model.rows, model.cols, deterministic)


def load_az_predict(blob: bytes, deterministic: bool = True) -> Callable[[Any, Any], int]:
    """Load a saved AZ checkpoint blob for inference only → ``predict(flat_obs, mask) -> action``.

    Rebuilds the net standalone from the blob's geometry (no pyspiel game needed here — the caller
    supplies the obs + legal mask per move), the AZ parallel to ``board_engine.load_board_predict``."""
    data = load_az_model(blob, device="cpu")
    net = _net_class()(
        data["planes"], data["rows"], data["cols"], data["n_actions"],
        data["channels"], data["blocks"],
    )
    net.load_state_dict(data["state_dict"])
    net.eval()
    return _wrap_net_predict(net, data["planes"], data["rows"], data["cols"], deterministic)


# ---------------------------------------------------------------------------------------------------
# Neural-guided MCTS self-play (the data generator) + the supervised net update.
# ---------------------------------------------------------------------------------------------------

_EVALUATOR_CLASS: Any = None


def _evaluator_class() -> Any:
    """Lazily define the neural :class:`mcts.Evaluator` (keeps open_spiel.mcts off the import path)."""
    global _EVALUATOR_CLASS
    if _EVALUATOR_CLASS is not None:
        return _EVALUATOR_CLASS

    from open_spiel.python.algorithms import mcts

    class _NeuralEvaluator(mcts.Evaluator):  # type: ignore[misc]
        """Feeds the CNN's value + policy into OpenSpiel MCTS in place of a random rollout.

        ``evaluate`` returns the zero-sum per-player value array MCTS backpropagates; ``prior`` returns
        the softmaxed legal-move policy MCTS uses for PUCT. Both come from ONE forward, cached by board
        string within a search so the back-to-back evaluate→prior calls on a freshly expanded leaf — and
        any transposition — share a single GPU forward (a large speedup over two forwards per node)."""

        def __init__(self, model: AZModel) -> None:
            self._model = model
            self._cache: dict[str, tuple[np.ndarray, float]] = {}

        def _infer(self, state: Any) -> tuple[np.ndarray, float, int]:
            import torch

            player = int(state.current_player())
            key = str(state)
            hit = self._cache.get(key)
            if hit is not None:
                return hit[0], hit[1], player
            obs = np.asarray(state.observation_tensor(player), dtype=np.float32)
            x = torch.as_tensor(obs).reshape(
                1, self._model.planes, self._model.rows, self._model.cols
            ).to(self._model.device)
            with torch.no_grad():
                logits, value = self._model.net(x)
            logits_np = logits[0].detach().cpu().numpy().astype(np.float64)
            v = float(value.item())
            if len(self._cache) < 50_000:  # bounded; cleared per game by the self-play loop
                self._cache[key] = (logits_np, v)
            return logits_np, v, player

        def evaluate(self, state: Any) -> np.ndarray:
            logits, v, player = self._infer(state)
            arr = np.zeros(self._model.num_players, dtype=np.float64)
            arr[player] = v
            arr[1 - player] = -v  # 2-player zero-sum (board games are all 2-player here)
            return arr

        def prior(self, state: Any) -> list[tuple[int, float]]:
            if state.is_chance_node():
                return state.chance_outcomes()
            logits, _v, player = self._infer(state)
            legal = state.legal_actions(player)
            sub = logits[legal]
            sub = np.exp(sub - sub.max())
            probs = sub / sub.sum()
            return list(zip(legal, probs, strict=True))

    _EVALUATOR_CLASS = _NeuralEvaluator
    return _EVALUATOR_CLASS


def self_play_game(
    model: AZModel,
    sims: int,
    c_puct: float,
    dirichlet_alpha: float,
    dirichlet_frac: float,
    temp_moves: int,
    rng: np.random.Generator,
    seed: int,
) -> tuple[list[SelfPlayExample], list[float]]:
    """Play one neural-guided-MCTS self-play game → ``(examples, game_returns)``.

    Each move runs ``sims`` PUCT simulations (Dirichlet noise at the root for exploration), records the
    **visit-count distribution** as the policy target, then picks the move by sampling that distribution
    for the first ``temp_moves`` plies (opening diversity) and greedily after. ``game_returns`` is
    ``state.returns()``; the trainer pairs each example's player with its outcome for the value target."""
    from open_spiel.python.algorithms import mcts

    evaluator = _evaluator_class()(model)
    bot = mcts.MCTSBot(
        model.game,
        uct_c=c_puct,
        max_simulations=sims,
        evaluator=evaluator,
        solve=False,  # visit counts (not a solved one-hot) are the AZ policy target
        dirichlet_noise=(dirichlet_alpha, dirichlet_frac),
        random_state=np.random.RandomState(seed),
        child_selection_fn=mcts.SearchNode.puct_value,
    )
    examples: list[SelfPlayExample] = []
    state = model.game.new_initial_state()
    move_idx = 0
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            state.apply_action(int(rng.choice([a for a, _ in outcomes])))
            continue
        player = int(state.current_player())
        root = bot.mcts_search(state)
        visits = np.zeros(model.n_actions, dtype=np.float64)
        for child in root.children:
            visits[child.action] = child.explore_count
        total = visits.sum()
        target = visits / total if total > 0 else visits
        obs = np.asarray(state.observation_tensor(player), dtype=np.float32).reshape(
            model.planes, model.rows, model.cols
        )
        examples.append((obs, target.astype(np.float32), player))
        if move_idx < temp_moves and total > 0:
            action = int(rng.choice(model.n_actions, p=target))
        else:
            action = int(target.argmax())
        state.apply_action(action)
        move_idx += 1
    return examples, list(state.returns())


def az_move_fn(
    model: AZModel, sims: int, c_puct: float = 2.0, seed: int | None = 0
) -> Callable[[Any], int]:
    """A strong ``(state) -> action`` move via **neural-guided MCTS** — the net's *real* playing strength.

    Where ``build_az_predict`` returns the raw policy head (one forward, used for the fast cosmetic
    preview), this runs ``sims`` of MCTS with the CNN as the evaluator and returns the most-visited
    move — the authentic AlphaZero inference, much stronger than the bare policy head. Used for the
    honest eval-vs-reference curve and for the human's Play-vs-net opponent. ``seed=None`` gives a
    varied opponent each game (the human-play convention); a fixed seed is a reproducible eval.

    The bot is built once and reused across moves (MCTS rebuilds its tree each call), and the
    evaluator's per-search forward cache carries transpositions — so this is as cheap as ``sims``
    net forwards per move, minus cache hits."""
    import numpy as np
    from open_spiel.python.algorithms import mcts

    evaluator = _evaluator_class()(model)
    bot = mcts.MCTSBot(
        model.game,
        uct_c=c_puct,
        max_simulations=sims,
        evaluator=evaluator,
        solve=False,
        random_state=np.random.RandomState(seed),
        child_selection_fn=mcts.SearchNode.puct_value,
    )

    def move(state: Any) -> int:
        return int(bot.step(state))

    return move


def train_on_buffer(
    model: AZModel,
    optimizer: Any,
    buffer: list[tuple[np.ndarray, np.ndarray, float]],
    batch_size: int,
    steps: int,
    rng: np.random.Generator,
) -> float:
    """Run ``steps`` minibatch SGD updates on the replay ``buffer`` → mean loss.

    Loss = policy cross-entropy against the MCTS visit distribution (illegal moves carry 0 target mass,
    so the full-softmax CE ignores them) + value MSE against the game outcome — the AlphaZero objective.
    """
    import torch
    import torch.nn.functional as F

    if not buffer:
        return 0.0
    model.net.train()
    n = len(buffer)
    total_loss = 0.0
    for _ in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        obs = torch.as_tensor(
            np.stack([buffer[i][0] for i in idx]), dtype=torch.float32, device=model.device
        )
        target_pi = torch.as_tensor(
            np.stack([buffer[i][1] for i in idx]), dtype=torch.float32, device=model.device
        )
        target_v = torch.as_tensor(
            np.array([buffer[i][2] for i in idx], dtype=np.float32), device=model.device
        )
        logits, value = model.net(obs)
        log_probs = F.log_softmax(logits, dim=1)
        policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value.squeeze(1), target_v)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    model.net.eval()  # leave it in eval mode for the next self-play / snapshot
    return total_loss / max(1, steps)


def best_device() -> str:
    """``"cuda"`` if a GPU is available (the G4b device path), else ``"cpu"``."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = [
    "AZModel",
    "SelfPlayExample",
    "az_move_fn",
    "best_device",
    "build_az_predict",
    "build_model_from_blob",
    "load_az_model",
    "load_az_predict",
    "self_play_game",
    "train_on_buffer",
]
