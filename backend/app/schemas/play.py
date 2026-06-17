"""Play-session contracts — defined once here (pydantic), mirrored in frontend/src/api/types.ts.

A *play session* is one interactive episode of a human-playable env, controlled either by a
human (actions arrive over WS as ``{type:"action"}``) or by a loaded checkpoint (AI watch).
It is deliberately game-agnostic: the same shapes carry CartPole today and slot a competitive
game like Pong in later (a future ``side`` field would pick which paddle the human drives).

Frames over WS:
* ``{type:"play_status"}`` — lifecycle snapshot (also returned by ``/api/play/*``)
* ``{type:"play_frame"}``  — one rendered episode image (base64 JPEG)
* ``{type:"play_result"}`` — the final score + :class:`~app.schemas.skill.SkillRating`
"""

from typing import Literal

from pydantic import BaseModel

from app.schemas.preview import BoardState, GridLayout
from app.schemas.skill import SkillRating

# Who controls the agent: a human at the keyboard, or a loaded checkpoint playing itself.
PlayMode = Literal["human", "ai"]
PlayState = Literal["idle", "playing", "finished", "stopped", "error"]
# Board-game (G6a) AI opponent strength → MCTS simulation count (see board_engine.STRENGTH_SIMS).
BoardStrength = Literal["easy", "medium", "hard"]

# BoardState moved to app.schemas.preview (a render payload shared by play_frame + the training-preview
# frame, G6b); re-exported here so existing ``from app.schemas.play import BoardState`` imports still work.
__all__ = ["BoardState", "PlayConfig", "PlayResult", "PlayStatus", "PlayFrame", "PlayActionMessage"]


class PlayConfig(BaseModel):
    """Start request for ``POST /api/play/start``.

    ``checkpoint_id`` is required for ``mode="ai"`` (the model that plays) and ignored for
    ``mode="human"``. ``seed`` makes the episode reproducible; ``None`` lets the env pick.
    ``speed`` (1×–20×) paces playback exactly like the training preview.

    ``idle_action`` is the action the session holds when the human gives no input — the env's
    "do nothing" (MountainCar/Acrobot 1 = no force/torque, LunarLander 0 = no thrust, a continuous
    env 0 = no torque/force). The keymap (frontend/src/content/playKeymaps.ts) is its source of
    truth; ``None`` means the env has no idle (CartPole always moves) and the session falls back to
    action 0. A float / list of floats carries a continuous (box) idle command.
    """

    env_id: str = "cartpole"
    mode: PlayMode = "human"
    checkpoint_id: str | None = None
    seed: int | None = None
    speed: float = 1.0
    idle_action: int | float | list[float] | None = None
    # Board games (G6a) only: which player the human controls (0 = first to move) and how strong the
    # MCTS opponent is. Ignored by every other env. ``mode="ai"`` on a board env is an AI-vs-AI watch.
    side: int = 0
    ai_strength: BoardStrength = "medium"


class PlaySpeedRequest(BaseModel):
    """Body for ``POST /api/play/speed`` — change a live session's playback pacing (the speed
    selector while a session is running). Clamped to the play range server-side."""

    speed: float


class PlayResult(BaseModel):
    """Final outcome of a finished session — pushed over WS as {type:"play_result", ...}."""

    type: Literal["play_result"] = "play_result"
    env_id: str
    mode: PlayMode
    score: float
    steps: int
    # Continuous-score envs (CartPole, …) carry a skill rating; board games (G6a) are 3-valued
    # (win/draw/loss vs an AI), not a continuous return, so they carry ``outcome`` and leave
    # ``rating`` null — the UI shows a W/D/L card instead of the misleading continuous skill %.
    rating: SkillRating | None = None
    outcome: Literal["win", "draw", "loss"] | None = None


class PlayStatus(BaseModel):
    """Lifecycle snapshot — returned by ``/api/play/*`` and pushed as {type:"play_status", ...}."""

    type: Literal["play_status"] = "play_status"
    state: PlayState
    env_id: str | None = None
    mode: PlayMode | None = None
    checkpoint_id: str | None = None
    seed: int | None = None
    speed: float = 1.0
    step: int = 0
    score: float = 0.0
    result: PlayResult | None = None  # set once the episode ends
    error: str | None = None


class PlayFrame(BaseModel):
    """One rendered episode frame, pushed over WS as {type:"play_frame", ...}.

    Same ``image`` convention as the training preview's :class:`~app.schemas.preview.FrameMessage`
    (base64 JPEG, no data-URI prefix); a distinct ``type`` so the client can tell a play frame
    from a training-preview frame. Built by hand in the play loop, but this model is the contract.
    """

    type: Literal["play_frame"] = "play_frame"
    step: int
    score: float
    # Either a server-rendered image (width/height/image) OR client-render state — never both.
    width: int | None = None
    height: int | None = None
    image: str | None = None
    state: list[float] | None = None  # client-render state (e.g. CartPole [x, theta]), drawn client-side
    action: int | None = None  # the discrete action just applied (lets the client draw the firing thruster)
    # Per-episode scene geometry the client can't derive from the obs — currently LunarLander's random
    # moon surface as obs-space [x, y] points (None for envs whose scene is fixed/derivable).
    terrain: list[list[float]] | None = None
    # Static board layout for a grid-world (Toy Text), so the client can draw the board (None elsewhere).
    grid: GridLayout | None = None
    # Board-game state (G6a) for the client-side board renderer; None for every non-board env.
    board: BoardState | None = None


class PlayActionMessage(BaseModel):
    """Inbound human input over WS: {type:"action", action:<int|float|number[]>}.

    ``action`` is a discrete action index for a discrete env (CartPole: 0=left, 1=right) or a
    continuous command — a float / list of floats — for a box env (Pendulum: a torque in [-2, 2]).
    Latency-tolerant — the session holds the latest received action and reuses it until the next
    one arrives, so dropped/late frames just repeat the prior input rather than stall.
    """

    type: Literal["action"] = "action"
    action: int | float | list[float]
