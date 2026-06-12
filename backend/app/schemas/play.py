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

from app.schemas.skill import SkillRating

# Who controls the agent: a human at the keyboard, or a loaded checkpoint playing itself.
PlayMode = Literal["human", "ai"]
PlayState = Literal["idle", "playing", "finished", "stopped", "error"]


class PlayConfig(BaseModel):
    """Start request for ``POST /api/play/start``.

    ``checkpoint_id`` is required for ``mode="ai"`` (the model that plays) and ignored for
    ``mode="human"``. ``seed`` makes the episode reproducible; ``None`` lets the env pick.
    ``speed`` (1×–20×) paces playback exactly like the training preview.
    """

    env_id: str = "cartpole"
    mode: PlayMode = "human"
    checkpoint_id: str | None = None
    seed: int | None = None
    speed: float = 1.0


class PlayResult(BaseModel):
    """Final outcome of a finished session — pushed over WS as {type:"play_result", ...}."""

    type: Literal["play_result"] = "play_result"
    env_id: str
    mode: PlayMode
    score: float
    steps: int
    rating: SkillRating


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
    width: int
    height: int
    image: str


class PlayActionMessage(BaseModel):
    """Inbound human input over WS: {type:"action", action:<int>}.

    ``action`` is a discrete action index for the current env (CartPole: 0=left, 1=right).
    Latency-tolerant — the session holds the latest received action and reuses it until the
    next one arrives, so dropped/late frames just repeat the prior input rather than stall.
    """

    type: Literal["action"] = "action"
    action: int
