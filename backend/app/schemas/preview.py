"""Preview (frame-streaming) contracts — pydantic models mirrored in frontend/src/api/types.ts.

The env preview is decoupled from training: a separate streamer renders the live policy on
its own env and pushes ``{type:"frame"}`` frames over WS, throttled and paced by ``speed``.
Defined once here so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel


class AgentSprite(BaseModel):
    """One agent's render state for the multi-agent "swarm" canvas (PettingZoo, G7a/ADR-038).

    Positions are world-space ``[x, y]`` (roughly centred on the origin); the client autoscales the
    whole scene to fit. ``role`` is ``"agent"`` (cooperative) or ``"adversary"`` (predator) and
    drives the colour; ``size`` is the entity's collision radius in the same world units.
    """

    x: float
    y: float
    role: str
    size: float


class WorldEntity(BaseModel):
    """A landmark for the swarm canvas — a coverage ``"target"`` (simple_spread) or an ``"obstacle"``
    (a collidable landmark). Same world-space coordinates + ``size`` as :class:`AgentSprite`."""

    x: float
    y: float
    kind: str
    size: float


class GridLayout(BaseModel):
    """Static board layout for a client-rendered grid-world (Toy Text), streamed with each frame.

    The dynamic part (agent / passenger / destination position) rides in the frame's ``state``; this
    describes the fixed board the client draws under it. ``cells`` is row-major (length ``rows*cols``),
    each entry one of: ``"normal"``, ``"start"``, ``"goal"``, ``"hole"`` (FrozenLake), ``"cliff"``
    (CliffWalking) or ``"stop"`` (Taxi pickup/drop-off points). ``kind`` selects the client renderer.
    """

    kind: Literal["frozenlake", "cliffwalking", "taxi"]
    rows: int
    cols: int
    cells: list[str]


class FrameMessage(BaseModel):
    """One rendered env frame, pushed over WS as {type:"frame", ...}.

    ``image`` is base64-encoded JPEG with no data-URI prefix; the client prepends
    ``data:image/jpeg;base64,``. Built by hand in the streamer's hot loop for speed,
    but this model is the contract.
    """

    type: Literal["frame"] = "frame"
    episode: int
    step: int
    reward: float
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
    # Multi-agent (PettingZoo) render state: per-agent sprites + landmark entities for the swarm
    # canvas (None for every single-agent env). Streamed each frame so a late joiner always has it.
    agents: list[AgentSprite] | None = None
    world: list[WorldEntity] | None = None


class PreviewState(BaseModel):
    """Current preview settings — returned by /api/preview and pushed as {type:"preview"}."""

    type: Literal["preview"] = "preview"
    visual: bool
    speed: float
    active: bool  # a training run is attached (frames may be flowing)


class PreviewConfig(BaseModel):
    """Partial update for POST /api/preview. Unset fields are left unchanged.

    ``speed`` is clamped to [1, 20] server-side rather than rejected, so an out-of-range
    value is forgiven instead of 422'd.
    """

    visual: bool | None = None
    speed: float | None = None


class PreviewWatch(BaseModel):
    """Start/stop a *training-free* preview of an env — the "watch the ecosystem" mode (G7b).

    Drives the same streamer with **no** published policy → a random-action rollout, so a
    multi-agent env that is neither human-playable nor yet trainable (heterogeneous ``simple_tag``,
    whose per-species trainer lands in G7b-2) is still watchable as a moving swarm. ``on`` toggles
    the watch; ``env_id`` names the env to render. ``checkpoint_id`` (optional) loads a saved model and
    drives the watch with it — **Watch AI** for a non-playable multi-agent env (simple_tag): the saved
    ecosystem plays itself (both species' trained brains). ``None`` → the random "watch the ecosystem".
    """

    env_id: str
    on: bool
    checkpoint_id: str | None = None
