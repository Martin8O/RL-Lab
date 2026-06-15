"""Preview (frame-streaming) contracts — pydantic models mirrored in frontend/src/api/types.ts.

The env preview is decoupled from training: a separate streamer renders the live policy on
its own env and pushes ``{type:"frame"}`` frames over WS, throttled and paced by ``speed``.
Defined once here so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel


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
