"""Preview (frame-streaming) contracts — pydantic models mirrored in frontend/src/api/types.ts.

The env preview is decoupled from training: a separate streamer renders the live policy on
its own env and pushes ``{type:"frame"}`` frames over WS, throttled and paced by ``speed``.
Defined once here so backend and frontend agree on one source of truth.
"""

from typing import Literal

from pydantic import BaseModel


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
    width: int
    height: int
    image: str


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
