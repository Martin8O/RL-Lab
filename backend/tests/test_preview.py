"""Unit tests for the preview streamer: frame encoding, settings/clamp, and a real
CartPole render smoke test. The full WS render loop is verified manually (it needs the
event loop + a live model), per the B4 Definition of Done.
"""

import base64
from io import BytesIO

import numpy as np
from app.schemas.preview import PreviewState
from app.services.connection_manager import ConnectionManager
from app.services.preview_streamer import PreviewStreamer, encode_frame
from PIL import Image


def test_encode_frame_roundtrips_to_jpeg() -> None:
    rgb = np.random.default_rng(0).integers(0, 255, size=(40, 60, 3)).astype(np.uint8)
    image_b64, width, height = encode_frame(rgb)
    assert (width, height) == (60, 40)
    decoded = Image.open(BytesIO(base64.b64decode(image_b64)))
    assert decoded.format == "JPEG"
    assert decoded.size == (60, 40)  # PIL size is (width, height)


def test_streamer_state_defaults_and_clamps() -> None:
    streamer = PreviewStreamer(ConnectionManager())

    state = streamer.state()
    assert isinstance(state, PreviewState)
    assert state.visual is True
    assert state.active is False
    assert state.speed == 1.0

    # speed is clamped into [1, 20] rather than rejected
    assert streamer.set_speed(99).speed == 20.0
    assert streamer.set_speed(0.1).speed == 1.0
    assert streamer.set_speed(5).speed == 5.0

    # visual toggle reflected in state (no loop spawns: no run is attached)
    assert streamer.set_visual(False).visual is False
    assert streamer.set_visual(True).visual is True


def test_render_real_cartpole_frame() -> None:
    """A real CartPole rgb_array render encodes to a non-trivial JPEG of matching size."""
    import gymnasium as gym

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        env.step(env.action_space.sample())
        rgb = np.asarray(env.render(), dtype=np.uint8)
    finally:
        env.close()

    assert rgb.ndim == 3 and rgb.shape[2] == 3
    image_b64, width, height = encode_frame(rgb)
    assert len(image_b64) > 100
    assert (width, height) == (rgb.shape[1], rgb.shape[0])
