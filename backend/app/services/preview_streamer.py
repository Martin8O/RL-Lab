"""Decoupled env-preview streamer.

Renders the *live* training policy on its own throwaway env and pushes JPEG frames over
WS. The trainer itself stays completely headless and full-speed (preserving B2's
reproducibility); turning the preview off stops all rendering — the cookbook's
"visual off ⇒ faster training" point — while turning it on costs only a modest,
non-collapsing amount of throughput because the render cadence is decoupled and throttled.

Design notes:
- A single daemon thread owns one ``gym.make(..., render_mode="rgb_array")`` env, loops
  episodes using ``predict`` (the manager hands us the live model), encodes frames to
  base64 JPEG and broadcasts them throttled to ``_SEND_FPS_CAP``.
- ``speed`` (1×–20×) paces the simulation via the inter-step sleep, so the slider visibly
  changes how fast the pole moves; WS send rate stays capped regardless of speed.
- ``predict`` reads the live model's weights while training mutates them. For a plain
  CartPole MLP on CPU this is benign (in-place Adam updates never reallocate tensor
  storage, ``predict`` runs under ``no_grad`` with ``deterministic=True`` so it draws no
  RNG and cannot perturb the training trajectory). Any transient failure falls back to a
  random action so the preview can never disturb training.

Kept torch/gym-free at import time so /health etc. stay fast; gymnasium is imported lazily
inside the worker thread.
"""

import asyncio
import base64
import threading
import time
from collections.abc import Callable
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from app.core.logging import get_logger
from app.schemas.preview import PreviewState
from app.services.client_render import client_state, grid_layout, terrain
from app.services.connection_manager import ConnectionManager, manager

logger = get_logger(__name__)

PredictFn = Callable[[Any], Any]

_SEND_FPS_CAP = 30.0  # max frames/sec sent over WS (anti-flood)
_DEFAULT_RENDER_FPS = 30.0  # fallback if the env exposes no render_fps
_JPEG_QUALITY = 70
_MIN_SPEED = 1.0
_MAX_SPEED = 20.0


def encode_frame(rgb: np.ndarray) -> tuple[str, int, int]:
    """RGB uint8 HxWx3 array → (base64 JPEG string, width, height)."""
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return base64.b64encode(buf.getvalue()).decode("ascii"), width, height


class PreviewStreamer:
    """Owns the single preview render loop and mirrors its state over WebSocket."""

    def __init__(self, connection_manager: ConnectionManager) -> None:
        self._cm = connection_manager
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._visual = True
        self._speed = 1.0
        self._run_active = False
        self._paused = False
        self._env_id: str | None = None
        self._predict: PredictFn | None = None

    # -- wiring -----------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the event loop so worker-thread broadcasts can reach it."""
        self._loop = loop

    # -- settings (driven by the REST API) --------------------------------------

    def state(self) -> PreviewState:
        with self._lock:
            return PreviewState(
                visual=self._visual, speed=self._speed, active=self._run_active
            )

    def set_visual(self, on: bool) -> PreviewState:
        with self._lock:
            self._visual = on
        self._ensure_loop()  # start the loop if a run is active and we just turned on
        state = self.state()
        self._broadcast(state.model_dump())
        return state

    def set_speed(self, speed: float) -> PreviewState:
        with self._lock:
            self._speed = max(_MIN_SPEED, min(_MAX_SPEED, float(speed)))
        state = self.state()
        self._broadcast(state.model_dump())
        return state

    # -- run lifecycle (driven by the training manager) -------------------------

    def attach_run(self, env_id: str) -> None:
        """A training run started: remember its env and (if visual on) begin streaming."""
        with self._lock:
            self._run_active = True
            self._paused = False
            self._env_id = env_id
            self._predict = None
        self._ensure_loop()
        self._broadcast(self.state().model_dump())

    def set_policy(self, predict: PredictFn) -> None:
        """Publish the live model's predict fn; until then the loop uses random actions."""
        with self._lock:
            self._predict = predict

    def set_paused(self, on: bool) -> None:
        """Freeze/unfreeze the preview in lock-step with training pause/resume, so a paused
        run also visibly freezes the env (otherwise the pole keeps moving and pause looks broken)."""
        with self._lock:
            self._paused = on

    def detach_run(self) -> None:
        """The run ended: the loop observes this, stops and closes its env."""
        with self._lock:
            self._run_active = False
            self._predict = None
        self._broadcast(self.state().model_dump())

    # -- worker thread ----------------------------------------------------------

    def _ensure_loop(self) -> None:
        """Spawn the render loop if a run is active, visual is on and it isn't running."""
        with self._lock:
            if not (self._run_active and self._visual):
                return
            if self._thread is not None and self._thread.is_alive():
                return
            env_id = self._env_id
            if env_id is None:
                return
            self._thread = threading.Thread(
                target=self._run, args=(env_id,), name="preview-streamer", daemon=True
            )
            self._thread.start()

    def _run(self, env_id: str) -> None:
        from app.envs.factory import make_env  # lazy: keep gym out of startup

        try:
            # Shared factory — same wrappers as training (incl. the discrete-obs one-hot), so the
            # decoupled predict fn the trainer publishes gets the obs shape it was built for.
            env = make_env(env_id, render_mode="rgb_array")
        except Exception:  # noqa: BLE001 — a bad render env must not crash anything
            logger.exception("Preview env creation failed for %s", env_id)
            return

        render_fps = float(env.metadata.get("render_fps", _DEFAULT_RENDER_FPS))
        base_dt = 1.0 / (render_fps or _DEFAULT_RENDER_FPS)
        send_interval = 1.0 / _SEND_FPS_CAP
        episode = 0
        last_sent = 0.0
        try:
            while self._active_and_visual():
                episode += 1
                obs, _ = env.reset()
                ep_reward = 0.0
                step = 0
                done = False
                while not done and self._active_and_visual():
                    if self._is_paused():  # training paused → freeze the last frame
                        time.sleep(0.05)
                        continue
                    action = self._choose_action(env, obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    ep_reward += float(reward)
                    step += 1
                    done = bool(terminated or truncated)

                    now = time.monotonic()
                    if now - last_sent >= send_interval or done:
                        last_sent = now
                        self._emit_frame(env, episode, step, ep_reward, obs, action)

                    time.sleep(base_dt / self._current_speed())
        finally:
            env.close()

    def _emit_frame(self, env: Any, episode: int, step: int, reward: float, obs: Any, action: Any) -> None:
        # Client-rendered envs draw from raw state — skip rgb render + JPEG. ``action`` (the discrete
        # action just applied, or None) lets the client draw the firing thruster (LunarLander plumes).
        act = int(action) if isinstance(action, (int, np.integer)) else None
        state = client_state(env, obs)
        if state is not None:
            frame = {
                "type": "frame", "episode": episode, "step": step,
                "reward": reward, "state": state, "action": act,
            }
            scene = terrain(env)  # LunarLander streams its real moon surface; None elsewhere
            if scene is not None:
                frame["terrain"] = scene
            board = grid_layout(env)  # Toy Text streams its static board; None elsewhere
            if board is not None:
                frame["grid"] = board
            self._broadcast(frame)
            return
        try:
            rgb = np.asarray(env.render(), dtype=np.uint8)
            image, width, height = encode_frame(rgb)
        except Exception:  # noqa: BLE001 — drop a bad frame, keep the loop alive
            logger.debug("Preview frame render/encode failed", exc_info=True)
            return
        # Matches schemas.preview.FrameMessage; built by hand to avoid per-frame validation.
        self._broadcast(
            {
                "type": "frame",
                "episode": episode,
                "step": step,
                "reward": reward,
                "width": width,
                "height": height,
                "image": image,
            }
        )

    def _choose_action(self, env: Any, obs: Any) -> Any:
        with self._lock:
            predict = self._predict
        if predict is None:
            return env.action_space.sample()
        try:
            # The predict fn already returns the right shape for the env: an int for a discrete
            # action space, a clipped float vector for a continuous (box) one — pass it straight
            # to env.step() (no int() cast, which would truncate a continuous action to 0/1).
            return predict(obs)
        except Exception:  # noqa: BLE001 — never let inference contention disturb the run
            logger.debug("Preview predict failed; using random action", exc_info=True)
            return env.action_space.sample()

    def _active_and_visual(self) -> bool:
        with self._lock:
            return self._run_active and self._visual

    def _is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def _current_speed(self) -> float:
        with self._lock:
            return self._speed

    def _broadcast(self, frame: dict) -> None:
        """Schedule a WS broadcast on the bound loop from any thread (best-effort)."""
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._cm.broadcast(frame), loop)
        except Exception:  # noqa: BLE001 — never let a dead loop kill the preview
            logger.debug("Preview WS broadcast skipped (loop unavailable)", exc_info=True)


# Module singleton, wired to the shared connection manager.
preview_streamer = PreviewStreamer(manager)
