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

    def start_watch(self, env_id: str) -> None:
        """Begin a *training-free* preview of ``env_id`` — the "watch the ecosystem" mode (G7b).

        Reuses the run-preview machinery with **no** published policy, so the loop steps the env
        with random actions (``_choose_*`` fall back to ``action_space.sample()``). Used for a
        multi-agent env whose per-species trainer isn't built yet (heterogeneous ``simple_tag``,
        G7b-1): neither human-playable nor trainable, but still watchable as a moving swarm. The
        lifecycle is identical to a training run's ``attach_run`` — a later real run just re-attaches
        and publishes its policy — so the two can never collide (the watch envs are training-gated).
        """
        # Switching directly between two watch-only envs would otherwise hit the stale-thread race
        # (``_ensure_loop`` skips while the *old* env's thread is still alive). Stop + join it first so
        # the new env always gets a fresh render loop. (Training never hits this — env switches there
        # are gated behind stopping the run.)
        with self._lock:
            busy_other = (
                self._thread is not None and self._thread.is_alive() and self._env_id != env_id
            )
        if busy_other:
            self.detach_run()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=2.0)
        self.attach_run(env_id)

    def stop_watch(self) -> None:
        """End a watch started by :meth:`start_watch` (the loop observes it, stops, closes its env)."""
        self.detach_run()

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
        from app.envs.registry import get_env
        from app.services.ma_env import is_multi_agent

        # Multi-agent (PettingZoo) envs are a different shape — N agents in one shared world — so they
        # run their own parallel rollout loop + swarm-frame emit (the 5th seam, ADR-038).
        spec = get_env(env_id)
        if is_multi_agent(spec):
            self._run_ma(env_id)
            return
        # Image-obs envs (Atari, G4b) need the shared AtariWrapper + frame-stack vec env so the obs
        # shape matches the CnnPolicy snapshot — a different env API + a raw-colour render path.
        if spec is not None and spec.obs_type == "image":
            self._run_image(env_id, spec)
            return

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

    def _run_ma(self, env_id: str) -> None:
        """Preview loop for a multi-agent (PettingZoo) env — the 5th seam (ADR-038).

        Steps the *parallel* env with the shared decoupled policy applied to **every** agent
        (parameter sharing), and broadcasts a swarm frame (per-agent + landmark world positions)
        the client draws on a canvas. Like the single-agent loop it reads the policy *snapshot* the
        trainer publishes (numpy forward), never the live SB3 model (ADR-019).
        """
        from app.services.ma_env import make_parallel_env

        try:
            env = make_parallel_env(env_id)
        except Exception:  # noqa: BLE001 — a bad MA env must not crash anything
            logger.exception("Preview MA env creation failed for %s", env_id)
            return

        meta = getattr(env, "metadata", {}) or {}
        render_fps = float(meta.get("render_fps", _DEFAULT_RENDER_FPS)) or _DEFAULT_RENDER_FPS
        base_dt = 1.0 / render_fps
        send_interval = 1.0 / _SEND_FPS_CAP
        episode = 0
        last_sent = 0.0
        try:
            while self._active_and_visual():
                episode += 1
                obs, _ = env.reset()
                ep_reward = 0.0
                step = 0
                while env.agents and self._active_and_visual():
                    if self._is_paused():  # training paused → freeze the last frame
                        time.sleep(0.05)
                        continue
                    actions = self._choose_ma_actions(env, obs)
                    obs, rewards, _, _, _ = env.step(actions)
                    ep_reward += float(np.mean(list(rewards.values()))) if rewards else 0.0
                    step += 1

                    now = time.monotonic()
                    if now - last_sent >= send_interval or not env.agents:
                        last_sent = now
                        self._emit_ma_frame(env, episode, step, ep_reward)
                    time.sleep(base_dt / self._current_speed())
        finally:
            env.close()

    def _choose_ma_actions(self, env: Any, obs: dict[str, Any]) -> dict[str, Any]:
        """One action per live agent: the shared policy applied to each agent's obs (random until
        the trainer publishes a policy). A flaky predict falls back to a random action per-agent."""
        with self._lock:
            predict = self._predict
        actions: dict[str, Any] = {}
        for agent in env.agents:
            if predict is None:
                actions[agent] = env.action_space(agent).sample()
                continue
            try:
                actions[agent] = predict(obs[agent])
            except Exception:  # noqa: BLE001 — never let inference contention disturb the run
                logger.debug("Preview MA predict failed; using random action", exc_info=True)
                actions[agent] = env.action_space(agent).sample()
        return actions

    def _emit_ma_frame(self, env: Any, episode: int, step: int, reward: float) -> None:
        from app.services.ma_env import agent_sprites, world_entities

        # Matches schemas.preview.FrameMessage (agents/world fields); built by hand to avoid
        # per-frame validation, like the single-agent emit above.
        self._broadcast(
            {
                "type": "frame",
                "episode": episode,
                "step": step,
                "reward": reward,
                "agents": agent_sprites(env),
                "world": world_entities(env),
            }
        )

    def _run_image(self, env_id: str, spec: Any) -> None:
        """Preview loop for an image-obs env (Atari, G4b).

        Drives the **shared** Atari vec env at ``n_envs=1`` — the exact AtariWrapper + frame-stack the
        CnnPolicy trained on — so the decoupled snapshot's obs shape always matches. The snapshot is a
        read-only CPU torch forward the trainer publishes (ADR-019), never the live CUDA model; until
        it arrives the loop uses random actions. The JPEG shows the **raw colour** frame (``WarpFrame``
        only rewrites the observation), not the 84×84 grayscale the policy consumes. A SB3 vec env
        auto-resets on ``done``, so the loop just counts episodes instead of calling ``reset`` itself.
        """
        from app.envs.atari import make_atari

        try:
            venv = make_atari(spec.gym_id, 1, make_kwargs=spec.make_kwargs)
        except Exception:  # noqa: BLE001 — a bad render env must not crash anything
            logger.exception("Preview image env creation failed for %s", env_id)
            return

        base_dt = 1.0 / _DEFAULT_RENDER_FPS
        send_interval = 1.0 / _SEND_FPS_CAP
        episode = 1
        last_sent = 0.0
        try:
            obs = venv.reset()
            ep_reward = 0.0
            step = 0
            while self._active_and_visual():
                if self._is_paused():  # training paused → freeze the last frame
                    time.sleep(0.05)
                    continue
                action = self._choose_image_action(venv, obs)
                obs, reward, dones, _ = venv.step(np.asarray([action]))
                ep_reward += float(reward[0])
                step += 1
                done = bool(dones[0])  # vec env has already auto-reset obs into the next episode

                now = time.monotonic()
                if now - last_sent >= send_interval or done:
                    last_sent = now
                    self._emit_image_frame(venv, episode, step, ep_reward)

                if done:
                    episode += 1
                    ep_reward = 0.0
                    step = 0
                time.sleep(base_dt / self._current_speed())
        finally:
            venv.close()

    def _choose_image_action(self, venv: Any, obs: Any) -> int:
        """The CNN snapshot's action over the single stacked obs (random until it's published)."""
        with self._lock:
            predict = self._predict
        if predict is None:
            return int(venv.action_space.sample())
        try:
            return int(predict(obs[0]))
        except Exception:  # noqa: BLE001 — never let inference contention disturb the run
            logger.debug("Preview image predict failed; using random action", exc_info=True)
            return int(venv.action_space.sample())

    def _emit_image_frame(self, venv: Any, episode: int, step: int, reward: float) -> None:
        try:
            rgb = np.asarray(venv.render(mode="rgb_array"), dtype=np.uint8)
            image, width, height = encode_frame(rgb)
        except Exception:  # noqa: BLE001 — drop a bad frame, keep the loop alive
            logger.debug("Preview image frame render/encode failed", exc_info=True)
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
