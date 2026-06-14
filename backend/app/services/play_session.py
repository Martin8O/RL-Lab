"""Interactive play session — one episode of a human-playable env, streamed over WS.

The mirror image of the training preview streamer, but the *actions* come from outside the
loop: a human at the keyboard (``mode="human"``, actions arrive over WS as ``{type:"action"}``)
or a loaded checkpoint playing itself (``mode="ai"``). The session renders the episode to JPEG
frames (``{type:"play_frame"}``), and on the episode's natural end rates the score into a skill
band (``{type:"play_result"}``) via :mod:`app.services.skill`.

Game-agnostic by construction so a competitive game (Pong) slots in later: today CartPole has
one agent; a future env would add a ``side`` to pick which agent the human drives while the AI
takes the other. Latency-tolerant — the loop reuses the last received human action until a new
one arrives, so dropped or late input simply repeats rather than stalls.

Kept torch/gym-free at import time (gymnasium is imported lazily inside the worker thread; the
AI policy is built lazily in :func:`app.services.policy.predict_from_checkpoint`) so /health and
the rest of the REST surface stay fast to boot.
"""

import asyncio
import threading
import time
from typing import Any

import numpy as np

from app.core.logging import get_logger
from app.envs.registry import get_env
from app.schemas.play import PlayConfig, PlayMode, PlayResult, PlayState, PlayStatus
from app.services import skill
from app.services.checkpoints import CheckpointStore, checkpoint_store
from app.services.client_render import cart_state
from app.services.connection_manager import ConnectionManager, manager
from app.services.policy import PolicyLoadError, PredictFn, predict_from_checkpoint
from app.services.preview_streamer import encode_frame

logger = get_logger(__name__)

_SEND_FPS_CAP = 30.0  # max frames/sec sent over WS (anti-flood)
_DEFAULT_RENDER_FPS = 30.0  # fallback if the env exposes no render_fps
# Play allows deep slow-motion (down to 0.1×) — unlike the training preview (min 1×) — so a
# beginner can actually balance CartPole at human reaction times (even 0.25× steps ~12×/s).
# Upper bound matches the preview.
_MIN_SPEED = 0.1
_MAX_SPEED = 20.0


class PlayError(RuntimeError):
    """Base class for play-session start failures (mapped to HTTP codes by the API)."""


class AlreadyPlayingError(PlayError):
    """Raised when a start is attempted while a session is already active."""


class InvalidPlayConfigError(PlayError):
    """Raised when the env is unknown / not human-playable / the AI request is malformed."""


class PlayCheckpointNotFoundError(PlayError):
    """Raised when an AI session names a checkpoint id that does not exist."""


def _clamp_speed(speed: float) -> float:
    return max(_MIN_SPEED, min(_MAX_SPEED, float(speed)))


class PlaySession:
    """Owns the single active play session and mirrors its state over WebSocket."""

    def __init__(
        self,
        connection_manager: ConnectionManager,
        checkpoints: CheckpointStore = checkpoint_store,
    ) -> None:
        self._cm = connection_manager
        self._ckpt = checkpoints
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

        self._state: PlayState = "idle"
        self._env_id: str | None = None
        self._mode: PlayMode | None = None
        self._checkpoint_id: str | None = None
        self._seed: int | None = None
        self._speed = 1.0
        self._step = 0
        self._score = 0.0
        self._result: PlayResult | None = None
        self._error: str | None = None

        self._predict: PredictFn | None = None
        self._latest_action = 0  # latest human action, held between WS frames
        self._n_actions: int | None = None  # discrete action count, known once the env is made
        self._stop = False

    # -- wiring -----------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the event loop so worker-thread broadcasts can reach it."""
        self._loop = loop

    # -- lifecycle --------------------------------------------------------------

    def start(self, config: PlayConfig) -> PlayStatus:
        """Begin one interactive episode. Validates + loads the AI policy synchronously so
        bad configs surface to the REST caller before any thread spins up."""
        spec = get_env(config.env_id)
        if spec is None:
            raise InvalidPlayConfigError(f"Unknown environment '{config.env_id}'")
        if not spec.human_playable:
            raise InvalidPlayConfigError(
                f"Environment '{config.env_id}' is not human-playable"
            )

        predict: PredictFn | None = None
        if config.mode == "ai":
            predict = self._load_ai_policy(config)

        with self._lock:
            if self._state == "playing":
                raise AlreadyPlayingError("A play session is already active")
            self._state = "playing"
            self._env_id = config.env_id
            self._mode = config.mode
            self._checkpoint_id = config.checkpoint_id
            self._seed = config.seed
            self._speed = _clamp_speed(config.speed)
            self._predict = predict
            # Hold the env's idle action (no-op) until the human presses a key — otherwise the
            # default 0 means "push left" on MountainCar/Acrobot, shoving the agent before any
            # input. CartPole has no idle (idle_action None) so 0 is as good as any there.
            self._latest_action = config.idle_action if config.idle_action is not None else 0
            self._n_actions = None
            self._step = 0
            self._score = 0.0
            self._result = None
            self._error = None
            self._stop = False

        self._thread = threading.Thread(
            target=self._run,
            args=(spec.gym_id, config.seed),
            name="play-session",
            daemon=True,
        )
        self._thread.start()
        self._broadcast(self.status().model_dump())
        return self.status()

    def _load_ai_policy(self, config: PlayConfig) -> PredictFn:
        if not config.checkpoint_id:
            raise InvalidPlayConfigError("AI play requires a checkpoint_id")
        loaded = self._ckpt.load(config.checkpoint_id)
        if loaded is None:
            raise PlayCheckpointNotFoundError(
                f"Checkpoint '{config.checkpoint_id}' not found"
            )
        if loaded.config.env_id != config.env_id:
            raise InvalidPlayConfigError(
                f"Checkpoint was trained on '{loaded.config.env_id}', "
                f"not '{config.env_id}' — cannot play it here"
            )
        try:
            return predict_from_checkpoint(loaded)
        except PolicyLoadError as exc:
            raise InvalidPlayConfigError(str(exc)) from exc

    def submit_action(self, action: int) -> None:
        """Record the latest human action (from a WS ``{type:"action"}`` frame).

        A no-op unless a session is actually playing, so stray input is harmless.
        """
        with self._lock:
            if self._state != "playing":
                return
            self._latest_action = int(action)

    def set_speed(self, speed: float) -> PlayStatus:
        """Change playback pacing mid-session (the speed selector while a session runs).

        The worker loop reads ``_current_speed()`` every step, so the new pace takes effect on
        the next frame — for both human and AI sessions. Harmless when idle (no loop is reading
        it). Clamped to the play range.
        """
        with self._lock:
            self._speed = _clamp_speed(speed)
            return self._status_locked()

    def stop(self) -> PlayStatus:
        """Abort the active session; the loop observes this and tears down its env."""
        with self._lock:
            if self._state != "playing":
                return self._status_locked()
            self._stop = True
            self._state = "stopped"
        self._broadcast(self.status().model_dump())
        return self.status()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to finish (used by tests)."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # -- worker thread ----------------------------------------------------------

    def _run(self, gym_id: str, seed: int | None) -> None:
        import gymnasium as gym  # lazy: keep gym out of startup

        try:
            env = gym.make(gym_id, render_mode="rgb_array")
        except Exception:  # noqa: BLE001 — a bad env must surface as state, not crash
            logger.exception("Play env creation failed for %s", gym_id)
            self._finalize(0.0, 0, completed=False, error="Could not create play environment")
            return

        with self._lock:
            self._n_actions = self._discrete_n(env)
        render_fps = float(env.metadata.get("render_fps", _DEFAULT_RENDER_FPS))
        base_dt = 1.0 / (render_fps or _DEFAULT_RENDER_FPS)
        send_interval = 1.0 / _SEND_FPS_CAP

        score = 0.0
        step = 0
        last_sent = 0.0
        completed = False
        error: str | None = None
        try:
            obs, _ = env.reset(seed=seed)
            self._emit_frame(env, step, score)  # show the starting state immediately
            done = False
            while not done and not self._stopped():
                obs, reward, terminated, truncated, _ = env.step(
                    self._choose_action(env, obs)
                )
                score += float(reward)
                step += 1
                done = bool(terminated or truncated)
                with self._lock:
                    self._step = step
                    self._score = score

                now = time.monotonic()
                if now - last_sent >= send_interval or done:
                    last_sent = now
                    self._emit_frame(env, step, score)
                time.sleep(base_dt / self._current_speed())
            completed = done
        except Exception:  # noqa: BLE001 — never let a step/render fault crash the thread
            logger.exception("Play session loop failed")
            error = "Play session crashed"
        finally:
            env.close()
        self._finalize(score, step, completed=completed, error=error)

    def _choose_action(self, env: Any, obs: Any) -> int:
        with self._lock:
            mode = self._mode
            predict = self._predict
            held = self._latest_action
            n = self._n_actions
        if mode == "ai" and predict is not None:
            try:
                return int(np.asarray(predict(obs)).flatten()[0])
            except Exception:  # noqa: BLE001 — a flaky predict falls back to a random action
                logger.debug("AI predict failed; using random action", exc_info=True)
                return int(env.action_space.sample())
        if n is not None:  # human: keep the held action inside the valid discrete range
            return max(0, min(n - 1, held))
        return held

    def _finalize(
        self, score: float, steps: int, *, completed: bool, error: str | None
    ) -> None:
        """Settle the terminal state once the loop exits and broadcast the outcome."""
        with self._lock:
            env_id = self._env_id
            mode = self._mode
            stopped = self._stop
        if error is not None:
            with self._lock:
                self._state = "error"
                self._error = error
            self._broadcast(self.status().model_dump())
            return
        if stopped:
            # stop() already set + broadcast the "stopped" state; nothing to rate.
            return
        rating = skill.rate(env_id, score) if env_id is not None else None
        result = (
            PlayResult(
                env_id=env_id or "",
                mode=mode or "human",
                score=score,
                steps=steps,
                rating=rating,
            )
            if rating is not None
            else None
        )
        with self._lock:
            self._state = "finished"
            self._step = steps
            self._score = score
            self._result = result
        if result is not None:
            self._broadcast(result.model_dump())
        self._broadcast(self.status().model_dump())

    # -- status / helpers -------------------------------------------------------

    def status(self) -> PlayStatus:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> PlayStatus:
        return PlayStatus(
            state=self._state,
            env_id=self._env_id,
            mode=self._mode,
            checkpoint_id=self._checkpoint_id,
            seed=self._seed,
            speed=self._speed,
            step=self._step,
            score=self._score,
            result=self._result,
            error=self._error,
        )

    @staticmethod
    def _discrete_n(env: Any) -> int | None:
        n = getattr(env.action_space, "n", None)
        return int(n) if n is not None else None

    def _emit_frame(self, env: Any, step: int, score: float) -> None:
        # CartPole is drawn client-side from raw state — skip the rgb render + JPEG entirely.
        state = cart_state(env)
        if state is not None:
            self._broadcast({"type": "play_frame", "step": step, "score": score, "state": state})
            return
        try:
            rgb = np.asarray(env.render(), dtype=np.uint8)
            image, width, height = encode_frame(rgb)
        except Exception:  # noqa: BLE001 — drop a bad frame, keep the loop alive
            logger.debug("Play frame render/encode failed", exc_info=True)
            return
        # Matches schemas.play.PlayFrame; built by hand to avoid per-frame validation.
        self._broadcast(
            {
                "type": "play_frame",
                "step": step,
                "score": score,
                "width": width,
                "height": height,
                "image": image,
            }
        )

    def _stopped(self) -> bool:
        with self._lock:
            return self._stop

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
        except Exception:  # noqa: BLE001 — never let a dead loop kill the session
            logger.debug("Play WS broadcast skipped (loop unavailable)", exc_info=True)


# Module singleton, wired to the shared connection manager.
play_session = PlaySession(manager)
