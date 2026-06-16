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
from app.services.client_render import client_state, grid_layout, terrain
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
        # Latest human action, held between WS frames. For a discrete env this is an action index
        # (int); for a continuous (box) env it is the analog command from the keymap (a float, e.g.
        # full torque one way) which _choose_action wraps into the env's action vector.
        self._latest_action: Any = 0
        # Turn-based human play (grid-worlds): the agent advances one step per key press, so a single
        # received action is consumed once (here) rather than held. None ⇒ no pending move.
        self._turn_based = False
        self._pending_action: Any = None
        self._n_actions: int | None = None  # discrete action count, known once the env is made
        # Continuous (box) action space, captured once the env is made (None ⇒ discrete env).
        self._box_low: np.ndarray | None = None
        self._box_high: np.ndarray | None = None
        self._box_shape: tuple[int, ...] | None = None
        # How much longer a play episode runs vs training (EnvSpec.play_step_scale) — also widens
        # the skill floor so the rating span matches the longer episode.
        self._play_step_scale = 1
        # Per-env extra slow-down on the human-play step interval (EnvSpec.human_play_slowdown); 1.0
        # for almost everything, >1 for fall-fast high-fps envs (MuJoCo Hopper/Walker2d) so a person
        # gets more real seconds before the topple. Applies to human mode only.
        self._human_play_slowdown = 1.0
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
            # input. CartPole has no idle (idle_action None) so 0 is as good as any there. For a
            # continuous env the idle is the analog rest command (0 = no torque/force).
            self._latest_action = config.idle_action if config.idle_action is not None else 0
            self._turn_based = spec.turn_based
            self._pending_action = None
            self._n_actions = None
            self._box_low = self._box_high = self._box_shape = None
            self._play_step_scale = spec.play_step_scale
            self._human_play_slowdown = spec.human_play_slowdown
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

    def submit_action(self, action: float | list[float]) -> None:
        """Record the latest human action (from a WS ``{type:"action"}`` frame).

        Stored raw — an int/float action index for a discrete env, or a float / list of floats
        (the analog command) for a continuous (box) env; :meth:`_choose_action` interprets it per
        the env's action space. A no-op unless a session is actually playing, so stray input is
        harmless.
        """
        with self._lock:
            if self._state != "playing":
                return
            self._latest_action = action
            self._pending_action = action  # one-shot move for turn-based grid play (ignored otherwise)

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
        # Shared factory: applies the registry's variant kwargs, the discrete-obs one-hot wrapper
        # (so a loaded AI policy gets the obs shape it trained on) and the play_step_scale episode
        # extension — only lengthening truncation; early termination (flag/goal/crash) still ends
        # the episode normally. Lazy import keeps gym out of startup.
        from app.envs.factory import make_env

        try:
            env = make_env(
                self._env_id or gym_id, gym_id,
                render_mode="rgb_array", play_scale=self._play_step_scale,
            )
        except Exception:  # noqa: BLE001 — a bad env must surface as state, not crash
            logger.exception("Play env creation failed for %s", gym_id)
            self._finalize(0.0, 0, completed=False, error="Could not create play environment")
            return

        with self._lock:
            self._n_actions = self._discrete_n(env)
            self._capture_action_space(env)
        render_fps = float(env.metadata.get("render_fps", _DEFAULT_RENDER_FPS))
        base_dt = 1.0 / (render_fps or _DEFAULT_RENDER_FPS)
        send_interval = 1.0 / _SEND_FPS_CAP
        # Human play must not advance the simulation faster than it shows frames (1 step ≤ 1 sent
        # frame). A high-render_fps env — MuJoCo Hopper/Walker2d run at 125 steps/s — otherwise falls
        # over in ~1 s, before a person can react or even see a leg move (the other MuJoCo envs run at
        # 20–50 fps and play fine). Cap the human base step rate at the frame-send rate; the speed
        # slider still scales it (down to 0.1× for very deliberate play). AI play keeps the env's own
        # real-time rate so a trained demo looks natural; turn-based human play ignores base_dt entirely.
        # On top of the cap, fall-fast envs apply a per-env slow-down (human_play_slowdown) so a person
        # gets more real seconds before an unpreventable topple (MuJoCo Hopper/Walker2d ≈5× longer).
        if self._mode == "human":
            base_dt = max(base_dt, send_interval) * self._human_play_slowdown

        score = 0.0
        step = 0
        last_sent = 0.0
        completed = False
        error: str | None = None
        # Grid-worlds the human plays turn-based: advance one step per key press instead of stepping
        # continuously at the render rate (a human can't react to 30 grid moves/second). The AI and
        # the preview still step continuously, paced by the speed slider.
        turn_based_human = self._turn_based and self._mode == "human"
        try:
            obs, _ = env.reset(seed=seed)
            self._emit_frame(env, step, score, obs, None)  # show the starting state immediately
            done = False
            while not done and not self._stopped():
                if turn_based_human:
                    pending = self._take_pending_action()
                    if pending is None:
                        time.sleep(0.03)  # wait for a key press — don't advance the episode
                        continue
                    action: Any = max(0, min((self._n_actions or 1) - 1, int(pending)))
                else:
                    action = self._choose_action(env, obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                step += 1
                done = bool(terminated or truncated)
                with self._lock:
                    self._step = step
                    self._score = score

                now = time.monotonic()
                if now - last_sent >= send_interval or done:
                    last_sent = now
                    self._emit_frame(env, step, score, obs, action)
                if not turn_based_human:
                    time.sleep(base_dt / self._current_speed())
            completed = done
        except Exception:  # noqa: BLE001 — never let a step/render fault crash the thread
            logger.exception("Play session loop failed")
            error = "Play session crashed"
        finally:
            env.close()
        self._finalize(score, step, completed=completed, error=error)

    def _choose_action(self, env: Any, obs: Any) -> Any:
        with self._lock:
            mode = self._mode
            predict = self._predict
            held = self._latest_action
            n = self._n_actions
            box_low = self._box_low
            box_high = self._box_high
            box_shape = self._box_shape
        if mode == "ai" and predict is not None:
            try:
                out = predict(obs)
            except Exception:  # noqa: BLE001 — a flaky predict falls back to a random action
                logger.debug("AI predict failed; using random action", exc_info=True)
                return env.action_space.sample()
            if box_low is not None:  # continuous: a clipped action vector in [low, high]
                return np.clip(
                    np.asarray(out, dtype=np.float32).reshape(box_shape), box_low, box_high
                )
            return int(np.asarray(out).flatten()[0])
        if box_low is not None:  # human, continuous: wrap the analog command into the action vector
            arr = np.asarray(held, dtype=np.float32).reshape(-1)
            if arr.size == 1 and box_shape is not None:  # scalar command → fill the action shape
                arr = np.full(box_shape, arr[0], dtype=np.float32)
            return np.clip(arr.reshape(box_shape), box_low, box_high)
        if n is not None:  # human, discrete: keep the held action inside the valid range
            return max(0, min(n - 1, int(held)))
        return held

    def _finalize(
        self, score: float, steps: int, *, completed: bool, error: str | None
    ) -> None:
        """Settle the terminal state once the loop exits and broadcast the outcome."""
        with self._lock:
            env_id = self._env_id
            mode = self._mode
            stopped = self._stop
            min_scale = float(self._play_step_scale)
        if error is not None:
            with self._lock:
                self._state = "error"
                self._error = error
            self._broadcast(self.status().model_dump())
            return
        if stopped:
            # stop() already set + broadcast the "stopped" state; nothing to rate.
            return
        rating = skill.rate(env_id, score, min_scale) if env_id is not None else None
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

    def _capture_action_space(self, env: Any) -> None:
        """Record the env's action bounds if it is continuous (box); leave them None if discrete.

        Caller holds the lock. A Box space has ``low``/``high``/``shape`` and no ``n``; a Discrete
        one has ``n`` (already captured into ``_n_actions``), so ``_box_low`` stays None there.
        """
        space = env.action_space
        if getattr(space, "n", None) is not None:
            return
        self._box_low = np.asarray(space.low, dtype=np.float32)
        self._box_high = np.asarray(space.high, dtype=np.float32)
        self._box_shape = tuple(int(d) for d in space.shape)

    def _emit_frame(self, env: Any, step: int, score: float, obs: Any, action: Any) -> None:
        # Client-rendered envs draw from raw state — skip rgb render + JPEG. ``action`` (the discrete
        # action just applied, or None) lets the client draw the firing thruster (LunarLander plumes).
        act = int(action) if isinstance(action, (int, np.integer)) else None
        state = client_state(env, obs)
        if state is not None:
            frame = {"type": "play_frame", "step": step, "score": score, "state": state, "action": act}
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

    def _take_pending_action(self) -> Any:
        """Pop the one-shot move for turn-based human play (None if no key has been pressed yet)."""
        with self._lock:
            action = self._pending_action
            self._pending_action = None
            return action

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
