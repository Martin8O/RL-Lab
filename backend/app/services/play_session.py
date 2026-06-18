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
from typing import Any, Literal

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
# Board games (G6a): wall-clock pause after an MCTS move so a human can follow the turn-based play
# (the human's own click applies instantly). Scaled by the speed slider like every other pacing.
_BOARD_MCTS_DELAY = 0.6


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
        # Board games (G6a): the human's side (0 = first player) + the MCTS opponent strength id.
        self._side = 0
        self._ai_strength = "medium"
        # Board games (G6b): a trained net opponent — a masked predict(obs, mask)->action loaded from a
        # checkpoint. None ⇒ the opponent is the built-in MCTS (G6a). Set when a board play config names
        # a checkpoint_id (human-vs-net, or net-vs-net for an "ai" watch).
        self._board_net: Any = None
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

        from app.services.board_engine import is_board_game

        # Board games (G6a) are routed to the OpenSpiel subsystem: the built-in AI is a training-free
        # MCTS, not a checkpoint, so an "ai" board session is an AI-vs-AI *watch* that needs no policy.
        predict: PredictFn | None = None
        if config.mode == "ai" and not is_board_game(spec):
            predict = self._load_ai_policy(config)
        # Board games (G6b): if a checkpoint is named, the opponent is the trained net (human-vs-net, or
        # net-vs-net for an "ai" watch) instead of the MCTS — loaded synchronously so a bad pick surfaces
        # to the REST caller. No checkpoint ⇒ the G6a MCTS at ``ai_strength``.
        board_net: Any = None
        if is_board_game(spec) and config.checkpoint_id:
            board_net = self._load_board_net(config)

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
            # Board games (G6a): which side the human takes + the MCTS opponent strength.
            self._side = config.side
            self._ai_strength = config.ai_strength
            self._board_net = board_net  # G6b: trained net opponent (None ⇒ MCTS)
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

    def _load_board_net(self, config: PlayConfig) -> Any:
        """Load a board checkpoint as a ``(state) -> action`` opponent move fn (the play lane's opponent).

        Both board trainers' nets are returned as a single ``(state) -> action`` so ``_run_board`` is
        algorithm-agnostic: G6b MaskablePPO is the masked policy (via ``board_move_fn``); G6f AlphaZero
        plays at full strength with **neural-MCTS** (``az_move_fn``) — the same search it was trained
        with, so the trained net is a genuinely strong human opponent, not the bare policy head. A
        ``None`` seed gives a varied opponent each game (the human-play convention). Validated
        synchronously, like the AI policy, so a bad pick surfaces to the REST caller.
        """
        from app.envs.registry import get_env
        from app.services import az_net, board_engine

        loaded = self._ckpt.load(config.checkpoint_id or "")
        if loaded is None:
            raise PlayCheckpointNotFoundError(f"Checkpoint '{config.checkpoint_id}' not found")
        if loaded.config.env_id != config.env_id:
            raise InvalidPlayConfigError(
                f"Checkpoint was trained on '{loaded.config.env_id}', "
                f"not '{config.env_id}' — cannot play it here"
            )
        spec = get_env(config.env_id)
        try:
            game = board_engine.load_game(spec.gym_id if spec else config.env_id)
            # AlphaZero (G6f): a CNN played with neural-MCTS, not a MaskablePPO zip.
            if loaded.config.algo == "alphazero":
                model, _ = az_net.build_model_from_blob(loaded.blob, game, device="cpu")
                az_hp = loaded.config.alphazero
                sims = az_hp.play_simulations if az_hp is not None else 60
                return az_net.az_move_fn(model, sims, seed=None)
            return board_engine.board_move_fn(game, board_engine.load_board_predict(loaded.blob))
        except Exception as exc:  # noqa: BLE001 — any deserialize failure → a clear, typed error
            raise InvalidPlayConfigError(f"Could not load board model: {exc}") from exc

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
        # Image-obs envs (Atari) can't be played by an AI on the raw render env: a CnnPolicy consumes
        # the 84×84×4 frame stack, not the raw 210×160×3 RGB the env emits, so model.predict on the
        # raw obs would shape-error. AI mode runs a dedicated loop on the shared AtariWrapper +
        # frame-stack vec env (G4b's make_atari) so the obs shape matches the checkpoint (G4c). Human
        # image play needs no policy (the person supplies the action), so it stays on the raw make_env
        # + JPEG path below — untouched.
        spec = get_env(self._env_id) if self._env_id else None
        # Board games (G6a) are a turn-based OpenSpiel subsystem, not a gym.Env — route both human
        # (human vs MCTS) and ai (MCTS-vs-MCTS watch) to the dedicated board loop. Never make_env'd.
        from app.services.board_engine import is_board_game

        if is_board_game(spec):
            self._run_board(spec, seed)
            return
        if self._mode == "ai" and spec is not None and spec.obs_type == "image":
            self._run_image_ai(spec, seed)
            return

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

    def _run_image_ai(self, spec: Any, seed: int | None) -> None:
        """AI-play loop for an image-obs env (Atari, G4c).

        A CnnPolicy consumes the 84×84×4 frame stack, not the raw 210×160×3 RGB, so AI play builds
        the **shared** Atari vec env (``make_atari`` at ``n_envs=1`` — the exact AtariWrapper +
        frame-stack the policy trained on, G4b) and feeds ``obs[0]`` to the checkpoint's predict fn.
        The JPEG still shows the **raw colour** frame (``WarpFrame`` only rewrites the observation),
        exactly like human play. One episode, then rate — the same shape as the vector AI path.

        Score = the summed step reward. ``AtariWrapper`` clips reward to its sign, but every
        symmetric duel game scores ±1 per point (Pong −21…21, Boxing/Tennis/…), so the clipped sum
        IS the true game score there — and those are the games where a skill reading is meaningful;
        it matches human play (raw env) and the [min_score, solved_score] meter exactly. (For the
        high-scoring arcade games the clip means the reading reflects the training-shaped reward; a
        raw-score eval env for those is out of scope here — true human-vs-net is G7c anyway.)
        """
        from app.envs.atari import make_atari

        try:
            # AI play keeps the configured seed (a reproducible demo); make_atari seeds the vec env.
            venv = make_atari(spec.gym_id, 1, make_kwargs=spec.make_kwargs, seed=seed)
        except Exception:  # noqa: BLE001 — a bad env must surface as state, not crash the thread
            logger.exception("Play image env creation failed for %s", spec.gym_id)
            self._finalize(0.0, 0, completed=False, error="Could not create play environment")
            return

        with self._lock:
            self._n_actions = self._discrete_n(venv)  # getattr-based, mypy-safe (Atari = Discrete(18))
        # Pace like the preview's image loop: a fixed 30 fps base scaled by the speed slider. The
        # Atari render_fps + the wrapper's 4-frame skip don't map cleanly onto one sleep, and 30 fps
        # reads as natural real-time arcade play. AI play isn't frame-rate-capped (that cap is a
        # human-reaction concern); the speed selector still slows it to 0.1× for a closer look.
        base_dt = 1.0 / _DEFAULT_RENDER_FPS
        send_interval = 1.0 / _SEND_FPS_CAP

        score = 0.0
        step = 0
        last_sent = 0.0
        completed = False
        error: str | None = None
        try:
            obs = venv.reset()
            self._emit_image_frame(venv, step, score)  # show the starting state immediately
            done = False
            while not done and not self._stopped():
                action = self._choose_image_action(venv, obs)
                obs, reward, dones, _ = venv.step(np.asarray([action]))
                score += float(reward[0])
                step += 1
                done = bool(dones[0])  # the vec env auto-resets, but we end the episode here
                with self._lock:
                    self._step = step
                    self._score = score

                now = time.monotonic()
                if now - last_sent >= send_interval or done:
                    last_sent = now
                    self._emit_image_frame(venv, step, score)
                time.sleep(base_dt / self._current_speed())
            completed = done
        except Exception:  # noqa: BLE001 — never let a step/render fault crash the thread
            logger.exception("Play image session loop failed")
            error = "Play session crashed"
        finally:
            venv.close()
        self._finalize(score, step, completed=completed, error=error)

    def _choose_image_action(self, venv: Any, obs: Any) -> int:
        """The checkpoint's CNN action over the single stacked obs (random fallback if it faults)."""
        with self._lock:
            predict = self._predict
        if predict is None:  # AI mode always loads a policy in start(); stay safe regardless
            return int(venv.action_space.sample())
        try:
            return int(predict(obs[0]))  # obs[0] = the 84×84×4 stack; SB3 predict transposes it
        except Exception:  # noqa: BLE001 — a flaky predict falls back to a random action
            logger.debug("AI image predict failed; using random action", exc_info=True)
            return int(venv.action_space.sample())

    def _emit_image_frame(self, venv: Any, step: int, score: float) -> None:
        """Broadcast the raw-colour vec-env frame as a play_frame JPEG (the human-play image path)."""
        try:
            rgb = np.asarray(venv.render(mode="rgb_array"), dtype=np.uint8)
            image, width, height = encode_frame(rgb)
        except Exception:  # noqa: BLE001 — drop a bad frame, keep the loop alive
            logger.debug("Play image frame render/encode failed", exc_info=True)
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

    def _run_board(self, spec: Any, seed: int | None) -> None:
        """Turn-based board loop (G6a) — drive an OpenSpiel ``pyspiel.State`` ply by ply.

        ``mode="human"`` = the human (``self._side``) vs an MCTS opponent on the other side; the
        human's move is the clicked legal action, taken via the existing turn-based pending-action
        path (so ``submit_action`` / WS routing are reused). ``mode="ai"`` = an MCTS-vs-MCTS watch.
        Each ply broadcasts the board payload; on terminal the zero-sum outcome is rated. The MCTS
        seed follows the play convention: a fixed seed (AI watch) is a reproducible demo, ``None``
        (human play) gives a varied opponent each game. Game-agnostic — no Tic-Tac-Toe specifics.
        """
        import numpy as np

        from app.services import board_engine

        try:
            game = board_engine.load_game(spec.gym_id)
        except Exception:  # noqa: BLE001 — a bad game must surface as state, not crash the thread
            logger.exception("Board game load failed for %s", spec.gym_id)
            self._finalize(0.0, 0, completed=False, error="Could not load board game")
            return

        sims = board_engine.strength_sims(self._ai_strength)
        # Human controls one side in human mode; both AI sides otherwise (an AI-vs-AI watch).
        human_side = self._side if self._mode == "human" else None
        # The AI opponent: a trained net (if a checkpoint was loaded) plays every AI-controlled side;
        # otherwise the training-free MCTS at the chosen difficulty (G6a). With a net + mode="ai" this is
        # a net-vs-net watch; with no net it is the MCTS-vs-MCTS watch. `_board_net` is already a
        # (state) -> action move fn (G6b masked policy or G6f neural-MCTS — see _load_board_net), legal
        # by construction, so it's used directly here.
        net_move = self._board_net
        bots: dict[int, board_engine.MctsOpponent] = {}
        if net_move is None:
            for player in range(game.num_players()):
                if human_side is not None and player == human_side:
                    continue
                bot_seed = None if seed is None else seed + player
                bots[player] = board_engine.MctsOpponent(game, sims, bot_seed)
        # A chance node sampler (dice etc.) — never hit by Tic-Tac-Toe, but keeps the loop general.
        rng = np.random.default_rng(seed)
        # Whose outcome to rate: the human's side in human play, else player 0's (a symmetric watch).
        rating_player = human_side if human_side is not None else 0

        last_action: int | None = None
        step = 0
        error: str | None = None
        state: Any = None
        try:
            state = game.new_initial_state()
            self._emit_board_frame(state, step, 0.0, last_action)  # show the empty board immediately
            while not state.is_terminal() and not self._stopped():
                if state.is_chance_node():  # general (backgammon dice…); TTT has none
                    outcomes = state.chance_outcomes()
                    action = int(rng.choice([a for a, _ in outcomes]))
                elif human_side is not None and state.current_player() == human_side:
                    pending = self._take_pending_action()
                    if pending is None:
                        time.sleep(0.03)  # wait for the human to click a cell
                        continue
                    action = int(pending)
                    if action not in state.legal_actions():
                        continue  # reject an illegal click; keep waiting for a legal one
                else:
                    action = (
                        net_move(state) if net_move is not None
                        else bots[state.current_player()].step(state)
                    )
                    # Pace AI moves so a human can follow them; the speed slider scales it. The
                    # human's own move applies instantly (no sleep on that branch).
                    time.sleep(_BOARD_MCTS_DELAY / self._current_speed())
                # Snapshot the board before the move so a diff-decoded game (chess) can report the last
                # move's from/to cells for the highlight (SAN can't be re-decoded post-move). Cheap; the
                # other games ignore prev_cells. board.board_cells imported lazily inside _emit_board_frame.
                prev_cells = board_engine.board_cells(state)
                state.apply_action(action)
                last_action = action
                step += 1
                self._emit_board_frame(state, step, 0.0, last_action, prev_cells)
        except Exception:  # noqa: BLE001 — never let a step fault crash the thread
            logger.exception("Board play loop failed")
            error = "Board session crashed"
        self._finalize_board(state, step, rating_player, error=error)

    def _emit_board_frame(
        self, state: Any, step: int, score: float, last_action: int | None,
        prev_cells: list[str] | None = None,
    ) -> None:
        """Broadcast one board ply as a play_frame carrying the BoardState payload (no JPEG).

        ``prev_cells`` (the board before ``last_action``) lets a diff-decoded game (chess) report the
        last move's from/to for the highlight; ignored by every other game."""
        from app.services import board_engine

        self._broadcast(
            {
                "type": "play_frame",
                "step": step,
                "score": score,
                "board": board_engine.board_payload(state, last_action, prev_cells),
            }
        )

    def _finalize_board(
        self, state: Any, steps: int, rating_player: int, *, error: str | None
    ) -> None:
        """Settle a finished board game — a 3-valued win/draw/loss outcome, not a skill rating."""
        from app.services import board_engine

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
            return  # stop() already set + broadcast "stopped"; nothing to rate
        label: Literal["win", "draw", "loss"]
        if state is not None and state.is_terminal():
            score, label = board_engine.outcome(state, rating_player)
        else:
            score, label = 0.0, "draw"
        result = PlayResult(
            env_id=env_id or "",
            mode=mode or "human",
            score=score,
            steps=steps,
            rating=None,  # board games show a W/D/L card, not the continuous skill meter
            outcome=label,  # one of win/draw/loss (typed via board_engine.outcome)
        )
        with self._lock:
            self._state = "finished"
            self._step = steps
            self._score = score
            self._result = result
        self._broadcast(result.model_dump())
        self._broadcast(self.status().model_dump())

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
