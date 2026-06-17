"""Competitive multi-agent self-play trainer — per-species PPO for simple_tag (G7b-2, ADR-048).

simple_tag has **two species** with different observation sizes (predator 16 vs prey 14) and opposite
rewards, so the homogeneous parameter-sharing bridge that drives simple_spread (G7a) can't train it —
SuperSuit needs identical spaces. This trainer learns **one shared policy per species** instead, by
**frozen-opponent alternating self-play** (iterated best response):

* Two persistent ``MlpPolicy`` PPO models are built, one per species (predators share one brain, prey
  share another — parameter sharing *within* a species via :func:`ma_env.make_species_vec_env`).
* The run is split into ``rounds`` rounds. In each round every species gets one learning turn: it
  optimises against the **frozen** numpy snapshot (ADR-019) of the *other* species, which is injected
  into the env by :class:`ma_env._SpeciesParallelEnv`. The learner sees a stationary opponent, so each
  turn is a standard single-agent PPO problem — maximal reuse of :mod:`trainer_ppo` (the interruptible
  update, the numpy preview snapshot, the metrics callback).
* After each turn the learner's fresh snapshot becomes the opponent's frozen policy for the next turn,
  so the two species co-evolve in alternation (the arms race, just discretised into rounds).

The decoupled preview (ADR-019) gets **both** species' numpy snapshots — the learner's live one and the
opponent's frozen one — so the swarm renders real-vs-real predators and prey as it trains. The
checkpoint packs **both** ``model.zip`` blobs into one ``species.zip`` so Save/Load round-trips the whole
ecosystem (the manager/store stay ML-free — it's just bytes).

(A simultaneous two-PPO / IPPO variant — both species learning at once — is the parked future
enhancement; frozen self-play was chosen for stable per-species curves and trainer reuse, ADR-048.)
"""

import functools
import io
import time
import zipfile
from collections.abc import Callable
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from app.schemas.training import (
    MultiAgentMetrics,
    SelfPlayHyperparams,
    SpeciesMetrics,
    TrainConfig,
    TrainState,
)
from app.services.checkpoints import CheckpointArtifact
from app.services.ma_env import (
    close_env,
    make_species_vec_env,
    random_obstacle_count,
    species_present,
)
from app.services.train_control import TrainControl
from app.services.trainer_ppo import (
    _ACTIVATIONS,
    _build_numpy_predict,
    _ep_means,
    _InterruptiblePPO,
)

# Published to the decoupled preview: a {role -> predict fn} map so each agent is driven by its own
# species' policy (predators by the predator net, prey by the prey net).
PredictFn = Callable[[object], Any]
PoliciesPublisher = Callable[[dict[str, PredictFn]], None]
MetricsSink = Callable[[MultiAgentMetrics], None]
SnapshotSink = Callable[[CheckpointArtifact], None]


def _opponent_of(role: str, roles: list[str]) -> str:
    """The other species (simple_tag has exactly two: predators vs prey)."""
    return next(r for r in roles if r != role)


def _build_model(config: TrainConfig, learner_role: str) -> _InterruptiblePPO:
    """One per-species PPO over a learner-only env (opponent random at build time — the policy just
    needs the right obs/action spaces here; each round rebuilds the env with the live frozen opponent
    baked in, since SuperSuit cloudpickle-clones the env and can't be mutated in place afterwards)."""
    hp = config.hyperparams
    env = make_species_vec_env(config.env_id, learner_role, opponent_predict=None)
    return _InterruptiblePPO(
        "MlpPolicy",
        env,
        seed=None,  # the SuperSuit vec env can't be seeded; the run seeds python/numpy/torch globally
        learning_rate=hp.learning_rate,
        gamma=hp.gamma,
        clip_range=hp.clip_range,
        ent_coef=hp.ent_coef,
        n_steps=hp.n_steps,
        batch_size=hp.batch_size,
        policy_kwargs={
            "net_arch": [hp.neurons_per_layer] * hp.n_hidden_layers,
            "activation_fn": _ACTIVATIONS[hp.activation],
        },
        device="cpu",
        verbose=0,
    )


def _pack_models(models: dict[str, _InterruptiblePPO], roles: list[str]) -> bytes:
    """Serialize both species' models into one ``species.zip`` (``adversary.zip`` + ``agent.zip``).

    Each entry is itself an SB3 ``model.zip`` (already deflated), so the outer archive is STORED to
    avoid pointless double compression. Kept to bytes so the manager / checkpoint store stay ML-free."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for role in roles:
            inner = io.BytesIO()
            models[role].save(inner)
            zf.writestr(f"{role}.zip", inner.getvalue())
    return buf.getvalue()


def _load_models(
    resume_blob: bytes, config: TrainConfig, roles: list[str]
) -> dict[str, _InterruptiblePPO]:
    """Rebuild both species' PPO models from a packed ``species.zip`` (the resume path).

    Each species gets a fresh learner-only env (random opponent — each round rebuilds it with the real
    frozen opponent), so ``PPO.load``'s space check matches; ``num_timesteps`` is restored so a resumed
    run continues each species' step counter."""
    models: dict[str, _InterruptiblePPO] = {}
    with zipfile.ZipFile(io.BytesIO(resume_blob)) as zf:
        for role in roles:
            data = zf.read(f"{role}.zip")
            env = make_species_vec_env(config.env_id, role, opponent_predict=None)
            models[role] = _InterruptiblePPO.load(io.BytesIO(data), env=env, device="cpu")
    return models


def load_species_predicts(blob: bytes) -> dict[str, PredictFn]:
    """Build a {species role -> numpy predict fn} map from a saved ``species.zip`` — for **Watch AI**.

    Loads each species' policy for *inference only* (``env=None`` — the saved zip carries the obs/action
    spaces, so no env is built and no pygame is touched), then takes the same decoupled numpy snapshot
    the live preview uses (ADR-019). The preview streamer applies each species' fn to its own agents, so
    a saved ecosystem can be watched playing itself without any training run (G7b-2 follow-up)."""
    predicts: dict[str, PredictFn] = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            if name.endswith(".zip"):
                role = name[:-4]  # "adversary.zip" -> "adversary", matching ma_env.agent_role
                model = _InterruptiblePPO.load(io.BytesIO(zf.read(name)), env=None, device="cpu")
                predicts[role] = _build_numpy_predict(model)
    return predicts


class _RoundCallback(BaseCallback):
    """Honours pause/stop during a species' learning turn and fires ``on_rollout`` at each rollout
    boundary (a quiescent point) so the preview snapshot + the ecosystem chart refresh mid-round."""

    def __init__(self, control: TrainControl, on_rollout: Callable[[], None]) -> None:
        super().__init__()
        self._control = control
        self._on_rollout = on_rollout

    def _on_step(self) -> bool:
        self._control.wait_if_paused()
        return not self._control.stop_requested

    def _on_rollout_end(self) -> None:
        self._on_rollout()


def train_tag(
    config: TrainConfig,
    gym_id: str,
    control: TrainControl,
    on_metrics: MetricsSink,
    on_policies: PoliciesPublisher,
    on_snapshot: SnapshotSink | None = None,
    resume_blob: bytes | None = None,
) -> TrainState:
    """Train both species of a competitive simple_tag env by frozen-opponent self-play.

    Blocks the calling thread; the manager runs this off the event loop. Emits a
    :class:`MultiAgentMetrics` frame (both species at once) at every rollout boundary and round end,
    publishes both species' decoupled numpy snapshots to the preview, and (if ``on_snapshot`` is given)
    snapshots the packed two-model checkpoint so Save/Load round-trips the ecosystem. ``gym_id`` is
    unused (the scenario is resolved from the registry by ``ma_env``); it is kept for a uniform trainer
    signature with the manager's other ``train_*`` entry points.
    """
    from stable_baselines3.common.utils import set_random_seed

    roles = species_present(config.env_id)  # e.g. ["adversary", "agent"] — predators first
    rounds = max(1, (config.self_play or SelfPlayHyperparams()).rounds)
    set_random_seed(config.seed)  # policy-level reproducibility (the SuperSuit vec env can't be seeded)
    obs_rng = np.random.default_rng(config.seed)  # drives the per-round random obstacle count (2…6)

    models = (
        _load_models(resume_blob, config, roles)
        if resume_blob is not None
        else {role: _build_model(config, role) for role in roles}
    )
    # Each species' current frozen snapshot (the opponent's injected policy + the preview policy).
    snap: dict[str, PredictFn] = {role: _build_numpy_predict(models[role]) for role in roles}
    on_policies(dict(snap))  # initial preview: both species (untrained, ~random) — never the live model

    # Last known per-species means: the learner's refresh each rollout, the frozen species keeps its
    # last value so every frame carries both lines.
    last_rew: dict[str, float | None] = dict.fromkeys(roles)
    last_len: dict[str, float | None] = dict.fromkeys(roles)
    species_steps: dict[str, int] = {role: int(models[role].num_timesteps) for role in roles}

    per_round = max(1, config.total_timesteps // (rounds * len(roles)))
    started_at = time.monotonic()
    # Honest cumulative budget for the progress display: where the run started (0 fresh, the restored
    # counters on resume) plus this run's budget. So a resumed run reports total = start + budget (not
    # the bare per-run budget), and the save card never shows steps exceeding the total. (The actual
    # steps still overshoot slightly per round — SB3 collects whole rollouts — which the bar clamps.)
    total_target = sum(species_steps.values()) + config.total_timesteps

    def emit(round_no: int, learner: str) -> None:
        # Refresh the learner's snapshot (the new frozen opponent + the preview policy) and publish
        # both species so the swarm renders real-vs-real. ADR-019: a numpy snapshot, never the live model.
        snap[learner] = _build_numpy_predict(models[learner])
        on_policies(dict(snap))
        rew, length = _ep_means(models[learner])
        if rew is not None:
            last_rew[learner], last_len[learner] = rew, length
        species_steps[learner] = int(models[learner].num_timesteps)
        total = sum(species_steps.values())
        predator = last_rew.get("adversary")  # the headline line (high-score / archive read this)
        on_metrics(
            MultiAgentMetrics(
                round=round_no,
                total_rounds=rounds,
                learning_role=learner,
                species=[
                    SpeciesMetrics(
                        role=role,
                        ep_rew_mean=last_rew[role],
                        ep_len_mean=last_len[role],
                        timesteps=species_steps[role],
                    )
                    for role in roles
                ],
                ep_rew_mean=predator,
                timesteps=total,
                total_timesteps=total_target,
                elapsed=time.monotonic() - started_at,
            )
        )
        if on_snapshot is not None:
            on_snapshot(
                CheckpointArtifact(
                    algo="ppo",
                    blob=_pack_models(models, roles),
                    artifact_name="species.zip",
                    reward=predator,
                    timesteps=total,
                    total_timesteps=total_target,
                    iteration=round_no,
                )
            )

    try:
        for r in range(1, rounds + 1):
            for learner in roles:
                if control.stop_requested:
                    return "stopped"
                opponent = _opponent_of(learner, roles)
                # Rebuild the learner's env with the opponent's *current* frozen snapshot baked in
                # (SuperSuit cloudpickle-clones the env, so the opponent can't be mutated in place), and
                # a fresh random obstacle count for this turn (2…6; the obs size stays fixed).
                env = make_species_vec_env(
                    config.env_id, learner, opponent_predict=snap[opponent],
                    obstacle_count=random_obstacle_count(obs_rng),
                )
                model = models[learner]
                model.set_env(env)
                model.stop_check = lambda: control.stop_requested
                model.learn(
                    per_round,
                    callback=_RoundCallback(control, functools.partial(emit, r, learner)),
                    reset_num_timesteps=False,
                )
                close_env(env)  # lock-guarded close — serialises the global pygame.quit()
                emit(r, learner)  # final frame for the turn (the last rollout may have been partial)
        return "stopped" if control.stop_requested else "finished"
    finally:
        for m in models.values():
            if m.env is not None:
                close_env(m.env)
