"""G7a — the multi-agent (PettingZoo / MPE) family + the 5th extensibility seam (ADR-038).

Every other env is single-agent (one Gymnasium ``step()`` loop). A multi-agent env is a different
shape — N agents in one shared world, each with its own obs/action/reward (PettingZoo's *parallel*
API) — so it rides a dedicated adapter (``app.services.ma_env``) and trainer/preview branches rather
than the single-agent ``make_env`` factory. ``simple_spread`` is the canonical homogeneous-agents
case, which is exactly what SuperSuit's parameter-sharing bridge needs: one shared ``MlpPolicy``
trained over all N agents at once.

Covers the registry rows, the adapter (parallel env + the SuperSuit→SB3 vec-env bridge), the
swarm render-state extraction (per-agent + landmark world positions → the frame contract), the
negative skill bands, the watch-only gate (not human-playable), and a real smoke train through
``train_ppo`` (the same path the manager uses) confirming ``ep_rew_mean`` populates on CPU.
"""

from typing import Any

import numpy as np
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.preview import AgentSprite, WorldEntity
from app.schemas.training import PPOHyperparams, SelfPlayHyperparams, TrainConfig
from app.services import ma_env
from app.services.train_control import TrainControl
from app.services.trainer_ppo import train_ppo
from app.services.trainer_tag import _load_models, train_tag
from fastapi.testclient import TestClient

client = TestClient(app)

MPE = ["mpe_spread", "mpe_spread_swarm"]


# -- registry ---------------------------------------------------------------


def test_mpe_family_registered() -> None:
    families = {e.id: e.family for e in list_envs()}
    for eid in MPE:
        spec = get_env(eid)
        assert spec is not None, f"{eid} not registered"
        assert families[eid] == "petting_zoo"
        assert spec.gym_id == "simple_spread_v3"  # the mpe2 scenario module name
        assert spec.obs_type == "vector"  # per-agent vector obs, stacked by the SuperSuit bridge
        assert spec.action_space == "discrete"  # Discrete(5): stay / ±x / ±y
        assert spec.supported_algos == ["ppo"]  # parameter-sharing PPO only (no evo / Q-learning)
        assert spec.hw_requirement == "cpu"  # trains on CPU now; the GPU desktop scales it (G7b)
        assert spec.human_playable is False  # a swarm has no single human driver — watch + train only
        assert spec.competitive is False
        assert spec.solved_score < 0 and spec.min_score < spec.solved_score  # negative coverage range


def test_mpe_make_kwargs_carry_agent_count() -> None:
    assert get_env("mpe_spread").make_kwargs["N"] == 3  # type: ignore[union-attr]
    assert get_env("mpe_spread_swarm").make_kwargs["N"] == 6  # type: ignore[union-attr]


# -- the adapter (the 5th seam) ---------------------------------------------


def test_is_multi_agent_flag() -> None:
    assert ma_env.is_multi_agent(get_env("mpe_spread")) is True
    assert ma_env.is_multi_agent(get_env("cartpole")) is False
    assert ma_env.is_multi_agent(None) is False


def test_make_parallel_env_has_n_agents() -> None:
    env = ma_env.make_parallel_env("mpe_spread")
    try:
        obs, _ = env.reset(seed=0)
        assert len(env.agents) == 3  # N=3 homogeneous agents
        a0 = env.agents[0]
        assert env.observation_space(a0).shape == (18,)  # per-agent vector obs
        assert int(env.action_space(a0).n) == 5
        assert obs[a0].shape == (18,)
    finally:
        env.close()


def test_make_vec_env_bridges_to_sb3() -> None:
    """SuperSuit stacks the 3 homogeneous agents as 3 SB3 sub-envs (parameter sharing)."""
    vec = ma_env.make_vec_env("mpe_spread")
    try:
        assert vec.num_envs == 3  # one sub-env per agent
        assert vec.observation_space.shape == (18,)
        assert int(vec.action_space.n) == 5
    finally:
        vec.close()


def test_swarm_render_state_extraction() -> None:
    """agent_sprites / world_entities return the per-agent + landmark world positions the swarm
    canvas draws, and each entry validates against the frame contract (AgentSprite / WorldEntity)."""
    env = ma_env.make_parallel_env("mpe_spread")
    try:
        env.reset(seed=0)
        sprites = ma_env.agent_sprites(env)
        entities = ma_env.world_entities(env)
        assert len(sprites) == 3 and len(entities) == 3  # 3 agents, 3 landmark targets
        for s in sprites:
            AgentSprite(**s)  # validates x/y/role/size
            assert s["role"] == "agent"  # simple_spread agents are cooperative (no adversary)
            assert np.isfinite(s["x"]) and np.isfinite(s["y"])
        for e in entities:
            WorldEntity(**e)
            assert e["kind"] == "target"  # simple_spread landmarks are non-collidable coverage points
    finally:
        env.close()


# -- skill bands (watch-only, but the endpoint still derives bands) ----------


def test_mpe_skill_bands_are_negative() -> None:
    skill = client.get("/api/skill/mpe_spread").json()
    assert skill["min_score"] == -50.0 and skill["max_score"] == -15.0
    assert skill["bands"][-1]["id"] == "superhuman"


# -- not human-playable -----------------------------------------------------


def test_mpe_play_start_rejected() -> None:
    """A swarm isn't human-playable, so starting a play session is rejected (not a 5xx)."""
    res = client.post(
        "/api/play/start",
        json={"env_id": "mpe_spread", "mode": "human", "speed": 1.0},
    )
    assert 400 <= res.status_code < 500


# -- the seam through the PPO trainer (parameter sharing) --------------------


def test_ppo_trains_on_mpe_via_supersuit_bridge() -> None:
    """train_ppo builds the SuperSuit vec env, trains one shared policy over all agents, and
    ep_rew_mean (the per-agent episode return) populates — the same path the manager uses, on CPU."""
    metrics: list = []
    terminal = train_ppo(
        TrainConfig(
            env_id="mpe_spread", algo="ppo", seed=0, total_timesteps=768,
            hyperparams=PPOHyperparams(n_steps=128, batch_size=64),
        ),
        "simple_spread_v3", TrainControl(),
        metrics.append, lambda _p: None,
    )
    assert terminal == "finished"
    assert len(metrics) >= 1
    # VecMonitor populates the episode buffer (25-cycle episodes finish well within a rollout).
    assert metrics[-1].ep_rew_mean is not None
    assert metrics[-1].timesteps >= 768


# -- Predator–Prey (simple_tag) — heterogeneous species, per-species self-play (G7b-2) -------------

MPE_TAG = ["mpe_tag", "mpe_tag_pack"]


def test_mpe_tag_registered_trainable() -> None:
    for eid in MPE_TAG:
        spec = get_env(eid)
        assert spec is not None, f"{eid} not registered"
        assert spec.family == "petting_zoo"
        assert spec.gym_id == "simple_tag_v3"  # the mpe2 scenario module name
        assert spec.supported_algos == ["ppo"]
        assert spec.competitive is True  # predators vs. prey → per-species self-play trainer (G7b-2)
        assert spec.human_playable is False  # a swarm has no single human driver (play is G7b-3)
        assert spec.train_implemented is True  # frozen self-play trainer (G7b-2, trainer_tag.py)
        assert spec.hw_requirement == "cpu"
        # The prey (second species) has its own skill scale for the two-line ecosystem chart.
        assert spec.prey_min_score is not None and spec.prey_solved_score is not None
        assert spec.prey_min_score < spec.prey_solved_score  # negative floor up to a near-0 good end
        # The self-play round schedule is an exposed tunable (in the ppo block since algo stays "ppo").
        assert "rounds" in spec.hyperparams["ppo"]


def test_mpe_tag_make_kwargs_carry_species_counts() -> None:
    k = get_env("mpe_tag").make_kwargs  # type: ignore[union-attr]
    assert k["num_adversaries"] == 3 and k["num_good"] == 1 and k["num_obstacles"] == 2
    kp = get_env("mpe_tag_pack").make_kwargs  # type: ignore[union-attr]
    assert kp["num_adversaries"] == 6 and kp["num_good"] == 2


def test_mpe_tag_is_heterogeneous_with_roles_and_obstacles() -> None:
    """simple_tag has two species with DIFFERENT obs sizes (why parameter sharing won't do, G7b-2),
    and the render extraction tags predators 'adversary' + collidable landmarks 'obstacle'."""
    env = ma_env.make_parallel_env("mpe_tag")
    try:
        env.reset(seed=0)
        adversaries = [a for a in env.agents if a.startswith("adversary")]
        good = [a for a in env.agents if a.startswith("agent")]
        assert len(adversaries) == 3 and len(good) == 1
        # Heterogeneous: predator obs is larger than prey obs (the homogeneity break, G7b-2).
        assert env.observation_space(adversaries[0]).shape != env.observation_space(good[0]).shape

        sprites = ma_env.agent_sprites(env)
        roles = sorted({s["role"] for s in sprites})
        assert roles == ["adversary", "agent"]  # both species drive the swarm colours
        for s in sprites:
            AgentSprite(**s)
        entities = ma_env.world_entities(env)
        assert entities and all(e["kind"] == "obstacle" for e in entities)  # collidable landmarks
        for e in entities:
            WorldEntity(**e)
    finally:
        env.close()


def test_make_species_vec_env_isolates_each_species() -> None:
    """The per-species bridge exposes ONLY the learner species (homogeneous again → SuperSuit can
    parameter-share it): 3 predators of obs 16, or 1 prey of obs 14 — the heterogeneity that needs
    two policies. The opponent species is stepped by the frozen policy inside the wrapper."""
    adv_vec = ma_env.make_species_vec_env("mpe_tag", "adversary")
    try:
        assert adv_vec.num_envs == 3  # 3 predators stacked as 3 sub-envs (parameter sharing)
        assert adv_vec.observation_space.shape == (16,)
        assert int(adv_vec.action_space.n) == 5
    finally:
        ma_env.close_env(adv_vec)
    prey_vec = ma_env.make_species_vec_env("mpe_tag", "agent")
    try:
        assert prey_vec.num_envs == 1  # the single prey
        assert prey_vec.observation_space.shape == (14,)  # DIFFERENT obs size → can't parameter-share
    finally:
        ma_env.close_env(prey_vec)


def test_species_helpers() -> None:
    assert ma_env.species_present("mpe_tag") == ["adversary", "agent"]  # predators first (headline)
    assert ma_env.agent_role("adversary_2") == "adversary"
    assert ma_env.agent_role("agent_0") == "agent"
    assert ma_env.is_competitive_ma(get_env("mpe_tag")) is True
    assert ma_env.is_competitive_ma(get_env("mpe_spread")) is False  # cooperative, not competitive
    assert ma_env.is_competitive_ma(get_env("cartpole")) is False


def test_train_tag_self_play_smoke() -> None:
    """train_tag learns one shared policy per species by frozen-opponent alternating rounds, emitting a
    two-species frame, publishing BOTH species' preview policies, and snapshotting a packed two-model
    checkpoint that round-trips. The same path the manager uses for a competitive MA run, on CPU."""
    frames: list = []
    policy_pubs: list = []
    snapshots: list = []
    config = TrainConfig(
        env_id="mpe_tag", algo="ppo", seed=0, total_timesteps=2000,
        hyperparams=PPOHyperparams(n_steps=128, batch_size=64),
        self_play=SelfPlayHyperparams(rounds=2),
    )
    terminal = train_tag(
        config, "simple_tag_v3", TrainControl(),
        frames.append, policy_pubs.append, snapshots.append,
    )
    assert terminal == "finished"
    assert frames, "no ecosystem metrics frames"
    # Every frame carries BOTH species; both learning roles appear across the run (each gets a turn).
    assert all({s.role for s in f.species} == {"adversary", "agent"} for f in frames)
    assert {f.learning_role for f in frames} == {"adversary", "agent"}
    assert all(set(p.keys()) == {"adversary", "agent"} for p in policy_pubs)  # both preview policies
    # The packed checkpoint round-trips into two correctly-shaped models (16-obs predator, 14-obs prey).
    assert snapshots and snapshots[-1].artifact_name == "species.zip"
    models = _load_models(snapshots[-1].blob, config, ["adversary", "agent"])
    try:
        assert models["adversary"].observation_space.shape == (16,)
        assert models["agent"].observation_space.shape == (14,)
    finally:
        for m in models.values():
            ma_env.close_env(m.env)


def test_preview_watch_endpoint_toggles_active() -> None:
    """The training-free 'watch the ecosystem' endpoint marks the preview active/inactive. Visual is
    turned off first so no render thread spawns (keeps the test fast + thread-free)."""
    client.post("/api/preview", json={"visual": False})
    try:
        on = client.post("/api/preview/watch", json={"env_id": "mpe_tag", "on": True}).json()
        assert on["active"] is True
        off = client.post("/api/preview/watch", json={"env_id": "mpe_tag", "on": False}).json()
        assert off["active"] is False
    finally:
        client.post("/api/preview", json={"visual": True})


# -- SISL "cooperative swarm" — Pursuit (ADR-075) --------------------------------------------------
#
# A second PettingZoo family alongside MPE. Pursuit is homogeneous + cooperative, so it rides the
# EXISTING parameter-sharing path (trainer_ppo → make_vec_env) with no new trainer — the only new
# seams are SISL scenario loading (pettingzoo.sisl) and the server-JPEG render (ma_render="image",
# since SISL has no MPE world object for the position-based swarm canvas).


def test_pursuit_registered() -> None:
    spec = get_env("pursuit")
    assert spec is not None, "pursuit not registered"
    assert spec.family == "petting_zoo"
    assert spec.gym_id == "pursuit_v4"  # the pettingzoo.sisl scenario module name
    assert spec.obs_type == "vector"  # (7,7,3) local view, FLOAT → flattened by MlpPolicy (CPU, not Cnn)
    assert spec.action_space == "discrete"  # Discrete(5): stay + 4 cardinal moves
    assert spec.supported_algos == ["ppo"]  # parameter-sharing PPO only
    assert spec.competitive is False  # homogeneous cooperative → the simple_spread lane (one shared brain)
    assert spec.human_playable is False  # a swarm has no single human driver — watch + train only
    assert spec.train_implemented is True
    assert spec.hw_requirement == "cpu"  # small MlpPolicy trains on CPU
    assert spec.ma_render == "image"  # SISL → server-JPEG render (no MPE world to read positions from)
    assert spec.min_score < spec.solved_score  # the do-nothing floor sits below a good cooperative return
    # The config chosen for ACTIVE hunting (measured, ADR-075): surround=False + n_catch=1 (a single
    # pursuer tags an evader by reaching its square) so pursuers hunt independently and fan out (spread
    # ≈ 7.8/8), and shared_reward=False (each scored for its OWN catches) — the local reward that lets PPO
    # actually learn (the shared team reward hits a credit-assignment wall and trains worse than random).
    assert spec.make_kwargs["surround"] is False
    assert spec.make_kwargs["n_catch"] == 1
    assert spec.make_kwargs["shared_reward"] is False
    assert spec.make_kwargs["n_evaders"] == 16 and spec.make_kwargs["n_pursuers"] == 8


def test_sisl_scenario_loads_from_pettingzoo_sisl() -> None:
    """ma_env._load_scenario resolves a SISL id from pettingzoo.sisl (the generalised loader), the way
    an MPE id resolves from mpe2 — proving the new family shares one code path."""
    module = ma_env._load_scenario("pursuit_v4")
    assert hasattr(module, "parallel_env")  # the PettingZoo parallel-env factory


def test_pursuit_make_vec_env_bridges_to_sb3() -> None:
    """SuperSuit stacks the 8 homogeneous pursuers as 8 SB3 sub-envs (parameter sharing); the (7,7,3)
    local-view obs flows through unchanged (MlpPolicy flattens it)."""
    vec = ma_env.make_vec_env("pursuit")
    try:
        assert vec.num_envs == 8  # one sub-env per pursuer
        assert vec.observation_space.shape == (7, 7, 3)  # per-agent local view
        assert int(vec.action_space.n) == 5
    finally:
        ma_env.close_env(vec)


def test_pursuit_ppo_smoke_and_cooperative_watch_load() -> None:
    """train_ppo runs the cooperative SISL env through the SuperSuit bridge (the manager's path, on
    CPU) and snapshots a single shared model.zip; load_preview_predict then round-trips that checkpoint
    into a working predict over the (7,7,3) obs — the Watch-AI loader for a cooperative swarm."""
    from app.services.trainer_ppo import load_preview_predict

    metrics: list = []
    snapshots: list = []
    terminal = train_ppo(
        TrainConfig(
            env_id="pursuit", algo="ppo", seed=0, total_timesteps=2048,
            hyperparams=PPOHyperparams(n_steps=128, batch_size=64),
        ),
        "pursuit_v4", TrainControl(),
        metrics.append, lambda _p: None, on_snapshot=snapshots.append,
    )
    assert terminal == "finished"
    assert metrics, "no metrics frames from the SISL bridge"
    assert metrics[-1].timesteps >= 2048  # 8 stacked sub-envs advance the shared counter
    # The cooperative checkpoint is a single shared model.zip (NOT a per-species species.zip).
    assert snapshots and snapshots[-1].artifact_name == "model.zip"
    predict = load_preview_predict(snapshots[-1].blob)
    obs = np.zeros((7, 7, 3), dtype=np.float32)  # a pursuer's local view
    action = predict(obs)
    assert isinstance(action, int) and 0 <= action < 5  # a valid Discrete(5) move


# -- SISL Multiwalker — the CONTINUOUS cooperative swarm (ADR-076) ---------------------------------
#
# Multiwalker is the continuous-control sibling of Pursuit: three homogeneous walkers carry one
# package, so it rides the SAME parameter-sharing path (trainer_ppo → make_vec_env) and the SAME SISL
# seams (pettingzoo.sisl loading + ma_render="image") — the only new thing is the Box(4) action space,
# which the cooperative MA path routes end to end (the preview's box-aware predict + the SuperSuit
# bridge passing the Box space through).


def test_multiwalker_registered() -> None:
    spec = get_env("multiwalker")
    assert spec is not None, "multiwalker not registered"
    assert spec.family == "petting_zoo"
    assert spec.gym_id == "multiwalker_v9"  # the pettingzoo.sisl scenario module name
    assert spec.obs_type == "vector"  # (31,) sensor vector, FLOAT → flattened by MlpPolicy (CPU, not Cnn)
    assert spec.action_space == "box"  # Box(4): continuous leg-joint torques — the NEW bit vs pursuit
    assert spec.supported_algos == ["ppo"]  # parameter-sharing PPO only
    assert spec.competitive is False  # homogeneous cooperative → the simple_spread/pursuit lane
    assert spec.human_playable is False  # twelve leg joints across three robots — watch + train only
    assert spec.train_implemented is True
    assert spec.hw_requirement == "cpu"  # small MlpPolicy trains on CPU
    assert spec.ma_render == "image"  # SISL → server-JPEG render (Box2D pygame, no MPE world)
    # min_score=0 is the no-progress baseline, NOT the −100 fall floor: the forward reward is
    # potential-based (telescopes to package displacement), so a frozen don't-fall-but-don't-walk pose
    # scores ≈ 0; anchoring the meter at −100 would read that frozen policy as ~71% (the ADR-026 trap).
    assert spec.min_score == 0.0
    assert spec.min_score < spec.solved_score  # 0 < 40: a real forward traverse sits above no-progress
    # Config: three walkers, and shared_reward=False (local per-walker reward) per the Pursuit
    # credit-assignment lesson — cleaner gradient for parameter-sharing PPO than the shared team mean.
    assert spec.make_kwargs["n_walkers"] == 3
    assert spec.make_kwargs["shared_reward"] is False


def test_multiwalker_make_vec_env_bridges_to_sb3() -> None:
    """SuperSuit stacks the 3 homogeneous walkers as 3 SB3 sub-envs (parameter sharing); the (31,)
    sensor obs and the continuous Box(4) action space flow through unchanged."""
    vec = ma_env.make_vec_env("multiwalker")
    try:
        assert vec.num_envs == 3  # one sub-env per walker
        assert vec.observation_space.shape == (31,)  # per-walker sensor vector
        assert getattr(vec.action_space, "n", None) is None  # a Box, not a Discrete
        assert vec.action_space.shape == (4,)  # four leg-joint torques per walker
    finally:
        ma_env.close_env(vec)


def test_multiwalker_ppo_smoke_and_box_watch_load() -> None:
    """train_ppo runs the continuous SISL env through the SuperSuit bridge (the manager's cooperative
    path, on CPU) and snapshots one shared model.zip; load_preview_predict round-trips it into a working
    BOX predict — a clipped float vector of shape (4,), the continuous-MA Watch-AI loader."""
    from app.services.trainer_ppo import load_preview_predict

    metrics: list = []
    snapshots: list = []
    terminal = train_ppo(
        TrainConfig(
            env_id="multiwalker", algo="ppo", seed=0, total_timesteps=2048,
            hyperparams=PPOHyperparams(n_steps=128, batch_size=64),
        ),
        "multiwalker_v9", TrainControl(),
        metrics.append, lambda _p: None, on_snapshot=snapshots.append,
    )
    assert terminal == "finished"
    assert metrics, "no metrics frames from the SISL bridge"
    assert metrics[-1].timesteps >= 2048  # 3 stacked sub-envs advance the shared counter
    assert snapshots and snapshots[-1].artifact_name == "model.zip"  # cooperative single shared brain
    predict = load_preview_predict(snapshots[-1].blob)
    obs = np.zeros((31,), dtype=np.float32)  # a walker's sensor vector
    action = predict(obs)
    # A continuous (box) action: a float vector of shape (4,) clipped into the [-1, 1] joint range —
    # NOT an int. (Casting this to int — the old _choose_ma_actions bug — would crash → random.)
    action = np.asarray(action, dtype=np.float32)
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_preload_scenario_is_idempotent() -> None:
    """ma_env.preload_scenario imports the env's pygame-importing scenario module single-threaded
    (the launch-path guard against the trainer↔preview pygame import deadlock, ADR-076). It must be
    a no-op-safe, idempotent best-effort call — safe to run twice, and a no-op for an unknown id."""
    ma_env.preload_scenario("multiwalker")
    ma_env.preload_scenario("multiwalker")  # idempotent — a second call must not raise
    ma_env.preload_scenario("not_a_real_env")  # unknown id → silently returns, no raise


def test_choose_ma_actions_passes_box_vectors_through() -> None:
    """The preview's _choose_ma_actions must hand a continuous (box) predict's vector straight to the
    env, not cast it to int (which raises on a length-4 array → silent fall-back to random, so the
    trained Multiwalker swarm would never show). Guards the ADR-076 box-aware fix with a fake env."""
    from app.services.preview_streamer import preview_streamer

    class _FakeBoxEnv:
        agents = ["walker_0", "walker_1"]

        def action_space(self, _agent: str) -> Any:  # only reached on the random fallback (a failure here)
            raise AssertionError("box action should pass through, not fall back to action_space.sample()")

    vec = np.array([0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    preview_streamer.set_policy(lambda _obs: vec)  # a cooperative box predict (same vector for all agents)
    try:
        obs = {"walker_0": np.zeros(31, dtype=np.float32), "walker_1": np.zeros(31, dtype=np.float32)}
        actions = preview_streamer._choose_ma_actions(_FakeBoxEnv(), obs)
        assert set(actions) == {"walker_0", "walker_1"}
        for a in actions.values():
            assert np.allclose(np.asarray(a, dtype=np.float32), vec)  # the box vector, untouched
    finally:
        preview_streamer.detach_run()  # reset the published policy (clears _predict / _policies)
