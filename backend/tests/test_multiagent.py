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

import numpy as np
from app.envs.registry import get_env, list_envs
from app.main import app
from app.schemas.preview import AgentSprite, WorldEntity
from app.schemas.training import PPOHyperparams, TrainConfig
from app.services import ma_env
from app.services.train_control import TrainControl
from app.services.trainer_ppo import train_ppo
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


# -- Predator–Prey (simple_tag) — heterogeneous species, watch-only first step (G7b-1) -------------

MPE_TAG = ["mpe_tag", "mpe_tag_pack"]


def test_mpe_tag_registered_watch_only() -> None:
    for eid in MPE_TAG:
        spec = get_env(eid)
        assert spec is not None, f"{eid} not registered"
        assert spec.family == "petting_zoo"
        assert spec.gym_id == "simple_tag_v3"  # the mpe2 scenario module name
        assert spec.supported_algos == ["ppo"]
        assert spec.competitive is True  # predators vs. prey
        assert spec.human_playable is False  # a swarm has no single human driver (play is G7b-3)
        assert spec.train_implemented is False  # per-species trainer not built yet (G7b-2)
        assert spec.hw_requirement == "cpu"


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


def test_mpe_tag_training_is_gated() -> None:
    """train_implemented=False → the manager backstop rejects a training start (not a 5xx)."""
    res = client.post(
        "/api/train/start",
        json={"env_id": "mpe_tag", "algo": "ppo", "seed": 0, "total_timesteps": 1000,
              "hyperparams": {}},
    )
    assert 400 <= res.status_code < 500


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
