"""G2a — Toy Text family + the discrete-observation (one-hot) seam.

Covers the registry rows, the OneHotObservation wrapper / shared factory, the client-render
state + grid layout for the grid-worlds, and a smoke train of both existing algorithms through
the seam (PPO + neuroevolution on a discrete-obs env).
"""

from app.envs.factory import OneHotObservation, make_env
from app.envs.registry import get_env, list_envs
from app.schemas.training import EvolutionHyperparams, PPOHyperparams, TrainConfig
from app.services.client_render import client_state, grid_layout
from app.services.train_control import TrainControl
from app.services.trainer_evolution import train_evolution
from app.services.trainer_ppo import train_ppo

TOY_TEXT = ["frozenlake", "frozenlake_noslip", "frozenlake8x8", "taxi", "cliffwalking"]


# -- registry ---------------------------------------------------------------


def test_toy_text_envs_registered_with_discrete_obs() -> None:
    families = {e.id: e.family for e in list_envs()}
    for eid in TOY_TEXT:
        spec = get_env(eid)
        assert spec is not None, f"{eid} not registered"
        assert families[eid] == "toy_text"
        assert spec.obs_type == "discrete"
        assert spec.action_space == "discrete"
        assert spec.supported_algos == ["ppo", "neuroevolution"]
        assert spec.turn_based is True
        assert spec.human_playable is True


def test_toy_text_score_scales() -> None:
    # Verified-in-venv numbers (gym reward_threshold / random-policy floors), not guesses.
    assert get_env("frozenlake").solved_score == 0.7
    assert get_env("frozenlake").min_score == 0.0  # reward 0/1 → fills like CartPole, no "measuring"
    assert get_env("taxi").solved_score == 8.0
    # Floor = the 200-step "did nothing / no delivery" return, so a stuck/idle agent reads ~0% on the
    # meter (not mid-scale). A deeper floor wrongly flattered a do-nothing run (see registry comment).
    assert get_env("taxi").min_score == -200.0
    assert get_env("cliffwalking").solved_score == -13.0
    assert get_env("cliffwalking").min_score == -200.0  # 200-step "never reached the goal" floor


# -- the one-hot seam (factory) ---------------------------------------------


def test_one_hot_wrapper_shapes_a_discrete_obs() -> None:
    env = make_env("frozenlake")
    try:
        assert isinstance(env, OneHotObservation)
        assert env.observation_space.shape == (16,)  # 4×4 grid → 16 states
        obs, _ = env.reset(seed=0)
        assert obs.shape == (16,) and obs.sum() == 1.0 and obs[0] == 1.0  # one-hot of the start cell
    finally:
        env.close()


def test_make_env_applies_variant_kwargs() -> None:
    # 8×8 map (64 states) and the deterministic no-slip variant share one gym_id; make_kwargs differ.
    big = make_env("frozenlake8x8")
    assert big.observation_space.shape == (64,)
    big.close()
    noslip = make_env("frozenlake_noslip")
    assert noslip.unwrapped.spec.kwargs.get("is_slippery") is False
    noslip.close()


def test_cliffwalking_gets_an_episode_cap() -> None:
    # CliffWalking has NO native TimeLimit — the factory must impose one (else a run can loop forever).
    assert make_env("cliffwalking").spec.max_episode_steps == 200
    # Play extends it by play_step_scale; a native-limit env keeps its limit at scale 1.
    assert make_env("cliffwalking", play_scale=3).spec.max_episode_steps == 600
    assert make_env("taxi").spec.max_episode_steps == 200


# -- client render (grid state + static board) ------------------------------


def test_client_state_grid_positions() -> None:
    fl = make_env("frozenlake")
    fl.reset(seed=0)
    assert client_state(fl) == [0.0, 0.0]  # start cell → row 0, col 0
    fl.close()
    taxi = make_env("taxi")
    taxi.reset(seed=0)
    state = client_state(taxi)
    assert state is not None and len(state) == 4  # [taxi_row, taxi_col, passenger_loc, destination]
    taxi.close()


def test_grid_layout_frozenlake_from_desc() -> None:
    env = make_env("frozenlake")
    layout = grid_layout(env)
    env.close()
    assert layout is not None
    assert layout["kind"] == "frozenlake" and layout["rows"] == 4 and layout["cols"] == 4
    assert len(layout["cells"]) == 16
    assert layout["cells"][0] == "start" and layout["cells"][15] == "goal"
    assert layout["cells"][5] == "hole"  # 'SFFF/FHFH/...' → (1,1) is a hole


def test_grid_layout_cliffwalking_and_taxi() -> None:
    cliff = make_env("cliffwalking")
    cl = grid_layout(cliff)
    cliff.close()
    assert cl is not None and cl["rows"] == 4 and cl["cols"] == 12
    assert cl["cells"][36] == "start" and cl["cells"][47] == "goal"
    assert all(cl["cells"][i] == "cliff" for i in range(37, 47))

    taxi = make_env("taxi")
    tx = grid_layout(taxi)
    taxi.close()
    assert tx is not None and tx["rows"] == 5 and tx["cols"] == 5
    assert {0, 4, 20, 23} == {i for i, c in enumerate(tx["cells"]) if c == "stop"}  # R, G, Y, B


def test_non_grid_env_has_no_grid_layout() -> None:
    env = make_env("cartpole")
    assert grid_layout(env) is None
    env.close()


# -- the seam through both trainers -----------------------------------------


def test_ppo_trains_on_discrete_obs_via_one_hot() -> None:
    """PPO's MlpPolicy consumes the one-hot vector — the discrete-obs env trains with no engine change."""
    metrics: list = []
    train_ppo(
        TrainConfig(
            env_id="frozenlake_noslip", algo="ppo", seed=0, total_timesteps=512,
            hyperparams=PPOHyperparams(n_steps=128, batch_size=64),
        ),
        "FrozenLake-v1", TrainControl(),
        metrics.append, lambda _p: None,
    )
    assert len(metrics) == 4  # 512 / 128 rollouts
    assert metrics[-1].timesteps == 512


def test_evolution_trains_on_discrete_obs_with_episode_cap() -> None:
    """Neuroevolution's numpy genome consumes the one-hot vector; CliffWalking's imposed cap means
    the unbounded env can't hang the scoring loop."""
    frames: list = []
    train_evolution(
        TrainConfig(
            env_id="cliffwalking", algo="neuroevolution", seed=0,
            evolution=EvolutionHyperparams(
                population_size=6, top_k_parents=3, mutation_rate=0.1,
                crossover_rate=0.5, generations=2, episodes=1,
            ),
        ),
        "CliffWalking-v1", TrainControl(),
        frames.append, lambda _f: None,
    )
    assert [f.generation for f in frames] == [1, 2]
