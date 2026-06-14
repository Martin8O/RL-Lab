from typing import Any, Literal

from pydantic import BaseModel


class Bilingual(BaseModel):
    en: str
    cz: str


class HyperparamDef(BaseModel):
    type: Literal["float", "int", "categorical"]
    default: Any
    recommended: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: list[str] | None = None


class EnvSpec(BaseModel):
    id: str
    gym_id: str
    display_name: Bilingual
    description: Bilingual
    family: str
    obs_type: Literal["vector", "image"]
    action_space: Literal["discrete", "box"]
    supported_algos: list[str]
    # algo_id -> param_id -> definition
    hyperparams: dict[str, dict[str, HyperparamDef]]
    # Score that counts as "solved" (100% of the goal). Drives the run-history archive
    # threshold (a run must reach ≥10% of this to be kept) and the "steps-to-solve" metric.
    solved_score: float
    # The bottom of the skill scale — the score that reads as 0% on the meter. 0 for envs whose
    # reward climbs from zero (CartPole), but negative for shaped envs that *start* deep in the
    # red (LunarLander begins ~-200 and a crash is ~-100), so the meter shows real progress
    # through the negative range instead of being pinned at 0% until the score turns positive.
    min_score: float = 0.0
    # Recommended PPO training budget for this env (the ★ default in the sidebar). Harder envs
    # need far more steps than CartPole, so this is per-env data; the sidebar builds its step
    # dropdown as a ladder around this value (×0.2 … ×4) and the store seeds it on env switch.
    default_total_timesteps: int
    human_playable: bool
    competitive: bool
    difficulty: Literal["beginner", "intermediate", "advanced"]
    hw_requirement: Literal["cpu", "gpu"]


_REGISTRY: dict[str, EnvSpec] = {}


def register(spec: EnvSpec) -> None:
    _REGISTRY[spec.id] = spec


def get_env(env_id: str) -> EnvSpec | None:
    return _REGISTRY.get(env_id)


def list_envs() -> list[EnvSpec]:
    return list(_REGISTRY.values())


def _standard_hyperparams() -> dict[str, dict[str, HyperparamDef]]:
    """The PPO + neuroevolution tunables shared by every simple vector/discrete env.

    These are the standard SB3 PPO defaults plus our neuroevolution defaults — sensible
    starting points for CartPole, LunarLander and the classic-control family alike (harder
    envs just need more steps, which is carried separately by ``default_total_timesteps`` and
    explained per-env in ``content/parameters.ts``). Built fresh per call so each EnvSpec owns
    its own definitions. Adding a new vector/discrete game reuses this verbatim — the param
    *surface* is identical; only the per-env *guidance* (content) and budget differ.
    """
    return {
        "ppo": {
            "learning_rate": HyperparamDef(
                type="float", default=3e-4, recommended=3e-4,
                min=1e-5, max=1e-2,
            ),
            "gamma": HyperparamDef(
                type="float", default=0.99, recommended=0.99,
                min=0.9, max=0.9999, step=0.001,
            ),
            "clip_range": HyperparamDef(
                type="float", default=0.2, recommended=0.2,
                min=0.05, max=0.4, step=0.01,
            ),
            "ent_coef": HyperparamDef(
                type="float", default=0.0, recommended=0.0,
                min=0.0, max=0.1, step=0.001,
            ),
            "n_steps": HyperparamDef(
                type="int", default=2048, recommended=2048,
                min=128, max=4096, step=128,
            ),
            "batch_size": HyperparamDef(
                type="int", default=64, recommended=64,
                min=32, max=512, step=32,
            ),
            "n_hidden_layers": HyperparamDef(
                type="int", default=2, recommended=2,
                min=1, max=4, step=1,
            ),
            "neurons_per_layer": HyperparamDef(
                type="int", default=64, recommended=64,
                min=16, max=512, step=16,
            ),
            "activation": HyperparamDef(
                type="categorical", default="tanh", recommended="tanh",
                choices=["tanh", "relu"],
            ),
        },
        "neuroevolution": {
            "population_size": HyperparamDef(
                type="int", default=50, recommended=50,
                min=10, max=200, step=10,
            ),
            "top_k_parents": HyperparamDef(
                type="int", default=10, recommended=10,
                min=2, max=50, step=1,
            ),
            "mutation_rate": HyperparamDef(
                type="float", default=0.1, recommended=0.1,
                min=0.01, max=1.0, step=0.01,
            ),
            "crossover_rate": HyperparamDef(
                type="float", default=0.5, recommended=0.5,
                min=0.0, max=1.0, step=0.05,
            ),
            "generations": HyperparamDef(
                type="int", default=30, recommended=30,
                min=5, max=200, step=5,
            ),
        },
    }


# ---------------------------------------------------------------------------
# CartPole-v1
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="cartpole",
        gym_id="CartPole-v1",
        display_name=Bilingual(en="CartPole-v1", cz="CartPole-v1"),
        description=Bilingual(
            en="Balance a pole on a cart by pushing left or right. The classic RL benchmark.",
            cz="Udržte tyč na vozíku pohybem doleva nebo doprava. Klasická RL úloha.",
        ),
        family="classic_control",
        obs_type="vector",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution"],
        hyperparams=_standard_hyperparams(),
        solved_score=500.0,  # CartPole-v1 caps at 500; ≥10% (50) is kept in run history
        default_total_timesteps=50_000,  # solves CartPole on CPU in well under a minute
        human_playable=True,
        competitive=False,
        difficulty="beginner",
        hw_requirement="cpu",
    )
)


# ---------------------------------------------------------------------------
# LunarLander-v3  (Box2D family — the CPU "canary" that proves the registry is
# data-driven: a vector-obs + discrete-action env like CartPole, so it reuses the
# exact MlpPolicy/CPU PPO path and the numpy neuroevolution path with no engine
# changes. Only setup cost was the gymnasium[box2d] wheel.)
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="lunarlander",
        gym_id="LunarLander-v3",
        display_name=Bilingual(en="LunarLander-v3", cz="LunarLander-v3"),
        description=Bilingual(
            en="Fire three thrusters to land the module gently on the pad between the flags. "
            "A Box2D physics classic — an 8-number state and four discrete actions.",
            cz="Pomocí tří trysek jemně přistaňte s modulem na plošině mezi vlajkami. "
            "Klasická úloha s fyzikou Box2D — stav o osmi číslech a čtyři diskrétní akce.",
        ),
        family="box2d",
        obs_type="vector",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution"],
        # Same hyperparameter surface as CartPole — the standard SB3 PPO + neuroevolution
        # defaults (LunarLander just needs more steps; see the per-env Total Steps note in
        # content/parameters.ts).
        hyperparams=_standard_hyperparams(),
        solved_score=200.0,  # LunarLander-v3 is "solved" at avg reward 200; ≥10% (20) is kept
        min_score=-200.0,  # 0% reference: a flailing/spinning agent bottoms out around -200 (a
        # bare crash is ~-100, worse runs go to -300); the meter fills from -200 up to +200
        default_total_timesteps=500_000,  # much harder than CartPole — needs a far larger budget
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


# ---------------------------------------------------------------------------
# Classic Control family — the rest of the discrete/vector envs (G1a). Like
# CartPole, these reuse the exact MlpPolicy/CPU PPO path and the numpy
# neuroevolution path with no engine changes (vector obs + discrete actions);
# only their reward scale differs (both give -1 per step, so scores are negative
# and the skill meter fills through the red — hence the negative min_score). The
# continuous-action members (Pendulum, MountainCarContinuous) are deferred to G1b
# because a `box` action space is a real engine seam, not a data row.
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="mountaincar",
        gym_id="MountainCar-v0",
        display_name=Bilingual(en="MountainCar-v0", cz="MountainCar-v0"),
        description=Bilingual(
            en="Drive an underpowered car up a steep hill. The engine is too weak to climb "
            "directly, so the agent must rock back and forth to build momentum — the classic "
            "sparse-reward exploration puzzle (a two-number state, three discrete actions).",
            cz="Vyjeďte se slabým autíčkem na strmý kopec. Motor je příliš slabý na přímý výjezd, "
            "takže agent musí houpáním sem a tam nabrat setrvačnost — klasická úloha s řídkou "
            "odměnou kladoucí důraz na zkoumání (stav o dvou číslech, tři diskrétní akce).",
        ),
        family="classic_control",
        obs_type="vector",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution"],
        hyperparams=_standard_hyperparams(),
        solved_score=-110.0,  # MountainCar-v0 reward_threshold; reward is -1/step (max 200 steps)
        min_score=-200.0,  # 0% reference: never reaching the flag = -1 × 200 steps (worst case)
        default_total_timesteps=200_000,  # sparse reward — needs lots of practice + exploration
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


register(
    EnvSpec(
        id="acrobot",
        gym_id="Acrobot-v1",
        display_name=Bilingual(en="Acrobot-v1", cz="Acrobot-v1"),
        description=Bilingual(
            en="Swing a two-link arm up until its free end rises above the bar. Torque can only "
            "be applied at the middle joint, so the agent must pump like a child on a swing to "
            "build height (a six-number state, three discrete actions).",
            cz="Rozhoupejte dvoukloubové rameno tak, aby jeho volný konec vystoupal nad tyč. "
            "Točivý moment lze přidat jen v prostředním kloubu, takže agent musí „pumpovat“ jako "
            "dítě na houpačce, aby nabral výšku (stav o šesti číslech, tři diskrétní akce).",
        ),
        family="classic_control",
        obs_type="vector",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution"],
        hyperparams=_standard_hyperparams(),
        solved_score=-100.0,  # Acrobot-v1 reward_threshold; reward is -1/step (max 500 steps)
        min_score=-500.0,  # 0% reference: never swinging up = -1 × 500 steps (worst case)
        default_total_timesteps=200_000,  # PPO reaches the goal within a few hundred thousand steps
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)
