from typing import Any, Literal

from pydantic import BaseModel, Field


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
    # "vector" = a fixed-length float observation (CartPole); "image" = pixels (Atari, later);
    # "discrete" = a single integer state (Toy Text — which grid cell / which Taxi configuration).
    # A discrete obs reaches the vector-obs policies/genomes through a one-hot wrapper applied by
    # app.envs.factory.make_env (the discrete-observation seam, G2).
    obs_type: Literal["vector", "image", "discrete"]
    action_space: Literal["discrete", "box"]
    supported_algos: list[str]
    # algo_id -> param_id -> definition
    hyperparams: dict[str, dict[str, HyperparamDef]]
    # Extra kwargs passed to gym.make for per-variant rows that share one gym_id (e.g. FrozenLake's
    # map_name / is_slippery). Empty for envs whose id maps 1:1 to a gym id.
    make_kwargs: dict[str, Any] = Field(default_factory=dict)
    # An explicit episode step cap for envs Gymnasium leaves *unbounded* (CliffWalking has no native
    # TimeLimit → a poor policy would loop forever). None = use the env's own TimeLimit. Play scales
    # this by play_step_scale like every other env.
    episode_step_limit: int | None = None
    # Turn-based human play: the agent advances one step per key press (grid-worlds), instead of the
    # loop stepping continuously at the render rate. The AI/preview still step continuously (paced by
    # speed). False = the usual real-time control (CartPole, LunarLander, …).
    turn_based: bool = False
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
    # PLAY sessions (human + AI) multiply the env's max_episode_steps by this so a person has time
    # to actually play short envs; training keeps the standard length. 1 = no change.
    play_step_scale: int = 1
    # Whether the skill meter's 0% floor (min_score) is widened by play_step_scale for play. True for
    # STEP-PENALTY envs whose failure floor (≈ −1 × max_steps) genuinely grows with episode length
    # (MountainCar/Acrobot −1/step, Pendulum per-step cost). False for shaped/terminal-reward envs
    # whose failure score does NOT scale with steps (LunarLander: a crash ends the episode early at
    # ≈ −100 regardless of the cap) — widening their floor would inflate the displayed skill (a crash
    # reading as "above average"). Decouples "longer play episode" from "deeper failure floor".
    floor_scales_with_steps: bool = True
    # Sparse 0/1 reward (FrozenLake): the episode pays 0 the whole way and +1 only on reaching the
    # goal, so the *running cumulative* score during PLAY is not a valid skill reading (it sits at 0
    # until the final step). The play skill meter shows "measuring…" until the episode ends for these
    # — like it already does for shaped/penalty envs (min_score < 0) whose partial score also isn't a
    # reading. False = the running score is a valid lower bound (CartPole climbs from 0).
    sparse_reward: bool = False
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


def _standard_hyperparams(q_episodes: int = 5_000) -> dict[str, dict[str, HyperparamDef]]:
    """The PPO + neuroevolution + Q-learning tunables shared by every simple vector/discrete env.

    These are the standard SB3 PPO defaults plus our neuroevolution defaults — sensible
    starting points for CartPole, LunarLander and the classic-control family alike (harder
    envs just need more steps, which is carried separately by ``default_total_timesteps`` and
    explained per-env in ``content/parameters.ts``). Built fresh per call so each EnvSpec owns
    its own definitions. Adding a new vector/discrete game reuses this verbatim — the param
    *surface* is identical; only the per-env *guidance* (content) and budget differ.

    The ``q_learning`` block is included for every env but only *exposed* where the env lists
    ``q_learning`` in ``supported_algos`` (Toy Text, whose discrete obs is the table's native
    input). ``q_episodes`` is that env's ★ recommended episode budget (Q-learning's "Total Steps"):
    a small deterministic maze wants a few thousand, Taxi's 500 states want far more.
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
        "q_learning": {
            "learning_rate": HyperparamDef(  # α — the table-update step (far larger than PPO's lr)
                type="float", default=0.1, recommended=0.1,
                min=0.01, max=1.0, step=0.01,
            ),
            "gamma": HyperparamDef(
                type="float", default=0.99, recommended=0.99,
                min=0.8, max=0.999, step=0.001,
            ),
            "epsilon_start": HyperparamDef(
                type="float", default=1.0, recommended=1.0,
                min=0.1, max=1.0, step=0.05,
            ),
            "epsilon_end": HyperparamDef(
                type="float", default=0.05, recommended=0.05,
                min=0.0, max=0.5, step=0.01,
            ),
            "epsilon_decay": HyperparamDef(  # fraction of the budget to anneal ε over, then hold
                type="float", default=0.5, recommended=0.5,
                min=0.1, max=1.0, step=0.05,
            ),
            "episodes": HyperparamDef(  # the training budget (this algorithm's "Total Steps")
                type="int", default=q_episodes, recommended=q_episodes,
                min=500, max=50_000, step=500,
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
        play_step_scale=3,  # Box2D landing takes a while by hand — give a human ~3× the episode length
        floor_scales_with_steps=False,  # shaped/terminal reward: a crash ends early ≈-100, doesn't scale with the 3× cap
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
        play_step_scale=3,  # 200 steps is over in seconds by hand — give a human 3× longer to play
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
        play_step_scale=3,  # 500 steps is brief by hand — give a human 3× longer to swing it up
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


# ---------------------------------------------------------------------------
# Classic Control family — the continuous-action members (G1b). Unlike the
# discrete envs above, these have a `box` (continuous) action space, which is a
# real engine seam rather than a data row: the trained policy outputs a real
# number (a torque / a throttle) instead of picking one of N buttons, so the
# play session, the AI-policy loader, the preview forward and the neuroevolution
# genome all had to learn to emit + step a continuous action. PPO's MlpPolicy
# already switches to a Gaussian action head for a box space automatically; the
# numpy forwards (preview + neuroevolution) tanh-scale their output into
# [low, high]. Still vector-obs + CPU, so they reuse the standard hyperparams.
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="pendulum",
        gym_id="Pendulum-v1",
        display_name=Bilingual(en="Pendulum-v1", cz="Pendulum-v1"),
        description=Bilingual(
            en="Swing a frictionless pendulum upright and hold it there, using a continuous "
            "torque you dial anywhere from full one way to full the other. The first "
            "continuous-control task — there are no buttons, the agent chooses a real number "
            "each step (a three-number state, one continuous action).",
            cz="Vyhoupněte bezfrikční kyvadlo do svislé polohy a udržte ho tam pomocí spojitého "
            "točivého momentu, který plynule nastavíte od plného v jednom směru po plný v "
            "druhém. První úloha se spojitým řízením — nejsou žádná tlačítka, agent v každém "
            "kroku volí reálné číslo (stav o třech číslech, jedna spojitá akce).",
        ),
        family="classic_control",
        obs_type="vector",
        action_space="box",
        supported_algos=["ppo", "neuroevolution"],
        hyperparams=_standard_hyperparams(),
        # Pendulum-v1 has NO official gym reward_threshold (verified: gym.spec(...).reward_threshold
        # is None). Reward is a per-step cost (angle² + 0.1·speed² + 0.001·torque²), so the return is
        # always negative and the best achievable is near 0. -150 is the widely-used "near-optimal /
        # solved" return for a 200-step episode (a strong PPO/SAC agent lands roughly -150…-250).
        solved_score=-150.0,
        # 0% reference: a flailing/do-nothing agent scores around -1200…-1400 (measured), worst runs
        # reach ~-1700; -1600 is a representative floor so the meter fills through the deep negatives.
        min_score=-1600.0,
        default_total_timesteps=200_000,  # PPO learns a good swing-up-and-hold within a few 100k steps
        play_step_scale=3,  # 200 steps is over in seconds by hand — give a human 3× longer to play
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


register(
    EnvSpec(
        id="mountaincarcontinuous",
        gym_id="MountainCarContinuous-v0",
        display_name=Bilingual(
            en="MountainCarContinuous-v0", cz="MountainCarContinuous-v0"
        ),
        description=Bilingual(
            en="The mountain-car hill again, but the throttle is now continuous — the agent dials "
            "how hard and which way to push instead of three buttons. Reaching the flag pays a big "
            "+100 bonus and using force costs a little, so the reward is sparse: the classic "
            "exploration puzzle in continuous form (a two-number state, one continuous action).",
            cz="Znovu kopec s autíčkem, ale plyn je teď spojitý — agent nastavuje, jak silně a "
            "kterým směrem tlačit, místo tří tlačítek. Dosažení vlajky vyplatí velký bonus +100 a "
            "použití síly stojí trochu, takže odměna je řídká: klasická úloha na zkoumání ve "
            "spojité podobě (stav o dvou číslech, jedna spojitá akce).",
        ),
        family="classic_control",
        obs_type="vector",
        action_space="box",
        supported_algos=["ppo", "neuroevolution"],
        hyperparams=_standard_hyperparams(),
        solved_score=90.0,  # MountainCarContinuous-v0 reward_threshold (reach the flag = +100 bonus)
        # 0% reference: a do-nothing agent scores 0.0 (no force cost, never solves); the reward only
        # climbs above ~0 once the flag is reached (+100), so the meter behaves like CartPole's
        # (fills from 0 up). Wasted-force runs go slightly negative and simply read as 0%.
        min_score=0.0,
        # Sparse-reward exploration trap: vanilla PPO often never reaches the flag at this budget and
        # sits near 0 — neuroevolution's population search tends to discover the goal more reliably.
        # Documented honestly in content/parameters.ts; the budget is a starting point, not a promise.
        default_total_timesteps=100_000,
        human_playable=True,
        competitive=False,
        difficulty="advanced",
        hw_requirement="cpu",
    )
)


# ---------------------------------------------------------------------------
# Toy Text family (G2a) — the discrete-observation envs. Unlike every env above,
# the observation is a single integer (which grid cell / which Taxi state), not a
# vector. That is a new seam: app.envs.factory.OneHotObservation turns the int into
# a length-n one-hot vector so the *same* MlpPolicy (PPO) and numpy genome
# (neuroevolution) used for CartPole apply with no engine change. (Tabular
# Q-learning — the native consumer of a discrete state — lands in G2b.) These are
# grid-worlds, so they are client-rendered (SVG board) and human-played turn-based:
# one move per key press. Scores were verified in the venv (random-policy floors,
# gym reward_threshold, deprecated ids) per the new-env pre-delivery checklist.
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="frozenlake",
        gym_id="FrozenLake-v1",
        display_name=Bilingual(en="FrozenLake (4×4)", cz="FrozenLake (4×4)"),
        description=Bilingual(
            en="Cross a frozen 4×4 lake from start to goal without falling through a hole. The ice "
            "is slippery, so a move can slide you sideways — you reach the goal only some of the "
            "time. Reward is 1 for reaching the goal, 0 otherwise; 'solved' is a 70% success rate.",
            cz="Přejděte zamrzlé jezero 4×4 ze startu do cíle, aniž byste se propadli dírou. Led "
            "klouže, takže vás krok může smeknout do strany — do cíle se dostanete jen občas. "
            "Odměna je 1 za dosažení cíle, jinak 0; „vyřešeno“ je 70% úspěšnost.",
        ),
        family="toy_text",
        obs_type="discrete",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution", "q_learning"],
        hyperparams=_standard_hyperparams(q_episodes=8_000),  # slippery 16-state lake
        sparse_reward=True,  # 0/1 reward → play meter "measures" until the goal is reached
        solved_score=0.7,  # gym reward_threshold: a 70% success rate (reward is 1 on reaching goal)
        min_score=0.0,  # reward is 0/1 — a failing agent scores 0, so the meter fills 0→0.7
        default_total_timesteps=200_000,  # slippery + sparse reward needs lots of episodes
        play_step_scale=1,  # turn-based human play (one move per key press) — no time pressure
        turn_based=True,
        human_playable=True,
        competitive=False,
        difficulty="beginner",
        hw_requirement="cpu",
    )
)


register(
    EnvSpec(
        id="frozenlake_noslip",
        gym_id="FrozenLake-v1",
        display_name=Bilingual(
            en="FrozenLake (4×4, no slip)", cz="FrozenLake (4×4, bez kluzu)"
        ),
        description=Bilingual(
            en="The 4×4 frozen lake with the ice made non-slippery: every move goes exactly where "
            "you point it. The deterministic version — the gentlest grid-world, where a learner can "
            "find the single safe path to the goal and solve it almost perfectly.",
            cz="Jezero 4×4 s ledem nastaveným jako neklouzavý: každý krok jde přesně tam, kam "
            "míříte. Deterministická verze — nejjednodušší mřížkový svět, kde se dá najít jediná "
            "bezpečná cesta do cíle a vyřešit ho téměř dokonale.",
        ),
        family="toy_text",
        obs_type="discrete",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution", "q_learning"],
        hyperparams=_standard_hyperparams(q_episodes=3_000),  # deterministic 16-state maze — fast
        sparse_reward=True,  # 0/1 reward → play meter "measures" until the goal is reached
        make_kwargs={"is_slippery": False},
        solved_score=0.7,  # same gym threshold; a deterministic agent reaches ~1.0 success
        min_score=0.0,
        default_total_timesteps=50_000,  # deterministic + tiny — solves fast
        play_step_scale=1,
        turn_based=True,
        human_playable=True,
        competitive=False,
        difficulty="beginner",
        hw_requirement="cpu",
    )
)


register(
    EnvSpec(
        id="frozenlake8x8",
        gym_id="FrozenLake-v1",
        display_name=Bilingual(en="FrozenLake (8×8)", cz="FrozenLake (8×8)"),
        description=Bilingual(
            en="The bigger 8×8 frozen lake — more ice, more holes and a longer slippery path to the "
            "goal. Same rules as the 4×4 (reach the goal for reward 1; 'solved' is a 70% success "
            "rate) but a much harder exploration problem.",
            cz="Větší zamrzlé jezero 8×8 — víc ledu, víc děr a delší kluzká cesta do cíle. Stejná "
            "pravidla jako u 4×4 (dosáhnout cíle za odměnu 1; „vyřešeno“ je 70% úspěšnost), ale "
            "mnohem těžší úloha na zkoumání.",
        ),
        family="toy_text",
        obs_type="discrete",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution", "q_learning"],
        hyperparams=_standard_hyperparams(q_episodes=20_000),  # 64 states + slip — needs the most
        sparse_reward=True,  # 0/1 reward → play meter "measures" until the goal is reached
        make_kwargs={"map_name": "8x8"},
        solved_score=0.7,
        min_score=0.0,
        default_total_timesteps=400_000,  # 64 states + sparse reward — needs the most practice
        play_step_scale=1,
        turn_based=True,
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


register(
    EnvSpec(
        id="taxi",
        gym_id="Taxi-v3",
        display_name=Bilingual(en="Taxi-v3", cz="Taxi-v3"),
        description=Bilingual(
            en="Drive a taxi on a 5×5 grid: pick up the passenger at one marked stop and drop them "
            "at another. Every step costs −1, a correct drop-off pays +20, and an illegal pickup or "
            "drop-off costs −10, so a good driver scores around +8 (the 'solved' mark).",
            cz="Řiďte taxík na mřížce 5×5: vyzvedněte cestujícího na jedné označené zastávce a "
            "vysaďte ho na jiné. Každý krok stojí −1, správné vysazení vyplatí +20 a nelegální "
            "vyzvednutí či vysazení stojí −10, takže dobrý řidič dosáhne kolem +8 (hranice „vyřešeno“).",
        ),
        family="toy_text",
        obs_type="discrete",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution", "q_learning"],
        hyperparams=_standard_hyperparams(q_episodes=20_000),  # 500 states — Q-learning's showcase
        solved_score=8.0,  # gym reward_threshold
        # 0% reference = "ran out the 200-step episode without ever delivering" ≈ −200 (−1/step, no
        # +20). This is the right floor for the skill meter: an idle/stuck agent that delivers nothing
        # reads ~0% (it has shown no skill), and only an actual delivery (the +20 makes the score climb
        # toward +8) lifts the meter. Worse runs (lots of illegal −10 pickups/drop-offs go to ~−800)
        # simply clamp to 0%. A larger floor (−800) was wrong — it made a do-nothing −200 read ~74%.
        min_score=-200.0,
        default_total_timesteps=500_000,  # 500 states, six actions — needs a sizeable budget
        play_step_scale=1,  # turn-based; 200 steps is ample for a human to deliver a fare
        turn_based=True,
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


register(
    EnvSpec(
        id="cliffwalking",
        gym_id="CliffWalking-v1",  # v0 is deprecated in this gymnasium — verified in the venv
        display_name=Bilingual(en="CliffWalking", cz="CliffWalking"),
        description=Bilingual(
            en="Walk from the bottom-left corner to the bottom-right goal along the edge of a cliff. "
            "Every step costs −1; stepping onto the cliff costs −100 and sends you back to the "
            "start. The optimal route hugs the cliff edge for a return of about −13.",
            cz="Dojděte z levého dolního rohu do cíle vpravo dole podél okraje útesu. Každý krok "
            "stojí −1; vstup na útes stojí −100 a vrátí vás na start. Optimální cesta vede těsně "
            "podél okraje útesu s návratem kolem −13.",
        ),
        family="toy_text",
        obs_type="discrete",
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution", "q_learning"],
        hyperparams=_standard_hyperparams(q_episodes=5_000),  # 48 states, dense reward — solves cleanly
        solved_score=-13.0,  # optimal return (no gym reward_threshold); the risky cliff-edge path
        # 0% reference = "ran out the 200-step cap without reaching the goal" = −200 (−1/step, no
        # cliff falls). This is the right floor for the skill meter: an agent that just sits/wanders
        # (e.g. PPO's common local optimum — go up to dodge the cliff, then get stuck) reads ~0%, and
        # only actually reaching the goal lifts the meter toward 100% (optimal −13). Catastrophic runs
        # that keep falling off the cliff (−100 each → ≪ −200) clamp to 0%. A larger floor (−2000) was
        # wrong — it made a stuck −200 read ~91% (near-superhuman), hiding that nothing was achieved.
        min_score=-200.0,
        default_total_timesteps=200_000,
        episode_step_limit=200,  # CliffWalking has NO native TimeLimit — cap it so a run can't hang
        play_step_scale=1,  # turn-based; 200 steps is ample to reach the goal by hand
        turn_based=True,
        human_playable=True,
        competitive=False,
        difficulty="intermediate",
        hw_requirement="cpu",
    )
)


# ---------------------------------------------------------------------------
# Atari family (ALE) — G4a "install + human-play on CPU now" batch.
#
# These are the first **image-observation** envs: the observation is a 210×160×3
# RGB frame, not a vector/discrete state. That breaks two CartPole-shaped
# assumptions the rest of the registry never hit, so the family is *data-rows +
# gating*, not free like Toy Text:
#   * obs_type="image" → a future trainer needs a CnnPolicy + a GPU (the
#     trainer_ppo._build_model seam). So training is **gated**: hw_requirement="gpu"
#     and the UI disables Run while no CUDA device is present (see /api/system). Atari
#     trains in G4b on the desktop; until then these are **human-playable now** — human
#     play needs no neural net, only env stepping + the JPEG render path, both of which
#     already exist (client_render returns None for an image obs → server JPEG).
#   * the numpy neuroevolution genome + tabular Q-learning can't consume an image, so
#     supported_algos=["ppo"] (PPO-only) as data — the same opt-out pattern CarRacing uses.
#
# Every Atari env is built with full_action_space=True (make_kwargs) so all 18 ALE
# actions sit at fixed indices across *every* game — a single shared keyboard map
# (content/playKeymaps.ts ATARI_KEYMAP) plays them all, instead of each game's
# game-specific minimal action set landing at different indices.
#
# Per-game skill bands come from [min_score, solved_score] like every other env (no
# bespoke band table needed): symmetric games (Pong −21…21, Boxing/Tennis ±) get a
# negative floor; one-directional arcade scores fill 0 → a "really good" target. The
# whole batch was venv-verified (create + reset + step + render) per the new-env checklist.
# Adding more of the 100+ ALE titles later is a one-row-per-game data change here.
# ---------------------------------------------------------------------------


def _atari_spec(
    env_id: str,
    display: str,
    game: str,
    difficulty: Literal["beginner", "intermediate", "advanced"],
    min_score: float,
    solved_score: float,
    desc_en: str,
    desc_cz: str,
) -> EnvSpec:
    """Build one Atari EnvSpec from a data row (the family is otherwise identical)."""
    return EnvSpec(
        id=env_id,
        gym_id=f"ALE/{game}-v5",
        display_name=Bilingual(en=display, cz=display),  # arcade titles are proper nouns
        description=Bilingual(en=desc_en, cz=desc_cz),
        family="atari",
        obs_type="image",
        action_space="discrete",
        supported_algos=["ppo"],  # image obs → CnnPolicy/GPU only; evo+Q-learning can't consume pixels
        hyperparams=_standard_hyperparams(),
        # All 18 ALE actions at fixed indices → one shared keyboard map across the whole family.
        make_kwargs={"full_action_space": True},
        solved_score=solved_score,
        min_score=min_score,
        default_total_timesteps=10_000_000,  # a realistic Atari PPO budget (gated to the GPU desktop)
        play_step_scale=1,  # real-time arcade play; episodes end on game-over, the speed slider paces it
        human_playable=True,
        competitive=False,
        difficulty=difficulty,
        hw_requirement="gpu",  # training needs a GPU; the UI gates Run, human play stays available now
    )


# id, display, ALE game, difficulty, min_score, solved_score, description EN, description CZ
_ATARI_GAMES: list[
    tuple[str, str, str, Literal["beginner", "intermediate", "advanced"], float, float, str, str]
] = [
    ("pong", "Pong", "Pong", "beginner", -21.0, 21.0,
     "The original video game: bounce the ball past the built-in opponent's paddle while defending "
     "your own side. First to 21 wins; your score is your points minus the opponent's (−21 to +21).",
     "Úplně první videohra: odrážejte míček za pálku vestavěného soupeře a zároveň braňte svou stranu. "
     "Kdo první nasbírá 21 bodů, vyhrává; skóre je vaše body mínus soupeřovy (−21 až +21)."),
    ("breakout", "Breakout", "Breakout", "beginner", 0.0, 120.0,
     "Bounce a ball off a paddle to smash every brick in the wall above. Don't let the ball fall past "
     "the paddle — each brick destroyed scores points, and clearing the wall is the goal.",
     "Odrážejte míček pálkou a rozbijte všechny cihly ve zdi nahoře. Nenechte míček propadnout pod "
     "pálku — každá rozbitá cihla přidá body a cílem je smést celou zeď."),
    ("spaceinvaders", "Space Invaders", "SpaceInvaders", "beginner", 0.0, 2000.0,
     "Move your laser cannon left and right and shoot down descending rows of alien invaders before "
     "they reach the ground, using the shields for cover. Each alien hit scores points.",
     "Posouvejte laserové dělo doleva a doprava a sestřelujte klesající řady mimozemšťanů dřív, než "
     "dosednou na zem; kryjte se za štíty. Každý zásah přidá body."),
    ("mspacman", "Ms. Pac-Man", "MsPacman", "beginner", 0.0, 6000.0,
     "Steer Ms. Pac-Man around the maze eating all the dots while dodging four ghosts. Grab a power "
     "pellet to turn the tables and eat the ghosts for bonus points.",
     "Veďte Ms. Pac-Man bludištěm, snězte všechny tečky a vyhýbejte se čtyřem duchům. Sebráním "
     "speciální kuličky se karta obrátí a duchy můžete za body sníst."),
    ("qbert", "Q*bert", "Qbert", "intermediate", 0.0, 15000.0,
     "Hop Q*bert across a pyramid of cubes to change every cube to the target colour, while avoiding "
     "enemies that chase you off the edge. Change all cubes to clear the stage.",
     "Skákejte s Q*bertem po pyramidě kostek a přebarvěte každou na cílovou barvu; vyhýbejte se "
     "nepřátelům, kteří vás shazují z okraje. Přebarvením všech kostek postoupíte dál."),
    ("seaquest", "Seaquest", "Seaquest", "intermediate", 0.0, 20000.0,
     "Pilot a submarine to shoot enemy sharks and subs while rescuing divers. Watch your oxygen — "
     "surface in time or you drown. Rescues and kills both score points.",
     "Řiďte ponorku, střílejte nepřátelské žraloky a ponorky a zachraňujte potápěče. Hlídejte kyslík — "
     "vynořte se včas, jinak se utopíte. Záchrany i zásahy přidávají body."),
    ("enduro", "Enduro", "Enduro", "intermediate", 0.0, 700.0,
     "An endurance race: pass as many cars as you can over day-and-night cycles without crashing. Your "
     "score is the number of cars you overtake — keep the throttle down and weave through traffic.",
     "Vytrvalostní závod: předjeďte co nejvíc aut během střídání dne a noci, aniž byste havarovali. "
     "Skóre je počet předjetých aut — držte plyn a kličkujte mezi vozy."),
    ("beamrider", "Beam Rider", "BeamRider", "intermediate", 0.0, 8000.0,
     "Defend a grid of laser beams: shoot waves of enemy ships sweeping across the lanes and dodge "
     "their fire. Survive each sector's wave to advance. Every enemy destroyed scores points.",
     "Braňte mřížku laserových paprsků: střílejte vlny nepřátelských lodí klouzajících po drahách a "
     "uhýbejte jejich palbě. Přežijte vlnu v každém sektoru. Každý zničený nepřítel přidá body."),
    ("asteroids", "Asteroids", "Asteroids", "intermediate", 0.0, 10000.0,
     "Fly a ship in open space, blasting drifting asteroids into smaller pieces and dodging the "
     "fragments and flying saucers. Clear the field to advance; each rock destroyed scores points.",
     "Pilotujte loď v otevřeném prostoru, rozstřelujte plující asteroidy na menší kusy a uhýbejte "
     "úlomkům a létajícím talířům. Vyčištěním pole postoupíte; každý kámen přidá body."),
    ("asterix", "Asterix", "Asterix", "beginner", 0.0, 8000.0,
     "Collect helpful objects while dodging the deadly ones that move across the rows. Grab the good "
     "items for points and avoid being hit — survive and collect as much as you can.",
     "Sbírejte užitečné předměty a vyhýbejte se smrtícím, které se pohybují po řadách. Dobré předměty "
     "dávají body; nenechte se zasáhnout — přežijte a posbírejte co nejvíc."),
    ("alien", "Alien", "Alien", "intermediate", 0.0, 7000.0,
     "Trapped in a spaceship's corridors, destroy the alien eggs while three aliens hunt you. Use your "
     "flamethrower and a power-up to fight back. Destroying eggs and aliens scores points.",
     "Uvězněni v chodbách kosmické lodi ničte vejce vetřelců, zatímco vás pronásledují tři vetřelci. "
     "Použijte plamenomet a posilu k obraně. Ničení vajec a vetřelců přidává body."),
    ("amidar", "Amidar", "Amidar", "intermediate", 0.0, 1700.0,
     "Move along a grid painting its lines while avoiding roaming enemies. Outline a box to fill it for "
     "points; clear the board without being caught.",
     "Pohybujte se po mřížce a vybarvujte její čáry, zatímco se vyhýbáte bloudícím nepřátelům. Obkroužení "
     "obdélníku ho vyplní za body; vyčistěte plochu, aniž vás chytí."),
    ("assault", "Assault", "Assault", "beginner", 0.0, 2500.0,
     "Defend against a mothership raining down enemy ships from above. Move your cannon and shoot them "
     "before they overwhelm you, watching your gun's heat. Every enemy destroyed scores points.",
     "Braňte se mateřské lodi, která shora chrlí nepřátelské stíhače. Posouvejte dělo a sestřelujte je, "
     "než vás zahltí; hlídejte přehřátí zbraně. Každý zničený nepřítel přidá body."),
    ("atlantis", "Atlantis", "Atlantis", "beginner", 0.0, 30000.0,
     "Defend the underwater city of Atlantis with three gun emplacements, shooting down waves of enemy "
     "ships before they destroy your installations. Every ship shot down scores points.",
     "Braňte podmořské město Atlantis třemi dělostřeleckými pozicemi a sestřelujte vlny nepřátelských "
     "lodí dřív, než zničí vaše stavby. Každá sestřelená loď přidá body."),
    ("bankheist", "Bank Heist", "BankHeist", "intermediate", 0.0, 1000.0,
     "Drive through a maze of city streets robbing banks while dodging police cars. Drop dynamite to "
     "block pursuers and manage your fuel. Each bank robbed scores points.",
     "Projíždějte bludištěm městských ulic, vykrádejte banky a unikejte policejním autům. Dynamitem "
     "blokujte pronásledovatele a hlídejte palivo. Každá vyloupená banka přidá body."),
    ("battlezone", "Battle Zone", "BattleZone", "intermediate", 0.0, 35000.0,
     "A first-person tank battle on a wireframe battlefield: hunt and destroy enemy tanks and missiles "
     "from your cockpit while avoiding their fire. Each enemy destroyed scores points.",
     "Tanková bitva z pohledu první osoby na drátěném bojišti: z kokpitu hledejte a ničte nepřátelské "
     "tanky a rakety a vyhýbejte se jejich palbě. Každý zničený nepřítel přidá body."),
    ("berzerk", "Berzerk", "Berzerk", "intermediate", 0.0, 1600.0,
     "Escape a maze of rooms full of robots, shooting them down while avoiding the walls (which are "
     "deadly) and the relentless Evil Otto. Each robot destroyed scores points.",
     "Unikejte bludištěm místností plných robotů, sestřelujte je a vyhýbejte se stěnám (jsou smrtící) i "
     "neúnavnému Evil Ottovi. Každý zničený robot přidá body."),
    ("bowling", "Bowling", "Bowling", "beginner", 0.0, 160.0,
     "Ten-pin bowling: time your throw and curve the ball to knock down as many pins as possible across "
     "ten frames. A perfect game is 300; a good score is around 160.",
     "Bowling: načasujte hod a zatočte míčem, abyste v deseti kolech srazili co nejvíc kuželek. Dokonalá "
     "hra je 300; dobré skóre je kolem 160."),
    ("boxing", "Boxing", "Boxing", "beginner", -100.0, 100.0,
     "A top-down boxing match: land more punches than the opponent before time runs out. Your score is "
     "your hits minus theirs (−100 to +100); a knockout ends it early.",
     "Box z ptačí perspektivy: zasaďte víc úderů než soupeř, než vyprší čas. Skóre je vaše zásahy mínus "
     "soupeřovy (−100 až +100); knokaut zápas ukončí dřív."),
    ("carnival", "Carnival", "Carnival", "beginner", 0.0, 5000.0,
     "A shooting gallery: blast the targets, ducks and bonus items moving across the rows, but don't run "
     "out of bullets. Each target hit scores points.",
     "Pouťová střelnice: sestřelujte terče, kachny a bonusové předměty pohybující se po řadách, ale "
     "nevyčerpejte náboje. Každý zásah přidá body."),
    ("centipede", "Centipede", "Centipede", "intermediate", 0.0, 12000.0,
     "Blast a centipede winding down through a field of mushrooms before it reaches you, while fending "
     "off spiders and fleas. Every segment and creature shot scores points.",
     "Sestřelte stonožku klikatící se dolů polem hub dřív, než vás dostihne, a odrážejte pavouky a "
     "blechy. Každý zasažený článek i tvor přidá body."),
    ("choppercommand", "Chopper Command", "ChopperCommand", "intermediate", 0.0, 10000.0,
     "Fly a helicopter to protect a convoy of trucks, shooting down enemy planes and choppers in the "
     "desert. Each enemy aircraft destroyed scores points.",
     "Pilotujte vrtulník chránící konvoj náklaďáků a sestřelujte nepřátelská letadla a vrtulníky v "
     "poušti. Každý zničený stroj přidá body."),
    ("crazyclimber", "Crazy Climber", "CrazyClimber", "intermediate", 0.0, 35000.0,
     "Climb a skyscraper hand over hand, opening and closing windows, while dodging falling objects and "
     "obstacles. The higher you climb, the more points you score.",
     "Šplhejte po mrakodrapu rukama nahoru, otvírejte a zavírejte okna a uhýbejte padajícím předmětům a "
     "překážkám. Čím výš vyšplháte, tím víc bodů."),
    ("demonattack", "Demon Attack", "DemonAttack", "beginner", 0.0, 4000.0,
     "Defend an icy planet from waves of diving demons that split and swoop. Move your cannon and shoot "
     "them before they reach you. Each demon destroyed scores points.",
     "Braňte ledovou planetu před vlnami střemhlav útočících démonů, kteří se dělí a klesají. Posouvejte "
     "dělo a sestřelujte je, než vás dostihnou. Každý démon přidá body."),
    ("doubledunk", "Double Dunk", "DoubleDunk", "intermediate", -24.0, 24.0,
     "Two-on-two basketball: pick a play, then pass, shoot and dunk to outscore the built-in opponent. "
     "Your score is your baskets minus theirs.",
     "Basketbal dva na dva: zvolte rozehru a pak přihrávejte, střílejte a smečujte, abyste přestříleli "
     "vestavěného soupeře. Skóre je vaše koše mínus soupeřovy."),
    ("elevatoraction", "Elevator Action", "ElevatorAction", "intermediate", 0.0, 10000.0,
     "A secret agent rides elevators down a building, shooting enemy agents and collecting documents "
     "behind red doors before escaping at the bottom. Kills and documents score points.",
     "Tajný agent sjíždí výtahy dolů budovou, střílí nepřátelské agenty a sbírá dokumenty za červenými "
     "dveřmi, než dole unikne. Zásahy i dokumenty přidávají body."),
    ("fishingderby", "Fishing Derby", "FishingDerby", "intermediate", -99.0, 50.0,
     "A fishing contest against the built-in angler: drop your line, hook fish and reel them in before "
     "your rival — and watch for the shark. Your score is your catch minus theirs.",
     "Rybářský závod proti vestavěnému rybáři: spusťte vlasec, zasekněte ryby a vytáhněte je dřív než "
     "soupeř — a pozor na žraloka. Skóre je váš úlovek mínus soupeřův."),
    ("freeway", "Freeway", "Freeway", "beginner", 0.0, 32.0,
     "Guide a chicken across a busy ten-lane highway without being hit by traffic. Each successful "
     "crossing scores a point; reach the top as many times as you can before time runs out.",
     "Proveďte slepici přes rušnou desetiproudou dálnici, aniž ji srazí auto. Každé úspěšné přejití dá "
     "bod; dostaňte se nahoru co nejvíckrát, než vyprší čas."),
    ("frostbite", "Frostbite", "Frostbite", "intermediate", 0.0, 4500.0,
     "Hop across drifting ice floes to build an igloo, collecting fish and avoiding the freezing water, "
     "birds and bears. Then enter the igloo to clear the level. Each safe jump and fish scores points.",
     "Skákejte po plovoucích krách a stavte iglú; sbírejte ryby a vyhýbejte se mrazivé vodě, ptákům a "
     "medvědům. Vstupem do iglú postoupíte. Každý bezpečný skok a ryba přidá body."),
    ("gopher", "Gopher", "Gopher", "beginner", 0.0, 8000.0,
     "Protect three carrots from a burrowing gopher: whack it with your shovel as it pops out of holes "
     "and fill the tunnels before it steals the crop. Each hit scores points.",
     "Chraňte tři mrkve před hrabajícím se sysel: praštěte ho lopatou, jakmile vykoukne z díry, a "
     "zasypávejte tunely dřív, než úrodu ukradne. Každý zásah přidá body."),
    ("gravitar", "Gravitar", "Gravitar", "advanced", 0.0, 3000.0,
     "Pilot a ship against real gravity, flying into planets to shoot bunkers and tractor-beam fuel "
     "while fighting inertia. A hard one — careful thrust is everything. Destroying targets scores points.",
     "Pilotujte loď proti skutečné gravitaci, nalétávejte na planety, ničte bunkry a vlečným paprskem "
     "sbírejte palivo, zatímco bojujete se setrvačností. Těžká hra — vše je o jemném tahu. Cíle dávají body."),
    ("hero", "H.E.R.O.", "Hero", "intermediate", 0.0, 25000.0,
     "Fly into collapsing mine shafts with a backpack rotor to rescue trapped miners, blasting walls and "
     "enemies with your laser and dynamite while managing power. Rescues and progress score points.",
     "Vlétněte se zádovým rotorem do hroutících se důlních šachet a zachraňujte uvězněné horníky; laserem "
     "a dynamitem ničte stěny a nepřátele a hlídejte energii. Záchrany a postup přidávají body."),
    ("icehockey", "Ice Hockey", "IceHockey", "intermediate", -20.0, 20.0,
     "Two-on-two ice hockey: pass, check and shoot to score more goals than the built-in team. Your "
     "score is your goals minus theirs.",
     "Lední hokej dva na dva: přihrávejte, napadejte a střílejte, abyste dali víc gólů než vestavěný "
     "tým. Skóre je vaše góly mínus soupeřovy."),
    ("jamesbond", "James Bond 007", "Jamesbond", "intermediate", 0.0, 1000.0,
     "Drive a multi-purpose vehicle through scrolling missions, shooting enemies and obstacles, jumping "
     "gaps and dodging fire across famous Bond scenes. Targets destroyed score points.",
     "Řiďte víceúčelové vozidlo posouvajícími se misemi, střílejte nepřátele a překážky, přeskakujte "
     "propasti a uhýbejte palbě ve slavných bondovských scénách. Zničené cíle dávají body."),
    ("kangaroo", "Kangaroo", "Kangaroo", "intermediate", 0.0, 8000.0,
     "A mother kangaroo climbs ladders and platforms to rescue her joey, punching monkeys and dodging "
     "thrown apples on the way up. Hits and fruit collected score points.",
     "Maminka klokanice šplhá po žebřících a plošinách za svým mládětem, cestou boxuje opice a uhýbá "
     "házeným jablkům. Zásahy a sebrané ovoce dávají body."),
    ("krull", "Krull", "Krull", "intermediate", 0.0, 8000.0,
     "Based on the film: cross several action screens — fighting through a web, hurling the Glaive and "
     "battling the Beast's slayers — to win. Defeating enemies scores points.",
     "Podle filmu: projděte několik akčních obrazovek — probojujte se pavučinou, vrhejte Glaive a "
     "bojujte s Bestiinými vrahy — abyste zvítězili. Porážení nepřátel dává body."),
    ("kungfumaster", "Kung-Fu Master", "KungFuMaster", "intermediate", 0.0, 23000.0,
     "Fight your way up a multi-floor temple with punches and kicks, beating waves of henchmen and a "
     "boss on each floor to rescue the captive. Each enemy defeated scores points.",
     "Probojujte se údery a kopy vzhůru vícepatrovým chrámem, na každém patře poražte vlny poskoků a "
     "bosse a zachraňte zajatkyni. Každý poražený nepřítel přidá body."),
    ("montezumarevenge", "Montezuma's Revenge", "MontezumaRevenge", "advanced", 0.0, 4000.0,
     "A notoriously hard exploration platformer: navigate a pyramid of rooms collecting keys to open "
     "doors, dodging skulls, snakes and traps. Treasures and progress score points.",
     "Pověstně těžká průzkumná plošinovka: procházejte pyramidou místností, sbírejte klíče k otevření "
     "dveří a vyhýbejte se lebkám, hadům a pastem. Poklady a postup přidávají body."),
    ("namethisgame", "Name This Game", "NameThisGame", "intermediate", 0.0, 8000.0,
     "An underwater shooter: defend your air hose from a giant octopus and a shark while collecting "
     "oxygen, blasting threats with your diver. Each hit scores points.",
     "Podmořská střílečka: braňte svou vzduchovou hadici před obří chobotnicí a žralokem a sbírejte "
     "kyslík; potápěčem ničte hrozby. Každý zásah přidá body."),
    ("phoenix", "Phoenix", "Phoenix", "intermediate", 0.0, 8000.0,
     "A vertical shooter through waves of alien birds up to a mothership boss you must crack open with a "
     "shield to beat. Move, shoot and shield. Each enemy destroyed scores points.",
     "Vertikální střílečka skrz vlny mimozemských ptáků až k mateřské lodi, kterou musíte přes její štít "
     "rozbít. Pohybujte se, střílejte a kryjte se. Každý zničený nepřítel přidá body."),
    ("pitfall", "Pitfall!", "Pitfall", "advanced", -300.0, 5000.0,
     "Guide Pitfall Harry through a jungle, swinging on vines and leaping over logs, crocodiles and tar "
     "pits to collect treasure against the clock. Treasures score points; hazards cost them.",
     "Veďte Pitfall Harryho džunglí, houpejte se na lianách a přeskakujte klády, krokodýly a dehtové "
     "jámy, abyste o závod s časem nasbírali poklady. Poklady dávají body, nástrahy je berou."),
    ("pooyan", "Pooyan", "Pooyan", "beginner", 0.0, 5000.0,
     "A mother pig rides a basket up and down a cliff, shooting arrows at wolves floating down on "
     "balloons before they reach the ground. Each wolf popped scores points.",
     "Maminka prasnice jezdí v koši nahoru a dolů po útesu a střílí šípy po vlcích snášejících se na "
     "balonech dřív, než dosednou. Každý sestřelený vlk přidá body."),
    ("privateeye", "Private Eye", "PrivateEye", "advanced", 0.0, 70000.0,
     "Drive a detective's car across the city collecting clues and stolen items to solve a case and "
     "catch the culprit before time runs out. Clues and arrests score points.",
     "Projíždějte detektivovým autem městem, sbírejte stopy a ukradené předměty, vyřešte případ a "
     "dopadněte pachatele dřív, než vyprší čas. Stopy a zatčení dávají body."),
    ("riverraid", "River Raid", "Riverraid", "intermediate", 0.0, 14000.0,
     "Fly a jet up a winding river, shooting ships, helicopters and bridges while refuelling over fuel "
     "depots — run dry and you crash. Each target destroyed scores points.",
     "Leťte tryskáčem proti proudu klikaté řeky, ničte lodě, vrtulníky a mosty a doplňujte palivo nad "
     "sklady — bez paliva havarujete. Každý zničený cíl přidá body."),
    ("roadrunner", "Road Runner", "RoadRunner", "beginner", 0.0, 30000.0,
     "Run as the Road Runner down the highway eating birdseed and outsmarting Wile E. Coyote, dodging "
     "trucks and traps. Seed eaten and the Coyote foiled score points.",
     "Běžte jako Uličník po dálnici, zobejte zrní a přelstěte kojota Wila E.; uhýbejte náklaďákům a "
     "pastem. Snězené zrní a přechytračený kojot dávají body."),
    ("robotank", "Robotank", "Robotank", "intermediate", 0.0, 50.0,
     "Command a tank in first-person night-and-fog combat, hunting enemy tanks squadron by squadron "
     "while damage knocks out your sensors. Your score is enemy tanks destroyed.",
     "Velte tanku v boji z první osoby za noci a mlhy a likvidujte nepřátelské tanky letku po letce, "
     "zatímco poškození vyřazuje vaše senzory. Skóre je počet zničených tanků."),
    ("skiing", "Skiing", "Skiing", "advanced", -30000.0, -5000.0,
     "Race downhill through the slalom gates as fast as you can. Your score is your time (lower is "
     "better, shown here as a large negative number) — missing gates adds a penalty.",
     "Sjíždějte co nejrychleji slalomovými brankami. Skóre je váš čas (čím nižší, tím lepší, zde jako "
     "velké záporné číslo) — vynechané branky přidávají penalizaci."),
    ("solaris", "Solaris", "Solaris", "advanced", 0.0, 12000.0,
     "A deep-space combat-and-navigation epic: jump between quadrants on a galactic map, dogfight enemy "
     "fleets and defend planets to find Solaris. Battles and rescues score points.",
     "Vesmírná epopej o boji a navigaci: skákejte mezi kvadranty na galaktické mapě, svádějte souboje s "
     "nepřátelskými flotilami a braňte planety, abyste našli Solaris. Boje a záchrany dávají body."),
    ("stargunner", "Star Gunner", "StarGunner", "intermediate", 0.0, 12000.0,
     "A fast side-scrolling shooter: skim over a planet's surface blasting waves of enemy craft and "
     "dodging their fire. Each enemy destroyed scores points.",
     "Rychlá horizontální střílečka: klouzejte nad povrchem planety, ničte vlny nepřátelských strojů a "
     "uhýbejte jejich palbě. Každý zničený nepřítel přidá body."),
    ("tennis", "Tennis", "Tennis", "intermediate", -24.0, 24.0,
     "A game of tennis against the built-in player: position yourself and time your swing to win rallies "
     "and games. Your score is your games won minus your opponent's.",
     "Tenis proti vestavěnému hráči: zaujměte pozici a načasujte úder, abyste vyhrávali výměny a hry. "
     "Skóre je vaše vyhrané hry mínus soupeřovy."),
    ("timepilot", "Time Pilot", "TimePilot", "intermediate", 0.0, 10000.0,
     "Dogfight through eras of history, shooting down enemy aircraft from biplanes to spaceships in an "
     "open, free-scrolling sky and beating each era's boss. Each kill scores points.",
     "Svádějte letecké souboje napříč epochami dějin, sestřelujte stroje od dvojplošníků po kosmické "
     "lodě na volně se posouvající obloze a porazte bosse každé éry. Každý sestřel přidá body."),
    ("tutankham", "Tutankham", "Tutankham", "intermediate", 0.0, 250.0,
     "Explore an Egyptian tomb maze as an archaeologist, shooting creatures, grabbing treasure and "
     "finding keys to unlock the exit. Treasure and kills score points.",
     "Prozkoumávejte jako archeolog bludiště egyptské hrobky, střílejte tvory, sbírejte poklady a "
     "hledejte klíče k odemčení východu. Poklady a zásahy dávají body."),
    ("upndown", "Up'n Down", "UpNDown", "intermediate", 0.0, 15000.0,
     "Drive a dune buggy along looping tracks, jumping over and onto other cars and collecting flags. "
     "Land on rivals to knock them out. Flags and takedowns score points.",
     "Řiďte buginu po klikatých tratích, přeskakujte ostatní vozy i na ně doskakujte a sbírejte vlajky. "
     "Dopadem na soupeře je vyřadíte. Vlajky a vyřazení dávají body."),
    ("venture", "Venture", "Venture", "advanced", 0.0, 1500.0,
     "Explore a dungeon as Winky the smiley adventurer, entering rooms to grab treasure and shoot "
     "monsters while hall-roaming Hallmonsters chase you. Treasure and kills score points.",
     "Prozkoumávejte kobku jako usměvavý dobrodruh Winky, vcházejte do místností pro poklady a střílejte "
     "příšery, zatímco vás v chodbách honí Hallmonsteři. Poklady a zásahy dávají body."),
    ("videopinball", "Video Pinball", "VideoPinball", "beginner", 0.0, 40000.0,
     "A classic pinball table: work the flippers to keep the ball alive, hit the bumpers and targets for "
     "points and rack up a high score without draining.",
     "Klasický pinball: ovládejte pálky, udržte míček ve hře, trefujte odrazníky a terče pro body a "
     "nasbírejte co nejvyšší skóre, aniž míček propadne."),
    ("wizardofwor", "Wizard of Wor", "WizardOfWor", "intermediate", 0.0, 8000.0,
     "Battle through dungeon mazes shooting monsters that grow more aggressive and turn invisible, "
     "racing to clear each chamber. Each monster destroyed scores points.",
     "Probojujte se bludišti kobek a střílejte příšery, které jsou stále agresivnější a stávají se "
     "neviditelnými; vyčistěte co nejrychleji každou komnatu. Každá zničená příšera přidá body."),
    ("yarsrevenge", "Yars' Revenge", "YarsRevenge", "intermediate", 0.0, 25000.0,
     "As an insect-like Yar, nibble or shoot through a barrier to reach the Qotile, then fire the Zorlon "
     "Cannon to destroy it — all while dodging its destroyer missile. Progress scores points.",
     "Jako hmyzí Yar prokousejte nebo prostřílejte bariéru k Qotile a pak ho zničte dělem Zorlon — to "
     "vše za uhýbání jeho ničivé střele. Postup přidává body."),
    ("zaxxon", "Zaxxon", "Zaxxon", "intermediate", 0.0, 10000.0,
     "Fly an isometric space fortress run, shooting turrets, fuel tanks and enemy fighters while "
     "judging your altitude over the walls. Each target destroyed scores points.",
     "Proleťte v izometrickém pohledu vesmírnou pevností, ničte věže, palivové nádrže a nepřátelské "
     "stíhače a odhadujte výšku nad zdmi. Každý zničený cíl přidá body."),
    ("frogger", "Frogger", "Frogger", "beginner", 0.0, 1000.0,
     "Hop a frog across a busy road and then a river of logs and turtles to reach the safe homes at the "
     "top, without being run over or falling in. Each frog home reached scores points.",
     "Skákejte se žábou přes rušnou silnici a pak přes řeku klád a želv k bezpečným domkům nahoře, aniž "
     "vás přejede auto nebo spadnete do vody. Každý dosažený domek přidá body."),
    ("galaxian", "Galaxian", "Galaxian", "beginner", 0.0, 6000.0,
     "Shoot a formation of alien ships that peel off to dive-bomb you. Move and fire, picking them off "
     "before they hit you. Each alien destroyed scores points (divers are worth more).",
     "Sestřelujte formaci mimozemských lodí, které se odpojují a střemhlav na vás útočí. Pohybujte se a "
     "střílejte a sundávejte je dřív, než vás zasáhnou. Každý nepřítel přidá body (útočníci víc)."),
    ("defender", "Defender", "Defender", "intermediate", 0.0, 20000.0,
     "Fly a ship over a scrolling planet, shooting alien landers before they abduct the humans below and "
     "mutate — use the radar and smart bombs. Kills and rescues score points.",
     "Leťte lodí nad posouvající se planetou a střílejte mimozemské únosce dřív, než unesou lidi dole a "
     "zmutují — využijte radar a chytré bomby. Zásahy a záchrany dávají body."),
    ("kaboom", "Kaboom!", "Kaboom", "beginner", 0.0, 1000.0,
     "Catch the bombs dropped by the Mad Bomber with a stack of buckets, sliding left and right. The "
     "bombs fall ever faster — miss one and a bucket blows up. Each bomb caught scores points.",
     "Chytejte bomby, které shazuje Šílený bombarďák, hromádkou kbelíků a klouzejte doleva a doprava. "
     "Bomby padají stále rychleji — když jednu minete, kbelík vybuchne. Každá chycená bomba přidá body."),
    ("tetris", "Tetris", "Tetris", "beginner", 0.0, 200.0,
     "The falling-blocks classic: rotate and slide tetrominoes to complete full horizontal lines, which "
     "clear for points. The stack rises as you play — don't let it reach the top.",
     "Klasika s padajícími kostkami: otáčejte a posouvejte tetromina tak, abyste doplnili celé vodorovné "
     "řady, které za body zmizí. Hromada roste — nenechte ji dosáhnout vrcholu."),
    ("pacman", "Pac-Man", "Pacman", "beginner", 0.0, 6000.0,
     "The arcade original: eat every dot in the maze while four ghosts chase you, and grab a power pellet "
     "to turn the tables and eat them for bonus points.",
     "Arkádový originál: snězte v bludišti všechny tečky, zatímco vás honí čtyři duchové, a sebráním "
     "speciální kuličky se karta obrátí — duchy můžete za body sníst."),
]

for _row in _ATARI_GAMES:
    register(_atari_spec(*_row))


# ---------------------------------------------------------------------------
# MiniGrid family (Farama) — G2c "new CPU grid family".
#
# The native observation is a Dict (a 7×7×3 partial-view "image" + the agent's
# facing direction + a natural-language mission string), NOT a vector. The shared
# factory wraps every minigrid env in ``minigrid.wrappers.FlatObsWrapper``, which
# flattens that Dict into a length-2835 ``Box(uint8)`` vector — so the SAME
# ``MlpPolicy`` (PPO) and numpy genome (neuroevolution) used for CartPole apply with
# no engine change (the same idea as the Toy Text one-hot seam, a different wrapper).
# Hence ``obs_type="vector"`` and ``hw_requirement="cpu"`` (verified: PPO solves
# Empty-5x5 to ~0.95 in ~6k steps on the laptop CPU, no SB3 warning on the uint8 obs)
# — unlike Atari, these are NOT GPU-gated and train here now.
#
# Reward is SPARSE: 0 until the goal, then ``1 − 0.9·(steps/max_steps)`` on success
# (≈0.9–0.97), so ``solved_score=0.95``, ``min_score=0.0``, ``sparse_reward=True`` (the
# play meter "measures" until the episode ends, like FrozenLake — ADR-030). The envs
# have NO native gym ``TimeLimit`` (``gym.spec().max_episode_steps`` is None) but the
# MiniGrid env truncates itself at its internal ``max_steps``, so no ``episode_step_limit``
# is needed (unlike CliffWalking).
#
# Grid-worlds, so played TURN-BASED (one move per key press, reusing the G2a path); but
# unlike Toy Text they are rendered SERVER-SIDE as a JPEG (the family is not in
# client_render, so ``client_state`` returns None → ``env.render()`` rgb_array → JPEG —
# exactly like Atari, minus the retro skin). Action space is ``Discrete(7)``: turn-left,
# turn-right, forward, pickup, drop, toggle, done; DoorKey/KeyCorridor need pickup +
# toggle. ``supported_algos`` keeps PPO + neuroevolution (the latter is weak on the big
# 2835-dim obs — documented honestly in content/parameters.ts). Adding more of the 80+
# MiniGrid levels later is a one-row-per-game data change here.
# ---------------------------------------------------------------------------


def _minigrid_spec(
    env_id: str,
    gym_id: str,
    display: str,
    difficulty: Literal["beginner", "intermediate", "advanced"],
    default_total_timesteps: int,
    desc_en: str,
    desc_cz: str,
) -> EnvSpec:
    """Build one MiniGrid EnvSpec from a data row (the family is otherwise identical)."""
    return EnvSpec(
        id=env_id,
        gym_id=gym_id,
        display_name=Bilingual(en=display, cz=display),  # MiniGrid level names are proper nouns
        description=Bilingual(en=desc_en, cz=desc_cz),
        family="minigrid",
        obs_type="vector",  # after FlatObsWrapper (applied in the shared factory)
        action_space="discrete",
        supported_algos=["ppo", "neuroevolution"],
        hyperparams=_standard_hyperparams(),
        solved_score=0.95,  # no gym reward_threshold; success pays 1 − 0.9·steps/max ≈ 0.9–0.97
        min_score=0.0,  # sparse 0/1-shaped reward — a failing agent scores 0, the meter fills 0 → 0.95
        sparse_reward=True,  # 0 until the goal → play meter "measures" until the episode ends (ADR-030)
        default_total_timesteps=default_total_timesteps,
        play_step_scale=1,  # turn-based human play — no time pressure
        turn_based=True,
        human_playable=True,
        competitive=False,
        difficulty=difficulty,
        hw_requirement="cpu",  # FlatObs + MlpPolicy trains on CPU now (not GPU-gated like Atari)
    )


# id, gym_id, display, difficulty, default_total_timesteps, description EN, description CZ
_MINIGRID_GAMES: list[
    tuple[str, str, str, Literal["beginner", "intermediate", "advanced"], int, str, str]
] = [
    ("minigrid_empty", "MiniGrid-Empty-5x5-v0", "MiniGrid Empty 5×5", "beginner", 100_000,
     "Reach the green goal square in a small empty room. The simplest MiniGrid level — turn to face a "
     "direction, then step forward — and the gentlest introduction to its sparse reward: you score only "
     "on reaching the goal, and the sooner you get there the higher the score.",
     "Dojděte na zelené cílové pole v malé prázdné místnosti. Nejjednodušší úroveň MiniGridu — otočte se "
     "směrem a pak udělejte krok vpřed — a nejmírnější úvod do její řídké odměny: bod získáte jen za "
     "dosažení cíle a čím dřív tam dojdete, tím vyšší skóre."),
    ("minigrid_fourrooms", "MiniGrid-FourRooms-v0", "MiniGrid Four Rooms", "intermediate", 500_000,
     "Find the goal in a layout of four rooms joined by narrow gaps. Far more exploration than the empty "
     "room: the goal and your start are placed randomly, so you must travel between rooms through the "
     "doorways to find it. Sparse reward — you score only on reaching the goal.",
     "Najděte cíl v rozložení čtyř místností spojených úzkými průchody. Mnohem víc zkoumání než prázdná "
     "místnost: cíl i váš start jsou umístěny náhodně, takže musíte procházet mezi místnostmi přes dveře, "
     "abyste cíl našli. Řídká odměna — bodujete jen za dosažení cíle."),
    ("minigrid_doorkey", "MiniGrid-DoorKey-5x5-v0", "MiniGrid Door & Key", "intermediate", 300_000,
     "Pick up the key, use it to unlock the door, then reach the goal on the far side. The first level "
     "with a sub-goal you must do in order — find and grab the key, open the locked door, then cross — a "
     "classic test of multi-step exploration. Sparse reward: only completing it scores.",
     "Seberte klíč, odemkněte jím dveře a pak dojděte k cíli na druhé straně. První úroveň s dílčím cílem, "
     "který musíte splnit v pořadí — najít a vzít klíč, otevřít zamčené dveře a přejít — klasický test "
     "vícekrokového zkoumání. Řídká odměna: boduje jen dokončení."),
    ("minigrid_keycorridor", "MiniGrid-KeyCorridorS3R1-v0", "MiniGrid Key Corridor", "advanced", 500_000,
     "Pick up the coloured ball locked away in a room. You must first find a key hidden behind another "
     "door, then use it to unlock the room with the ball — a hierarchical goal that needs real exploration "
     "of the corridor and its rooms. The hardest of these four; sparse reward.",
     "Seberte barevný míček zamčený v jedné z místností. Nejdřív musíte najít klíč ukrytý za jinými dveřmi "
     "a teprve jím odemknout místnost s míčkem — hierarchický cíl vyžadující skutečné prozkoumání chodby a "
     "jejích místností. Nejtěžší ze čtyř; řídká odměna."),
]

for _mg_row in _MINIGRID_GAMES:
    register(_minigrid_spec(*_mg_row))
