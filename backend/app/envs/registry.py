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
