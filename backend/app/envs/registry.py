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
    # Competitive multi-agent only (simple_tag, G7b-2): the SECOND species' skill scale, so the
    # two-line ecosystem chart can read each species against its own [floor, good] range. ``min_score``/
    # ``solved_score`` are the **predator** (adversary) scale (the headline); these are the **prey**
    # (agent) scale — a deep "frequently caught" floor and a "mostly escapes" good end (prey returns are
    # negative). None for every single-species env. Competitive scales are inherently approximate (each
    # species' raw return depends on how strong the opponent currently is), so these are reference lines,
    # not exact skill — the two reward *curves* are the real signal.
    prey_min_score: float | None = None
    prey_solved_score: float | None = None
    # Recommended PPO training budget for this env (the ★ default in the sidebar). Harder envs
    # need far more steps than CartPole, so this is per-env data; the sidebar builds its step
    # dropdown as a ladder around this value (×0.2 … ×4) and the store seeds it on env switch.
    default_total_timesteps: int
    # PLAY sessions (human + AI) multiply the env's max_episode_steps by this so a person has time
    # to actually play short envs; training keeps the standard length. 1 = no change.
    play_step_scale: int = 1
    # Extra slow-down on the per-step interval for HUMAN play only (AI play is unaffected). Most envs
    # need none (1.0). It exists for envs that end on an early *termination* a human can't prevent
    # (a fall), where play_step_scale can't help — extending the step cap does nothing when the episode
    # ends on the fall, not the cap. The only lever left is wall-clock: stepping slower gives a person
    # more real seconds to react before the inevitable fall. MuJoCo Hopper/Walker2d render at 125 fps
    # and topple in ~1 s; even capped at the 30 fps frame rate that is only ~8 s, so they set this ~8
    # (≈5× longer wall-clock once the fixed per-frame render cost is included). The speed slider still
    # scales on top of it.
    human_play_slowdown: float = 1.0
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
    # Whether this env's TRAINING code path actually exists yet. True for every env whose obs reaches
    # the implemented MlpPolicy trainer (all vector/discrete envs — incl. the GPU-gated *vector* heavies
    # BipedalWalker + MuJoCo, which train correctly with MlpPolicy and are gated only because they need
    # millions of steps). False for the **image-observation** envs (Atari, CarRacing): their training
    # needs the CnnPolicy + frame-stack + CUDA seam that is NOT built yet (G4b / G3c-train), so a "Train"
    # there would build an MlpPolicy on pixels and crash. Decouples "needs a GPU" from "trainer not
    # implemented": a GPU machine un-gates the vector heavies (hw_requirement="gpu" + gpu present) but
    # MUST keep image envs gated until the CnnPolicy seam lands. The training-manager start() enforces
    # this as a backstop and the UI shows a distinct "coming later" note. Flip to True per-env in G4b.
    train_implemented: bool = True


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
# Box2D family — BipedalWalker (G3b-play, "install + human-play on CPU now,
# training GPU-gated"). Like LunarLander these are vector-obs Box2D physics envs,
# but the action space is **continuous** Box(4) — the four leg-joint torques
# [hip1, knee1, hip2, knee2], each in [-1, 1] — so they ride the G1b continuous-box
# seam (box-aware predict/play/preview; the numpy forwards tanh-scale into [low,
# high]) plus the existing server-JPEG render. No new engine code: human play is
# data + content. Human play sends a *per-joint vector* (one key per joint/direction,
# summed client-side; the WS action frame + play_session already accept list[float]).
#
# Training is **gated** (hw_requirement="gpu") like Atari, but for a different reason:
# the obs is a vector (MlpPolicy, no CnnPolicy needed), yet learning to walk takes
# millions of steps — impractical on the laptop CPU. So Run is disabled here until the
# desktop (G3b-train); human play needs no training and stays available now.
#
# supported_algos=["ppo"] (PPO-only, opted out of neuroevolution as data, like
# CarRacing): the box action itself IS supported by the G1b numpy genome, but
# population search is impractical on a hard 4-DoF locomotion task — so this is a
# difficulty gate, not a seam limitation.
#
# Skill: solved_score=300 (the gym reward_threshold). min_score=-100 is the
# fall/timeout baseline (ADR-026), NOT the deeper random floor: a do-nothing/flailing
# agent falls for ≈ -92…-115 (venv-measured), and -100 (the fall penalty) is the honest
# 0% floor so a non-walker reads ~0% — exactly LunarLander-shaped. floor_scales_with_steps
# is False (a fall ends the episode early ≈ -100; the failure score does not grow with the
# step cap), and play_step_scale=1 (1600 steps at 50 fps ≈ 32 s is already long enough by hand).
# ---------------------------------------------------------------------------


def _bipedal_spec(
    env_id: str,
    display: str,
    difficulty: Literal["beginner", "intermediate", "advanced"],
    default_total_timesteps: int,
    make_kwargs: dict[str, Any],
    desc_en: str,
    desc_cz: str,
) -> EnvSpec:
    """Build one BipedalWalker EnvSpec from a data row (the variants share everything but the
    terrain — Hardcore adds ladders/stumps/pits via ``make_kwargs={"hardcore": True}``)."""
    return EnvSpec(
        id=env_id,
        gym_id="BipedalWalker-v3",  # both variants share the gym id; Hardcore differs by make_kwargs
        display_name=Bilingual(en=display, cz=display),
        description=Bilingual(en=desc_en, cz=desc_cz),
        family="box2d",
        obs_type="vector",
        action_space="box",  # continuous Box(4): four leg-joint torques — the G1b seam
        supported_algos=["ppo"],  # PPO-only (evolution opted out as data — hard 4-DoF locomotion)
        hyperparams=_standard_hyperparams(),
        make_kwargs=make_kwargs,
        solved_score=300.0,  # BipedalWalker-v3 reward_threshold (walk smoothly to the far end)
        min_score=-100.0,  # fall/timeout baseline (ADR-026): a non-walker falls ≈ -92…-115; -100 = 0% floor
        default_total_timesteps=default_total_timesteps,  # the ★ PPO budget when GPU training lands
        play_step_scale=1,  # 1600 steps @ 50 fps ≈ 32 s — already long enough to play by hand
        floor_scales_with_steps=False,  # shaped/terminal: a fall ends early ≈ -100, doesn't scale with steps
        human_playable=True,
        competitive=False,
        difficulty=difficulty,
        hw_requirement="gpu",  # training deferred to the desktop (millions of steps); play available now
    )


register(
    _bipedal_spec(
        "bipedalwalker",
        "BipedalWalker-v3",
        "advanced",
        5_000_000,  # hard continuous locomotion — a realistic PPO budget for the GPU desktop
        {},
        "Teach a two-legged robot to walk across gently uneven ground without falling over. A "
        "Box2D physics task with a 24-number state (joint angles, hull tilt, ground contacts and "
        "lidar) and four continuous leg-joint torques. Moving forward earns reward, using the "
        "motors costs a little, and a fall ends the run with a −100 penalty; walking smoothly to "
        "the far end scores around +300 (the 'solved' mark).",
        "Naučte dvounohého robota přejít mírně nerovný terén, aniž by upadl. Úloha s fyzikou Box2D "
        "se stavem o 24 číslech (úhly kloubů, náklon trupu, kontakty se zemí a lidar) a čtyřmi "
        "spojitými momenty v kloubech nohou. Pohyb vpřed dává odměnu, použití motorů něco stojí a "
        "pád ukončí běh penalizací −100; plynulá chůze až na konec dá kolem +300 (hranice „vyřešeno“).",
    )
)


register(
    _bipedal_spec(
        "bipedalwalkerhardcore",
        "BipedalWalker-v3 (Hardcore)",
        "advanced",
        10_000_000,  # notoriously hard — needs an even larger budget than the standard course
        {"hardcore": True},
        "The same two-legged robot, but now the terrain is an obstacle course of ladders, stumps "
        "and pits to climb over, step around and leap across. Same 24-number state and four "
        "continuous leg torques, same +reward for progress and −100 for a fall — but far harder, "
        "and one of the classic 'hard' continuous-control benchmarks.",
        "Stejný dvounohý robot, ale terén je teď překážková dráha plná žebříků, pařezů a jam, "
        "které je třeba přelézt, obejít a přeskočit. Stejný stav o 24 číslech a čtyři spojité "
        "momenty nohou, stejná odměna za postup a −100 za pád — ale mnohem těžší, a jeden z "
        "klasických „těžkých“ benchmarků spojitého řízení.",
    )
)


# ---------------------------------------------------------------------------
# CarRacing-v3  (Box2D family — G3c-play: "install + human-play on CPU now,
# training GPU-gated"). This is the env the seam roadmap flagged as the LAST
# int→box case: **image obs (96×96×3) AND a continuous Box(3) action** (steer
# ∈ [-1,1], gas ∈ [0,1], brake ∈ [0,1]). For *human play* both halves are
# already solved seams — the image obs rides the existing server-JPEG render
# path (like Atari/MiniGrid; client_state returns None → env.render() → JPEG),
# and the box action rides the G1b/G3b continuous-box play path (play_session
# reshapes the held analog command into the action vector + clips it). Human
# play sends a *per-joint vector* via the G3b multi-key keymap (←/→ steer, ↑
# gas, ↓ brake, summed client-side), so this needs NO new engine code — pure
# data + content, exactly like BipedalWalker.
#
# Training is **gated** (hw_requirement="gpu") — but unlike BipedalWalker (a
# vector env gated only by step count), CarRacing genuinely needs the CnnPolicy
# seam for its image obs (the laptop's MlpPolicy/CPU trainer can't consume
# pixels). That seam lands on the desktop (G3c-train); the GPU gate both blocks
# the impractical CPU run AND stands in for the missing CnnPolicy until then.
# Human play needs no training and is available now.
#
# supported_algos=["ppo"] — PPO-only, evolution opted out as data (the numpy
# genome is an MLP over a flat vector; it can't take pixels either). Skill:
# solved_score=900 (the community "solved" mark — visit ~all track tiles at
# +1000/N while paying −0.1/frame). min_score=-100 is the do-nothing/off-track
# floor (ADR-026): idle for the full episode costs ≈ −0.1 × 1000 ≈ −100, and
# leaving the playfield ends the episode with a −100 penalty — so a non-driver
# reads ~0%. floor_scales_with_steps=False (the −100 off-field penalty is
# terminal, not a per-step floor that grows with the cap) and play_step_scale=1
# (1000 steps @ 50 fps ≈ 20 s, and the play-speed slider goes to 0.1× for more
# real time — extending the episode would only deepen the −0.1/frame penalty).
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="carracing",
        gym_id="CarRacing-v3",
        display_name=Bilingual(en="CarRacing-v3", cz="CarRacing-v3"),
        description=Bilingual(
            en="Drive a race car around a randomly generated track from a top-down view, staying on "
            "the road and visiting every track tile as fast as you can. A Box2D classic and the "
            "first pixels-in game you can play here: the state is the 96×96 colour image itself, and "
            "you steer with a continuous wheel plus gas and brake pedals. Each tile visited pays "
            "+1000/N, every frame costs a little, and leaving the track is heavily penalised — a "
            "clean lap scores around +900 (the 'solved' mark).",
            cz="Řiďte závodní auto po náhodně generované trati z pohledu shora, držte se na silnici a "
            "co nejrychleji projeďte každý dílek trati. Klasika Box2D a první „hra z pixelů“, kterou "
            "si tu zahrajete: stavem je přímo barevný obraz 96×96 a řídíte spojitým volantem plus "
            "plynem a brzdou. Každý projetý dílek vyplatí +1000/N, každý snímek něco stojí a sjetí z "
            "trati je tvrdě penalizováno — čisté kolo dá kolem +900 (hranice „vyřešeno“).",
        ),
        family="box2d",
        obs_type="image",  # 96×96×3 pixels → server-JPEG render for play; CnnPolicy training is G3c-train
        action_space="box",  # continuous Box(3): steer [-1,1], gas [0,1], brake [0,1] — the G1b/G3b seam
        supported_algos=["ppo"],  # PPO-only (evolution opted out as data — a flat-vector genome can't take pixels)
        hyperparams=_standard_hyperparams(),
        make_kwargs={"continuous": True},  # the continuous steer/gas/brake variant (vs Discrete(5))
        solved_score=900.0,  # the community "solved" mark (visit ~all tiles at +1000/N, less frame cost)
        min_score=-100.0,  # do-nothing/off-track floor (ADR-026): idle ≈ −100, leaving the field penalises −100
        default_total_timesteps=1_000_000,  # the ★ CnnPolicy PPO budget when GPU training lands (G3c-train)
        play_step_scale=1,  # 1000 steps @ 50 fps ≈ 20 s; the play-speed slider (to 0.1×) gives more real time
        floor_scales_with_steps=False,  # the −100 off-field penalty is terminal, not a per-step floor
        human_playable=True,
        competitive=False,
        difficulty="advanced",
        hw_requirement="gpu",  # image obs needs the CnnPolicy seam (G3c-train, desktop); play available now
        train_implemented=False,  # image obs → CnnPolicy seam not built yet (G3c-train); stays gated even on a GPU
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
#   * obs_type="image" → training needs a CnnPolicy + CUDA (the trainer_ppo._build_model
#     branch + the shared app/envs/atari.py frame-stack vec env, built in G4b). So training
#     stays hw_requirement="gpu": the UI disables Run while no CUDA device is present (see
#     /api/system) and the manager rejects it. On the RTX 5070 desktop Atari trains now;
#     human play needs no neural net (env stepping + the JPEG render path) so it ran from G4a.
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
        hw_requirement="gpu",  # CnnPolicy training needs a GPU; the UI gates Run on a CPU box, human play stays
        train_implemented=True,  # G4b: CnnPolicy + AtariWrapper/frame-stack + CUDA trainer built — trains on a GPU
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


# ---------------------------------------------------------------------------
# Multi-agent family (PettingZoo / MPE) — G7a "the 5th seam".
#
# The first **multi-agent** envs: N agents act in one shared world, each with its own
# observation / action / reward (PettingZoo's *parallel* API). That doesn't fit the
# single-agent Gymnasium ``step()`` loop the whole registry has assumed, so it is NOT a
# plain registry row — it rides a dedicated adapter (``app.services.ma_env``) and trainer/
# preview branches (ADR-038). The rows here still describe the env as data; the seam is the
# code those rows route to.
#
# These are **Simple Spread** (cooperative coverage): the agents must spread out to cover all
# the landmark targets while avoiding collisions, sharing a global reward. The agents are
# **homogeneous** (identical obs/action spaces), which is exactly what SuperSuit's
# parameter-sharing bridge needs — one shared ``MlpPolicy`` trained over all N agents at once.
# Per-agent obs is a vector (Box) and the action is ``Discrete(5)`` (stay / ±x / ±y), so the
# SAME CPU ``MlpPolicy`` PPO path used for CartPole applies once the SuperSuit bridge stacks the
# agents — hence ``obs_type="vector"``, ``hw_requirement="cpu"`` (trains on CPU now; the RTX 5070
# desktop just *scales* it in G7b). ``supported_algos=["ppo"]`` — neuroevolution / tabular
# Q-learning have no multi-agent path. Rendered client-side as a "swarm" canvas from streamed
# per-agent positions (``client_state`` is bypassed; the frame carries ``agents``/``world``).
#
# Heterogeneous species (simple_tag predators vs. prey — different obs/action spaces) need
# per-species policies and are deferred to G7b. Scores were venv-measured per the new-env
# checklist: a random policy averages ≈ −26 (N=3) / −38 (N=6) per agent, so ``min_score`` is the
# scattered/no-coverage floor and ``solved_score`` a genuinely good cooperative coverage return.
# These envs are watch-and-train only (``human_playable=False``) — a single human can't drive a
# whole swarm; competitive/cooperative human play arrives with the 2-player envs in G7c.
# ---------------------------------------------------------------------------


def _mpe_spread_spec(
    env_id: str,
    n_agents: int,
    display: str,
    difficulty: Literal["beginner", "intermediate", "advanced"],
    min_score: float,
    solved_score: float,
    default_total_timesteps: int,
    desc_en: str,
    desc_cz: str,
) -> EnvSpec:
    """Build one Simple Spread EnvSpec from a data row (the variants differ only by agent count)."""
    return EnvSpec(
        id=env_id,
        gym_id="simple_spread_v3",  # the mpe2 scenario module name (resolved by app.services.ma_env)
        display_name=Bilingual(en=display, cz=display),
        description=Bilingual(en=desc_en, cz=desc_cz),
        family="petting_zoo",
        obs_type="vector",  # per-agent vector obs, stacked across agents by the SuperSuit bridge
        action_space="discrete",  # Discrete(5): stay / left / right / down / up
        supported_algos=["ppo"],  # parameter-sharing PPO only; evo / Q-learning have no MA path
        hyperparams=_standard_hyperparams(),
        # PettingZoo parallel_env kwargs (consumed by ma_env.make_parallel_env / make_vec_env).
        make_kwargs={"N": n_agents, "max_cycles": 25, "continuous_actions": False},
        solved_score=solved_score,  # a genuinely good cooperative coverage return (negative)
        min_score=min_score,  # scattered / no-coverage floor (random ≈ −26…−38 per agent)
        default_total_timesteps=default_total_timesteps,  # ★ budget; the GPU desktop scales it (G7b)
        play_step_scale=1,
        floor_scales_with_steps=False,  # shaped per-step reward; the floor doesn't scale with the cap
        human_playable=False,  # a swarm has no single human driver — watch + train only (G7a)
        competitive=False,
        difficulty=difficulty,
        hw_requirement="cpu",  # parameter-sharing PPO trains on CPU now (not GPU-gated)
    )


# id, agents, display, difficulty, min_score, solved_score, default_total_timesteps, desc EN, desc CZ
_MPE_GAMES: list[
    tuple[str, int, str, Literal["beginner", "intermediate", "advanced"], float, float, int, str, str]
] = [
    ("mpe_spread", 3, "Simple Spread (3 agents)", "intermediate", -50.0, -15.0, 500_000,
     "Three agents must spread out to cover three landmark targets in a shared 2-D world, "
     "cooperating so each target ends up occupied while avoiding collisions with each other. "
     "Every agent shares one global reward (closer coverage scores higher, bumps cost a little), "
     "and they all learn from a single shared brain — the gentlest introduction to multi-agent RL "
     "and emergent teamwork.",
     "Tři agenti se musí rozprostřít a pokrýt tři cílové značky ve sdíleném 2-D světě; "
     "spolupracují tak, aby každý cíl někdo obsadil, a přitom se vyhýbají vzájemným srážkám. "
     "Všichni agenti sdílejí jednu společnou odměnu (lepší pokrytí dává víc, srážky trochu stojí) "
     "a učí se z jediného sdíleného „mozku“ — nejmírnější úvod do více­agentního RL a vznikající "
     "týmové spolupráce."),
    ("mpe_spread_swarm", 6, "Simple Spread (6 agents)", "advanced", -60.0, -25.0, 1_000_000,
     "The same cooperative coverage task scaled up to a six-agent swarm with six targets — far more "
     "coordination, more potential collisions and a harder credit-assignment problem, but still one "
     "shared policy driving every agent. A vivid showcase of parameter-sharing at swarm scale.",
     "Stejná kooperativní úloha pokrytí, ale zvětšená na roj šesti agentů se šesti cíli — mnohem víc "
     "koordinace, víc možných srážek a těžší přiřazení zásluh, ale stále jedna sdílená strategie "
     "řídící každého agenta. Názorná ukázka sdílení parametrů v měřítku roje."),
]

for _mpe_row in _MPE_GAMES:
    register(_mpe_spread_spec(*_mpe_row))


# ---------------------------------------------------------------------------
# Multi-agent — Predator–Prey (simple_tag) — G7b "the ecosystem" (heterogeneous species).
#
# Unlike Simple Spread (homogeneous, one shared brain), simple_tag has **two species** with
# *different* obs sizes and *opposite* rewards: fast **predators** (``adversary`` — they share a
# reward for touching the prey) chase a faster **prey** (``agent`` — penalised on contact), around
# a couple of solid **obstacles**. That breaks SuperSuit's parameter-sharing bridge (it needs
# identical spaces), so the trainer learns **one shared policy per species** by **frozen-opponent
# alternating self-play** — G7b-2, ``app.services.trainer_tag`` (ADR-048). Training is now built
# (``train_implemented=True``); the run is still ``algo=="ppo"`` in the UI but routes to the
# self-play trainer in the manager. ``human_playable=False`` stays (a swarm has no single human
# driver; competitive human play is G7b-3).
#
# ``min_score`` / ``solved_score`` are the **predator** (adversary) scale — the headline reward line;
# ``prey_min_score`` / ``prey_solved_score`` are the **prey** (agent) scale (returns are negative —
# a deep "caught" floor up to a "mostly escapes" near-0 good end). Measured from random-policy
# rollouts (predator mean ≈ 3.6 / 16.7, prey mean ≈ −16 / −22 for 3v1 / 6v2); competitive scales are
# inherently approximate (each species' raw return depends on how strong the *current* opponent is),
# so they are chart reference lines, not exact skill — the two reward *curves* are the real signal.
# ---------------------------------------------------------------------------


def _self_play_hyperparams() -> dict[str, dict[str, HyperparamDef]]:
    """The standard PPO knobs (one per-species net) plus the self-play ``rounds`` schedule (G7b-2).

    simple_tag trains two shared policies by frozen-opponent alternating rounds (ADR-048); each
    species' net is a normal ``MlpPolicy`` so it reuses the standard PPO block verbatim, and the only
    extra tunable is how many times the two species alternate. ``rounds`` rides in the ``ppo`` block
    because the run is still ``algo=="ppo"`` in the UI."""
    hp = _standard_hyperparams()
    hp["ppo"]["rounds"] = HyperparamDef(
        type="int", default=8, recommended=8, min=2, max=20, step=1,
    )
    return hp


def _board_hyperparams() -> dict[str, dict[str, HyperparamDef]]:
    """Standard PPO knobs with a small ``ent_coef`` ★ — the board trainer (MaskablePPO vs the MCTS
    teacher, G6b) benefits from a little exploration so the masked policy doesn't collapse onto one
    line too early. The self-play ``rounds`` schedule is internal for board (not a UI knob), so this
    is the plain PPO block, not ``_self_play_hyperparams``."""
    hp = _standard_hyperparams()
    hp["ppo"]["ent_coef"] = HyperparamDef(
        type="float", default=0.01, recommended=0.01, min=0.0, max=0.1, step=0.001,
    )
    return hp


def _mpe_tag_spec(
    env_id: str,
    n_adversaries: int,
    n_good: int,
    n_obstacles: int,
    display: str,
    difficulty: Literal["beginner", "intermediate", "advanced"],
    min_score: float,
    solved_score: float,
    prey_min_score: float,
    prey_solved_score: float,
    default_total_timesteps: int,
    desc_en: str,
    desc_cz: str,
) -> EnvSpec:
    """Build one Predator–Prey (simple_tag) EnvSpec — heterogeneous species, per-species self-play (G7b-2)."""
    return EnvSpec(
        id=env_id,
        gym_id="simple_tag_v3",  # the mpe2 scenario module name (resolved by app.services.ma_env)
        display_name=Bilingual(en=display, cz=display),
        description=Bilingual(en=desc_en, cz=desc_cz),
        family="petting_zoo",
        obs_type="vector",  # per-agent vector obs — sizes DIFFER by species (16 vs 14) → per-species nets
        action_space="discrete",  # Discrete(5): stay / left / right / down / up
        supported_algos=["ppo"],  # per-species PPO self-play (G7b-2); evo / Q-learning have no MA path
        hyperparams=_self_play_hyperparams(),
        # PettingZoo parallel_env kwargs (consumed by ma_env.make_parallel_env). simple_tag takes
        # explicit species counts + obstacles instead of Simple Spread's single ``N``.
        make_kwargs={
            "num_good": n_good,
            "num_adversaries": n_adversaries,
            "num_obstacles": n_obstacles,
            "max_cycles": 25,
            "continuous_actions": False,
            # Observe the 2 NEAREST obstacles (zero-padded), so the obs size stays fixed (16/14) even
            # though the obstacle count varies 2…6 per round/session — the variable-obstacle seam (G7b-2).
            "num_landmark_neighbors": 2,
        },
        solved_score=solved_score,  # predator (adversary) scale — the headline reward line
        min_score=min_score,  # predator floor (a do-nothing predator catches nothing ≈ 0)
        prey_min_score=prey_min_score,  # prey "frequently caught" floor (returns are negative)
        prey_solved_score=prey_solved_score,  # prey "mostly escapes" good end (near 0)
        default_total_timesteps=default_total_timesteps,
        play_step_scale=1,
        floor_scales_with_steps=False,
        human_playable=False,  # a swarm has no single human driver; competitive play is G7b-3
        competitive=True,  # predators vs. prey → per-species self-play trainer (ADR-048)
        difficulty=difficulty,
        hw_requirement="cpu",  # small env; per-species PPO trains on CPU (the GPU desktop scales it)
        train_implemented=True,  # per-species frozen self-play trainer (G7b-2, trainer_tag.py)
    )


# id, n_adversaries, n_good, n_obstacles, display, difficulty, pred_min, pred_solved, prey_min, prey_solved, steps, desc EN, desc CZ
_MPE_TAG_GAMES: list[
    tuple[str, int, int, int, str, Literal["beginner", "intermediate", "advanced"], float, float, float, float, int, str, str]
] = [
    ("mpe_tag", 3, 1, 2, "Predator–Prey (3 vs 1)", "intermediate", 0.0, 80.0, -80.0, 0.0, 500_000,
     "Three cooperating predators chase a single, faster prey around a shared 2-D world dotted with "
     "two obstacles. The predators share a reward for every touch of the prey; the prey is penalised "
     "for being caught and for fleeing off-screen — so the two species learn opposite goals. The "
     "classic, accessible cousin of OpenAI's hide-and-seek and the gateway to emergent herding and "
     "ambushing. Each species trains its own shared brain by alternating self-play.",
     "Tři spolupracující predátoři honí jedinou, rychlejší kořist ve sdíleném 2-D světě se dvěma "
     "překážkami. Predátoři dostávají společnou odměnu za každý dotyk kořisti; kořist je trestána za "
     "chycení i za útěk mimo obrazovku — oba druhy se tak učí opačné cíle. Klasický a přístupný "
     "bratranec hry na schovávanou od OpenAI a brána ke vznikajícímu obkličování a léčkám. "
     "Každý druh si střídavým self-play trénuje vlastní sdílený „mozek“."),
    ("mpe_tag_pack", 6, 2, 2, "Predator–Prey (6 vs 2)", "advanced", 0.0, 120.0, -100.0, 0.0, 1_000_000,
     "The same predator–prey chase scaled up to a six-predator pack hunting two prey — richer pack "
     "coordination, more chances for the prey to split the hunters and escape, and a harder "
     "credit-assignment problem for both species. A vivid ecosystem to watch once each species has "
     "its own trained brain.",
     "Stejná honička predátor–kořist zvětšená na šestičlennou smečku lovící dvě kořisti — bohatší "
     "koordinace smečky, víc příležitostí pro kořist rozdělit lovce a uniknout a těžší přiřazení "
     "zásluh pro oba druhy. Názorný ekosystém ke sledování, jakmile každý druh dostane vlastní "
     "natrénovaný „mozek“."),
]

for _mpe_tag_row in _MPE_TAG_GAMES:
    register(_mpe_tag_spec(*_mpe_tag_row))


# ---------------------------------------------------------------------------
# MuJoCo family (continuous control / robotics) — G5a "install + human-play on CPU
# now, training GPU-gated" (the Atari/BipedalWalker pattern).
#
# These are vector-obs + **continuous Box** physics envs (image-free, MlpPolicy), so
# they are DATA ROWS that reuse two existing seams with no engine code:
#   * the G1b/G3b continuous-box action seam — box-aware predict/play/preview, and a
#     PER-JOINT vector play keymap (each held key contributes a torque vector that the
#     frontend sums; play_session reshapes + clips it), exactly like BipedalWalker; and
#   * the server-JPEG render path — MuJoCo is NOT in client_render, so client_state()
#     returns None and the streamer renders env.render()→rgb_array→JPEG (like Atari).
# The risk the prompt flagged (offscreen rgb_array on Windows needing a GL backend) was
# checked FIRST: all six envs render a 480×480×3 frame on the laptop with the bundled
# glfw/pyopengl, so G5a builds here — no GL rabbit-hole, nothing deferred for rendering.
#
# Training is **gated** (hw_requirement="gpu"). Unlike CarRacing (which genuinely needs
# the CnnPolicy seam for its image obs), MuJoCo is a vector env that *could* train on the
# CPU — but a good gait takes a few million steps, impractical on the laptop, so it is
# step-count-gated to the desktop exactly like BipedalWalker. The RTX 5070 is also where
# SAC/TD3 (the algorithms that shine on MuJoCo, S5) will land. Human play needs no trained
# model and is available now. supported_algos=["ppo"] — neuroevolution is opted out as data
# (population search is impractical on hard multi-joint locomotion), like the box2d heavies.
#
# Skill: min_score is the venv-measured **idle (zero-torque) baseline** per the ADR-026 rule
# (a do-nothing agent must read ~0%, NOT the deeper random/flailing floor). MuJoCo's quirk is
# that "idle" is POSITIVE for the locomotion envs — the per-step "healthy" bonus accrues while
# a standing robot has not yet fallen (Ant stands the full episode ≈ +990; Hopper/Walker fall
# after ≈ +100–150) — so each env's floor is its own measured idle return, not a shared 0.
# HalfCheetah/Swimmer idle ≈ 0 (never terminate; random flailing goes negative → clamps to 0%).
# floor_scales_with_steps is False for the locomotion envs (a fall is terminal; the floor does
# not deepen with the cap) and play_step_scale=1 (their native ~1000 steps is a fine play
# length). Reacher is the exception: a 50-step arm-reach with a per-step distance penalty (a
# genuine step-penalty env, like Pendulum), so floor_scales_with_steps=True and play_step_scale=6
# lengthens its very short episode for a human (the floor widens with it). solved_score is each
# env's gym reward_threshold where one exists (Hopper 3800, HalfCheetah 4800, Ant 6000, Reacher
# −3.75, Swimmer 360); Walker2d has none, so 3500 is the widely-cited strong-PPO mark.
# ---------------------------------------------------------------------------


def _mujoco_spec(
    env_id: str,
    gym_id: str,
    display: str,
    difficulty: Literal["beginner", "intermediate", "advanced"],
    min_score: float,
    solved_score: float,
    default_total_timesteps: int,
    play_step_scale: int,
    floor_scales_with_steps: bool,
    desc_en: str,
    desc_cz: str,
) -> EnvSpec:
    """Build one MuJoCo EnvSpec from a data row (the family is otherwise identical: vector obs,
    continuous box action, PPO-only, GPU-gated training, human-playable now)."""
    return EnvSpec(
        id=env_id,
        gym_id=gym_id,
        display_name=Bilingual(en=display, cz=display),  # the gym ids are the conventional names
        description=Bilingual(en=desc_en, cz=desc_cz),
        family="mujoco",
        obs_type="vector",  # a fixed-length float state → MlpPolicy (no CnnPolicy); server-JPEG render
        action_space="box",  # continuous per-joint torques in [-1, 1] — the G1b/G3b continuous-box seam
        supported_algos=["ppo"],  # PPO-only (evolution opted out as data — hard multi-joint locomotion)
        hyperparams=_standard_hyperparams(),
        solved_score=solved_score,
        min_score=min_score,  # venv-measured idle (zero-torque) baseline → a do-nothing reads ~0% (ADR-026)
        default_total_timesteps=default_total_timesteps,  # the ★ PPO budget when GPU training lands
        play_step_scale=play_step_scale,
        floor_scales_with_steps=floor_scales_with_steps,
        human_playable=True,
        competitive=False,
        difficulty=difficulty,
        hw_requirement="gpu",  # millions of steps → desktop; play available now (like BipedalWalker)
    )


# id, gym_id, display, difficulty, min_score, solved_score, budget, play_scale, floor_scales, desc EN, desc CZ
_MUJOCO_GAMES: list[
    tuple[str, str, str, Literal["beginner", "intermediate", "advanced"], float, float, int, int, bool, str, str]
] = [
    ("hopper", "Hopper-v5", "Hopper-v5", "advanced", 120.0, 3800.0, 1_000_000, 1, False,
     "Teach a one-legged robot to hop forward as far as it can without toppling over. A MuJoCo "
     "physics task with an 11-number state (joint angles and velocities) and three continuous "
     "joint torques — thigh, knee and ankle. Hopping forward earns reward, staying upright earns "
     "a small bonus each step, and a fall ends the run; a good hopper scores around +3800 (the "
     "'solved' mark).",
     "Naučte jednonohého robota skákat vpřed co nejdál, aniž by se převrátil. Úloha s fyzikou "
     "MuJoCo se stavem o 11 číslech (úhly a rychlosti kloubů) a třemi spojitými momenty v kloubech "
     "— stehno, koleno a kotník. Skákání vpřed dává odměnu, udržení vzpřímené polohy přidává každý "
     "krok malý bonus a pád běh ukončí; dobrý skokan dosáhne kolem +3800 (hranice „vyřešeno“)."),
    ("walker2d", "Walker2d-v5", "Walker2d-v5", "advanced", 80.0, 3500.0, 1_000_000, 1, False,
     "Teach a two-legged robot to walk forward as far as it can without falling. A MuJoCo task "
     "with a 17-number state and six continuous joint torques — a thigh, knee and foot on each "
     "leg. Forward progress and staying upright earn reward, a fall ends the run; a smooth walk "
     "scores well into the thousands (around +3500 counts as a strong gait).",
     "Naučte dvounohého robota chodit vpřed co nejdál, aniž by upadl. Úloha s fyzikou MuJoCo se "
     "stavem o 17 číslech a šesti spojitými momenty v kloubech — stehno, koleno a chodidlo na "
     "každé noze. Postup vpřed a udržení vzpřímené polohy dávají odměnu, pád běh ukončí; plynulá "
     "chůze dosáhne klidně tisíců (kolem +3500 je už silná chůze)."),
    ("halfcheetah", "HalfCheetah-v5", "HalfCheetah-v5", "advanced", 0.0, 4800.0, 1_000_000, 1, False,
     "Teach a two-legged 'cheetah' to run forward as fast as possible. A MuJoCo task with a "
     "17-number state and six continuous joint torques (a thigh, shin and foot at the front and "
     "the back). It never falls over — the whole challenge is a fast, efficient gait. Reward is "
     "the forward speed minus a small effort cost, and a strong run scores around +4800.",
     "Naučte dvounohého „geparda“ běžet vpřed co nejrychleji. Úloha s fyzikou MuJoCo se stavem o "
     "17 číslech a šesti spojitými momenty v kloubech (stehno, holeň a chodidlo vepředu i vzadu). "
     "Nikdy se nepřevrátí — celou výzvou je rychlý, úsporný běh. Odměnou je rychlost vpřed minus "
     "malá cena za námahu a silný běh dosáhne kolem +4800."),
    ("ant", "Ant-v5", "Ant-v5", "advanced", 980.0, 6000.0, 2_000_000, 1, False,
     "Teach a four-legged robot to walk forward across the plane. A MuJoCo task with a large "
     "105-number state and eight continuous joint torques — a hip and an ankle on each of four "
     "legs. Forward progress and staying healthy earn reward, flipping over ends the run; a good "
     "walker scores around +6000 (the 'solved' mark).",
     "Naučte čtyřnohého robota chodit vpřed po rovině. Úloha s fyzikou MuJoCo s velkým stavem o "
     "105 číslech a osmi spojitými momenty v kloubech — kyčel a kotník na každé ze čtyř nohou. "
     "Postup vpřed a udržení „zdraví“ dávají odměnu, převrácení běh ukončí; dobrý chodec dosáhne "
     "kolem +6000 (hranice „vyřešeno“)."),
    ("reacher", "Reacher-v5", "Reacher-v5", "intermediate", -12.0, -3.75, 200_000, 6, True,
     "Steer a two-jointed robot arm so its tip touches a target dot, then hold it there. A short "
     "MuJoCo task with a 10-number state and two continuous joint torques. Every step costs the "
     "distance to the target plus a little for effort, so a quick, steady reach scores best (near "
     "−3.75, the 'solved' mark); the episode is only 50 steps long.",
     "Naveďte dvoukloubové robotické rameno tak, aby se jeho špička dotkla cílové tečky, a pak ji "
     "tam udržte. Krátká úloha s fyzikou MuJoCo se stavem o 10 číslech a dvěma spojitými momenty v "
     "kloubech. Každý krok stojí vzdálenost k cíli plus trochu za námahu, takže nejlépe boduje "
     "rychlé a klidné dosažení (kolem −3,75, hranice „vyřešeno“); epizoda trvá jen 50 kroků."),
    ("swimmer", "Swimmer-v5", "Swimmer-v5", "intermediate", 0.0, 360.0, 1_000_000, 1, False,
     "Teach a three-link 'swimmer' to glide forward through a viscous fluid by rippling its two "
     "joints. A MuJoCo task with an 8-number state and two continuous joint torques. Reward is the "
     "forward speed minus a small effort cost, so a good swimming rhythm — alternating the joints "
     "in time — scores around +360.",
     "Naučte třídílného „plavce“ klouzat vpřed viskózní tekutinou vlněním svých dvou kloubů. Úloha "
     "s fyzikou MuJoCo se stavem o 8 číslech a dvěma spojitými momenty v kloubech. Odměnou je "
     "rychlost vpřed minus malá cena za námahu, takže dobrý plavecký rytmus — střídání kloubů v "
     "pravý čas — dosáhne kolem +360."),
]

for _mj_row in _MUJOCO_GAMES:
    register(_mujoco_spec(*_mj_row))


# ---------------------------------------------------------------------------
# Board games (G6a — OpenSpiel turn-based self-play subsystem, ADR-050). The 7th seam: a
# 2-player, turn-based, perfect-info, zero-sum game with legal-move masking and self-play —
# OpenSpiel's pyspiel.State API, NOT a gym.Env. A board row is a discoverable picker entry but
# is **routed to app/services/board_engine.py via is_board_game** (mirroring is_multi_agent for
# PettingZoo), so it never goes through app.envs.factory.make_env. Only Tic-Tac-Toe ships, both
# human-playable (vs a training-free MCTS opponent, G6a) and NOW trainable: the neural board trainer
# landed in G6b (train_implemented=True) — MaskablePPO learns by playing the MCTS teacher (ADR-051),
# so the agent's skill curve, Save/Load, Watch-AI and Play-vs-your-net all come alive. obs_type=
# "vector" is an inert tag here (the observation_tensor IS a flat vector, but make_env never runs for
# a board env); supported_algos=["ppo"] is the simple_tag precedent — competitive self-play is
# surfaced as "ppo" and the manager routes board+ppo → app/services/trainer_board.py via is_board_game
# (the parallel to is_competitive_ma → train_tag). hyperparams uses the standard PPO block (ent_coef
# ★ 0.01); the round schedule is internal. CPU — no GPU gate.
# ---------------------------------------------------------------------------

register(
    EnvSpec(
        id="tictactoe",
        gym_id="tic_tac_toe",  # the OpenSpiel short name (resolved by app.services.board_engine)
        display_name=Bilingual(en="Tic-Tac-Toe", cz="Piškvorky 3×3"),
        description=Bilingual(
            en="The classic 3×3 game: take turns placing your mark and try to get three in a row. "
            "Play against a built-in AI that searches ahead (Monte-Carlo Tree Search) — pick a side and "
            "a difficulty — or **train your own neural net** to play (it learns by playing the search AI) "
            "and then face it. With perfect play on both sides every game is a draw, which makes it the "
            "perfect, testable first board game.",
            cz="Klasická hra 3×3: střídavě pokládáte své značky a snažíte se dostat tři v řadě. Hrajte "
            "proti vestavěné AI, která prohledává tahy dopředu (Monte-Carlo stromové prohledávání) — "
            "vyberte si stranu a obtížnost — nebo si **natrénujte vlastní neuronovou síť** (učí se hrou "
            "proti prohledávací AI) a pak se jí postavte. Při dokonalé hře obou stran je každá partie "
            "remíza, což z ní dělá ideální, ověřitelnou první deskovou hru.",
        ),
        family="board",
        obs_type="vector",  # inert tag — board games are routed, never made via make_env
        action_space="discrete",  # Discrete(9): place a mark in one of the nine cells
        supported_algos=["ppo"],  # surfaced as "ppo"; routed to the board trainer by is_board_game (G6b)
        hyperparams=_board_hyperparams(),  # standard PPO knobs (ent_coef ★ 0.01); rounds is internal
        # The learning chart plots eval-vs-reference-MCTS ∈ [−1, 1] as ep_rew_mean, so the meter scale
        # already matches: solved = +1 (win), min = −1 (loss); a well-trained TTT net converges toward 0
        # (draws — the game's ceiling). Board PLAY still shows a W/D/L card, not the continuous meter.
        solved_score=1.0,
        min_score=-1.0,
        default_total_timesteps=100_000,  # ★ budget — a near-optimal (drawing) net in ~80–100k (G6b)
        play_step_scale=1,
        floor_scales_with_steps=False,
        turn_based=True,  # one move per click; the board subsystem drives the turn loop
        human_playable=True,  # play a side vs the MCTS AI or your trained net
        competitive=True,  # 2-player zero-sum → routed to the board trainer (G6b), like simple_tag
        difficulty="beginner",
        hw_requirement="cpu",  # MCTS + the MaskablePPO board trainer both run on CPU (no GPU gate)
        train_implemented=True,  # neural trainer landed in G6b (MaskablePPO vs the MCTS teacher, ADR-051)
    )
)

# Connect Four (G6c) — the SECOND board game, proving the subsystem is game-agnostic: a data row + a
# renderer glyph map, no engine code (board_engine / play_session / trainer_board / preview all key off
# the generic pyspiel API and resolve the game from gym_id). The one real difference vs Tic-Tac-Toe is
# that a move is a COLUMN (7 actions over a 6×7 board), not a cell — handled in the renderer via a data
# flag (content/boardGames.ts actionMode="column"), not here. It is a much bigger game (~10^13 states),
# so the board trainer scores/teaches it against the EASY MCTS (board_engine.BOARD_PROFILES, G6c) — the
# net demonstrably learns to beat the weak search bot on a CPU budget, and the honest skill curve climbs
# instead of sitting pinned at the loss floor — and its ★ budget is larger than TTT's.
register(
    EnvSpec(
        id="connect_four",
        gym_id="connect_four",  # the OpenSpiel short name (resolved by app.services.board_engine)
        display_name=Bilingual(en="Connect Four", cz="Čtyři v řadě"),
        description=Bilingual(
            en="Drop discs into a 7-column, 6-row grid and try to line up four of your colour — "
            "horizontally, vertically or diagonally — before the AI does. Play against a built-in AI "
            "that searches ahead (Monte-Carlo Tree Search) — pick a side and a difficulty — or **train "
            "your own neural net** to play (it learns by playing the search AI) and then face it. A "
            "bigger, deeper game than Tic-Tac-Toe, so the AI's tactics really show.",
            cz="Vhazujte žetony do mřížky o 7 sloupcích a 6 řadách a snažte se spojit čtyři své barvy "
            "v řadě — vodorovně, svisle nebo úhlopříčně — dřív než AI. Hrajte proti vestavěné AI, která "
            "prohledává tahy dopředu (Monte-Carlo stromové prohledávání) — vyberte si stranu a obtížnost "
            "— nebo si **natrénujte vlastní neuronovou síť** (učí se hrou proti prohledávací AI) a pak se "
            "jí postavte. Větší a hlubší hra než piškvorky, takže taktika AI opravdu vynikne.",
        ),
        family="board",
        obs_type="vector",  # inert tag — board games are routed, never made via make_env
        action_space="discrete",  # Discrete(7): drop a disc into one of the seven columns
        supported_algos=["ppo"],  # surfaced as "ppo"; routed to the board trainer by is_board_game
        hyperparams=_board_hyperparams(),  # standard PPO knobs (ent_coef ★ 0.01); rounds is internal
        # Same eval-vs-reference-MCTS ∈ [−1, 1] chart scale as TTT (solved = +1, min = −1); the trainer
        # scores Connect Four against the EASY MCTS (BOARD_PROFILES) so the curve is honest on CPU.
        solved_score=1.0,
        min_score=-1.0,
        default_total_timesteps=200_000,  # ★ budget — a bigger game than TTT needs more steps (≈7 min CPU)
        play_step_scale=1,
        floor_scales_with_steps=False,
        turn_based=True,  # one move per click; the board subsystem drives the turn loop
        human_playable=True,  # play a side vs the MCTS AI or your trained net
        competitive=True,  # 2-player zero-sum → routed to the board trainer, like simple_tag
        difficulty="intermediate",  # a deeper game than the beginner TTT
        hw_requirement="cpu",  # MCTS + the MaskablePPO board trainer both run on CPU (no GPU gate)
        train_implemented=True,  # the same game-agnostic neural board trainer (MaskablePPO vs MCTS, G6b)
    )
)

# Othello / Reversi (G6d) — the THIRD board game. Still data-only on the engine side (the board
# subsystem resolves the game from gym_id), but it exercises two generic wrinkles the smaller games
# didn't: (1) OpenSpiel prints its 8×8 board *decorated* (row/column labels, "-" for empty), so the
# board parser detokenises it (board_engine._board_grid, keyed by gym_id) — TTT/Connect Four stay
# byte-identical; (2) a **pass** move when a player has no legal placement (action 64 of 65), surfaced
# generically as BoardState.pass_action → a Pass button in the renderer. A click is still a single
# CELL placement (actionMode "cell"), and flips are automatic (the streamed board already carries the
# post-move discs, so the renderer needs no flip logic). Much bigger than Connect Four (~10^28 states),
# so it trains against the near-random NOVICE MCTS teacher (ramping to easy) and is scored vs the easy
# reference (board_engine.BOARD_PROFILES) — its honest curve climbs ≈−0.7→+0.2 on a CPU budget.
register(
    EnvSpec(
        id="othello",
        gym_id="othello",  # the OpenSpiel short name (resolved by app.services.board_engine)
        display_name=Bilingual(en="Othello", cz="Othello (Reversi)"),
        description=Bilingual(
            en="Place a disc on the 8×8 board so it traps a line of the AI's discs between your new "
            "disc and another of yours — every trapped disc flips to your colour. Whoever owns more "
            "discs when no moves remain wins. If you have no legal move you must pass. Play against a "
            "built-in AI that searches ahead (Monte-Carlo Tree Search) — pick a side and a difficulty "
            "— or **train your own neural net** to play (it learns by playing the search AI) and then "
            "face it. A classic of swings and reversals, far bigger than Connect Four.",
            cz="Položte žeton na desku 8×8 tak, aby mezi váš nový žeton a jiný váš uvěznil souvislou "
            "řadu žetonů AI — každý uvězněný žeton se obrátí na vaši barvu. Vyhrává ten, kdo má víc "
            "žetonů, až nejsou možné tahy. Pokud nemáte legální tah, musíte vynechat (pass). Hrajte "
            "proti vestavěné AI, která prohledává tahy dopředu (Monte-Carlo stromové prohledávání) — "
            "vyberte si stranu a obtížnost — nebo si **natrénujte vlastní neuronovou síť** (učí se hrou "
            "proti prohledávací AI) a pak se jí postavte. Klasika plná zvratů, mnohem větší než Čtyři v řadě.",
        ),
        family="board",
        obs_type="vector",  # inert tag — board games are routed, never made via make_env
        action_space="discrete",  # Discrete(65): 64 cell placements + a pass move
        supported_algos=["ppo"],  # surfaced as "ppo"; routed to the board trainer by is_board_game
        hyperparams=_board_hyperparams(),  # standard PPO knobs (ent_coef ★ 0.01); rounds is internal
        # Same eval-vs-reference-MCTS ∈ [−1, 1] chart scale as the other board games (solved = +1, min =
        # −1); trained novice→easy and scored vs easy (BOARD_PROFILES) so the curve climbs honestly.
        solved_score=1.0,
        min_score=-1.0,
        default_total_timesteps=150_000,  # ★ budget — a much bigger game; the curve climbs over ~5 min CPU
        play_step_scale=1,
        floor_scales_with_steps=False,
        turn_based=True,  # one move per click; the board subsystem drives the turn loop
        human_playable=True,  # play a side vs the MCTS AI or your trained net
        competitive=True,  # 2-player zero-sum → routed to the board trainer, like simple_tag
        difficulty="advanced",  # the deepest board game so far
        hw_requirement="cpu",  # MCTS + the MaskablePPO board trainer both run on CPU (no GPU gate)
        train_implemented=True,  # the same game-agnostic neural board trainer (MaskablePPO vs MCTS, G6b)
    )
)

# Hopper and Walker2d render at 125 fps and fall fast, so even with human play capped at the 30 fps
# frame rate a person gets only ~8 s before the topple — too short to actually play. Stretch their
# human-play wall-clock ~5× (the user's request) so there is real time to react; the other MuJoCo
# envs already run slow enough (20–50 fps) and never fall this fast, so they keep the default 1.0.
# Set post-construction (a single data tweak on two of six rows) rather than threading a mostly-1.0
# column through the whole _MUJOCO_GAMES table.
for _slow_id in ("hopper", "walker2d"):
    _slow_spec = get_env(_slow_id)
    if _slow_spec is not None:
        _slow_spec.human_play_slowdown = 8.0
