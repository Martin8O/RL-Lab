<p align="center">
  <img src="docs/hero.svg" alt="RL Lab — train, watch &amp; play reinforcement-learning agents" width="100%">
</p>

# RL Lab

A bilingual (CZ/EN), dark/light **reinforcement-learning workbench**: pick from **98 environments**, tune
parameters with beginner-friendly info popups, train with **7 algorithms**, watch the agent learn in real
time, compare runs, and **play against the trained AI** with a skill meter — all from a single browser
dashboard, from CartPole and Atari through MuJoCo physics to board games and multi-agent swarms.

> **Status:** all shipped families train on the GPU desktop. A single registry
> ([`backend/app/envs/registry.py`](backend/app/envs/registry.py)) is the source of truth; the per-prompt
> history lives in [`dev_history.md`](dev_history.md) and the decision index in [`docs/adr.md`](docs/adr.md).

---

> 📸 _Screenshots & demo GIFs go in `docs/media/` — capture the dashboard mid-training, a "watch the AI
> learn" loop, and a "play vs AI" skill-meter clip._

---

## Features

| Area | Detail |
|---|---|
| **Environment selector** | Category → game flyout, grouped by family; adding a vector/discrete game is data-only in the registry |
| **Parameter sliders** | ★ recommended markers + rich bilingual info popups (general + per-environment, CZ/EN) |
| **Seven algorithms** | **PPO** · custom **neuroevolution** · tabular **Q-learning** · **AlphaZero** (MaskablePPO vs an MCTS teacher) · **SAC** · **TD3** · **DQN**; per-env `supported_algos` gating + a ★ recommended algorithm per game |
| **Realtime charts** | Reward / loss / fitness, EMA smoothing, multi-run compare overlay with a "solved @" marker |
| **Live preview** | Decoupled rendering of the running policy (client-side SVG for vector envs, server JPEG for image/MiniGrid/MuJoCo) + visual on/off and time-acceleration — never perturbs training |
| **Save / Load / Export** | Filterable checkpoint manager (resume training) + run-history compare |
| **Play vs AI** | Human session over WebSocket + a skill meter (child → below-avg → average → above-avg → superhuman), named human/AI leaderboards |
| **Bilingual / themed** | CZ/EN toggle + dark/light toggle, persisted to localStorage; accessibility (aria-labels) enforced by a checker |

---

## Environments

**98 environments** across eight families. The registry
([`backend/app/envs/registry.py`](backend/app/envs/registry.py)) is the single source of truth — `EnvSpec`
flags do most of the work, so most new games are a data-only registry row.

| Family | Examples | Notes |
|---|---|---|
| **Classic Control** | CartPole, MountainCar, Acrobot, Pendulum | vector obs, discrete + continuous actions |
| **Toy Text (tabular)** | FrozenLake, Taxi, CliffWalking | discrete obs → tabular Q-learning + PPO/evo |
| **MiniGrid** | Empty, DoorKey, LavaGap, … | `Dict` obs flattened per family; turn-based |
| **Box2D (physics)** | LunarLander, BipedalWalker, CarRacing | continuous control; CarRacing is image + box |
| **Atari** | Breakout, Pong, Space Invaders, … | image obs → CNN policy on CUDA |
| **MuJoCo (robotics)** | Hopper, Walker2d, Ant, Humanoid, … | continuous torques; SAC is the ★ recommended algo |
| **Board games** | Tic-Tac-Toe, Connect Four, Breakthrough, … | OpenSpiel; MaskablePPO vs an MCTS teacher |
| **Multi-agent** | MPE (simple_spread, simple_tag) | PettingZoo + SuperSuit param-sharing / self-play |

See [`docs/adding-an-environment.md`](docs/adding-an-environment.md) and the
[five extensibility seams](docs/architecture.md#the-five-extensibility-seams).

---

## Tech stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11 · FastAPI (REST + WebSocket) · PyTorch · Stable-Baselines3 + `sb3-contrib` (MaskablePPO) · custom numpy neuroevolution · Gymnasium · OpenSpiel · PettingZoo · SuperSuit |
| **Frontend** | React · TypeScript · Vite · Tailwind · zustand · react-i18next |
| **Tooling** | ruff · mypy · pytest (backend) · eslint · vitest (frontend) · an i18n parity checker |

---

## Getting started (dev)

### Prerequisites

- **Python 3.11** — `winget install -e --id Python.Python.3.11` (system 3.14 is too new for the ML stack)
- **Node 20+** — `winget install OpenJS.NodeJS.LTS`

### One-time setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install ruff mypy pytest                       # dev tools
# GPU: swap the torch wheels to the CUDA 12.8 (Blackwell) index
pip install --index-url https://download.pytorch.org/whl/cu128 torch
python backend/verify_env.py                       # expects CUDA True + RTX 5070 + a PPO smoke test
cd frontend; npm install; cd ..
```

### Run it

```powershell
.\tasks.ps1 dev-backend     # FastAPI on http://127.0.0.1:8000 (hot-reload; API docs at /docs)
.\tasks.ps1 dev-frontend    # Vite on http://localhost:5173
```

Then open <http://localhost:5173>, pick an environment, and press **Run**. A standalone build (single exe)
is produced by `.\build-standalone.ps1 [-Zip]`.

### Quality gate

```powershell
.\tasks.ps1 lint     # ruff + mypy (backend) + eslint (frontend)
.\tasks.ps1 test     # pytest (backend) + vitest (frontend)
.\tasks.ps1 i18n     # en/cz key parity + every static t('key') resolvable
.\tasks.ps1 build    # tsc + vite production build
.\tasks.ps1 all      # lint + i18n + test + build  ← the one command to run before commit
```

### Environment variables

Copy `backend/.env.example` to `backend/.env` and adjust:

```
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=http://localhost:5173
```

---

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System & data flow (Mermaid), thread model, rendering paths, the five extensibility seams |
| [`docs/adding-an-environment.md`](docs/adding-an-environment.md) | The data-only path + the five seams + the pre-delivery checklist |
| [`docs/adding-an-algorithm.md`](docs/adding-an-algorithm.md) | The peer-trainer pattern (how the trainers plug into one manager) |
| [`docs/api.md`](docs/api.md) | REST endpoint + WebSocket frame reference |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeds, recorded config, the run archive, "reproduce this run" |
| [`docs/adr.md`](docs/adr.md) | Curated architecture-decision index |
| [`dev_history.md`](dev_history.md) | The changelog of record + full ADRs |

---

## Project structure

```
RL/
├── backend/
│   ├── app/
│   │   ├── api/          # REST routers (/api/*) + WS routing in main.py
│   │   ├── core/         # config, logging, path resolution
│   │   ├── envs/         # environment registry (the source of truth) + factory
│   │   ├── schemas/      # pydantic models = the contracts (mirrored in frontend types.ts)
│   │   ├── services/     # trainers, streamers, stores, training manager
│   │   └── training/     # training utilities
│   ├── tests/
│   ├── pyproject.toml    # ruff + mypy + pytest config
│   ├── requirements.txt
│   ├── .env.example
│   └── verify_env.py
├── frontend/
│   └── src/{components, api, store, i18n, content}
├── docs/                 # public documentation (architecture, guides, API, ADRs) + hero.svg
├── data/                 # models + checkpoints + runs + scores (gitignored)
├── tasks.ps1             # dev shortcuts
└── CLAUDE.md             # project guidance
```

---

## Hardware

Trained on a single desktop:

- **Intel Ultra 7 265K · RTX 5070 12 GB · 32 GB RAM · Windows 11**
- Torch **`2.11.0+cu128`** (Blackwell sm_120). GPU training is live for **every** family — BipedalWalker,
  the six MuJoCo robots, Atari, and CarRacing all run on the GPU.

`python backend/verify_env.py` checks for CUDA + the RTX 5070 and runs a PPO smoke test.

---

## License

Private project — not yet open-sourced. (An OSS license + contributor guide are planned for the public
release.)
