# RL All-in-One Dashboard

A bilingual (CZ/EN), dark/light **reinforcement-learning workbench**: pick an environment, tune
parameters with beginner-friendly info popups, train via **PPO** or **neuroevolution**, watch the agent
learn in real time, compare runs, and **play against the trained AI** with a skill meter — all from a
single browser dashboard. Built incrementally from CartPole (CPU) up toward Atari / Box2D / board games
(RTX desktop).

> **Status:** actively built in phases (see [`dev_history.md`](dev_history.md)). The CPU core is complete
> — six environments, both algorithms, full train → watch → play loop, persistence, i18n/a11y/theming,
> and a test + quality gate. GPU families (Atari, MuJoCo, board games) follow on the desktop.

---

<!--
  MEDIA SLOTS — add real captures here. The app must be running (`.\tasks.ps1 dev-backend` + `dev-frontend`).
  Recommended tool on Windows: ScreenToGif (https://www.screentogif.com/).

  1. Hero screenshot (dark theme), the full dashboard mid-training:
     ![RL Dashboard](docs/media/dashboard-dark.png)
  2. "Watch the AI learn" GIF — start a CartPole PPO run with the live preview on, ~6–10 s loop:
     ![Watch the AI learn](docs/media/train-cartpole.gif)
  3. "Play vs AI" GIF — human-play LunarLander, then the skill meter rating:
     ![Play vs AI](docs/media/play-vs-ai.gif)
  4. Light-theme screenshot (for the theming line below).
-->

> 📸 _Screenshots & demo GIFs go in `docs/media/` — see the comment above for the shot list._

---

## Features

| Area | Detail |
|---|---|
| **Environment selector** | Category → game flyout, grouped by family; adding a vector/discrete game is data-only in the registry |
| **Parameter sliders** | ★ recommended markers + rich bilingual info popups (general + per-environment, CZ/EN) |
| **Two algorithms** | **PPO** (Stable-Baselines3) and a custom **neuroevolution** with a Top-5 leaderboard; per-env `supported_algos` gating |
| **Realtime charts** | Reward / loss / fitness, EMA smoothing, multi-run compare overlay with a "solved @" marker |
| **Live preview** | Client-side SVG rendering of the running policy, visual on/off + time-acceleration (training is faster with visual off) — **decoupled** so it never perturbs training |
| **Save / Load / Export** | Checkpoint slots (resume training) + run-history compare (v1/v2/v3) |
| **Play vs AI** | Human session over WebSocket + a skill meter (child → below-avg → average → above-avg → superhuman), named human/AI leaderboards |
| **Bilingual / themed** | CZ/EN toggle + dark/light toggle, persisted to localStorage; accessibility (aria-labels) enforced by a checker |

---

## Environments

Six environments ship today (all CPU-trainable). The registry
([`backend/app/envs/registry.py`](backend/app/envs/registry.py)) is the single source of truth.

| Environment | Family | Obs | Action | Algorithms | Solved score | HW |
|---|---|---|---|---|---|---|
| CartPole-v1 | Classic Control | vector | discrete | PPO · neuroevolution | 500 | CPU |
| MountainCar-v0 | Classic Control | vector | discrete | PPO · neuroevolution | −110 | CPU |
| Acrobot-v1 | Classic Control | vector | discrete | PPO · neuroevolution | −100 | CPU |
| Pendulum-v1 | Classic Control | vector | **box** (continuous) | PPO · neuroevolution | −150 | CPU |
| MountainCarContinuous-v0 | Classic Control | vector | **box** (continuous) | PPO · neuroevolution | 90 | CPU |
| LunarLander-v3 | Box2D | vector | discrete | PPO · neuroevolution | 200 | CPU |

GPU families (Atari, MuJoCo, OpenSpiel board games, …) are designed for but not yet shipped — see the
[five extensibility seams](docs/architecture.md#the-five-extensibility-seams) and
[`docs/adding-an-environment.md`](docs/adding-an-environment.md).

---

## Tech stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11 · FastAPI (REST + WebSocket) · PyTorch · Stable-Baselines3 (PPO) · custom numpy neuroevolution · Gymnasium |
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
pip install ruff mypy pytest          # dev tools
python backend/verify_env.py          # expects "CUDA available: False" on a CPU laptop
cd frontend; npm install; cd ..
```

### Run it

```powershell
.\tasks.ps1 dev-backend     # FastAPI on http://127.0.0.1:8000 (hot-reload; API docs at /docs)
.\tasks.ps1 dev-frontend    # Vite on http://localhost:5173
```

Then open <http://localhost:5173>, pick an environment, and press **Run**.

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
| [`docs/adding-an-algorithm.md`](docs/adding-an-algorithm.md) | The peer-trainer pattern (how PPO and neuroevolution plug into one manager) |
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
│   │   ├── core/         # config, logging
│   │   ├── envs/         # environment registry (the source of truth)
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
├── docs/                 # public documentation (architecture, guides, API, ADRs)
├── data/                 # models + checkpoints + runs + scores (gitignored)
├── tasks.ps1             # dev shortcuts
└── CLAUDE.md             # project guidance
```

---

## Hardware notes

- **Laptop (now):** AMD Ryzen 5 2500U · no NVIDIA GPU → CPU-only. The six shipped environments train on
  CPU in seconds–minutes.
- **Desktop (planned):** Intel Ultra 7 265K · RTX 5070 12 GB → GPU families (Atari, MuJoCo, board games).
  Migration swaps the CPU torch wheels for `cu128` (Blackwell sm_120); a migration guide is planned for
  Phase F3.

---

## License

Private project — not yet open-sourced. (An OSS license + contributor guide are planned for the public
release; see Phase F4 in the backlog.)
