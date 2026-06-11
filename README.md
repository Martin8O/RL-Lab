# RL All-in-One Dashboard

A bilingual (CZ/EN), dark/light **reinforcement-learning workbench** where you pick an environment,
tune parameters with beginner-friendly info popups, train via **PPO** or **neuroevolution**, watch
the agent learn in real time, compare runs, and **play against the trained AI** with a skill meter —
all from a single browser dashboard.  Built incrementally from CartPole (CPU) up to Atari / Box2D /
board games (RTX 5070 desktop).

> **Status:** Phase A — foundation.  CartPole + PPO end-to-end is the current milestone.

---

<!--
  SCREENSHOT SLOT — replace this comment with:
  ![Dashboard screenshot](docs/screenshot.png)
  once the frontend is running (Phase A4 onwards).
-->

---

## Features (target)

| Area | Detail |
|---|---|
| **Environment selector** | Many RL games/simulations; adding a game is data-only in the registry |
| **Parameter sliders** | ★ recommended markers + detailed CZ/EN info popups (general + per-game) |
| **Realtime charts** | Reward / Loss / Fitness, EMA smoothing, window controls |
| **Live preview** | Visual on/off + time-acceleration; training is faster with visual off |
| **Two algorithms** | PPO (Stable-Baselines3) and custom neuroevolution with Top-5 leaderboard |
| **Save / Load / Export** | Checkpoints + run-history compare (v1/v2/v3) |
| **Play vs AI** | Human session + skill meter (child → below-avg → average → above-avg → superhuman) |
| **Bilingual / themed** | CZ/EN toggle + dark/light toggle, persisted to localStorage |

---

## Tech stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11 · FastAPI (REST + WebSocket) · PyTorch · Stable-Baselines3 · Gymnasium |
| **Frontend** | React · TypeScript · Vite · Tailwind · zustand · react-i18next |
| **Tooling** | ruff · mypy · pytest · (eslint / prettier on the frontend) |

---

## Getting started (dev)

### Prerequisites

- **Python 3.11** — `winget install -e --id Python.Python.3.11`  (system 3.14 is too new for the ML stack)
- **Node 20+** — `winget install OpenJS.NodeJS.LTS`

### One-time setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
pip install ruff mypy pytest          # dev tools
python backend/verify_env.py          # expects "CUDA available: False" on laptop
```

### Daily workflow — use `tasks.ps1`

```powershell
.\tasks.ps1 dev-backend     # start FastAPI on :8000 (hot-reload)
.\tasks.ps1 dev-frontend    # start Vite on :5173  (A4 onwards)
.\tasks.ps1 lint            # ruff + mypy
.\tasks.ps1 test            # pytest
.\tasks.ps1 all             # lint + test
```

Or run manually:

```powershell
# Backend
uvicorn app.main:app --app-dir backend --reload

# Frontend (A4 onwards)
cd frontend && npm run dev
```

### Environment variables

Copy `backend/.env.example` to `backend/.env` and adjust:

```
HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=http://localhost:5173
```

---

## Project structure

```
RL/
├── backend/
│   ├── app/
│   │   ├── api/          # REST routers
│   │   ├── core/         # config, logging
│   │   ├── envs/         # environment registry
│   │   ├── schemas/      # pydantic models / TS-mirrored contracts
│   │   ├── services/     # trainers, connection manager
│   │   └── training/     # training utilities
│   ├── tests/
│   ├── pyproject.toml    # ruff + mypy + pytest config
│   ├── requirements.txt
│   ├── .env.example
│   └── verify_env.py
├── frontend/             # React + Vite (scaffolded in A4)
├── docs/
│   └── MIGRATION.md      # desktop / RTX 5070 setup guide (F3)
├── data/                 # models + checkpoints (gitignored)
├── Local/                # Claude's private workspace (gitignored)
├── tasks.ps1             # dev shortcuts
└── CLAUDE.md             # project guidance for Claude Code
```

---

## Hardware notes

- **Laptop (now):** AMD Ryzen 5 2500U · no GPU → CPU-only, CartPole / MLP environments.
- **Desktop (soon):** Intel Ultra 7 265K · RTX 5070 12 GB → GPU environments (Atari, Box2D, board games).
  See `docs/MIGRATION.md` for the exact torch `cu128` swap.

---

## License

Private project.
