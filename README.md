<p align="center">
  <img src="docs/hero.svg" alt="RL Lab — train, watch &amp; play reinforcement-learning agents" width="100%">
</p>

<h1 align="center">RL Lab</h1>

<p align="center">
  <b>Build, train, watch, and play against reinforcement-learning agents across 100+ environments — from one browser dashboard.</b>
</p>

<p align="center">
  Pick a game, tune the knobs with beginner-friendly info popups, train with one of 9 algorithms,
  watch the agent learn in real time, compare runs like a research paper, then <i>play against your own AI</i>
  with a skill meter. Bilingual&nbsp;(CZ/EN), dark&nbsp;/&nbsp;light.
</p>

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-REST%20%2B%20WebSocket-009688?logo=fastapi&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11%20%C2%B7%20CUDA%2012.8-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="React 19" src="https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black">
  <img alt="TypeScript 6" src="https://img.shields.io/badge/TypeScript-6-3178C6?logo=typescript&logoColor=white">
  <img alt="Vite 8" src="https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white">
  <br>
  <img alt="100+ environments" src="https://img.shields.io/badge/environments-100%2B-6e7bff">
  <img alt="8 families" src="https://img.shields.io/badge/families-8-6e7bff">
  <img alt="9 algorithms" src="https://img.shields.io/badge/algorithms-9-6e7bff">
  <img alt="tests" src="https://img.shields.io/badge/tests-421%20py%20%C2%B7%2074%20ts-3fb950">
  <img alt="i18n" src="https://img.shields.io/badge/CZ%20%2F%20EN-bilingual-e3b341">
</p>

---

## Trained agents, live

Every clip below is a real policy trained inside RL Lab, replayed from a saved checkpoint — a neuroevolution
lander, SAC/PPO MuJoCo robots, a PPO race car, a cooperative multi-agent swarm, Atari from pixels, and an
AlphaZero board player. The badge on each shows the algorithm and the skill score that run reached.

<table>
  <tr>
    <td align="center"><img src="docs/media/breakout.gif" width="250" alt="Breakout — DQN"></td>
    <td align="center"><img src="docs/media/lunarlander.gif" width="250" alt="Lunar Lander — Neuroevolution"></td>
    <td align="center"><img src="docs/media/carracing.gif" width="250" alt="Car Racing — PPO"></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/media/checkers.gif" width="250" alt="Checkers — AlphaZero"></td>
    <td align="center"><img src="docs/media/bipedalwalker.gif" width="250" alt="Bipedal Walker — PPO"></td>
    <td align="center"><img src="docs/media/pong.gif" width="250" alt="Pong — PPO"></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/media/humanoid.gif" width="250" alt="Humanoid — SAC"></td>
    <td align="center"><img src="docs/media/pursuit.gif" width="250" alt="Pursuit — cooperative multi-agent PPO"></td>
    <td align="center"><img src="docs/media/walker2d.gif" width="250" alt="Walker2D — PPO"></td>
  </tr>
</table>

---

## The dashboard

One screen does it all: choose an environment, set hyperparameters, and press **Run**. The reward curve climbs
live, the agent renders next to it (decoupled from training, so watching never perturbs the run), and a skill
meter reads out where the agent sits between "idle" and "solved."

<p align="center">
  <img src="docs/media/dashboard-training.jpg" width="100%" alt="RL Lab dashboard mid-training: a Q-learning run on FrozenLake 8×8, the reward curve climbing toward the goal, the agent navigating the frozen grid, a live Q-table heatmap, CPU telemetry, and a skill meter">
</p>

<p align="center"><i>A Q-learning run on FrozenLake 8×8, mid-training — the reward curve climbing toward the goal, the decoupled grid preview, the live Q-table heatmap, CPU telemetry, and a skill meter.</i></p>

### Dark &amp; light, both first-class

Every surface uses a semantic design system (the "Laboratory" theme) so dark and light both look intentional —
tabular numerics in a monospaced face, 2px dividers between panels, 1px inside. Below, the **same tabular
Q-learning run on Taxi** learning in real time — the reward curve climbing from ≈ −800 toward the goal, the taxi
navigating the grid, the skill meter rising from *Child* to *Superhuman*, and the Q-table heatmap filling in.

<table>
  <tr>
    <td align="center"><img src="docs/media/taxi-dark.gif" width="100%" alt="Taxi Q-learning training, dark theme — reward curve climbing, Q-table filling"><br><b>Dark</b></td>
    <td align="center"><img src="docs/media/taxi-light.gif" width="100%" alt="Taxi Q-learning training, light theme — reward curve climbing, Q-table filling"><br><b>Light</b></td>
  </tr>
</table>

---

## Features

| Capability | What you get |
| :-- | :-- |
| 🎮 **100+ environments** | Eight families — Classic Control → Toy Text → MiniGrid → Box2D → Atari → MuJoCo → board games → multi-agent — all behind one data-driven registry. |
| 🧠 **9 algorithms** | PPO · neuroevolution · tabular Q-learning · AlphaZero · SAC · TD3 · DQN · A2C · QR-DQN, gated per-environment with a ★ recommended pick for each game. |
| 📈 **Live training** | Realtime reward / loss / fitness charts with EMA smoothing, a "solved @" marker, and a multi-run compare overlay. |
| 👀 **Watch it learn** | The running policy renders live — client-side SVG for vector envs, server-streamed frames for pixels / MuJoCo — with visual on/off and time-acceleration. |
| 🕹️ **Play vs your AI** | Take control over WebSocket and go head-to-head with the trained agent; a skill meter grades you Child → Below avg → Average → Above avg → Superhuman, with named leaderboards. |
| 🔬 **Data Lab** | A full experiment-analysis surface: seed sweeps, `rliable` aggregation (IQM, bootstrap CIs, performance profiles), a ranked summary table, and one-click export to CSV / Excel / LaTeX / TensorBoard / repro-card. |
| 💾 **Save / resume / export** | A filterable checkpoint manager — resume training from any snapshot, or export a run as a citable dataset. |
| 📚 **Learn as you go** | Every tunable ships a bilingual info popup (what it is, ★ recommended value, range, and a note for *this* game). |
| 🌍 **Bilingual &amp; themed** | CZ / EN and dark / light toggles, persisted; accessibility (aria-labels) enforced by a checker. |
| 🔁 **Reproducible** | Every run records its full config + seed; "reproduce this run" is a `curl` command in the repro card. |

---

## Environments

**100+ environments across eight families.** The registry
([`backend/app/envs/registry.py`](backend/app/envs/registry.py)) is the single source of truth — an `EnvSpec`
row does most of the work, so most new games are a data-only addition.

<p align="center">
  <img src="docs/media/env-picker.png" width="100%" alt="Environment picker flyout showing all eight families with game counts">
</p>

| Family | Count | Examples | Notes |
|---|:--:|---|---|
| **Classic Control** | 5 | CartPole, MountainCar, Acrobot, Pendulum | vector obs, discrete + continuous actions |
| **Toy Text** | 5 | FrozenLake, Taxi, CliffWalking | discrete obs → tabular Q-learning + PPO / evo |
| **MiniGrid** | 4 | Empty, DoorKey, KeyCorridor, FourRooms | `Dict` obs flattened per family; turn-based |
| **Box2D** | 4 | LunarLander, BipedalWalker, CarRacing | continuous control; CarRacing is image + box |
| **Atari** | 64 | Pong, Breakout, Ms. Pac-Man, Enduro … | image obs → CNN policy on CUDA |
| **MuJoCo** | 7 | Hopper, Walker2d, HalfCheetah, Ant, Humanoid … | continuous torques; SAC is the ★ pick |
| **Board games** | 6 | Tic-Tac-Toe, Connect Four, Othello, Breakthrough, Checkers, Chess | OpenSpiel; MaskablePPO / AlphaZero vs an MCTS teacher |
| **Multi-agent** | 7 | simple_spread, simple_tag, Pursuit, Multiwalker, Waterworld | PettingZoo + SuperSuit param-sharing / self-play |

See [`docs/adding-an-environment.md`](docs/adding-an-environment.md) and the
[extensibility seams](docs/architecture.md#the-extensibility-seams).

---

## Algorithms

Every algorithm plugs into one training manager behind a single peer-trainer seam. Each game
declares which algorithms it supports, and which one is ★ recommended.

| Algorithm | Kind | Best for | Notes |
| :-- | :-- | :-- | :-- |
| **PPO** | on-policy policy-gradient | almost everything | Stable-Baselines3; the universal baseline |
| **Neuroevolution** | evolutionary | small vector envs | custom numpy; no gradients, population-based |
| **Q-learning** | tabular value-based | discrete Toy Text | numpy; ships a `<canvas>` Q-table heatmap |
| **AlphaZero** | self-play + MCTS | board games | reimplemented in-repo, trained vs an MCTS teacher |
| **SAC** | off-policy actor-critic | continuous control (MuJoCo) | the ★ pick for robotics |
| **TD3** | off-policy deterministic | continuous control | twin critics + delayed updates |
| **DQN** | off-policy value-based | discrete + Atari | ε-greedy; the original deep-RL Atari algorithm (Mnih et al., 2015) |
| **A2C** | on-policy actor-critic | discrete + continuous | PPO's simpler predecessor — one un-clipped update per rollout |
| **QR-DQN** | distributional value-based | discrete + Atari | DQN that learns each action's whole return distribution (quantiles); a Rainbow ingredient |

New algorithms follow [`docs/adding-an-algorithm.md`](docs/adding-an-algorithm.md).

---

## Data Lab

Training gives you curves; the **Data Lab** gives you *conclusions*. Select any runs on disk and it overlays
their learning curves, collapses multiple seeds into a mean ± CI band, and — crucially — lets you **compare
algorithms head-to-head**. It computes the `rliable` metrics a modern RL paper reports — IQM, mean, median,
optimality gap (all with 95% stratified-bootstrap CIs), plus performance profiles and probability-of-improvement.
A ranked summary table sorts by AUC / final-% / time-to-solve, and one click exports the selection as CSV, Excel
(with native charts), a LaTeX booktabs table, a TensorBoard log dir, a standalone SVG figure, or a
reproducibility card with a config hash + BibTeX.

<p align="center">
  <img src="docs/media/data-lab.gif" width="100%" alt="Data Lab in action: runs are added one by one to a head-to-head comparison of PPO vs Neuroevolution on CartPole — learning curves overlay, seeds collapse into mean ± CI bands, and the rliable aggregate (IQM/mean/median/optimality-gap), performance profile, probability-of-improvement, and ranked summary table update live as the selection grows">
</p>

<p align="center"><i>Building a comparison live: pick runs on the left and the Data Lab overlays their curves, collapses seeds into a mean ± CI band, and recomputes the full <code>rliable</code> aggregate, performance profile, and ranked table on the fly — here <b>PPO</b> vs <b>Neuroevolution</b> on CartPole. Wide bands on few seeds are shown honestly; that width <b>is</b> the message.</i></p>

> See [`docs/reproducibility.md`](docs/reproducibility.md) for how runs are recorded and reproduced.

---

## Learn as you go

RL Lab is built to be *understood*, not just run. Every parameter has a data-driven popup — general explanation,
a ★ recommended value, the sane range, and a note specific to the game you're on — in both Czech and English.

<p align="center">
  <img src="docs/media/param-popup.png" width="100%" alt="A beginner-friendly info popup for the Learning Rate parameter: what it is, recommended value, range, and a note for CartPole">
</p>

---

## Board games &amp; self-play

Board games route through OpenSpiel and train with MaskablePPO or AlphaZero against a Monte-Carlo-Tree-Search
teacher whose strength you can dial. Watch two AIs play it out, or take a side yourself and test your skill
against the trained agent.

<p align="center">
  <img src="docs/media/board.png" width="100%" alt="Connect Four mid-game between two AIs, with AlphaZero marked as the recommended algorithm">
</p>

---

## Tech stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.11 · FastAPI (REST + WebSocket) · PyTorch · Stable-Baselines3 + `sb3-contrib` (MaskablePPO) · custom numpy neuroevolution · Gymnasium · OpenSpiel · PettingZoo · SuperSuit |
| **Frontend** | React 19 · TypeScript · Vite · Tailwind · zustand · react-i18next · hand-rolled SVG charts |
| **Tooling** | ruff · mypy · pytest (backend) · eslint · vitest (frontend) · an i18n parity checker |

---

## Getting started (dev)

### Prerequisites

- **Python 3.11** — `winget install -e --id Python.Python.3.11` (the system 3.14 is too new for the ML stack)
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
python backend/verify_env.py                       # expects CUDA True + a PPO smoke test
cd frontend; npm install; cd ..
```

### Run it

```powershell
.\tasks.ps1 dev-backend     # FastAPI on http://127.0.0.1:8000  (hot-reload; API docs at /docs)
.\tasks.ps1 dev-frontend    # Vite on http://localhost:5173
```

Then open <http://localhost:5173>, pick an environment, and press **Run**. A standalone single-executable build
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
| [`docs/architecture.md`](docs/architecture.md) | System &amp; data flow, thread model, rendering paths, the five extensibility seams |
| [`docs/adding-an-environment.md`](docs/adding-an-environment.md) | The data-only path + the seams + the pre-delivery checklist |
| [`docs/adding-an-algorithm.md`](docs/adding-an-algorithm.md) | How a trainer plugs into the one manager |
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
│   │   ├── services/     # trainers, streamers, stores, training manager, Data Lab analysis
│   │   └── training/     # training utilities
│   ├── tests/
│   └── verify_env.py
├── frontend/
│   └── src/{components, api, store, i18n, content}
├── docs/                 # public docs (architecture, guides, API, ADRs) + media
├── data/                 # models + checkpoints + runs (gitignored)
├── tasks.ps1             # dev shortcuts
└── CLAUDE.md             # project guidance
```

---

## Hardware

Developed and trained on a single desktop:

- **Intel Ultra 7 265K · RTX 5070 12 GB · 32 GB RAM · Windows 11**
- Torch **`2.11.0+cu128`** (Blackwell `sm_120`). GPU training is live for **every** family — BipedalWalker,
  the MuJoCo robots, Atari, and CarRacing all run on the GPU.

`python backend/verify_env.py` checks for CUDA + the GPU and runs a PPO smoke test. CPU-only machines run all
human-play paths and every CPU-trainable environment identically (only GPU training is gated out).

---

## Acknowledgements

Built on the shoulders of [Gymnasium](https://gymnasium.farama.org/),
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/), [PyTorch](https://pytorch.org/),
[OpenSpiel](https://github.com/google-deepmind/open_spiel),
[PettingZoo](https://pettingzoo.farama.org/) + [SuperSuit](https://github.com/Farama-Foundation/SuperSuit),
and the [`rliable`](https://github.com/google-research/rliable) methodology (Agarwal et al., NeurIPS 2021).

## License

© 2026 Martin Svoboda. Private project — not yet open-sourced; all rights reserved. (An OSS license +
contributor guide are planned for a public release.)
