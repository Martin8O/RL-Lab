# ReinforcedLearningProjects — RL All-in-One Dashboard

**Creating an all-in-one tool for building and testing one's RL neural network on many games.**

A bilingual (🇨🇿/🇬🇧) dashboard to **pick an environment, tune parameters, train, watch, compare, and
play against** reinforcement-learning agents — using both **PPO** and **neuroevolution**, across a
growing list of games (CartPole → Pong/Breakout → LunarLander/CarRacing → board games).

> Status: **early development** (Phase A — foundation). Built incrementally with Claude Code.

## Features (target)
- 🎮 **Environment selector** — many RL games/simulations from one place.
- 🎚️ **Parameter sliders** with ★ recommended markers and **detailed info popups** (general +
  per-game, beginner-friendly, CZ/EN).
- 📈 **Realtime charts** — reward / loss / fitness, with EMA smoothing and window controls.
- 🕹️ **Live environment preview** — visual on/off + time-acceleration; faster training when off.
- 🧬 **Two learning methods** — PPO (Stable-Baselines3) and neuroevolution (population/fitness/mutation),
  with a **Top-5 leaderboard** and Set-Parent.
- 💾 **Save / Load / Export** checkpoints and **compare runs** (v1/v2/v3).
- 🤖 **Play vs AI** with a **skill meter** (child → below-avg → average → above-avg → superhuman).
- 🌗 **Dark/light** + 🌐 **CZ/EN** toggles.

## Tech stack
- **Backend:** Python 3.11 · FastAPI (REST + WebSocket) · PyTorch · Stable-Baselines3 · Gymnasium
- **Frontend:** React · TypeScript · Vite · Tailwind · zustand · react-i18next

## Getting started (dev)
> Detailed, exact steps are added as each phase lands; full desktop/GPU setup will live in `docs/MIGRATION.md`.

```powershell
# Backend (Python 3.11)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python backend/verify_env.py            # sanity check (expects "CUDA available: False" on laptop)
uvicorn app.main:app --app-dir backend --reload

# Frontend
cd frontend
npm install
npm run dev
```

**Hardware notes.** Develops on CPU (laptop) for CartPole; GPU games (Atari/Box2D/board) target an
NVIDIA RTX 5070 desktop with PyTorch on the `cu128` channel. See `docs/MIGRATION.md`.

## License
Private project.
