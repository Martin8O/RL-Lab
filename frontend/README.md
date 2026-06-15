# Frontend — RL All-in-One Dashboard

The React + TypeScript + Vite UI for the [RL All-in-One Dashboard](../README.md). It renders the
controls, the live reward/fitness chart, the client-side environment preview, and the play-vs-AI
experience, talking to the FastAPI backend over REST + a WebSocket.

## Stack

- **React + TypeScript + Vite** · **Tailwind** for layout (components use inline styles referencing the
  *Laboratory* design tokens in `src/index.css`)
- **zustand** — app state · **react-i18next** — CZ/EN bilingual UI

## Layout (`src/`)

| Folder | What it holds |
|---|---|
| `components/` | UI — Sidebar, RewardChart, EnvPreview/EnvStages, Play controls, SkillMeter, modals |
| `api/` | `client.ts` (REST + WS) and `types.ts` (the contracts, mirroring `backend/app/schemas/`) |
| `store/` | the zustand store |
| `i18n/` | `en.json` / `cz.json` (parity enforced by the i18n checker) |
| `content/` | data-driven copy — parameter popups, play guides, keymaps, env categories |

## Develop

Run from the **repo root** via `tasks.ps1` (preferred — it wires the backend too):

```powershell
.\tasks.ps1 dev-frontend    # Vite dev server on http://localhost:5173
.\tasks.ps1 lint            # eslint (+ backend ruff/mypy)
.\tasks.ps1 test            # vitest (+ backend pytest)
.\tasks.ps1 i18n            # en/cz key parity
.\tasks.ps1 build           # tsc + vite production build
```

Or directly in this folder: `npm install`, then `npm run dev` / `npm run lint` / `npm run test` /
`npm run i18n:check` / `npm run build`.

## Conventions

- **No hard-coded user-facing strings** — add keys to both `i18n/en.json` and `i18n/cz.json` (including
  `aria-label`/`title`); interactive controls need an accessible name.
- **Use the semantic design tokens** (`--surface-*`, `--text-*`, `--accent`, `--viz-*`, …), not raw
  values; both dark and light must look correct. Render numerics in the mono font with tabular figures.
- **The frontend is hand-formatted** — Prettier is installed but **not** in the gate; never run it over
  the existing tree.

See [`../CLAUDE.md`](../CLAUDE.md) and [`../docs/`](../docs) for the full conventions and architecture.
