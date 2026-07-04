# API reference

> Living document (Phase F4). The backend is **FastAPI**; every REST body and WebSocket frame is a
> **pydantic model** in `backend/app/schemas/`, mirrored as a TypeScript type in
> `frontend/src/api/types.ts` — the contract is defined once. See [`architecture.md`](architecture.md).

**Interactive reference:** with the backend running (`.\tasks.ps1 dev-backend`), FastAPI serves
auto-generated OpenAPI docs at <http://127.0.0.1:8000/docs> (Swagger UI) and `/openapi.json`. This page
is the human-readable companion to it.

Base URL in dev: `http://127.0.0.1:8000`. CORS allows the Vite origin (`CORS_ORIGINS`, default
`http://localhost:5173`).

## REST endpoints

### Health
| Method | Path | Returns |
|---|---|---|
| GET | `/api/health` | `{status, version}` |
| GET | `/api/system` | `SystemInfo` — `{gpu_available}` (cached `torch.cuda` probe) |

`gpu_available` gates **GPU-only training** together with two `EnvSpec` fields (ADR-043). `hw_requirement`
(`"cpu"`/`"gpu"`) declares whether a CUDA device is needed; `train_implemented` declares whether the env's
trainer exists — it is now **`true` for every shipped env** (the `CnnPolicy` image trainers for Atari +
CarRacing landed, G4b / G3c-train). The two are independent: the UI disables Run — and `POST
/api/train/start` rejects with `400` — when an env needs a GPU that's absent. On a **CUDA machine** every
family trains, including the GPU heavies (BipedalWalker, the six MuJoCo robots, Atari, and CarRacing). Human
play of every gated env stays available.

### Environments
| Method | Path | Returns |
|---|---|---|
| GET | `/api/envs` | `EnvSpec[]` — the full registry |
| GET | `/api/envs/{env_id}` | `EnvSpec` |

### Training (`/api/train`)
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/train/start` | `TrainConfig` | `TrainStatus` |
| POST | `/api/train/sweep` | `SweepRequest` (config + `seeds[]` / `seed_count`) | `TrainStatus` (queues N seeds, run back-to-back under one `experiment_id`, ADR-085) |
| POST | `/api/train/pause` | — | `TrainStatus` |
| POST | `/api/train/resume` | — | `TrainStatus` |
| POST | `/api/train/stop` | — | `TrainStatus` |
| GET | `/api/train/status` | — | `TrainStatus` |

One run is active at a time. `TrainConfig.algo` selects one of **seven** algorithms — PPO (`hyperparams`),
off-policy **SAC / TD3 / DQN** (`sac` / `td3` / `dqn`), **neuroevolution** (`evolution`), tabular
**Q-learning** (`q_learning` — α/γ/ε-start/ε-end/ε-decay/episodes; discrete-obs Toy Text), or **AlphaZero**
(`alphazero`; board games) — each gated per-env by `supported_algos`; `seed` + the full config are echoed
back in `TrainStatus` for reproducibility. `TrainStatus` also retains `last_metrics` (PPO/SAC/TD3/DQN),
`last_evolution`, `last_q_learning` + `last_qtable`, and `last_ma_metrics` (multi-agent self-play) so a
reconnecting client repopulates panels at once.

### Preview (`/api/preview`)
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/preview` | — | `PreviewState` |
| POST | `/api/preview` | `PreviewConfig` (partial) | `PreviewState` |
| POST | `/api/preview/watch` | `PreviewWatch` (`env_id`, `on`, `checkpoint_id?`) | `PreviewState` |

Toggles the live visual (`visual`) and playback `speed` (clamped to `[1, 20]`). Decoupled from training.
`/preview/watch` drives a training-free preview of a multi-agent env: with a `checkpoint_id` it is
**Watch AI** (the saved `species.zip` plays itself — both species' brains); without one, a random rollout.

### Play vs AI (`/api/play`)
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/play/start` | `PlayConfig` | `PlayStatus` |
| POST | `/api/play/stop` | — | `PlayStatus` |
| POST | `/api/play/speed` | `PlaySpeedRequest` | `PlayStatus` (re-paces a **live** session) |
| GET | `/api/play/status` | — | `PlayStatus` |

`mode="human"` takes keyboard actions over WS; `mode="ai"` needs a `checkpoint_id`. Human play passes a
random `seed` (varied scene each game); AI play keeps the configured seed (reproducible demo).

### Checkpoints (`/api/checkpoints`)
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/checkpoints` | — | `CheckpointMeta[]` |
| POST | `/api/checkpoints` | save request | `CheckpointMeta` (saves the current run's latest snapshot) |
| POST | `/api/checkpoints/{id}/load` | — | `TrainStatus` (resume from a slot) |
| GET | `/api/checkpoints/{id}/export` | — | a `.zip` of the slot |
| DELETE | `/api/checkpoints/{id}` | — | `204` |

### Run history (`/api/runs`)
| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/runs` | — | `RunMeta[]` — **every** auto-archived finished/stopped run (ADR-088; low-skill filtering moved to the Data Lab picker, not a save-time gate) |
| GET | `/api/runs/{id}` | — | `RunDetail` (meta + config + metric series) |
| PATCH | `/api/runs/{id}` | `RunMetaPatch` | `RunMeta` — edit note / label / experiment / `excluded` (a `meta.json` sidecar; config/metrics untouched, ADR-091) |
| POST | `/api/runs/group` | `GroupRequest` | `RunMeta[]` — tag runs into a named experiment |
| POST | `/api/runs/delete` | `BulkDeleteRequest` | `BulkDeleteResult` — bulk delete |
| DELETE | `/api/runs/{id}` | — | `204` |

### Analysis · Data Lab (`/api/analysis`)
Server-side experiment analysis over the **full run archive** on disk (the frontend store is a capped ring
buffer, never the export source). Phase X / ADR-082–092.

| Method | Path | Query | Returns |
|---|---|---|---|
| GET | `/api/analysis/summary` | `run_ids` | `RunSummary[]` — per-run scalars (final %, AUC, steps-to-solve, peak %, collapse %) |
| GET | `/api/analysis/experiments` | — | `ExperimentInfo[]` — runs grouped into experiments |
| GET | `/api/analysis/aggregate` | `experiment_id` | `AggregateResponse` — rebin-then-average mean ± 95 % CI band across seeds |
| GET | `/api/analysis/rliable` | `run_ids` / `experiment_id` | `RliableResult` — IQM/mean/median/optimality-gap + bootstrap CIs, performance profiles, probability-of-improvement |
| GET | `/api/analysis/export.{csv,xlsx,repro,tex,svg,zip,npz}` | `run_ids`, `pivot`, `locale` | a downloadable dataset — tidy CSV · Excel (native charts) · repro card (config-hash + BibTeX) · LaTeX booktabs · SVG figure · TensorBoard log dir · `npz` runs×tasks matrix |

### Skill & leaderboards
| Method | Path | Returns |
|---|---|---|
| GET | `/api/skill/{env_id}` | `EnvSkill` — the five skill bands derived from `[min_score, solved_score]` |
| GET | `/api/highscores` · `/api/highscores/{env_id}` | training high scores (`HighScore`) |
| GET | `/api/playscores/{env_id}` | `PlayScores` — named human + AI boards (top-5) |
| POST | `/api/playscores/{env_id}` | `PlayScoreResult` — record a finished play score |

## WebSocket — `/ws`

One socket carries both directions. Outbound frames from the server are a **tagged union on `type`**;
the only inbound frame is the human play action.

### Inbound (client → server)
```jsonc
{ "type": "action", "action": 1 }            // discrete: action index
{ "type": "action", "action": [0.5] }        // box: continuous command vector
```
Latency-tolerant: the play session holds the latest action and reuses it until the next arrives. Any
other text is echoed back as `{echo: <text>}` (the original A3 contract).

### Outbound (server → client)
| `type` | Schema | When |
|---|---|---|
| `metrics` | `TrainingMetrics` | once per PPO rollout |
| `progress` | `TrainingProgress` | ~1 Hz, from a decoupled ticker thread |
| `hwstats` | `HwStatsFrame` | ~1 Hz CPU/GPU telemetry from the manager, for *any* active run (every algorithm) — feeds the HW panel; GPU fields `None` without `pynvml` (G4b) |
| `evolution` | `EvolutionMetrics` | once per neuroevolution generation (Top-5 + mutation histogram) |
| `q_learning` | `QLearningMetrics` | periodic tabular-Q report (x = `episode`; `ep_rew_mean` learning curve) |
| `qtable` | `QTableFrame` | the live `[n_states × n_actions]` Q-table heatmap snapshot (decoupled, unlogged) |
| `ma_metrics` | `MultiAgentMetrics` | competitive self-play (simple_tag, G7b-2): **both** species at once (`species[]` + `learning_role` + `round`/`total_rounds`) → the two-line ecosystem chart; `ep_rew_mean` mirrors the predator headline for high-score/archive |
| `status` | `TrainStatus` | training lifecycle changes |
| `frame` | `FrameMessage` | training-preview frame (server JPEG **or** client-render `state`) |
| `preview` | `PreviewState` | preview settings changed |
| `play_status` | `PlayStatus` | play lifecycle changes |
| `play_frame` | `PlayFrame` | one play episode frame (JPEG **or** `state` + optional `action`/`terrain`) |
| `play_result` | `PlayResult` | the episode ended (final score + `SkillRating`) |

### Frame payloads (rendering)
A `frame` / `play_frame` carries **either** a server-rendered image (`image` base64 JPEG + `width`/
`height`) **or** client-render data (`state`) — never both (ADR-018/022). Client-render frames may also
carry `action` (the just-applied action, e.g. to draw a firing thruster), `terrain` (per-episode scene
geometry the obs can't provide, e.g. LunarLander's random moon in obs-normalized coordinates), and `grid`
(a `GridLayout` — the static Toy Text board: `kind` + `rows`/`cols` + a row-major `cells` tag list).

### Reconnect
The `status` frame fires only on lifecycle changes, so a client that connects mid-run reconciles by
fetching `GET /api/train/status` (and `/api/play/status`) on every (re)connect (ADR-013).
