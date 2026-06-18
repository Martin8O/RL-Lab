import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useAppStore } from '../store/useAppStore'
import { fetchCheckpoints, sendPlayAction, setFrameHandler, setPlayFrameHandler, setPreview, startPlay, stopPlay, watchPreview } from '../api/client'
import type { AgentSprite, BoardState, CheckpointMeta, GridLayout, PlayFrame, PreviewFrame, WorldEntity } from '../api/types'
import { keymapFor } from '../content/playKeymaps'
import { DEFAULT_AGENT, DEFAULT_GRIDS, isGridEnv } from '../content/gridMaps'
import { boardMetaFor } from '../content/boardGames'
import PlayControls from './PlayControls'
import WatchInfo from './WatchInfo'
import SkillMeter from './SkillMeter'
import ParamInfo from './ParamInfo'
import {
  CART_X_LIMIT, CART_X_SCALE, MC_START, mcCarTransform,
  PEND_CX, PEND_CY, ACRO_CX, ACRO_CY, ACRO_JOINT_Y,
  LL_START, llLanderTransform, llTerrainPaths, LL_DEFAULT_TERRAIN, llX, llY,
} from './envGeometry'
import {
  CartPoleStage, MountainCarStage, PendulumStage, AcrobotStage, LunarLanderStage, GridStage, SwarmStage,
  BoardStage, type SwarmLegendItem,
} from './EnvStages'

const MIN_SPEED = 1
const MAX_SPEED = 20
// When 2+ engines are held at once, the discrete env can still only fire ONE per step, so we
// rapidly alternate them (≈ each at half thrust). ~30 ms keeps pace with the sim step rate.
const MULTI_KEY_ALTERNATE_MS = 30

// Client-side rendered envs draw from the raw physics state the backend streams (client_render.py)
// instead of a JPEG — crisper + lighter on CPU. The SVG stages + their geometry live in ./EnvStages;
// `clientKind` (below) maps each env id to its renderer. Keep both in sync with the backend.

const EyeOn = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2" />
  </svg>
)
const EyeOff = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M3 3l18 18M10.6 10.6A3 3 0 0014 14M6.9 6.9C4.2 8.5 2 12 2 12s3.5 7 10 7c2 0 3.7-.5 5.2-1.3M9.9 5.2A10 10 0 0112 5c6.5 0 10 7 10 7a17.7 17.7 0 01-2.2 3"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)
const ExpandGlyph = (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M8 3H5a2 2 0 00-2 2v3M16 3h3a2 2 0 012 2v3M8 21H5a2 2 0 01-2-2v-3M16 21h3a2 2 0 002-2v-3"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)
const CompressGlyph = (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
    <path d="M3 8h3a2 2 0 002-2V3M21 8h-3a2 2 0 01-2-2V3M3 16h3a2 2 0 012 2v3M21 16h-3a2 2 0 00-2 2v3"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
)

export default function EnvPreview() {
  const { t } = useTranslation()

  const visual        = useAppStore((s) => s.visual)
  const speed         = useAppStore((s) => s.speed)
  const setVisual     = useAppStore((s) => s.setVisual)
  const setSpeed      = useAppStore((s) => s.setSpeed)
  const trainState    = useAppStore((s) => s.trainState)
  const backendStatus = useAppStore((s) => s.backendStatus)
  const selectedEnvId = useAppStore((s) => s.selectedEnvId)
  const envs          = useAppStore((s) => s.envs)
  const locale        = useAppStore((s) => s.locale)
  const playState     = useAppStore((s) => s.playState)
  const playMode      = useAppStore((s) => s.playMode)
  const playSpeed     = useAppStore((s) => s.playSpeed)
  const boardSide     = useAppStore((s) => s.boardSide)
  const boardStrength = useAppStore((s) => s.boardStrength)
  const playActiveCheckpoint = useAppStore((s) => s.playActiveCheckpoint)
  const playResult    = useAppStore((s) => s.playResult)
  const setPlayMode   = useAppStore((s) => s.setPlayMode)
  const applyPlayStatus = useAppStore((s) => s.applyPlayStatus)

  const selectedEnv = envs.find((e) => e.id === selectedEnvId)
  const selectedFamily = selectedEnv?.family  // a stable string — used as a keyboard-effect dep instead
                                              // of the whole `envs` array (which changes reference/size)
  const envName = selectedEnv?.display_name[locale] ?? t('envpreview.title')

  const canvasRef     = useRef<HTMLCanvasElement | null>(null)
  const stageRef      = useRef<HTMLDivElement | null>(null)  // the stage box — target for fullscreen
  const cartGroupRef  = useRef<SVGGElement | null>(null)   // CartPole: horizontal cart travel
  const poleGroupRef  = useRef<SVGGElement | null>(null)   // CartPole: pole angle
  const carRef        = useRef<SVGGElement | null>(null)   // MountainCar: car along the hill
  const pendRodRef    = useRef<SVGGElement | null>(null)   // Pendulum: rod angle
  const acroLink1Ref  = useRef<SVGGElement | null>(null)   // Acrobot: first link angle
  const acroLink2Ref  = useRef<SVGGElement | null>(null)   // Acrobot: second link angle
  const landerRef     = useRef<SVGGElement | null>(null)   // LunarLander: lander pose
  const llMainRef     = useRef<SVGGElement | null>(null)   // LunarLander: main-engine plume
  const llLeftRef     = useRef<SVGGElement | null>(null)   // LunarLander: left-engine puff
  const llRightRef    = useRef<SVGGElement | null>(null)   // LunarLander: right-engine puff
  const llGroundRef   = useRef<SVGPathElement | null>(null) // LunarLander: filled moon ground
  const llSurfaceRef  = useRef<SVGPathElement | null>(null) // LunarLander: moon surface stroke
  const swarmCanvasRef = useRef<HTMLCanvasElement | null>(null) // multi-agent (MPE): the swarm canvas
  const llTerrainObs  = useRef<readonly (readonly number[])[] | null>(null) // last moon surface (obs pts)
  // True once a frame has actually arrived this session — lets a finished session linger, but
  // falls back to the idle (centred) cart after a reload that reconciled a finished session.
  const [hasFrame, setHasFrame] = useState(false)
  // Fullscreen state of the stage box (Esc exits natively); tracked so the toggle's icon flips.
  const [isFullscreen, setIsFullscreen] = useState(false)

  const runLive      = trainState === 'running' || trainState === 'paused' || trainState === 'stopping'
  const playVisible  = playState !== 'idle'
  // Watch AI (G7b-2 follow-up): a multi-agent swarm can't be hand-played, but a *saved* model can be
  // watched playing itself — `watching` is true while a checkpoint streams the trained ecosystem.
  const [watching, setWatching] = useState(false)
  // "live" = frames are (or just were) flowing for this env.
  const live         = (visual && runLive) || playState === 'playing' || (playVisible && hasFrame) || (watching && hasFrame)
  // Which client-side renderer to use (else a server JPEG on the canvas).
  const clientKind: 'cartpole' | 'mountaincar' | 'pendulum' | 'acrobot' | 'lunarlander' | 'grid' | 'mpe' | 'board' | null =
    selectedEnvId === 'cartpole' ? 'cartpole'
    : selectedEnvId === 'mountaincar' || selectedEnvId === 'mountaincarcontinuous' ? 'mountaincar'
    : selectedEnvId === 'pendulum' ? 'pendulum'
    : selectedEnvId === 'acrobot' ? 'acrobot'
    : selectedEnvId === 'lunarlander' ? 'lunarlander'
    : selectedEnv?.family === 'petting_zoo' ? 'mpe'  // multi-agent swarm canvas (G7a)
    : selectedEnv?.family === 'board' ? 'board'      // OpenSpiel board renderer (G6a)
    : isGridEnv(selectedEnvId) ? 'grid'
    : null
  const clientRender = clientKind !== null
  // Swarm legend — colour-coded per env, matching drawSwarm (visual-labels rule): a competitive
  // predator–prey world (simple_tag) shows Predators (red) / Prey (blue) / Obstacles (grey); the
  // cooperative coverage world (simple_spread) shows Agents (blue dots) / Targets (open rings).
  const swarmLegend: SwarmLegendItem[] = selectedEnv?.competitive
    ? [
        { color: 'var(--danger)', label: t('species.predator') },
        { color: 'var(--accent)', label: t('species.prey') },
        { color: 'var(--border-strong)', label: t('species.obstacles') },
      ]
    : [
        { color: 'var(--accent)', label: t('envpreview.swarm_agents') },
        { color: 'var(--text-muted)', label: t('envpreview.swarm_targets'), ring: true },
      ]
  // Atari (image obs → server JPEG) gets a one-time, game-agnostic retro/CRT skin (scanlines + glow +
  // bezel) so all ~60 games read as deliberately retro instead of "tiny ugly pixels". A true vector
  // re-skin would need per-game object extraction, so this is the family-wide alternative (G4a follow-up).
  const retroSkin = !clientRender && selectedEnv?.family === 'atari'
  // Toy Text grids re-render declaratively (low-frequency moves), unlike the imperative physics
  // stages: the latest streamed board + agent position live in React state here, tagged with the env
  // it belongs to so a stale board never flashes after an env switch.
  const [gridFrame, setGridFrame] = useState<{ envId: string; grid: GridLayout; agent: number[] } | null>(null)
  // Board games (G6a) re-render declaratively from the streamed BoardState (low-frequency turn-based
  // moves), tagged with the env it belongs to so a stale board never flashes after an env switch.
  const [boardFrame, setBoardFrame] = useState<{ envId: string; board: BoardState } | null>(null)

  // Sync persisted toggle/speed to the backend whenever it comes online (UI is source of truth).
  useEffect(() => {
    if (backendStatus !== 'online') return
    const { visual: v, speed: s } = useAppStore.getState()
    void setPreview({ visual: v, speed: s }).catch(() => {})
  }, [backendStatus])

  // Watch AI (G7b-2 follow-up): the saved checkpoints for THIS multi-agent env, picked in the footer
  // bar. Refetched after a run reaches a terminal state (a fresh save may have appeared).
  const isMaEnv = clientKind === 'mpe'
  const [maCheckpoints, setMaCheckpoints] = useState<CheckpointMeta[]>([])
  const [maCkpt, setMaCkpt] = useState<string>('')
  useEffect(() => {
    // Only fetch for a multi-agent env; a stale list for a non-MA env is harmless (the footer that
    // shows it only renders for clientKind 'mpe'), so no synchronous clear is needed here.
    if (!isMaEnv || backendStatus !== 'online') return
    void fetchCheckpoints().then((list) => {
      const mine = list.filter((c) => c.env_id === selectedEnvId)
      setMaCheckpoints(mine)
      setMaCkpt((prev) => (mine.some((c) => c.id === prev) ? prev : (mine[0]?.id ?? '')))
    }).catch(() => {})
  }, [isMaEnv, selectedEnvId, backendStatus, trainState])

  // The watch lifecycle. A ref mirrors `watching` so the env-switch / unmount cleanup can stop a
  // lingering watch on the shared streamer without re-subscribing the effect to `watching`.
  const watchStartedRef = useRef(false)
  const startWatchAi = useCallback(async () => {
    if (!selectedEnvId || !maCkpt) return
    watchStartedRef.current = true
    try { await watchPreview(selectedEnvId, true, maCkpt); setWatching(true) }
    catch { watchStartedRef.current = false }
  }, [selectedEnvId, maCkpt])
  const stopWatchAi = useCallback(async () => {
    watchStartedRef.current = false
    setWatching(false)
    try { await watchPreview('', false) } catch { /* ignore */ }
  }, [])
  // A starting training run (or a play session) takes over the shared streamer → reset the watch flag.
  useEffect(() => {
    if (watchStartedRef.current && (runLive || playState === 'playing')) {
      watchStartedRef.current = false
      setWatching(false)
    }
  }, [runLive, playState])
  // Stop a lingering watch when leaving this env (or unmounting), so it doesn't keep streaming.
  useEffect(() => () => {
    if (watchStartedRef.current) {
      watchStartedRef.current = false
      void watchPreview('', false).catch(() => {})
    }
  }, [selectedEnvId])

  // One frame sink for both training-preview and play frames. A frame carries either client
  // -render state (CartPole → drive the SVG cart, no React render) or a JPEG (→ canvas).
  useEffect(() => {
    const img = new Image()
    img.onload = () => {
      const canvas = canvasRef.current
      if (!canvas) return
      if (canvas.width !== img.naturalWidth || canvas.height !== img.naturalHeight) {
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
      }
      canvas.getContext('2d')?.drawImage(img, 0, 0)
    }
    const drawCart = (x: number, theta: number) => {
      const cg = cartGroupRef.current
      const pg = poleGroupRef.current
      if (cg) {
        const tx = Math.max(-CART_X_SCALE, Math.min(CART_X_SCALE, (x / CART_X_LIMIT) * CART_X_SCALE))
        cg.setAttribute('transform', `translate(${tx.toFixed(1)} 0)`)
      }
      if (pg) pg.setAttribute('transform', `rotate(${((theta * 180) / Math.PI).toFixed(2)} 300 190)`)
    }
    const drawCar = (pos: number) => {  // MountainCar: slide + tilt the car along the hill
      carRef.current?.setAttribute('transform', mcCarTransform(pos))
    }
    const drawPendulum = (theta: number) => {  // rotate the rod around the pivot (θ = 0 is upright)
      pendRodRef.current?.setAttribute('transform', `rotate(${((theta * 180) / Math.PI).toFixed(1)} ${PEND_CX} ${PEND_CY})`)
    }
    const drawAcrobot = (th1: number, th2: number) => {  // link1 around the pivot, link2 around the joint
      acroLink1Ref.current?.setAttribute('transform', `rotate(${((th1 * 180) / Math.PI).toFixed(1)} ${ACRO_CX} ${ACRO_CY})`)
      acroLink2Ref.current?.setAttribute('transform', `rotate(${((th2 * 180) / Math.PI).toFixed(1)} ${ACRO_CX} ${ACRO_JOINT_Y})`)
    }
    const applyTerrain = (pts: readonly (readonly number[])[]) => {  // store + redraw the moon
      llTerrainObs.current = pts
      const { ground, surface } = llTerrainPaths(pts)
      llGroundRef.current?.setAttribute('d', ground)
      llSurfaceRef.current?.setAttribute('d', surface)
    }
    const terrainScreenYAt = (screenX: number): number | null => {  // surface height under the lander
      const pts = llTerrainObs.current
      if (!pts || pts.length < 2) return null
      for (let i = 0; i < pts.length - 1; i++) {
        const x0 = llX(pts[i][0]), x1 = llX(pts[i + 1][0])
        if (screenX >= x0 && screenX <= x1) {
          const f = (screenX - x0) / (x1 - x0)
          return llY(pts[i][1]) + f * (llY(pts[i + 1][1]) - llY(pts[i][1]))
        }
      }
      return llY(pts[screenX < llX(pts[0][0]) ? 0 : pts.length - 1][1])
    }
    // Local extremities of the lander art — the two outer leg tips + the hull's lower corners — used
    // to keep its lowest point resting ON the surface (so legs never sink into the ground, even on a
    // tilted crash or a sloped touchdown). obs-y alone is relative to the pad, so on a hard/tilted
    // contact the body can penetrate before the episode ends; this clamp fixes that visually.
    const LANDER_FOOT_PTS: ReadonlyArray<readonly [number, number]> = [[-20, 21], [20, 21], [-10, 12], [10, 12]]
    const drawLunarLander = (s: number[], action: number | null | undefined, terrain: number[][] | null | undefined) => {
      if (terrain && terrain.length) applyTerrain(terrain)
      // s = obs [x, y, vx, vy, angle, …]. Plume per firing engine: exhaust shoots opposite to the push
      // — action 1 (left engine) pushes left → plume on the RIGHT; action 3 → LEFT; 2 = main = down.
      const angleDeg = -(s[4] * 180) / Math.PI
      const lx = llX(s[0])
      let ly = llY(s[1])
      const ty = terrainScreenYAt(lx)
      if (ty != null) {  // raise the lander so its lowest (rotated) extremity sits on the surface
        const rad = (angleDeg * Math.PI) / 180, sin = Math.sin(rad), cos = Math.cos(rad)
        let lowest = -Infinity
        for (const [px, py] of LANDER_FOOT_PTS) { const sy = px * sin + py * cos; if (sy > lowest) lowest = sy }
        const overlap = ly + lowest - ty
        if (overlap > 0) ly -= overlap
      }
      landerRef.current?.setAttribute('transform', `translate(${lx.toFixed(1)} ${ly.toFixed(1)}) rotate(${angleDeg.toFixed(1)})`)
      llMainRef.current?.setAttribute('opacity', action === 2 ? '1' : '0')
      llLeftRef.current?.setAttribute('opacity', action === 3 ? '1' : '0')
      llRightRef.current?.setAttribute('opacity', action === 1 ? '1' : '0')
    }
    // Multi-agent "swarm" (MPE): draw the per-agent + landmark world positions onto the canvas.
    // A fixed world half-range keeps the view stable when one agent drifts far (it clamps to the
    // arena edge instead of zooming everything out). Colours read from the live theme tokens.
    const SWARM_WORLD = 1.8
    const drawSwarm = (agents: AgentSprite[], world: WorldEntity[]) => {
      const canvas = swarmCanvasRef.current
      const ctx = canvas?.getContext('2d')
      if (!canvas || !ctx) return
      const S = canvas.width
      const pad = 30
      const scale = (S / 2 - pad) / SWARM_WORLD
      const mid = S / 2
      const clamp = (v: number) => Math.max(-SWARM_WORLD, Math.min(SWARM_WORLD, v))
      const toX = (x: number) => mid + clamp(x) * scale
      const toY = (y: number) => mid - clamp(y) * scale  // world +y is up → canvas -y
      const css = getComputedStyle(canvas)
      const col = (name: string, fb: string) => css.getPropertyValue(name).trim() || fb
      ctx.clearRect(0, 0, S, S)
      // arena frame
      ctx.strokeStyle = col('--border-default', '#444'); ctx.lineWidth = 1.5
      ctx.strokeRect(pad / 2, pad / 2, S - pad, S - pad)
      // landmarks: targets as open rings, obstacles as solid discs
      const targetCol = col('--text-muted', '#888')
      for (const e of world) {
        const r = Math.max(6, e.size * scale)
        ctx.beginPath(); ctx.arc(toX(e.x), toY(e.y), r, 0, Math.PI * 2)
        if (e.kind === 'obstacle') {
          ctx.fillStyle = col('--surface-3', '#555'); ctx.fill()
          ctx.strokeStyle = col('--border-strong', '#777'); ctx.lineWidth = 1.5; ctx.stroke()
        } else {
          ctx.strokeStyle = targetCol; ctx.lineWidth = 2.5; ctx.stroke()
          ctx.beginPath(); ctx.arc(toX(e.x), toY(e.y), 2.5, 0, Math.PI * 2)
          ctx.fillStyle = targetCol; ctx.fill()
        }
      }
      // agents: filled discs (cooperative = accent, adversary = danger), thin edge for contrast
      const agentCol = col('--accent', '#3b82f6')
      const advCol = col('--danger', '#e2453c')
      const edge = col('--surface-1', '#0c0c0c')
      for (const a of agents) {
        const r = Math.max(7, a.size * scale)
        ctx.beginPath(); ctx.arc(toX(a.x), toY(a.y), r, 0, Math.PI * 2)
        ctx.fillStyle = a.role === 'adversary' ? advCol : agentCol; ctx.fill()
        ctx.strokeStyle = edge; ctx.lineWidth = 2; ctx.stroke()
      }
    }
    const onFrame = (frame: PreviewFrame | PlayFrame) => {
      // Multi-agent swarm (MPE): draw all agents + landmarks to the canvas from the streamed state.
      if (clientKind === 'mpe') {
        if (frame.agents) {
          setHasFrame(true)
          drawSwarm(frame.agents, frame.world ?? [])
        }
        return
      }
      // Board games: declarative re-render from the streamed BoardState (turn-based, low-frequency).
      if (clientKind === 'board') {
        if (frame.board) {
          const envId = selectedEnvId ?? ''
          setHasFrame(true)
          setBoardFrame({ envId, board: frame.board })
        }
        return
      }
      // Toy Text grids: declarative re-render from the streamed board + agent (low-frequency moves).
      if (clientKind === 'grid') {
        if (frame.state) {
          const envId = selectedEnvId ?? ''
          setHasFrame(true)
          setGridFrame((prev) => ({
            envId,
            grid: frame.grid
              ?? (prev && prev.envId === envId ? prev.grid : undefined)
              ?? DEFAULT_GRIDS[envId] ?? DEFAULT_GRIDS.frozenlake,
            agent: frame.state!,
          }))
        }
        return
      }
      if (frame.state && frame.state.length >= 2) {
        setHasFrame(true)
        const s = frame.state
        if (clientKind === 'mountaincar') drawCar(s[0])             // [position, velocity]
        else if (clientKind === 'pendulum') drawPendulum(s[0])      // [theta, theta_dot]
        else if (clientKind === 'acrobot') drawAcrobot(s[0], s[1])  // [theta1, theta2]
        else if (clientKind === 'lunarlander') drawLunarLander(s, frame.action, frame.terrain)  // obs + action + moon
        else drawCart(s[0], s[1])                                   // [x, theta]
      } else if (frame.image) {
        setHasFrame(true)
        img.src = `data:image/jpeg;base64,${frame.image}`
      }
    }
    setFrameHandler(onFrame)
    setPlayFrameHandler(onFrame)
    return () => { setFrameHandler(null); setPlayFrameHandler(null) }
  }, [clientKind, selectedEnvId])

  // When nothing is live, reset to a resting pose (drop the last streamed transform): the cart
  // re-centres, the MountainCar car parks in the valley (its typical start) — also on env switch.
  useEffect(() => {
    if (!live) {
      cartGroupRef.current?.removeAttribute('transform')
      poleGroupRef.current?.removeAttribute('transform')
      carRef.current?.setAttribute('transform', mcCarTransform(MC_START))
      pendRodRef.current?.setAttribute('transform', `rotate(180 ${PEND_CX} ${PEND_CY})`)  // hang down
      acroLink1Ref.current?.removeAttribute('transform')  // rest pose = links hang straight down
      acroLink2Ref.current?.removeAttribute('transform')
      // LunarLander: park near the top centre, engines off, moon flat (real terrain arrives with play)
      landerRef.current?.setAttribute('transform', llLanderTransform(LL_START.x, LL_START.y, 0))
      llMainRef.current?.setAttribute('opacity', '0')
      llLeftRef.current?.setAttribute('opacity', '0')
      llRightRef.current?.setAttribute('opacity', '0')
      llTerrainObs.current = LL_DEFAULT_TERRAIN
      const { ground, surface } = llTerrainPaths(LL_DEFAULT_TERRAIN)
      llGroundRef.current?.setAttribute('d', ground)
      llSurfaceRef.current?.setAttribute('d', surface)
      // Multi-agent swarm: wipe the canvas back to an empty arena when nothing is live.
      const sc = swarmCanvasRef.current
      sc?.getContext('2d')?.clearRect(0, 0, sc.width, sc.height)
    }
  }, [live, clientKind])

  // Keyboard control for human play, driven by the per-env keymap (content/playKeymaps.ts).
  // Latency-tolerant — the backend holds the last action between WS frames. We track every held
  // bound key; with one engine held we send it, with two-or-more we alternate them (LunarLander's
  // discrete space fires one engine per step, so this approximates pressing both at once). When
  // all keys are released we send the env's idle action (LunarLander 0 = no thrust) or, for an env
  // with no idle (CartPole), keep the last action.
  useEffect(() => {
    if (!(playState === 'playing' && playMode === 'human')) return
    // Board games (G6a) are played by clicking a cell (BoardStage), not the keyboard — and the
    // default keymap would map arrow keys to action 0/1, which the board loop would read as illegal
    // cell clicks. So no keyboard wiring for a board env.
    if (clientKind === 'board') return
    const keymap = keymapFor(selectedEnvId, selectedFamily)
    const isFormField = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName
      return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA'
    }

    // Multi-joint continuous envs (BipedalWalker, Box(4)): each key carries a per-joint VECTOR
    // contribution (e.g. ← = [-1,0,0,0]); held keys are SUMMED element-wise into one action vector
    // and sent (the backend reshapes + clips it). No alternation — the joints are independent, unlike
    // the discrete one-engine-per-step case below. Releasing all keys sends the scalar idle (0), which
    // the backend fills into a zero-torque vector.
    const vectorBinding = keymap.bindings.find((b) => Array.isArray(b.action))
    if (vectorBinding) {
      const dim = (vectorBinding.action as number[]).length
      const vlookup = new Map<string, number[]>()
      for (const b of keymap.bindings)
        if (Array.isArray(b.action)) for (const k of b.keys) vlookup.set(k, b.action)
      const heldV: string[] = []
      const applyV = () => {
        if (heldV.length === 0) {
          if (keymap.idleAction !== null) sendPlayAction(keymap.idleAction)
          return
        }
        const vec = new Array<number>(dim).fill(0)
        for (const k of heldV) {
          const a = vlookup.get(k)
          if (a) for (let i = 0; i < dim; i++) vec[i] += a[i]
        }
        sendPlayAction(vec)
      }
      const onDownV = (e: KeyboardEvent) => {
        if (isFormField(e) || !vlookup.has(e.key)) return
        e.preventDefault()
        if (!heldV.includes(e.key)) heldV.push(e.key)
        applyV()
      }
      const onUpV = (e: KeyboardEvent) => {
        if (!vlookup.has(e.key)) return
        const i = heldV.indexOf(e.key)
        if (i >= 0) heldV.splice(i, 1)
        applyV()
      }
      window.addEventListener('keydown', onDownV)
      window.addEventListener('keyup', onUpV)
      return () => {
        window.removeEventListener('keydown', onDownV)
        window.removeEventListener('keyup', onUpV)
      }
    }

    // Scalar keymaps (discrete + single-torque continuous): key → one action value.
    const lookup = new Map<string, number>()
    for (const b of keymap.bindings) for (const k of b.keys) lookup.set(k, b.action as number)

    // Turn-based grid-worlds (Toy Text): send exactly one action per key press — no auto-repeat,
    // no idle. The backend steps the agent one cell per received action.
    if (keymap.turnBased) {
      const onPress = (e: KeyboardEvent) => {
        if (isFormField(e) || !lookup.has(e.key)) return
        e.preventDefault()
        sendPlayAction(lookup.get(e.key)!)
      }
      window.addEventListener('keydown', onPress)
      return () => window.removeEventListener('keydown', onPress)
    }

    const held: string[] = []  // ordered; last entry = most recently pressed
    let timer: number | null = null
    const stopTimer = () => { if (timer !== null) { clearInterval(timer); timer = null } }

    // Distinct actions currently demanded by the held keys, in press order.
    const heldActions = (): number[] => {
      const seen = new Set<number>(); const out: number[] = []
      for (const k of held) { const a = lookup.get(k)!; if (!seen.has(a)) { seen.add(a); out.push(a) } }
      return out
    }
    // Push current intent to the backend, (re)starting the alternation timer only while 2+ engines
    // are held; the timer reads heldActions() live so it adapts as keys come and go.
    const apply = () => {
      const actions = heldActions()
      if (actions.length >= 2) {
        if (timer === null) {
          let i = 0
          timer = window.setInterval(() => {
            const a = heldActions()
            if (a.length < 2) { stopTimer(); return }
            sendPlayAction(a[i % a.length]); i++
          }, MULTI_KEY_ALTERNATE_MS)
        }
        return
      }
      stopTimer()
      if (actions.length === 1) sendPlayAction(actions[0])
      else if (keymap.idleAction !== null) sendPlayAction(keymap.idleAction)
    }

    const onDown = (e: KeyboardEvent) => {
      if (isFormField(e) || !lookup.has(e.key)) return
      e.preventDefault()
      if (!held.includes(e.key)) held.push(e.key)
      apply()
    }
    const onUp = (e: KeyboardEvent) => {
      if (!lookup.has(e.key)) return
      const i = held.indexOf(e.key)
      if (i >= 0) held.splice(i, 1)
      apply()
    }
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    return () => {
      stopTimer()
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
    }
  }, [playState, playMode, selectedEnvId, selectedFamily, clientKind])

  // Fullscreen the stage box on demand; Esc (or the toggle again) exits. Track the actual
  // fullscreen element so the icon stays correct even when the user exits via Esc.
  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === stageRef.current)
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
  }, [])

  function toggleFullscreen() {
    if (document.fullscreenElement) void document.exitFullscreen().catch(() => {})
    else void stageRef.current?.requestFullscreen().catch(() => {})
  }

  // Minimal human Play / Stop usable from inside fullscreen, where the PlayControls bar (below the
  // stage) is hidden — so you can fullscreen first, then start. Mirrors PlayControls.handlePlay('human'):
  // human play uses a fresh random seed each game and the env's idle action from the keymap.
  async function startFullscreenPlay() {
    if (!selectedEnv?.human_playable) return
    setPlayMode('human')
    try {
      applyPlayStatus(await startPlay({
        env_id: selectedEnv.id, mode: 'human', checkpoint_id: null, seed: null,
        speed: playSpeed, idle_action: keymapFor(selectedEnv.id, selectedEnv.family).idleAction,
      }))
    } catch { /* the PlayControls bar surfaces errors; the overlay stays quiet */ }
  }

  async function stopFullscreenPlay() {
    try { applyPlayStatus(await stopPlay()) } catch { /* status reconciles via WS */ }
  }

  function toggleVisual() {
    const next = !visual
    setVisual(next)
    void setPreview({ visual: next }).catch(() => {})
  }

  function changeSpeed(next: number) {
    setSpeed(next)
    void setPreview({ speed: next }).catch(() => {})
  }

  const hint = visual ? t('envpreview.idle_hint') : t('envpreview.visual_off_hint')

  // Grid board to draw: the live streamed one (only if it belongs to the selected env), else the
  // default board with the agent at its start — mirrors how the physics stages reset when idle.
  const gridData = live && gridFrame && gridFrame.envId === selectedEnvId ? gridFrame : null
  const gridBoard = gridData?.grid ?? DEFAULT_GRIDS[selectedEnvId ?? ''] ?? DEFAULT_GRIDS.frozenlake
  const gridAgent = (gridData?.agent?.length ? gridData.agent : DEFAULT_AGENT[selectedEnvId ?? '']) ?? [0, 0]

  // Board game (G6a) to draw: the live streamed ply if it belongs to this env, else the idle board.
  const boardMeta = boardMetaFor(selectedEnvId)
  const liveBoard = boardFrame && boardFrame.envId === selectedEnvId ? boardFrame.board : null
  const board = liveBoard ?? boardMeta?.idle ?? null
  const boardPlaying = playState === 'playing'
  // It's the human's turn when a human session is live and the board is waiting on the human's side.
  const boardHumanTurn =
    boardPlaying && playMode === 'human' && !!board && !board.is_terminal && board.current_player === boardSide
  // Orientation (G6e): in a human context (idle preview or a human game, NOT an AI watch or training)
  // the board is drawn from the human's side; in watch/training it keeps OpenSpiel's default view.
  const boardHumanSide = clientKind === 'board' && playMode === 'human' && !runLive ? boardSide : null
  // The opponent faced, for the W/D/L banner: the trained net (G6b — the active session carries a
  // checkpoint) reads "your trained AI"; otherwise the built-in MCTS at the chosen difficulty.
  const vsNet = clientKind === 'board' && !!playActiveCheckpoint
  const aiLevelLabel = vsNet ? t('board.level_net') : t(`play.diff_${boardStrength}`)
  // Whose-turn / result text for the board's status line + banner (honest W/D/L, no skill %).
  type BoardBanner = { text: string; kind: 'win' | 'draw' | 'loss'; mark?: { glyph: string; color: string } }
  let boardStatus = t('board.pick_side')
  let boardBanner: BoardBanner | null = null
  if (clientKind === 'board' && board) {
    const pieceFor = (player: number) => Object.values(boardMeta?.pieces ?? {}).find((p) => p.player === player)
    const markFor = (player: number) => pieceFor(player)?.glyph ?? `#${player + 1}`
    // Watch / training winner banner: a colour-coded mark + "<mark> wins". The colour is what tells the
    // two players apart in same-glyph games (Connect Four), where "● wins" alone is unreadable.
    const winnerBanner = (winner: number | null): BoardBanner =>
      winner === null
        ? { kind: 'draw', text: t('board.result_draw_watch') }
        : { kind: 'draw', text: t('board.player_wins'),
            mark: { glyph: markFor(winner), color: pieceFor(winner)?.color ?? 'var(--text-strong)' } }
    if (runLive) {
      // Training preview (self-play) takes precedence over any *stale* finished play session, so a
      // previous game's result never hangs on the board while the net is training.
      boardStatus = t('board.training')
      if (board.is_terminal) boardBanner = winnerBanner(board.winner)
    } else if (playState === 'playing' || playState === 'finished') {
      const ended = board.is_terminal || playState === 'finished'
      if (ended) {
        if (playMode === 'human') {
          const won = board.winner === boardSide
          const lost = board.winner !== null && board.winner !== boardSide
          const kind = won ? 'win' : lost ? 'loss' : 'draw'
          boardBanner = { kind, text: t(`board.result_${kind}`, { level: aiLevelLabel }) }
        } else {
          boardBanner = winnerBanner(board.winner)
        }
      } else if (boardPlaying) {
        // A forced pass (Othello, G6d): the human has no placement, only the pass move — prompt them
        // to use the Pass button instead of "click to play". For now a pass ⇒ no other legal move.
        const mustPass = boardHumanTurn && board.pass_action != null && board.legal_actions.length <= 1
        boardStatus = playMode === 'human'
          ? (mustPass ? t('board.must_pass') : boardHumanTurn ? t('board.your_move') : t('board.ai_thinking'))
          : t('board.watching', { mark: markFor(board.current_player) })
      }
    }
  }
  // Use the just-finished result's outcome (the authoritative label) when present — but never while a
  // training run is live (a stale play result must not override the training status).
  if (!runLive && clientKind === 'board' && playState === 'finished' && playResult?.outcome && playMode === 'human') {
    boardBanner = { kind: playResult.outcome, text: t(`board.result_${playResult.outcome}`, { level: aiLevelLabel }) }
  }
  const onBoardCellClick = (action: number) => { if (boardHumanTurn) sendPlayAction(action) }

  return (
    <section style={{
      flex: '0 0 55%', display: 'flex', flexDirection: 'column',
      borderRight: '2px solid var(--border-default)', overflow: 'hidden',
    }}>
      {/* Header: title + visual toggle + speed (moved up so the stage grows to align with chart) */}
      <div style={{
        minHeight: 'var(--panel-head-h)', padding: '0 var(--space-4)',
        borderBottom: '1px solid var(--border-default)',
        background: 'var(--surface-1)', flexShrink: 0,
        display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
      }}>
        <span style={{ fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', letterSpacing: 'var(--ls-eyebrow)', textTransform: 'uppercase', color: 'var(--text-faint)' }}>
          {t('envpreview.title')}
        </span>

        <div style={{ flex: 1 }} />

        {/* CPU/GPU training badge — centred in this header while a run is live; GPU = accent/green,
            CPU = muted. Reflects the ACTUAL training device, not the hw_requirement gate: only an
            image-obs CnnPolicy trains on CUDA (Atari); every vector/discrete env trains its small
            MlpPolicy on the CPU even on a GPU box (it is genuinely faster there — measured 3×). The
            info popup explains the gate-vs-device distinction so an idle GPU here doesn't read as a bug. */}
        {runLive && (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }}>
            <span style={{
              display: 'inline-flex', alignItems: 'center', height: 'var(--control-sm)', padding: '0 11px',
              borderRadius: 'var(--radius-pill)', fontSize: 'var(--fs-label)', whiteSpace: 'nowrap',
              fontWeight: 'var(--fw-medium)', letterSpacing: 'var(--ls-tight)',
              background: selectedEnv?.obs_type === 'image' ? 'var(--success-surface)' : 'var(--surface-2)',
              border: `1px solid ${selectedEnv?.obs_type === 'image' ? 'var(--success)' : 'var(--border-default)'}`,
              color: selectedEnv?.obs_type === 'image' ? 'var(--success)' : 'var(--text-muted)',
            }}>
              {selectedEnv?.obs_type === 'image' ? t('envpreview.badge_gpu') : t('envpreview.badge_cpu')}
            </span>
            <ParamInfo paramId="training_device" label={t('envpreview.device_info')} />
          </span>
        )}

        <div style={{ flex: 1 }} />

        {/* Visual on/off — quiet (surface, never a bright fill); the eye carries the accent */}
        <button
          onClick={toggleVisual}
          aria-pressed={visual}
          title={visual ? t('envpreview.visual_on_hint') : t('envpreview.visual_off_hint')}
          style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            height: 'var(--control-sm)', padding: '0 10px',
            borderRadius: 'var(--radius-md)', cursor: 'pointer',
            fontSize: 'var(--fs-label)', fontWeight: 'var(--fw-medium)',
            background: 'var(--surface-2)', border: '1px solid var(--border-default)',
            color: visual ? 'var(--text-strong)' : 'var(--text-muted)', transition: 'var(--t-colors)',
          }}
        >
          <span aria-hidden style={{ display: 'inline-flex', color: visual ? 'var(--accent)' : 'var(--text-faint)' }}>
            {visual ? EyeOn : EyeOff}
          </span>
          {t('envpreview.visual')}
        </button>

        {/* Speed — compact, fixed-width track so it reads as a setting, not a control bar */}
        <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7, opacity: visual ? 1 : 0.45 }}>
          <span style={{ fontSize: 'var(--fs-label)', color: 'var(--text-muted)' }}>{t('envpreview.speed')}</span>
          <input
            type="range"
            min={MIN_SPEED} max={MAX_SPEED} step={1}
            value={speed}
            disabled={!visual}
            onChange={(e) => changeSpeed(parseInt(e.target.value, 10))}
            style={{ width: 88, cursor: visual ? 'pointer' : 'default' }}
            aria-label={t('envpreview.speed')}
          />
          <span style={{
            fontSize: 'var(--fs-label)', fontFamily: 'var(--font-mono)',
            fontFeatureSettings: 'var(--ff-tabular)', color: 'var(--text-strong)',
            minWidth: 28, textAlign: 'right',
          }}>
            {speed}×
          </span>
        </div>
      </div>

      {/* Stage: the live SVG cart (CartPole), or a JPEG canvas for image-rendered envs.
          stageRef is the fullscreen target (Esc exits); it fills the screen with the same flex
          centring + background, so the game scales up and the skill-meter overlay rides along. */}
      <div ref={stageRef} className="env-stage" style={{
        position: 'relative',
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: 'var(--chart-plot-bg)', overflow: 'hidden', padding: 'var(--space-5)',
      }}>
        {clientRender ? (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
            {clientKind === 'cartpole' ? (
              <CartPoleStage envName={envName} cartRef={cartGroupRef} poleRef={poleGroupRef} />
            ) : clientKind === 'mountaincar' ? (
              <MountainCarStage envName={envName} carRef={carRef} />
            ) : clientKind === 'pendulum' ? (
              <PendulumStage envName={envName} rodRef={pendRodRef} />
            ) : clientKind === 'acrobot' ? (
              <AcrobotStage envName={envName} link1Ref={acroLink1Ref} link2Ref={acroLink2Ref} />
            ) : clientKind === 'grid' ? (
              <GridStage envName={envName} grid={gridBoard} agent={gridAgent} />
            ) : clientKind === 'board' && board && boardMeta ? (
              <BoardStage
                envName={envName} board={board} meta={boardMeta}
                humanTurn={boardHumanTurn} humanSide={boardHumanSide} onCellClick={onBoardCellClick}
                statusText={boardStatus} banner={boardBanner}
              />
            ) : clientKind === 'mpe' ? (
              <SwarmStage
                envName={envName} canvasRef={swarmCanvasRef}
                legend={swarmLegend}
              />
            ) : (
              <LunarLanderStage
                envName={envName} landerRef={landerRef}
                mainPlumeRef={llMainRef} leftPlumeRef={llLeftRef} rightPlumeRef={llRightRef}
                groundRef={llGroundRef} surfaceRef={llSurfaceRef}
              />
            )}
            {!live && clientKind !== 'board' && (
              <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', maxWidth: 360 }}>
                {hint}
              </span>
            )}
          </div>
        ) : live ? (
          // inline-block shrink-wraps the canvas so the CRT scanline overlay (Atari only) covers
          // exactly the picture, not the letterbox bars around it.
          <div style={{ position: 'relative', height: '100%', display: 'inline-block' }}>
            <canvas
              ref={canvasRef}
              style={{
                // Scale the frame UP to fill the stage instead of sitting at its tiny native size
                // (Atari is 160×210 — unreadably small otherwise). Driving the size by height with
                // width:auto lets the canvas keep its intrinsic aspect ratio (a plain width/height
                // 100% collapses to 0 in this centring flex row); maxWidth caps a very wide frame.
                // `pixelated` upscales the retro pixels crisply rather than blurring them.
                display: 'block', height: '100%', width: 'auto', maxWidth: '100%',
                objectFit: 'contain', imageRendering: 'pixelated',
                borderRadius: 'var(--radius-md)',
                boxShadow: retroSkin ? '0 0 0 3px var(--surface-3), var(--shadow-md)' : 'var(--shadow-md)',
              }}
            />
            {retroSkin && (
              // CRT skin: faint scanlines + an inward vignette/glow. pointer-events:none so it never
              // eats keyboard focus or clicks; opacity kept low so the game stays clearly readable.
              <div aria-hidden style={{
                position: 'absolute', inset: 0, pointerEvents: 'none', borderRadius: 'var(--radius-md)',
                backgroundImage:
                  'repeating-linear-gradient(to bottom, rgba(0,0,0,0.16) 0px, rgba(0,0,0,0.16) 1px, transparent 1px, transparent 3px)',
                boxShadow: 'inset 0 0 36px rgba(0,0,0,0.5)',
                mixBlendMode: 'multiply',
              }} />
            )}
          </div>
        ) : (
          <span style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', textAlign: 'center', padding: '0 16px' }}>
            {hint}
          </span>
        )}
        {/* Play / Stop overlay — only in fullscreen, where the PlayControls bar is off-screen. Lets
            you go fullscreen first, then start, and stop without leaving fullscreen. */}
        {isFullscreen && selectedEnv?.human_playable && (
          <button
            onClick={playState === 'playing' ? stopFullscreenPlay : startFullscreenPlay}
            style={{
              position: 'absolute', left: 16, top: '50%', transform: 'translateY(-50%)', zIndex: 3,
              display: 'inline-flex', alignItems: 'center', gap: 7,
              height: 'var(--control-md)', padding: '0 16px', borderRadius: 'var(--radius-md)',
              cursor: 'pointer', fontSize: 'var(--fs-sm)', fontWeight: 'var(--fw-semibold)',
              border: '1px solid transparent', boxShadow: 'var(--shadow-md)', transition: 'var(--t-colors)',
              background: playState === 'playing' ? 'var(--danger-surface)' : 'var(--accent)',
              color: playState === 'playing' ? 'var(--danger)' : 'var(--accent-contrast)',
            }}
          >
            {playState === 'playing'
              ? <><span aria-hidden>■</span> {t('play.stop')}</>
              : <><span aria-hidden>▶</span> {t('play.play')}</>}
          </button>
        )}

        {/* Fullscreen toggle (top-right) — blow the game up to the whole screen; Esc returns. */}
        <button
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? t('envpreview.fullscreen_exit') : t('envpreview.fullscreen')}
          title={isFullscreen ? t('envpreview.fullscreen_exit') : t('envpreview.fullscreen')}
          style={{
            position: 'absolute', top: 12, right: 12, zIndex: 2,
            display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
            width: 'var(--control-sm)', height: 'var(--control-sm)', padding: 0, cursor: 'pointer',
            background: 'var(--surface-3)', borderWidth: 1, borderStyle: 'solid', borderColor: 'var(--border-default)',
            borderRadius: 'var(--radius-md)', color: 'var(--text-muted)',
            boxShadow: 'var(--shadow-sm)', transition: 'var(--t-colors)',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent-border)' }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.borderColor = 'var(--border-default)' }}
        >
          {isFullscreen ? CompressGlyph : ExpandGlyph}
        </button>

        {playState === 'playing' && playMode === 'human' && clientKind !== 'board' && (
          <div style={{
            position: 'absolute', top: 12, left: 12,
            padding: '4px 10px', borderRadius: 'var(--radius-pill)',
            background: 'var(--surface-3)', border: '1px solid var(--border-default)',
            color: 'var(--text-strong)', boxShadow: 'var(--shadow-sm)',
            fontSize: 'var(--fs-meta)', fontWeight: 'var(--fw-semibold)', pointerEvents: 'none',
          }}>
            ⌨ {t('play.playing_hint')}
          </div>
        )}

        {/* Skill meter floats as an overlay at the bottom of the stage (no footer row) — shown only
            while a play session is the live context, so it doesn't steal space from the panels below. */}
        <SkillMeter slot="play" overlay />
      </div>

      {/* Play vs AI (E2): controls. The skill meter is the single overlay inside the stage above.
          Multi-agent envs can't be driven by a single human (G7a), so the play bar is replaced by a
          watch-only "What am I watching?" affordance — keeping the env explanation the play bar used
          to carry (its How-to-play guide) instead of leaving the swarm unlabelled. */}
      {clientKind !== 'mpe' ? <PlayControls /> : (
        <WatchInfo
          checkpoints={maCheckpoints}
          selected={maCkpt}
          onSelect={setMaCkpt}
          watching={watching}
          onWatch={() => void startWatchAi()}
          onStop={() => void stopWatchAi()}
        />
      )}
    </section>
  )
}
