// #1 — the guided-tour step lists. Each step spotlights one real UI anchor (a `[data-tour="…"]`
// element) and shows a copy card; the first + last steps of each flow are centered (no target)
// welcome/finish cards. There are two flows: the dashboard tour and a separate, detailed Data Lab
// tour. The tour is **mode-aware**: a step tagged with `modes` only shows in those audience modes,
// and at runtime any step whose anchor isn't currently rendered (or has zero size) is skipped — so
// Simple mode (which hides the hyperparameters, the seed sweep, the Data Lab, …) automatically gets
// a shorter tour without a second hand-maintained list.
//
// Copy is data-driven i18n: each step id resolves `tour.step_<id>_title` / `tour.step_<id>_body`
// (both locales), so adding/renaming a step is a content edit here + two locale entries.

import type { AudienceMode } from '../store/useAppStore'

export type TourFlow = 'dashboard' | 'datalab'

export interface TourStep {
  /** Stable id — also the i18n key stem (`tour.step_<id>_title` / `_body`). */
  id: string
  /** CSS selector for the anchor to spotlight; omit for a centered card (welcome / finish). */
  target?: string
  /** Preferred card side relative to the anchor; the overlay flips/clamps it to fit the viewport. */
  placement?: 'top' | 'bottom' | 'left' | 'right'
  /** Restrict the step to these audience modes; omit ⇒ shown in both. */
  modes?: AudienceMode[]
}

// ── Dashboard flow ────────────────────────────────────────────────────────────
// Ordered along the app's natural spine: welcome → pick a game → method → tune → start → watch →
// progress → skill → play → records → (Data Lab) → the mode switch → language/theme → finish.
export const DASHBOARD_STEPS: TourStep[] = [
  { id: 'welcome' },
  { id: 'env',       target: '[data-tour="env"]',       placement: 'right' },
  { id: 'algo',      target: '[data-tour="algo"]',      placement: 'right' },
  { id: 'params',    target: '[data-tour="params"]',    placement: 'right', modes: ['advanced'] },
  { id: 'length',    target: '[data-tour="length"]',    placement: 'right', modes: ['simple'] },
  { id: 'run',       target: '[data-tour="run"]',       placement: 'right' },
  { id: 'preview',   target: '[data-tour="preview"]',   placement: 'top' },
  { id: 'chart',     target: '[data-tour="chart"]',     placement: 'top' },
  { id: 'skill',     target: '[data-tour="skill"]',     placement: 'top' },
  { id: 'play',      target: '[data-tour="play"]',      placement: 'top' },
  { id: 'records',   target: '[data-tour="records"]',   placement: 'top' },
  { id: 'datalab',   target: '[data-tour="datalab"]',   placement: 'bottom', modes: ['advanced'] },
  { id: 'mode',      target: '[data-tour="mode"]',      placement: 'bottom' },
  { id: 'langtheme', target: '[data-tour="langtheme"]', placement: 'bottom' },
  { id: 'finish' },
]

// ── Data Lab flow ───────────────────────────────────────────────────────────
// A separate, detailed tour of the analysis surface (Advanced-only — the Data Lab is hidden in
// Simple). Runs only while the Data Lab is open (its anchors live inside AnalysisSurface).
export const DATALAB_STEPS: TourStep[] = [
  { id: 'dl_welcome' },
  { id: 'dl_sources',  target: '[data-tour="dl-sources"]',  placement: 'right' },
  { id: 'dl_pivot',    target: '[data-tour="dl-pivot"]',    placement: 'bottom' },
  { id: 'dl_axis',     target: '[data-tour="dl-axis"]',     placement: 'bottom' },
  { id: 'dl_controls', target: '[data-tour="dl-controls"]', placement: 'bottom' },
  { id: 'dl_chart',    target: '[data-tour="dl-chart"]',    placement: 'top' },
  { id: 'dl_table',    target: '[data-tour="dl-table"]',    placement: 'top' },
  { id: 'dl_rliable',  target: '[data-tour="dl-rliable"]',  placement: 'left' },
  { id: 'dl_export',   target: '[data-tour="dl-export"]',   placement: 'left' },
  { id: 'dl_finish' },
]

export function stepsForFlow(flow: TourFlow): TourStep[] {
  return flow === 'datalab' ? DATALAB_STEPS : DASHBOARD_STEPS
}

/** Whether a step is allowed in the given audience mode (the DOM-presence check happens later). */
export function stepInMode(step: TourStep, mode: AudienceMode): boolean {
  return !step.modes || step.modes.includes(mode)
}
