import { beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import { DASHBOARD_STEPS, stepInMode } from '../content/tourSteps'
import Tour from './Tour'

// Every anchor the tour can spotlight — rendered so the DOM-presence filter passes; mode filtering
// then decides which show. (In jsdom there's no layout, so we stub non-zero rects below.)
const ANCHORS = ['env', 'algo', 'params', 'length', 'run', 'preview', 'chart', 'skill', 'play', 'records', 'datalab', 'mode', 'langtheme']
const DL_ANCHORS = ['dl-sources', 'dl-pivot', 'dl-axis', 'dl-controls', 'dl-chart', 'dl-table', 'dl-rliable', 'dl-export']

function Harness() {
  return (
    <>
      {[...ANCHORS, ...DL_ANCHORS].map((a) => (
        <div key={a} data-tour={a}>{a}</div>
      ))}
      <Tour />
    </>
  )
}

beforeEach(() => {
  // jsdom has no layout: give data-tour anchors a real size so isPresent() keeps them, and stub
  // scrollIntoView (unimplemented in jsdom).
  Element.prototype.scrollIntoView = vi.fn()
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Element.prototype.getBoundingClientRect = function (this: Element): any {
    const anchored = this.getAttribute && this.getAttribute('data-tour')
    return anchored
      ? { x: 10, y: 10, left: 10, top: 10, right: 110, bottom: 60, width: 100, height: 50, toJSON() {} }
      : { x: 0, y: 0, left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0, toJSON() {} }
  }
  // A picked mode + both tours already seen ⇒ no auto-open; tests drive startTour() explicitly.
  useAppStore.setState({
    envs: [cartpoleEnv], selectedEnvId: 'cartpole',
    modeChosen: true, tourSeen: true, datalabTourSeen: true, analysisOpen: false,
    tourOpen: false, tourFlow: 'dashboard',
  })
})

describe('tour step content', () => {
  it('gates params to Advanced and the training-length card to Simple', () => {
    const params = DASHBOARD_STEPS.find((s) => s.id === 'params')!
    const length = DASHBOARD_STEPS.find((s) => s.id === 'length')!
    expect(stepInMode(params, 'advanced')).toBe(true)
    expect(stepInMode(params, 'simple')).toBe(false)
    expect(stepInMode(length, 'simple')).toBe(true)
    expect(stepInMode(length, 'advanced')).toBe(false)
  })
})

describe('<Tour />', () => {
  it('does not render until opened', () => {
    render(<Harness />)
    expect(screen.queryByRole('dialog')).toBeNull()
  })

  it('shows more steps in Advanced than Simple (params + Data Lab are Advanced-only)', () => {
    useAppStore.setState({ mode: 'advanced' })
    render(<Harness />)
    act(() => useAppStore.getState().startTour())
    // Advanced keeps every anchor + welcome/finish minus the Simple-only length card.
    expect(screen.getByText('Step 1 of 14')).toBeInTheDocument()

    act(() => useAppStore.getState().closeTour())
    useAppStore.setState({ mode: 'simple', tourOpen: false })
    act(() => useAppStore.getState().startTour())
    // Simple drops params + Data Lab (Advanced-only) but adds the length card → two fewer.
    expect(screen.getByText('Step 1 of 13')).toBeInTheDocument()
  })

  it('navigates Next/Back and finishes, marking the tour seen', () => {
    useAppStore.setState({ mode: 'simple', tourSeen: false })
    render(<Harness />)
    act(() => useAppStore.getState().startTour())

    // Step 1 = welcome (centered, no Back button).
    expect(screen.getByText('Step 1 of 13')).toBeInTheDocument()
    expect(screen.queryByText('Back')).toBeNull()

    fireEvent.click(screen.getByText('Next'))
    expect(screen.getByText('Step 2 of 13')).toBeInTheDocument()
    expect(screen.getByText('Back')).toBeInTheDocument()

    fireEvent.click(screen.getByText('Back'))
    expect(screen.getByText('Step 1 of 13')).toBeInTheDocument()
  })

  it('Skip closes the tour and marks it seen so it never auto-opens again', () => {
    useAppStore.setState({ mode: 'simple', tourSeen: false })
    render(<Harness />)
    act(() => useAppStore.getState().startTour())
    expect(screen.getByRole('dialog')).toBeInTheDocument()

    fireEvent.click(screen.getByText('Skip tour'))
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(useAppStore.getState().tourOpen).toBe(false)
    expect(useAppStore.getState().tourSeen).toBe(true)
  })

  it('auto-opens once after a mode is picked when not yet seen', () => {
    useAppStore.setState({ mode: 'simple', modeChosen: true, tourSeen: false, tourOpen: false })
    render(<Harness />)
    // The auto-open effect fires on mount (envs loaded + modeChosen + !seen).
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByText('Step 1 of 13')).toBeInTheDocument()
  })
})

describe('<Tour /> — Data Lab flow', () => {
  it('runs the separate Data Lab step list when started with that flow', () => {
    useAppStore.setState({ mode: 'advanced' })
    render(<Harness />)
    act(() => useAppStore.getState().startTour('datalab'))
    // 8 Data Lab anchors + welcome/finish = 10; none are mode-gated.
    expect(screen.getByText('Step 1 of 10')).toBeInTheDocument()
    expect(screen.getByText('Welcome to the Data Lab 🔬')).toBeInTheDocument()
  })

  it('auto-opens once when the Data Lab is first opened', () => {
    useAppStore.setState({ mode: 'advanced', modeChosen: true, tourSeen: true, datalabTourSeen: false, analysisOpen: true, tourOpen: false })
    render(<Harness />)
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    expect(screen.getByText('Step 1 of 10')).toBeInTheDocument()
  })

  it('marks only the Data Lab tour seen on close (dashboard tour still pending)', () => {
    useAppStore.setState({ mode: 'advanced', tourSeen: false, datalabTourSeen: false })
    render(<Harness />)
    act(() => useAppStore.getState().startTour('datalab'))
    fireEvent.click(screen.getByText('Skip tour'))
    expect(useAppStore.getState().datalabTourSeen).toBe(true)
    expect(useAppStore.getState().tourSeen).toBe(false)
  })
})
