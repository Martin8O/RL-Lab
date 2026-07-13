import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import Sidebar from './Sidebar'

describe('<Sidebar />', () => {
  beforeEach(() => {
    // These cover the Advanced full-control UI (algo picker, hyperparameters, Run/sweep). The store
    // now defaults to Simple (#2b), so pin Advanced here; the Simple layout has its own tests below.
    useAppStore.setState({ envs: [cartpoleEnv], selectedEnvId: 'cartpole', mode: 'advanced', modeChosen: true })
  })

  it('renders the panel title and the algorithm options (a dropdown, env-gated)', () => {
    render(<Sidebar />)
    expect(screen.getByText('Parameters')).toBeInTheDocument()
    // The picker is a custom LabSelect (combobox trigger + portal listbox) — options render on open.
    const combo = screen.getByRole('combobox', { name: 'Algorithm' })
    fireEvent.click(combo)
    // cartpole's ★ recommended algo is ppo, so its option carries the marker.
    expect(screen.getByRole('option', { name: /★ PPO/ })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Neuroevolution' })).toBeInTheDocument()
  })

  it('marks the recommended algorithm and confirms when it is the selected one', () => {
    render(<Sidebar />)
    // ppo is both recommended and selected → the confirmation caption shows (no switch button).
    expect(screen.getByText('Recommended for this game')).toBeInTheDocument()
  })

  it('offers a one-click switch to the recommended algo when another is selected', () => {
    render(<Sidebar />)
    fireEvent.click(screen.getByRole('combobox', { name: 'Algorithm' }))
    fireEvent.click(screen.getByRole('option', { name: 'Neuroevolution' }))
    expect(useAppStore.getState().algo).toBe('neuroevolution')
    const recBtn = screen.getByRole('button', { name: 'Recommended: PPO' })
    fireEvent.click(recBtn)
    expect(useAppStore.getState().algo).toBe('ppo')
  })

  it('shows the total distinct-algorithm count beside the picker', () => {
    render(<Sidebar />)
    // the fixture has one env supporting ppo + neuroevolution → 2 distinct algorithms.
    expect(screen.getByLabelText('Total learning algorithms available in the app').textContent).toBe('2')
  })

  it('shows PPO hyperparameters by default and switches to evolution settings', () => {
    render(<Sidebar />)
    expect(screen.getByText('PPO Hyperparameters')).toBeInTheDocument()
    expect(screen.getByLabelText('Learning Rate')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('combobox', { name: 'Algorithm' }))
    fireEvent.click(screen.getByRole('option', { name: 'Neuroevolution' }))

    expect(screen.getByText('Evolution Settings')).toBeInTheDocument()
    expect(screen.getByLabelText('Population Size')).toBeInTheDocument()
    expect(useAppStore.getState().algo).toBe('neuroevolution')
  })

  it('enables Run when an environment is selected, disables it otherwise', () => {
    const { rerender } = render(<Sidebar />)
    expect(screen.getByRole('button', { name: 'Run' })).toBeEnabled()

    useAppStore.setState({ envs: [], selectedEnvId: null })
    rerender(<Sidebar />)
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled()
  })

  // #2b: Simple mode replaces the picker with a read-only algo badge, hides the hyperparameters, and
  // offers a Quick-start button + friendly training-length instead of the raw step ladder.
  it('Simple mode: forces the recommended algo as a badge, hides the picker + hyperparameters', () => {
    useAppStore.setState({ mode: 'simple', modeChosen: true, algo: 'ppo' })
    render(<Sidebar />)
    // No algorithm dropdown, and no learning-rate slider.
    expect(screen.queryByRole('combobox', { name: 'Algorithm' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Learning Rate')).not.toBeInTheDocument()
    // Friendly training-length + Quick-start instead.
    expect(screen.getByText('How long to train')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Quick-start' })).toBeEnabled()
  })
})
