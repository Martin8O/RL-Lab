import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import Sidebar from './Sidebar'

describe('<Sidebar />', () => {
  beforeEach(() => {
    useAppStore.setState({ envs: [cartpoleEnv], selectedEnvId: 'cartpole' })
  })

  it('renders the panel title and the algorithm options (a dropdown, env-gated)', () => {
    render(<Sidebar />)
    expect(screen.getByText('Parameters')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Algorithm' })).toBeInTheDocument()
    // cartpole's ★ recommended algo is ppo, so its option carries the marker.
    expect(screen.getByRole('option', { name: '★ PPO' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'Neuroevolution' })).toBeInTheDocument()
  })

  it('marks the recommended algorithm and confirms when it is the selected one', () => {
    render(<Sidebar />)
    // ppo is both recommended and selected → the confirmation caption shows (no switch button).
    expect(screen.getByText('Recommended for this game')).toBeInTheDocument()
  })

  it('offers a one-click switch to the recommended algo when another is selected', () => {
    render(<Sidebar />)
    fireEvent.change(screen.getByRole('combobox', { name: 'Algorithm' }), { target: { value: 'neuroevolution' } })
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

    fireEvent.change(screen.getByRole('combobox', { name: 'Algorithm' }), { target: { value: 'neuroevolution' } })

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
})
