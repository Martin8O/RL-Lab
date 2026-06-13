import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import Sidebar from './Sidebar'

describe('<Sidebar />', () => {
  beforeEach(() => {
    useAppStore.setState({ envs: [cartpoleEnv], selectedEnvId: 'cartpole' })
  })

  it('renders the panel title and both algorithm options', () => {
    render(<Sidebar />)
    expect(screen.getByText('Parameters')).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'PPO' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: 'Neuroevolution' })).toBeInTheDocument()
  })

  it('shows PPO hyperparameters by default and switches to evolution settings', () => {
    render(<Sidebar />)
    expect(screen.getByText('PPO Hyperparameters')).toBeInTheDocument()
    expect(screen.getByLabelText('Learning Rate')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('tab', { name: 'Neuroevolution' }))

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
