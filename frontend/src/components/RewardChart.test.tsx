import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import RewardChart from './RewardChart'

describe('<RewardChart />', () => {
  beforeEach(() => {
    useAppStore.setState({ envs: [cartpoleEnv], selectedEnvId: 'cartpole' })
  })

  it('renders the Reward / Loss / Fitness tabs', () => {
    render(<RewardChart />)
    expect(screen.getByRole('button', { name: 'Reward' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Loss' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Fitness' })).toBeInTheDocument()
  })

  it('shows the empty placeholder when there is no data, per tab', () => {
    render(<RewardChart />)
    expect(screen.getByText('Start training to see the live chart')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Fitness' }))
    expect(screen.getByText('Switch to Neuroevolution and Run to see fitness')).toBeInTheDocument()
  })
})
