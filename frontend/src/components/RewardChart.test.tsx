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

  it('shows an algorithm-aware empty message per tab', () => {
    // PPO (default): Reward is one of its tabs → "start training"; Fitness is not → "doesn't use…".
    useAppStore.setState({ algo: 'ppo' })
    render(<RewardChart />)
    expect(screen.getByText('Start training to see the live chart')).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: 'Fitness' }))
    expect(screen.getByText("PPO doesn't use the Fitness chart — see Reward / Loss")).toBeInTheDocument()
  })

  it('marks tabs not produced by the selected algorithm (Q-learning)', () => {
    useAppStore.setState({ algo: 'q_learning' })
    render(<RewardChart />)
    // Reward is Q-learning's tab → start-training hint; Loss is not → "doesn't use…".
    expect(screen.getByText('Start training to see the live chart')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Loss' }))
    expect(screen.getByText("Q-learning doesn't use the Loss chart — see Reward")).toBeInTheDocument()
  })
})
