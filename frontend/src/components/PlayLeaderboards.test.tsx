import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { samplePlayScores } from '../test/fixtures'
import PlayLeaderboards from './PlayLeaderboards'

describe('<PlayLeaderboards />', () => {
  it('shows the shared header, both boards, and an empty state when there are no scores', () => {
    render(<PlayLeaderboards />)
    expect(screen.getByText(/High Scores/)).toBeInTheDocument() // header reads "🏆 High Scores"
    expect(screen.getByText(/Human/)).toBeInTheDocument()
    expect(screen.getByText(/AI/)).toBeInTheDocument()
    // Both the human and AI boards are empty.
    expect(screen.getAllByText('No scores yet')).toHaveLength(2)
  })

  it('renders human and AI entries with rounded scores', () => {
    useAppStore.setState({ playScores: samplePlayScores })
    render(<PlayLeaderboards />)

    expect(screen.getByText('Ada')).toBeInTheDocument()
    expect(screen.getByText('Grace')).toBeInTheDocument()
    expect(screen.getByText('PPO 50k')).toBeInTheDocument()
    expect(screen.getByText('480')).toBeInTheDocument() // Ada's score, rounded
    expect(screen.getByText('500')).toBeInTheDocument() // AI score, rounded
    expect(screen.queryByText('No scores yet')).not.toBeInTheDocument()
  })
})
