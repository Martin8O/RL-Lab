import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

// Smoke render: the whole tree (TopBar / Sidebar / EnvPreview / RewardChart / BottomPanels /
// PlayScoreGate) composes and mounts without throwing. The WebSocket + fetch globals are stubbed
// in the test setup, so no real backend is needed.
describe('<App /> smoke render', () => {
  it('mounts the full dashboard shell', async () => {
    render(<App />)
    expect(await screen.findByText('RL Lab')).toBeInTheDocument() // TopBar brand
    expect(screen.getByText('Parameters')).toBeInTheDocument()           // Sidebar header
    expect(screen.getByText(/High Scores/)).toBeInTheDocument()          // bottom leaderboards
  })
})
