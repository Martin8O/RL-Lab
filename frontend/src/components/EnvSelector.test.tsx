import { beforeEach, describe, expect, it } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { useAppStore } from '../store/useAppStore'
import { cartpoleEnv } from '../test/fixtures'
import EnvSelector from './EnvSelector'
import type { EnvSpec } from '../api/types'

// A minimal Atari env sharing the CartPole fixture's shape; only family/id/name matter for the picker.
const pongEnv: EnvSpec = {
  ...cartpoleEnv,
  id: 'pong',
  gym_id: 'ALE/Pong-v5',
  display_name: { en: 'Pong', cz: 'Pong' },
  family: 'atari',
}

function openAtari() {
  render(<EnvSelector />)
  fireEvent.click(screen.getByRole('button')) // the picker trigger
  // Reveal the Atari games (the category is still hoverable/clickable to browse even when locked).
  fireEvent.click(screen.getByRole('menuitem', { name: /Atari/ }))
}

describe('<EnvSelector /> — Atari ale-py lock (R1)', () => {
  beforeEach(() => {
    useAppStore.setState({ envs: [cartpoleEnv, pongEnv], selectedEnvId: 'cartpole' })
  })

  it('locks the Atari category + games and shows the install hint when ale-py is missing', () => {
    useAppStore.setState({ atariAvailable: false })
    openAtari()

    expect(screen.getByRole('menuitem', { name: /Atari/ })).toHaveAttribute('aria-disabled', 'true')
    const pong = screen.getByRole('menuitem', { name: /Pong/ })
    expect(pong).toHaveAttribute('aria-disabled', 'true')

    // Hover a locked row → the neutral install-hint popup appears.
    fireEvent.mouseEnter(pong)
    expect(screen.getByRole('tooltip')).toHaveTextContent(/ale-py/)

    // Clicking a locked game must NOT select it — selection stays on CartPole.
    fireEvent.click(pong)
    expect(useAppStore.getState().selectedEnvId).toBe('cartpole')
  })

  it('leaves Atari selectable (no lock, no hint) when ale-py is available', () => {
    useAppStore.setState({ atariAvailable: true })
    openAtari()

    const atariCat = screen.getByRole('menuitem', { name: /Atari/ })
    expect(atariCat).not.toHaveAttribute('aria-disabled')
    const pong = screen.getByRole('menuitem', { name: /Pong/ })
    fireEvent.mouseEnter(pong)
    expect(screen.queryByRole('tooltip')).toBeNull()

    fireEvent.click(pong)
    expect(useAppStore.getState().selectedEnvId).toBe('pong')
  })
})
