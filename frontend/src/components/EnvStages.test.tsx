import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen } from '@testing-library/react'
import { BoardStage } from './EnvStages'
import { BOARD_GAMES } from '../content/boardGames'
import type { BoardState } from '../api/types'

// G6d — BoardStage's two Othello-specific bits: the disc glyphs render from the streamed grid, and a
// forced "pass" surfaces as a Pass button (reached only deep in an endgame, so a unit test is the
// reliable check). Everything else (cell clicks, last-move ring) is exercised by the smaller games.

const othelloMeta = BOARD_GAMES.othello

function passBoard(overrides: Partial<BoardState> = {}): BoardState {
  return {
    cells: Array<string>(64).fill('.'),
    rows: 8,
    cols: 8,
    legal_actions: [64], // only the pass move is legal (no placement)
    current_player: 0,
    last_action: null,
    is_terminal: false,
    winner: null,
    pass_action: 64,
    ...overrides,
  }
}

describe('<BoardStage /> (Othello, G6d)', () => {
  it('renders the filled ● and open ◯ discs from the idle opening board', () => {
    render(
      <BoardStage
        envName="Othello" board={othelloMeta.idle} meta={othelloMeta}
        humanTurn={false} onCellClick={() => {}} statusText="" banner={null}
      />,
    )
    // The standard opening has two discs of each colour at the centre.
    expect(screen.getAllByText('●').length).toBeGreaterThanOrEqual(2)
    expect(screen.getAllByText('◯').length).toBeGreaterThanOrEqual(2)
  })

  it('shows a Pass button on the human turn and submits the pass action when clicked', () => {
    const onCellClick = vi.fn()
    render(
      <BoardStage
        envName="Othello" board={passBoard()} meta={othelloMeta}
        humanTurn onCellClick={onCellClick} statusText="" banner={null}
      />,
    )
    const pass = screen.getByRole('button', { name: 'Pass' })
    fireEvent.click(pass)
    expect(onCellClick).toHaveBeenCalledWith(64) // the backend-detected pass action index
  })

  it('hides the Pass button when it is not the human turn or there is no pass move', () => {
    const { rerender } = render(
      <BoardStage
        envName="Othello" board={passBoard()} meta={othelloMeta}
        humanTurn={false} onCellClick={() => {}} statusText="" banner={null}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Pass' })).not.toBeInTheDocument()

    rerender(
      <BoardStage
        envName="Othello" board={passBoard({ pass_action: null, legal_actions: [19] })}
        meta={othelloMeta} humanTurn onCellClick={() => {}} statusText="" banner={null}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Pass' })).not.toBeInTheDocument()
  })
})
