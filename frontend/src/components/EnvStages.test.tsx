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
        humanTurn={false} humanSide={null} onCellClick={() => {}} statusText="" banner={null}
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
        humanTurn humanSide={0} onCellClick={onCellClick} statusText="" banner={null}
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
        humanTurn={false} humanSide={null} onCellClick={() => {}} statusText="" banner={null}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Pass' })).not.toBeInTheDocument()

    rerender(
      <BoardStage
        envName="Othello" board={passBoard({ pass_action: null, legal_actions: [19] })}
        meta={othelloMeta} humanTurn humanSide={null} onCellClick={() => {}} statusText="" banner={null}
      />,
    )
    expect(screen.queryByRole('button', { name: 'Pass' })).not.toBeInTheDocument()
  })
})

// G6e — BoardStage's move-mode (Breakthrough) interaction: a move is (from → to), so a piece is picked
// first, its destinations light up, and a destination click submits the matching action int. Uses a
// tiny 3×3 board (BoardStage is generic over rows/cols) with one 'b' piece that can go to two squares.
const breakthroughMeta = BOARD_GAMES.breakthrough

function moveBoard(): BoardState {
  const cells = Array<string>(9).fill('.')
  cells[4] = 'b' // centre piece (row 2, col 2), player 0
  return {
    cells,
    rows: 3,
    cols: 3,
    legal_actions: [100, 101],
    current_player: 0,
    last_action: null,
    is_terminal: false,
    winner: null,
    moves: [
      { action: 100, from_cell: 4, to_cell: 1 }, // up to row 1, col 2
      { action: 101, from_cell: 4, to_cell: 7 }, // down to row 3, col 2
    ],
  }
}

describe('<BoardStage /> move mode (Breakthrough, G6e)', () => {
  it('select-a-piece → highlight destinations → click a destination submits its action', () => {
    const onCellClick = vi.fn()
    render(
      <BoardStage
        envName="Breakthrough" board={moveBoard()} meta={breakthroughMeta}
        humanTurn humanSide={0} onCellClick={onCellClick} statusText="" banner={null}
      />,
    )
    // Before selecting, the piece is a "select" button and no destinations are offered.
    expect(screen.queryByRole('button', { name: 'Move to row 1, column 2' })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Select your piece at row 2, column 2' }))

    // Now both of that piece's destinations are clickable; clicking one submits its action int.
    expect(screen.getByRole('button', { name: 'Move to row 1, column 2' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'Move to row 3, column 2' }))
    expect(onCellClick).toHaveBeenCalledWith(101) // the action of the (from 4 → to 7) move
  })

  it('clicking the selected piece again deselects it (destinations disappear)', () => {
    render(
      <BoardStage
        envName="Breakthrough" board={moveBoard()} meta={breakthroughMeta}
        humanTurn humanSide={0} onCellClick={() => {}} statusText="" banner={null}
      />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Select your piece at row 2, column 2' }))
    expect(screen.getByRole('button', { name: 'Move to row 1, column 2' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /Selected piece at row 2, column 2/ }))
    expect(screen.queryByRole('button', { name: 'Move to row 1, column 2' })).not.toBeInTheDocument()
  })
})

// G6g — chess: the FEN piece letters distinguish white (UPPERCASE → outline glyphs) from black
// (lowercase → filled glyphs) by case, and a pawn promotion offers SEVERAL actions on one square that a
// picker disambiguates. Uses a tiny 3×3 board (BoardStage is generic over rows/cols) with one white pawn.
const chessMeta = BOARD_GAMES.chess

function chessPromoBoard(): BoardState {
  const cells = Array<string>(9).fill('.')
  cells[4] = 'P' // a white pawn (uppercase) at the centre — case is what marks it white (player 1)
  cells[0] = 'k' // a black king (lowercase, player 0) — proves case-sensitive lookup picks the filled glyph
  return {
    cells,
    rows: 3,
    cols: 3,
    legal_actions: [200, 201, 202, 203],
    current_player: 0,
    last_action: null,
    is_terminal: false,
    winner: null,
    moves: [ // four actions landing on the SAME square (a promotion), differing only by piece
      { action: 200, from_cell: 4, to_cell: 1, promotion: 'q' },
      { action: 201, from_cell: 4, to_cell: 1, promotion: 'r' },
      { action: 202, from_cell: 4, to_cell: 1, promotion: 'b' },
      { action: 203, from_cell: 4, to_cell: 1, promotion: 'n' },
    ],
  }
}

describe('<BoardStage /> chess (G6g)', () => {
  it('renders white/black pieces as the lichess SVGs, mapped from FEN case', () => {
    const { container } = render(
      <BoardStage envName="Chess" board={chessMeta.idle} meta={chessMeta}
        humanTurn={false} humanSide={null} onCellClick={() => {}} statusText="" banner={null} />,
    )
    const imgs = (file: string) => container.querySelectorAll(`img[src$="${file}.svg"]`).length
    expect(imgs('wK')).toBe(1) // white king (uppercase K → player 1 → cburnett wK)
    expect(imgs('bK')).toBe(1) // black king (lowercase k → player 0 → bK)
    expect(imgs('wP')).toBe(8) // eight white pawns
    expect(imgs('bP')).toBe(8) // eight black pawns
  })

  it('promotion: destination click opens a piece picker, and the chosen piece submits its action', () => {
    const onCellClick = vi.fn()
    render(
      <BoardStage envName="Chess" board={chessPromoBoard()} meta={chessMeta}
        humanTurn humanSide={1} onCellClick={onCellClick} statusText="" banner={null} />,
    )
    // Pick the pawn → its single destination lights up; clicking it opens the picker (does NOT submit yet).
    fireEvent.click(screen.getByRole('button', { name: 'Select your piece at row 2, column 2' }))
    fireEvent.click(screen.getByRole('button', { name: 'Move to row 1, column 2' }))
    expect(onCellClick).not.toHaveBeenCalled() // a promotion needs the piece choice first
    // All four promotion choices are offered; picking the queen submits that action int.
    expect(screen.getByRole('button', { name: 'Promote to rook' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Promote to knight' })).toBeInTheDocument()
    // The picker shows the human's OWN colour: white (player 1) promotes to the white queen ♕, not ♛.
    expect(screen.getByRole('button', { name: 'Promote to queen' })).toHaveTextContent('♕')
    fireEvent.click(screen.getByRole('button', { name: 'Promote to queen' }))
    expect(onCellClick).toHaveBeenCalledWith(200) // the action behind the =Q choice
  })

  // Orientation (G6e): the board flips 180° when the human plays the side that isn't `bottomPlayer`
  // (so their pieces sit at the bottom). breakthroughMeta.orient.bottomPlayer === 1.
  it('rotates the board 180° when the human plays the non-bottom side, not otherwise', () => {
    const grid = (c: HTMLElement) => c.querySelector('div[style*="grid-template-columns"]') as HTMLElement
    const first = render(
      <BoardStage envName="Breakthrough" board={moveBoard()} meta={breakthroughMeta}
        humanTurn={false} humanSide={0} onCellClick={() => {}} statusText="" banner={null} />,
    )
    expect(grid(first.container).style.transform).toBe('rotate(180deg)') // human is player 0 → flipped

    const second = render(
      <BoardStage envName="Breakthrough" board={moveBoard()} meta={breakthroughMeta}
        humanTurn={false} humanSide={1} onCellClick={() => {}} statusText="" banner={null} />,
    )
    expect(grid(second.container).style.transform).toBe('') // player 1 already at the bottom → no flip

    const watch = render(
      <BoardStage envName="Breakthrough" board={moveBoard()} meta={breakthroughMeta}
        humanTurn={false} humanSide={null} onCellClick={() => {}} statusText="" banner={null} />,
    )
    expect(grid(watch.container).style.transform).toBe('') // watch/training keeps the default view
  })
})
