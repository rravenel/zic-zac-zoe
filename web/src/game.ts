/**
 * Zic-Zac-Zoe Game Logic
 *
 * Rules:
 * - 6x6 grid
 * - 4 in a row wins
 * - 3 in a row loses
 * - Diagonals count
 * - X moves first
 */

export const BOARD_SIZE = 6;
export const WIN_LENGTH = 4;
export const LOSE_LENGTH = 3;

export enum Player {
  Empty = 0,
  X = 1,
  O = 2,
}

export enum GameResult {
  Ongoing = 0,
  XWins = 1,
  OWins = 2,
  Draw = 3,
}

export type BoardState = Player[];

/**
 * Create a new empty board
 */
export function createBoard(): BoardState {
  return new Array(BOARD_SIZE * BOARD_SIZE).fill(Player.Empty);
}

/**
 * Get the current player (X moves first, then alternates)
 */
export function getCurrentPlayer(board: BoardState): Player {
  const moveCount = board.filter((cell) => cell !== Player.Empty).length;
  return moveCount % 2 === 0 ? Player.X : Player.O;
}

/**
 * Get opponent of a player
 */
export function getOpponent(player: Player): Player {
  return player === Player.X ? Player.O : Player.X;
}

/**
 * Check if a cell is empty
 */
export function isEmpty(board: BoardState, index: number): boolean {
  return board[index] === Player.Empty;
}

/**
 * Get all legal moves (empty cells)
 */
export function getLegalMoves(board: BoardState): number[] {
  const moves: number[] = [];
  for (let i = 0; i < board.length; i++) {
    if (board[i] === Player.Empty) {
      moves.push(i);
    }
  }
  return moves;
}

/**
 * Make a move, returning a new board state
 */
export function makeMove(board: BoardState, index: number): BoardState {
  if (board[index] !== Player.Empty) {
    throw new Error(`Cell ${index} is not empty`);
  }
  const newBoard = [...board];
  newBoard[index] = getCurrentPlayer(board);
  return newBoard;
}

/**
 * Convert flat index to (row, col)
 */
export function indexToCoord(index: number): [number, number] {
  return [Math.floor(index / BOARD_SIZE), index % BOARD_SIZE];
}

/**
 * Convert (row, col) to flat index
 */
export function coordToIndex(row: number, col: number): number {
  return row * BOARD_SIZE + col;
}

/**
 * Check for consecutive pieces in a direction from a starting point.
 * Returns the indices of the connected pieces.
 */
function findConnected(
  board: BoardState,
  startRow: number,
  startCol: number,
  dRow: number,
  dCol: number,
  player: Player
): number[] {
  const indices: number[] = [];

  let r = startRow;
  let c = startCol;

  while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
    const idx = coordToIndex(r, c);
    if (board[idx] === player) {
      indices.push(idx);
      r += dRow;
      c += dCol;
    } else {
      break;
    }
  }

  return indices;
}

/**
 * Get the maximum consecutive count for a player in any direction,
 * along with the indices of the winning/losing line.
 */
function getMaxConsecutive(
  board: BoardState,
  player: Player
): { count: number; indices: number[] } {
  let maxCount = 0;
  let maxIndices: number[] = [];

  // Check all starting positions
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let col = 0; col < BOARD_SIZE; col++) {
      if (board[coordToIndex(row, col)] !== player) continue;

      // Check 4 directions (right, down, down-right, down-left)
      const directions = [
        [0, 1],
        [1, 0],
        [1, 1],
        [1, -1],
      ];

      for (const [dRow, dCol] of directions) {
        const indices = findConnected(board, row, col, dRow, dCol, player);
        if (indices.length > maxCount) {
          maxCount = indices.length;
          maxIndices = indices;
        }
      }
    }
  }

  return { count: maxCount, indices: maxIndices };
}

/**
 * Result of checking game state
 */
export interface GameCheckResult {
  result: GameResult;
  winningIndices: number[];
  losingIndices: number[];
  losingPlayer: Player | null;
}

/**
 * Check if the game has ended
 */
export function checkResult(board: BoardState): GameCheckResult {
  const xResult = getMaxConsecutive(board, Player.X);
  const oResult = getMaxConsecutive(board, Player.O);

  // Check for loss (exactly 3 in a row) - takes priority
  if (xResult.count === LOSE_LENGTH) {
    return {
      result: GameResult.OWins,
      winningIndices: [],
      losingIndices: xResult.indices,
      losingPlayer: Player.X,
    };
  }
  if (oResult.count === LOSE_LENGTH) {
    return {
      result: GameResult.XWins,
      winningIndices: [],
      losingIndices: oResult.indices,
      losingPlayer: Player.O,
    };
  }

  // Check for win (4+ in a row)
  if (xResult.count >= WIN_LENGTH) {
    return {
      result: GameResult.XWins,
      winningIndices: xResult.indices,
      losingIndices: [],
      losingPlayer: null,
    };
  }
  if (oResult.count >= WIN_LENGTH) {
    return {
      result: GameResult.OWins,
      winningIndices: oResult.indices,
      losingIndices: [],
      losingPlayer: null,
    };
  }

  // Check for draw (board full)
  const moveCount = board.filter((cell) => cell !== Player.Empty).length;
  if (moveCount === BOARD_SIZE * BOARD_SIZE) {
    return {
      result: GameResult.Draw,
      winningIndices: [],
      losingIndices: [],
      losingPlayer: null,
    };
  }

  return {
    result: GameResult.Ongoing,
    winningIndices: [],
    losingIndices: [],
    losingPlayer: null,
  };
}

/**
 * Fast check after a move (only examines lines through the move)
 */
export function checkResultFast(
  board: BoardState,
  lastMove: number
): GameCheckResult {
  const moveCount = board.filter((cell) => cell !== Player.Empty).length;
  const lastPlayer = moveCount % 2 === 1 ? Player.X : Player.O;
  const [lastRow, lastCol] = indexToCoord(lastMove);

  // Check all 4 directions through the last move
  const directions = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];

  let maxCount = 0;
  let maxIndices: number[] = [];

  for (const [dRow, dCol] of directions) {
    // Find connected in positive direction
    const positive = findConnected(
      board,
      lastRow + dRow,
      lastCol + dCol,
      dRow,
      dCol,
      lastPlayer
    );

    // Find connected in negative direction
    const negative = findConnected(
      board,
      lastRow - dRow,
      lastCol - dCol,
      -dRow,
      -dCol,
      lastPlayer
    );

    // Combine with the piece itself
    const indices = [...negative.reverse(), lastMove, ...positive];
    if (indices.length > maxCount) {
      maxCount = indices.length;
      maxIndices = indices;
    }
  }

  // Check loss (exactly 3)
  if (maxCount === LOSE_LENGTH) {
    const winner = lastPlayer === Player.X ? Player.O : Player.X;
    return {
      result: winner === Player.X ? GameResult.XWins : GameResult.OWins,
      winningIndices: [],
      losingIndices: maxIndices,
      losingPlayer: lastPlayer,
    };
  }

  // Check win (4+)
  if (maxCount >= WIN_LENGTH) {
    return {
      result: lastPlayer === Player.X ? GameResult.XWins : GameResult.OWins,
      winningIndices: maxIndices,
      losingIndices: [],
      losingPlayer: null,
    };
  }

  // Check draw
  if (moveCount === BOARD_SIZE * BOARD_SIZE) {
    return {
      result: GameResult.Draw,
      winningIndices: [],
      losingIndices: [],
      losingPlayer: null,
    };
  }

  return {
    result: GameResult.Ongoing,
    winningIndices: [],
    losingIndices: [],
    losingPlayer: null,
  };
}
