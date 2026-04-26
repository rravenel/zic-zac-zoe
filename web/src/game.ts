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
 * Calculate the score for a specific move based on the points variant rules.
 */
export function calculateMoveScore(
  board: BoardState,
  moveIndex: number,
  player: Player
): number {
  const [row, col] = indexToCoord(moveIndex);
  const directions = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];

  const lineScores: number[] = [];

  for (const [dr, dc] of directions) {
    // Scan in negative direction
    let l1 = 0;
    let r = row - dr;
    let c = col - dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
      if (board[coordToIndex(r, c)] === player) {
        l1++;
        r -= dr;
        c -= dc;
      } else {
        break;
      }
    }

    // Scan in positive direction
    let l2 = 0;
    r = row + dr;
    c = col + dc;
    while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
      if (board[coordToIndex(r, c)] === player) {
        l2++;
        r += dr;
        c += dc;
      } else {
        break;
      }
    }

    const totalL = l1 + l2 + 1;

    if (totalL > 1) {
      // Base Score logic
      let base = totalL === 3 ? 0 : totalL;

      // Bridge Bonus: 2x if move has neighbors on both sides of this axis
      if (l1 > 0 && l2 > 0) {
        base *= 2;
      }

      lineScores.push(base);
    }
  }

  const n = lineScores.length;
  if (n === 0) {
    return 1; // Lone tile rule
  }

  const sum = lineScores.reduce((a, b) => a + b, 0);
  return sum * n;
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
  // In the points variant, the game only ends when the board is full.
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
  _lastMove: number
): GameCheckResult {
  const moveCount = board.filter((cell) => cell !== Player.Empty).length;

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
