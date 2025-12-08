/**
 * Rule-based AI for Zic-Zac-Zoe (v3)
 *
 * Pure tactical rules, no neural network:
 * 1. Take winning moves (complete 4-in-a-row)
 * 2. Block opponent's winning threats (their XX_X patterns)
 * 3. Avoid moves that create 3-in-a-row for us
 * 4. Otherwise play randomly
 */

import { BoardState, Player, BOARD_SIZE, getLegalMoves } from "./game";

const EMPTY = Player.Empty;

// All lines of length 4+ on the 6x6 board
const ALL_LINES: number[][] = [];

// Generate all lines at module load
function initLines(): void {
  // Horizontal lines
  for (let row = 0; row < BOARD_SIZE; row++) {
    for (let startCol = 0; startCol <= BOARD_SIZE - 4; startCol++) {
      const line: number[] = [];
      for (let col = startCol; col < BOARD_SIZE; col++) {
        line.push(row * BOARD_SIZE + col);
      }
      ALL_LINES.push(line);
    }
  }

  // Vertical lines
  for (let col = 0; col < BOARD_SIZE; col++) {
    for (let startRow = 0; startRow <= BOARD_SIZE - 4; startRow++) {
      const line: number[] = [];
      for (let row = startRow; row < BOARD_SIZE; row++) {
        line.push(row * BOARD_SIZE + col);
      }
      ALL_LINES.push(line);
    }
  }

  // Diagonal down-right (\)
  for (let startRow = 0; startRow <= BOARD_SIZE - 4; startRow++) {
    for (let startCol = 0; startCol <= BOARD_SIZE - 4; startCol++) {
      const line: number[] = [];
      let r = startRow, c = startCol;
      while (r < BOARD_SIZE && c < BOARD_SIZE) {
        line.push(r * BOARD_SIZE + c);
        r++; c++;
      }
      if (line.length >= 4) ALL_LINES.push(line);
    }
  }

  // Diagonal up-right (/)
  for (let startRow = 3; startRow < BOARD_SIZE; startRow++) {
    for (let startCol = 0; startCol <= BOARD_SIZE - 4; startCol++) {
      const line: number[] = [];
      let r = startRow, c = startCol;
      while (r >= 0 && c < BOARD_SIZE) {
        line.push(r * BOARD_SIZE + c);
        r--; c++;
      }
      if (line.length >= 4) ALL_LINES.push(line);
    }
  }
}

initLines();

/**
 * Check if placing a piece at `move` creates exactly 3-in-a-row for `player`
 */
export function createsThree(board: BoardState, move: number, player: Player): boolean {
  // Simulate the move
  const testBoard = [...board];
  testBoard[move] = player;

  // Check all lines containing this move
  for (const line of ALL_LINES) {
    if (!line.includes(move)) continue;

    // Check all windows of 3 consecutive cells in this line
    for (let i = 0; i <= line.length - 3; i++) {
      const window = [line[i], line[i + 1], line[i + 2]];

      // Count player pieces in this window
      let count = 0;
      for (const idx of window) {
        if (testBoard[idx] === player) count++;
      }

      // If all 3 are ours, we created 3-in-a-row
      if (count === 3) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Check if placing a piece at `move` creates 4-in-a-row for `player`
 */
export function createsFour(board: BoardState, move: number, player: Player): boolean {
  // Simulate the move
  const testBoard = [...board];
  testBoard[move] = player;

  // Check all lines containing this move
  for (const line of ALL_LINES) {
    if (!line.includes(move)) continue;

    // Check all windows of 4 consecutive cells in this line
    for (let i = 0; i <= line.length - 4; i++) {
      const window = [line[i], line[i + 1], line[i + 2], line[i + 3]];

      // Count player pieces in this window
      let count = 0;
      for (const idx of window) {
        if (testBoard[idx] === player) count++;
      }

      // If all 4 are ours, we created 4-in-a-row
      if (count === 4) {
        return true;
      }
    }
  }
  return false;
}

/**
 * Find cells where opponent can complete 4-in-a-row (XX_X or X_XX patterns)
 */
export function findThreats(board: BoardState, opponent: Player): number[] {
  const threats: Set<number> = new Set();

  for (const line of ALL_LINES) {
    // Check all windows of 4 consecutive cells
    for (let i = 0; i <= line.length - 4; i++) {
      const window = [line[i], line[i + 1], line[i + 2], line[i + 3]];

      let opponentCount = 0;
      let emptyIdx = -1;
      let emptyCount = 0;

      for (const idx of window) {
        if (board[idx] === opponent) {
          opponentCount++;
        } else if (board[idx] === EMPTY) {
          emptyCount++;
          emptyIdx = idx;
        }
      }

      // Pattern: 3 opponent pieces + 1 empty = threat
      if (opponentCount === 3 && emptyCount === 1) {
        threats.add(emptyIdx);
      }
    }
  }

  return Array.from(threats);
}

/**
 * Rule-based AI move selection
 */
export function getRulesMove(board: BoardState, currentPlayer: Player): number {
  const opponent = currentPlayer === Player.X ? Player.O : Player.X;
  const legalMoves = getLegalMoves(board);

  if (legalMoves.length === 0) {
    throw new Error("No legal moves available");
  }

  // 1. Check for winning moves (complete 4-in-a-row)
  for (const move of legalMoves) {
    if (createsFour(board, move, currentPlayer)) {
      return move;
    }
  }

  // 2. Find opponent threats we must block
  const threats = findThreats(board, opponent);

  // 3. Filter out suicidal moves (ones that create 3-in-a-row for us)
  const safeMoves = legalMoves.filter(move => !createsThree(board, move, currentPlayer));

  // 4. If there are threats, try to block (but only with safe moves)
  if (threats.length > 0) {
    const safeBlocks = threats.filter(t => safeMoves.includes(t));
    if (safeBlocks.length > 0) {
      // Block the first threat with a safe move
      return safeBlocks[0];
    }
    // All blocks are suicidal - we're in trouble, play random safe move
  }

  // 5. Play random safe move
  if (safeMoves.length > 0) {
    return safeMoves[Math.floor(Math.random() * safeMoves.length)];
  }

  // 6. All moves are suicidal - pick random (we lose anyway)
  return legalMoves[Math.floor(Math.random() * legalMoves.length)];
}

/**
 * Check if rules AI is being used (via ?rules=1 URL param)
 */
export function isRulesAI(): boolean {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("rules") === "1";
}

// =============================================================================
// Checkmate Detection
// =============================================================================

/**
 * Get all 3-in-a-row patterns that would be created if player plays at move
 */
export function getThreePatterns(board: BoardState, move: number, player: Player): number[][] {
  const patterns: number[][] = [];
  const testBoard = [...board];
  testBoard[move] = player;

  for (const line of ALL_LINES) {
    if (!line.includes(move)) continue;

    for (let i = 0; i <= line.length - 3; i++) {
      const window = [line[i], line[i + 1], line[i + 2]];
      let count = 0;
      for (const idx of window) {
        if (testBoard[idx] === player) count++;
      }
      if (count === 3) {
        patterns.push(window);
      }
    }
  }
  return patterns;
}

/**
 * Get the 4-pattern for a threat cell (opponent's 3 pieces + the empty threat cell)
 */
export function getThreatPattern(board: BoardState, threatCell: number, opponent: Player): number[] | null {
  for (const line of ALL_LINES) {
    if (!line.includes(threatCell)) continue;

    for (let i = 0; i <= line.length - 4; i++) {
      const window = [line[i], line[i + 1], line[i + 2], line[i + 3]];
      if (!window.includes(threatCell)) continue;

      let opponentCount = 0;
      let emptyCount = 0;

      for (const idx of window) {
        if (board[idx] === opponent) opponentCount++;
        else if (board[idx] === EMPTY) emptyCount++;
      }

      if (opponentCount === 3 && emptyCount === 1) {
        return window;
      }
    }
  }
  return null;
}

export interface CheckmateResult {
  isCheckmate: boolean;
  loser: Player | null;
  // Patterns to highlight
  threatPatterns: number[][];   // 4-patterns from opponent threats
  suicidePatterns: number[][];  // 3-patterns from suicide moves
}

/**
 * Check if the current player is in checkmate.
 *
 * Checkmate conditions (prerequisite: no winning move available):
 * 1. Suicide block - opponent has 1+ threats, and 1+ blocking cell creates 3-in-a-row
 * 2. Multiple threats - opponent has 2+ distinct threat cells
 * 3. All moves suicide - every empty cell creates 3-in-a-row
 */
export function detectCheckmate(board: BoardState, currentPlayer: Player): CheckmateResult {
  const opponent = currentPlayer === Player.X ? Player.O : Player.X;
  const legalMoves = getLegalMoves(board);

  const noCheckmate: CheckmateResult = {
    isCheckmate: false,
    loser: null,
    threatPatterns: [],
    suicidePatterns: [],
  };

  if (legalMoves.length === 0) {
    return noCheckmate;
  }

  // Prerequisite: Check if current player has any winning moves
  for (const move of legalMoves) {
    if (createsFour(board, move, currentPlayer)) {
      // Player can win - not checkmate
      return noCheckmate;
    }
  }

  // Find opponent threats
  const threats = findThreats(board, opponent);

  // Find which moves are suicide (create 3-in-a-row)
  const suicideMoves = legalMoves.filter(move => createsThree(board, move, currentPlayer));

  // Trigger 1 & 2: Check threat-based checkmate
  if (threats.length > 0) {
    // Trigger 2: Multiple threats (can only block one)
    if (threats.length >= 2) {
      const threatPatterns: number[][] = [];
      for (const t of threats) {
        const pattern = getThreatPattern(board, t, opponent);
        if (pattern) threatPatterns.push(pattern);
      }
      return {
        isCheckmate: true,
        loser: currentPlayer,
        threatPatterns,
        suicidePatterns: [],
      };
    }

    // Trigger 1: Single threat but blocking is suicide
    // (only compute suicideBlocks when we have exactly 1 threat)
    const suicideBlocks = threats.filter(t => suicideMoves.includes(t));
    if (suicideBlocks.length > 0) {
      const threatPatterns: number[][] = [];
      const suicidePatterns: number[][] = [];

      for (const t of suicideBlocks) {
        const threatPattern = getThreatPattern(board, t, opponent);
        if (threatPattern) threatPatterns.push(threatPattern);

        const threePatterns = getThreePatterns(board, t, currentPlayer);
        suicidePatterns.push(...threePatterns);
      }

      return {
        isCheckmate: true,
        loser: currentPlayer,
        threatPatterns,
        suicidePatterns,
      };
    }
  }

  // Trigger 3: All moves are suicide (no threats needed)
  if (suicideMoves.length === legalMoves.length) {
    const suicidePatterns: number[][] = [];
    for (const move of suicideMoves) {
      const patterns = getThreePatterns(board, move, currentPlayer);
      suicidePatterns.push(...patterns);
    }
    return {
      isCheckmate: true,
      loser: currentPlayer,
      threatPatterns: [],
      suicidePatterns,
    };
  }

  return noCheckmate;
}
