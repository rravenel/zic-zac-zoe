/**
 * Tests for checkmate detection logic.
 * Run with: npx ts-node src/test-checkmate.ts
 * Or: npx vite-node src/test-checkmate.ts
 */

import { Player, BOARD_SIZE } from "./game";

import {
  createsThree,
  createsFour,
  findThreats,
  getThreePatterns,
  getThreatPattern,
  detectCheckmate,
} from "./rules-ai";

type BoardState = number[];

// =============================================================================
// Test Utilities
// =============================================================================

function createEmptyBoard(): BoardState {
  return new Array(BOARD_SIZE * BOARD_SIZE).fill(Player.Empty);
}

function boardFromString(s: string): BoardState {
  // Parse a board string like:
  // `. . . . . .
  //  . X X X . .
  //  . . . . . .
  //  . . . . . .
  //  . . . . . .
  //  . . . . . .`
  const board = createEmptyBoard();
  const chars = s.replace(/\s+/g, "");
  for (let i = 0; i < chars.length && i < 36; i++) {
    if (chars[i] === "X") board[i] = Player.X;
    else if (chars[i] === "O") board[i] = Player.O;
  }
  return board;
}

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string): void {
  if (condition) {
    console.log(`  ✓ ${message}`);
    passed++;
  } else {
    console.log(`  ✗ ${message}`);
    failed++;
  }
}

function assertEqual<T>(actual: T, expected: T, message: string): void {
  if (actual === expected) {
    console.log(`  ✓ ${message}`);
    passed++;
  } else {
    console.log(`  ✗ ${message}: expected ${expected}, got ${actual}`);
    failed++;
  }
}

// =============================================================================
// Tests: createsThree
// =============================================================================

function testCreatesThree(): void {
  console.log("\ncreatesThree:");

  // Test: Playing at index 2 creates X X X in row 0
  const board1 = boardFromString(`
    X X . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsThree(board1, 2, Player.X), "X at index 2 creates 3-in-a-row");
  assert(!createsThree(board1, 3, Player.X), "X at index 3 does not create 3-in-a-row");

  // Test: O creating 3-in-a-row vertically
  const board2 = boardFromString(`
    O . . . . .
    O . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsThree(board2, 12, Player.O), "O at index 12 creates vertical 3-in-a-row");

  // Test: Diagonal 3-in-a-row (down-right, long diagonal)
  const board3 = boardFromString(`
    X . . . . .
    . X . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsThree(board3, 14, Player.X), "X at index 14 creates diagonal 3-in-a-row");

  // Test: Short diagonal 2,7,12 (up-right, only 3 cells)
  const board4 = boardFromString(`
    . . O . . .
    . O . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsThree(board4, 12, Player.O), "O at index 12 creates short diagonal 2,7,12");

  // Test: Short diagonal 3,10,17 (down-right from col 3)
  const board5 = boardFromString(`
    . . . X . .
    . . . . X .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsThree(board5, 17, Player.X), "X at index 17 creates short diagonal 3,10,17");

  // Test: Short diagonal 5,10,15 (down-left from col 5)
  const board6 = boardFromString(`
    . . . . . X
    . . . . X .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsThree(board6, 15, Player.X), "X at index 15 creates short diagonal 5,10,15");

  // Test: 2-cell diagonal should NOT trigger 3-in-a-row
  const board7 = boardFromString(`
    . . . . . X
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(!createsThree(board7, 10, Player.X), "X at index 10 does NOT create 3 (only 2-cell diagonal)");
}

// =============================================================================
// Tests: createsFour
// =============================================================================

function testCreatesFour(): void {
  console.log("\ncreatesFour:");

  // Test: Playing at index 3 creates X X X X in row 0
  const board1 = boardFromString(`
    X X X . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(createsFour(board1, 3, Player.X), "X at index 3 creates 4-in-a-row");
  assert(!createsFour(board1, 4, Player.X), "X at index 4 does not create 4-in-a-row");

  // Test: 3 pieces don't make 4
  const board2 = boardFromString(`
    X X . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  assert(!createsFour(board2, 2, Player.X), "X at index 2 only creates 3, not 4");
}

// =============================================================================
// Tests: findThreats
// =============================================================================

function testFindThreats(): void {
  console.log("\nfindThreats:");

  // Test: O has 3 in a row with one gap - threat at index 3
  const board1 = boardFromString(`
    O O O . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const threats1 = findThreats(board1, Player.O);
  assert(threats1.includes(3), "Threat detected at index 3 (O O O _)");

  // Test: X X _ X pattern - threat at index 2
  const board2 = boardFromString(`
    X X . X . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const threats2 = findThreats(board2, Player.X);
  assert(threats2.includes(2), "Threat detected at index 2 (X X _ X)");

  // Test: No threats with only 2 pieces
  const board3 = boardFromString(`
    X X . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const threats3 = findThreats(board3, Player.X);
  assertEqual(threats3.length, 0, "No threats with only 2 pieces");
}

// =============================================================================
// Tests: detectCheckmate - Trigger 1 (Suicide Block)
// =============================================================================

function testCheckmateT1SuicideBlock(): void {
  console.log("\ndetectCheckmate - Trigger 1 (Suicide Block):");

  // O threatens at cell 3 (O O O _). X has pieces at 9,15 (column 3).
  // If X blocks at 3, it creates 3-in-a-row (3,9,15) - suicide!
  // X has no winning move, so this is checkmate.
  const board = boardFromString(`
    O O O . . .
    . . . X . .
    . . . X . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);

  const result = detectCheckmate(board, Player.X);
  assert(result.isCheckmate, "X is in checkmate (suicide block)");
  assertEqual(result.loser, Player.X, "Loser is X");
  assert(result.threatPatterns.length > 0, "Has threat patterns");
  assert(result.suicidePatterns.length > 0, "Has suicide patterns");
}

// =============================================================================
// Tests: detectCheckmate - Short Diagonal Suicide
// =============================================================================

function testCheckmateShortDiagonal(): void {
  console.log("\ndetectCheckmate - Short Diagonal Suicide:");

  // Real game scenario: X threatens at 12 (column 0: 0,6,12,18)
  // O must block at 12, but that creates O's 3-in-a-row on short diagonal 2,7,12
  // Board from screenshot:
  // X . O X . .
  // X O . O . .
  // . . X . . .
  // X O . X . .
  // . . . . . .
  // . . . . . O
  const board = boardFromString(`
    X . O X . .
    X O . O . .
    . . X . . .
    X O . X . .
    . . . . . .
    . . . . . O
  `);

  // Verify the setup: X has threat at 12
  const xThreats = findThreats(board, Player.X);
  assert(xThreats.includes(12), "X threatens at 12 (column 0)");

  // Verify: O playing at 12 creates 3-in-a-row (diagonal 2,7,12)
  assert(createsThree(board, 12, Player.O), "O at 12 creates 3-in-a-row (diagonal 2,7,12)");

  // O should be in checkmate
  const result = detectCheckmate(board, Player.O);
  assert(result.isCheckmate, "O is in checkmate (short diagonal suicide)");
  assertEqual(result.loser, Player.O, "Loser is O");
}

// =============================================================================
// Tests: detectCheckmate - Trigger 2 (Multiple Threats)
// =============================================================================

function testCheckmateT2MultipleThreats(): void {
  console.log("\ndetectCheckmate - Trigger 2 (Multiple Threats):");

  // O has two threats (at cells 3 and 9) - X can only block one.
  // X pieces are scattered with no winning patterns.
  const board = createEmptyBoard();

  // O threats: row 0 (threat at 3), row 1 (threat at 9)
  board[0] = Player.O;
  board[1] = Player.O;
  board[2] = Player.O;
  board[6] = Player.O;
  board[7] = Player.O;
  board[8] = Player.O;

  // X pieces scattered (no patterns)
  board[5] = Player.X;
  board[10] = Player.X;
  board[23] = Player.X;
  board[24] = Player.X;
  board[29] = Player.X;
  board[30] = Player.X;

  const result = detectCheckmate(board, Player.X);
  assert(result.isCheckmate, "X is in checkmate (multiple threats)");
  assertEqual(result.loser, Player.X, "Loser is X");
  assert(result.threatPatterns.length >= 2, "Has at least 2 threat patterns");
}

// =============================================================================
// Tests: detectCheckmate - Trigger 3 (All Moves Suicide)
// =============================================================================

function testCheckmateT3AllSuicide(): void {
  console.log("\ndetectCheckmate - Trigger 3 (All Moves Suicide):");

  // Only cells 14 and 20 are empty. Both are suicide (create 3-in-a-row).
  // No O threats exist. X has no winning moves.
  // - Cell 14: completes diagonal 0,7,14
  // - Cell 20: completes diagonal 6,13,20
  const board = createEmptyBoard();

  // X pieces for diagonal suicide patterns
  board[0] = Player.X;   // diagonal 0,7,14
  board[7] = Player.X;
  board[6] = Player.X;   // diagonal 6,13,20
  board[13] = Player.X;

  // X pieces to break O threat windows
  board[15] = Player.X;  // breaks [14,15,16,17]
  board[19] = Player.X;  // breaks [18,19,20,21]
  board[22] = Player.X;  // breaks [20,21,22,23]
  board[28] = Player.X;  // breaks [14,21,28,35]

  // Fill rest with O (except empty cells 14 and 20)
  for (let i = 0; i < 36; i++) {
    if (board[i] === Player.Empty && i !== 14 && i !== 20) {
      board[i] = Player.O;
    }
  }

  const result = detectCheckmate(board, Player.X);
  assert(result.isCheckmate, "X is in checkmate (all moves suicide)");
  assertEqual(result.loser, Player.X, "Loser is X");
  assertEqual(result.threatPatterns.length, 0, "No threat patterns (trigger 3)");
  assert(result.suicidePatterns.length > 0, "Has suicide patterns");
}

// =============================================================================
// Tests: detectCheckmate - Not Checkmate Cases
// =============================================================================

function testNotCheckmate(): void {
  console.log("\ndetectCheckmate - Not Checkmate:");

  // Empty board - not checkmate
  const board1 = createEmptyBoard();
  const result1 = detectCheckmate(board1, Player.X);
  assert(!result1.isCheckmate, "Empty board is not checkmate");

  // X has a winning move available
  const board2 = boardFromString(`
    X X X . . .
    O O O . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  // X can play at 3 to win (4-in-a-row), so not checkmate even if O threatens
  const result2 = detectCheckmate(board2, Player.X);
  assert(!result2.isCheckmate, "Not checkmate when winning move available");

  // Single threat but blocking is safe
  const board3 = boardFromString(`
    O O O . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  // O threatens at 3, X can safely block (no 3-in-a-row created)
  const result3 = detectCheckmate(board3, Player.X);
  assert(!result3.isCheckmate, "Not checkmate when safe block available");
}

// =============================================================================
// Tests: getThreePatterns
// =============================================================================

function testGetThreePatterns(): void {
  console.log("\ngetThreePatterns:");

  const board = boardFromString(`
    X X . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);

  const patterns = getThreePatterns(board, 2, Player.X);
  assert(patterns.length > 0, "Found 3-in-a-row pattern");
  assert(patterns.some(p => p.includes(0) && p.includes(1) && p.includes(2)),
    "Pattern includes indices 0,1,2");
}

// =============================================================================
// Tests: getThreatPattern
// =============================================================================

function testGetThreatPattern(): void {
  console.log("\ngetThreatPattern:");

  const board = boardFromString(`
    O O O . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);

  const pattern = getThreatPattern(board, 3, Player.O);
  assert(pattern !== null, "Found threat pattern");
  if (pattern) {
    assert(pattern.includes(0), "Pattern includes index 0");
    assert(pattern.includes(1), "Pattern includes index 1");
    assert(pattern.includes(2), "Pattern includes index 2");
    assert(pattern.includes(3), "Pattern includes index 3");
  }
}

// =============================================================================
// Run All Tests
// =============================================================================

function runTests(): void {
  console.log("=".repeat(60));
  console.log("Checkmate Detection Tests");
  console.log("=".repeat(60));

  testCreatesThree();
  testCreatesFour();
  testFindThreats();
  testGetThreePatterns();
  testGetThreatPattern();
  testCheckmateT1SuicideBlock();
  testCheckmateShortDiagonal();
  testCheckmateT2MultipleThreats();
  testCheckmateT3AllSuicide();
  testNotCheckmate();

  console.log("\n" + "=".repeat(60));
  console.log(`Results: ${passed} passed, ${failed} failed`);
  console.log("=".repeat(60));

  if (failed > 0) {
    process.exit(1);
  }
}

runTests();
