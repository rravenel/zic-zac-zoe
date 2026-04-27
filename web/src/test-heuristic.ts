/**
 * Tests for Tactical Heuristic AI logic.
 * Run with: npx vite-node src/test-heuristic.ts
 */

import { Player, BOARD_SIZE } from "./game";
import { getHeuristicMove } from "./ai";

type BoardState = number[];

// =============================================================================
// Test Utilities
// =============================================================================

function createEmptyBoard(): BoardState {
  return new Array(BOARD_SIZE * BOARD_SIZE).fill(Player.Empty);
}

function boardFromString(s: string): BoardState {
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
    passed++;
    // console.log(`  ✓ ${message}`);
  } else {
    failed++;
    console.error(`  ✗ FAILED: ${message}`);
  }
}

function test(name: string, fn: () => void): void {
  console.log(`Running: ${name}...`);
  fn();
}

// =============================================================================
// Test Scenarios
// =============================================================================

test("Scenario: Empty Board", () => {
  const board = createEmptyBoard();
  const result = getHeuristicMove(board, Player.X);
  assert(result === null, "getHeuristicMove should return null on an empty board");
});

test("Scenario: Point Threshold (4-token rule)", () => {
  // Setup: X has 2 pieces. A 3rd piece would form a line of 3 (0 points).
  // Even if it awarded points, it's < 4 tokens total, so heuristic should ignore.
  const board = boardFromString(`
    . . . . . .
    . X X . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const result = getHeuristicMove(board, Player.X);
  // adjacent cells (like [1, 3]) should be ignored because they don't meet 4-token threshold
  assert(result === null, "Heuristic should ignore moves not meeting 4-token threshold");
});

test("Scenario: High Value Build", () => {
  // Setup: X has 3 pieces. A 4th piece forms a line of 4 (4 points).
  // Total tokens involved = 4. Meets threshold.
  const board = boardFromString(`
    . . . . . .
    . X X X . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const result = getHeuristicMove(board, Player.X);
  assert(result !== null, "Heuristic should find a move for line of 4");
  if (result) {
    // It might pick 6 or 10 (end of line, 4 pts) or an intersection move like 14 (18 pts)
    assert(result.value >= 4, `Heuristic should find a move worth at least 4 pts (got ${result.value})`);
    assert(result.intent === "build", "Intent should be 'build'");
  }
});

test("Scenario: High Value Block", () => {
  // Setup: Human (O) has 3 pieces. AI (X) should block the 4th cell.
  const board = boardFromString(`
    . . . . . .
    . O O O . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const result = getHeuristicMove(board, Player.X);
  assert(result !== null, "Heuristic should find a blocking move");
  if (result) {
    assert(result.value >= 4, `Heuristic should find a block worth at least 4 pts (got ${result.value})`);
    assert(result.intent === "block", "Intent should be 'block'");
  }
});

test("Scenario: Tie-Break Priority (Human Block wins)", () => {
  // Setup: AI (X) has a move worth 4 pts, Human (O) has a move worth 4 pts.
  // Both are 4 tokens. Heuristic should prioritize the block.
  const board = boardFromString(`
    X X X . . .
    . . . . . .
    O O O . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const result = getHeuristicMove(board, Player.X);
  assert(result !== null, "Heuristic should find a move");
  if (result) {
    assert(result.intent === "block", `Heuristic should prioritize blocking Human (got ${result.intent})`);
  }
});

test("Scenario: Proximity Search", () => {
  // Setup: One piece at [0, 0].
  // Heuristic should only consider [0, 1], [1, 0], [1, 1].
  // (In reality, they will all fail the threshold filter, but we test candidates implicitly).
  // Actually, we can test that it doesn't crash and returns null if no valid moves found.
  const board = boardFromString(`
    X . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
    . . . . . .
  `);
  const result = getHeuristicMove(board, Player.X);
  assert(result === null, "Proximity search should result in null if no move meets threshold");
});

// =============================================================================
// Summary
// =============================================================================

console.log(`\nTests finished: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
else process.exit(0);
