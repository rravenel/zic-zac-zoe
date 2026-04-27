# Engineering Specification: Points Variant

This document outlines the technical implementation details for transitioning Zic-Zac-Zoe to a points-based system.

## 1. Game Logic Refactoring
- **Files:** `game.py` (Python), `web/src/game.ts` (TypeScript)
- **Scoring Engine:** Implement `calculate_move_score(board, move, player)`.
  - **Logic:** For each of the 4 axes, calculate length $L$.
  - **Base Score:** $L=3 \rightarrow 0$; $L=1 \rightarrow$ (see below); otherwise $L$.
  - **Bridge Bonus:** 2x Base Score if $L_1 > 0$ AND $L_2 > 0$.
  - **Multi-Line:** (Sum of Line Scores) × N, where N is count of axes with $L > 1$.
  - **Lone Tile:** If N=0 (no neighbors), the move scores exactly **1 point**.
- **End State:** Modify `check_result` to return terminal only when `board.is_full()`. Winner is determined by final score comparison.
- **Testing:**
  - Create `test_scoring_logic.py` and `web/src/test-scoring.ts` using a **Shared Golden Set** of test cases (including lone-tile, bridges, and multi-line intersections) to ensure 1:1 parity between engines.
  - Update `test_game.py` to reflect the fixed 36-move episode length and point-based terminal rewards.

## 2. State Management & Stats
- **Frontend State (`web/src/main.ts`):** 
  - Extend `GameState` with `playerXScore: number` and `playerOScore: number`.
  - **Stats:** Retain binary Win/Loss tracking in `localStorage`. A "Win" is simply having the higher score at the terminal state.
- **Backend Model (`game.py`):**
  - Update `Board` class to support score accumulation.
- **Testing:**
  - Update unit tests for `Board` (in `test_game.py`) to verify `score_x` and `score_o` are correctly tracked, copied, and included in hash/equality.
  - Verify stats update logic in the frontend, specifically asserting that a tie (equal scores) correctly results in no change to won/lost counters.

## 3. UI and Rendering (`web/src/main.ts`)
- **Header Refactor:** Replace existing rules container with the single-line scoreboard.
- **Display Update:** Update the scoreboard string on every move. 
  - **Formatting:** Use a 5-character right-aligned field for each score (e.g., `"    0"`, `" 1234"`). This preserves the 4-space visual gap at start and ensures at least one space remains next to the colon if the score reaches 4 digits.
  - **Stability:** Use monospaced font elements to prevent layout shift.
- **Visuals:** Disable any existing "last move" or "winning line" highlighting that relies on the old terminal state logic to simplify the prototype transition.
- **Testing:**
  - Manually verify layout stability in the browser.
  - Ensure UI-based "Game Over" triggers correctly only when the board is full.

## 4. AI Compatibility (`web/src/main.ts`)
- **Move Logic:** Update `makeAIMove` in `web/src/main.ts` to bypass the existing `getAIMove` and `getRulesMove` logic.
- **Implementation:** The AI should select a random index from the list returned by `getLegalMoves(state.board)`.
- **Disconnection:** The existing `ai.ts` and `rules-ai.ts` modules should remain unused by the game controller for this prototype phase.
- **Testing:**
  - Verify that the AI selects a valid legal move on every turn.
  - Ensure the AI continues to play until the board is completely full.

## General Testing Mandates
- **Fidelity:** All new logic must have corresponding tests.
- **Cleanup:** Tests for removed code/logic must be deleted immediately.
- **Maintenance:** Stale or incorrect tests are not permitted. The test suite must remain clean and 100% green.
