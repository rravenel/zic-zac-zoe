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

## 2. State Management & Stats
- **Frontend State (`web/src/main.ts`):** 
  - Extend `GameState` with `playerXScore: number` and `playerOScore: number`.
  - **Stats:** Retain binary Win/Loss tracking in `localStorage`. A "Win" is simply having the higher score at the terminal state.
- **Backend Model (`game.py`):**
  - Update `Board` class to support score accumulation.

## 3. UI and Rendering (`web/src/main.ts`)
- **Header Refactor:** Replace existing rules container with the single-line scoreboard.
- **Display Update:** Update the scoreboard string on every move. 
  - **Formatting:** Use a 5-character right-aligned field for each score (e.g., `"    0"`, `" 1234"`). This preserves the 4-space visual gap at start and ensures at least one space remains next to the colon if the score reaches 4 digits.
  - **Stability:** Use monospaced font elements to prevent layout shift.
- **Visuals:** Disable any existing "last move" or "winning line" highlighting that relies on the old terminal state logic to simplify the prototype transition.

## 4. AI Compatibility (`web/src/ai.ts`)
- **MCTS Update:** Ensure the search tree can handle the full-board depth. Limit search time/depth to keep AI moves responsive (~500ms).
- **Inference:** The model will continue to output move probabilities; the game controller will simply apply those moves regardless of the model's "value" estimation of the board.
