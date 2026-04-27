# Actionable Tasks: "Claim" Variant Implementation

## Phase 1: Configuration & State Infrastructure
- [ ] **Task 1.1: Config Manager Implementation**
  - Create `web/public/game_config.json` with all modes and settings.
  - Implement fetch/load logic in `web/src/main.ts` bootstrap.
  - Implement validation: filter for `active: true` and set `state.configError` if != 1.
- [ ] **Task 1.2: State Object Expansion**
  - Update `State` interface in `web/src/main.ts` with: `pendingMove`, `awaitingDecision`, `movesRemainingX/O`, `activeModeId`, `modeSettings`, and `configError`.
  - Initialize counters in `newGame()` based on loaded configuration.

## Phase 2: Core State Machine & Logic
- [ ] **Task 2.1: Interaction Lock & Pending State**
  - Update `handleCellClick` to guard against clicks during `awaitingDecision`.
  - Store `pendingMove`, set `awaitingDecision = true`, and decrement active player's move count.
- [ ] **Task 2.2: Scoring Engine Update**
  - Refactor `getScoringData` in `web/src/game.ts` to return both `totalScore` and `involvedCells`.
- [ ] **Task 2.3: Claim & Pass Handlers**
  - Implement `handleClaim()`: Calculate points, clear current player's tokens (if not board full), apply `.evaporate` class, and finalize turn.
  - Implement `handlePass()`: Finalize turn with 0 points.
- [ ] **Task 2.4: Termination Logic**
  - Update `checkGameOver()` to branch by `activeModeId` (Point Cap, Move Cap, Point Lead).
  - Implement "Board Full" auto-claim logic (must not clear tokens).

## Phase 3: UI Transformation
- [ ] **Task 3.1: Layout Restructuring**
  - Move `#scoreboard` from header to board anchor in `index.html`.
  - Implement `#mode-display` in header.
  - Add `#config-error` overlay logic and board visibility toggle (`display: none`).
- [ ] **Task 3.2: Control Button Update**
  - Replace setting buttons with `btn-claim` and `btn-pass`.
  - Wire buttons to `handleClaim()` and `handlePass()`.
- [ ] **Task 3.3: Visual Feedback & Animations**
  - Implement `.btn-blink` CSS animation for decision state.
  - Implement `.evaporate` CSS transition for token clearing.
  - Update `renderScoreboard()` to show `M: N` and `+N` indicators.
  - Ensure persistent scoring highlights on empty cells.

## Phase 4: AI & Refinement
- [ ] **Task 4.1: AI Decision Engine**
  - Implement `getDecision(points)` in `web/src/ai.ts` with linear probability scaling.
  - Integrate decision phase into the AI turn flow in `web/src/main.ts`.
- [ ] **Task 4.2: Development Tooling & QA**
  - Update console logger to include Decision type and Moves Left.
  - **QA Task:** Verify all 3 win conditions via config changes.
  - **QA Task:** Verify board-full auto-termination.
