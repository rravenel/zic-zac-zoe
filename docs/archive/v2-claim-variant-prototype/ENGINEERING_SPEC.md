# Engineering Specification: "Claim" Variant Implementation

This document details the technical implementation plan for the "Claim" game mechanic in Zic-Zac-Zoe.

## 1. Configuration Management
- **Source:** `web/public/game_config.json`
- **Interface:**
  ```typescript
  interface GameConfig {
    modes: {
      id: string;
      active: boolean;
      settings: Record<string, number>;
    }[];
  }
  ```
- **Initialization:**
  - Load via `fetch()` during the application bootstrap phase.
  - **Validation:** 
    - Filter `modes` where `active === true`.
    - If `activeCount !== 1`, set `state.configError = "CONFIGURATION ERROR: Multiple or No Active Modes"`.
  - **Injection:** Store the active mode and its settings in the global `state` object.
  - **UI Impact:** If `state.configError` is present, the `.board` container must be set to `display: none` and the error overlay shown.

## 2. State Machine Expansion
The game state must transition through a "Pending Decision" phase.

### Updated `GameState` Object
```typescript
interface State {
  // ... existing properties
  pendingMove: { r: number, c: number } | null;
  awaitingDecision: boolean;
  movesRemainingX: number;
  movesRemainingO: number;
  activeModeId: string;
  modeSettings: Record<string, number>;
  configError: string | null;
}
```

### Flow Control
0. **`newGame()`**:
   - Initialize `state.movesRemainingX` and `state.movesRemainingO` from the active mode's `limit_per_side` setting.
1. **`handleCellClick(r, c)`**:
   - If `state.awaitingDecision`, return (ignore clicks).
   - Place token: `state.board[r][c] = state.currentPlayer`.
   - Update state: `state.pendingMove = { r, c }`, `state.awaitingDecision = true`.
   - Decrement moves: `state.movesRemainingX/O--`.
   - Trigger `renderBoard()` and `updateUI()`.
2. **`handleClaim()`**:
   - Calculate points and contributing cells for `state.pendingMove`.
   - Update score: `state.score += points`.
   - Clear cells: 
     - If `!isBoardFull()`, set `state.board[r][c] = null` for every matching cell in the contributing lines (current player only).
     - Apply `.evaporate` CSS class to cells being cleared to trigger fade-out animation.
   - Finalize: `state.awaitingDecision = false`, `state.pendingMove = null`, call `endTurn()`.
3. **`handlePass()`**:
   - Record `+0` for the turn.
   - Finalize: `state.awaitingDecision = false`, `state.pendingMove = null`, call `endTurn()`.

## 3. UI Implementation

### Layout Changes (`index.html` & `style.css`)
- **Header:** Replace `#scoreboard` with `#mode-display` (centered text).
- **Board Area:**
  - Inject `#scoreboard` directly above the `.board` container.
  - Add `#config-error` overlay (hidden by default, shown if `state.configError` is set).
- **Controls:**
  - Remove `btn-mode`, `btn-player`, `btn-difficulty`.
  - Add `btn-claim` and `btn-pass` to `.controls-left`.

### Visual Feedback
- **Blinking:** Add `.btn-blink` CSS class using `@keyframes` for opacity pulse. Apply to `btn-claim` and `btn-pass` when `state.awaitingDecision` is true.
- **Evaporation Animation:** Add `.evaporate` CSS class with `opacity: 0` and a `transition: opacity 0.5s ease-out`.
- **Scoreboard:**
  - Modify `updateScoreboard()` to display `M: N` (Left-justified) and `+N` (Right-justified) in the per-turn row.
- **Highlights:** Ensure `scoringHighlights` persist on empty cells until the next turn of that player.

## 4. AI Probabilistic Logic (`ai.ts`)
The AI must now handle the decision phase.
- **`getDecision(points: number): 'claim' | 'pass'`**
  - If `points <= 1`, probability $P = 0.5$.
  - If `points >= 50`, probability $P = 0.99$.
  - Linear interpolation for values in between.
  - Use `Math.random() < P` to return decision.

## 5. Scoring Engine Adjustments (`game.ts`)
- **`getScoringData(board, move)`**: Must return an object containing:
  - `totalScore`: The move's calculated score.
  - `involvedCells`: Array of `{r, c}` coordinates contributing to the score across all axes.

## 6. Termination & Win Conditions
Update `checkGameOver()` to branch based on `state.activeModeId`:
- **`point_cap`**: Winner if `score >= target`.
- **`move_cap`**: Game ends when `movesRemainingX === 0 && movesRemainingO === 0`.
- **`point_lead`**: Winner if `Math.abs(scoreX - scoreO) >= margin`. Safety valve: triggers `move_cap` logic if moves hit zero first.
- **Board Full:** If no empty cells remain:
  - Automatically call `handleClaim()` for the current player.
  - Skip board clearing (logic inside `handleClaim` should check `isBoardFull`).
  - Set `state.gameOver = true`.

## 7. Development Tools
- **Console Log:** Update turn snapshots to include "Decision: Claim/Pass" and "Moves Left: X/O".
