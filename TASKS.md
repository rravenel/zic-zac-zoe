# Actionable Tasks: Points Variant Implementation

## Phase 1: Python Backend (Scoring Logic & Model)
- [ ] **Task 1.1:** Implement `calculate_move_score` in `game.py`.
  - **Approach:** Use "Directional Rays" (scan outward from move) for efficiency.
  - **Base Score:** $L=3 \rightarrow 0$; $L=1 \rightarrow 1$ (if $N=0$); otherwise $L$.
  - **Bridge Bonus:** 2x Base Score if neighbors exist on both sides of an axis.
  - **Multiplier $N$:** Count of **distinct axes** (max 4) where $L > 1$.
- [ ] **Task 1.2:** Refactor `check_result` and `check_result_fast` in `game.py`.
  - Return terminal only when `board.is_full()`.
- [ ] **Task 1.3:** Update `Board` class in `game.py`.
  - Add `score_x` and `score_o` to initialization and copies.
  - **CRITICAL:** Update `hash()` and `__eq__` to include scores so identical boards with different scores are distinct states.
- [ ] **Task 1.4:** Add unit tests in `test_scoring_logic.py` verifying all `FEATURE_SPEC` examples (The "Golden Set").
- [ ] **Task 1.5:** Update `test_game.py`.
  - Add tests for score-based state identity and accumulation.

## Phase 2: Web Logic (TypeScript)
- [ ] **Task 2.1:** Implement `calculateMoveScore` in `web/src/game.ts` (1:1 Port of Python).
- [ ] **Task 2.2:** Update `checkResult` and `checkResultFast` to terminal-on-full.
- [ ] **Task 2.3:** Update `GameState` interface in `web/src/main.ts` for scores.
- [ ] **Task 2.4:** Update move handling in `web/src/main.ts` to accumulate scores.
- [ ] **Task 2.5:** Verify AI inference loop handling of non-terminal states.
- [ ] **Task 2.6:** Update `resetGame` logic in `web/src/main.ts` to zero out scores.
- [ ] **Task 2.7:** Add unit tests in `web/src/test-scoring.ts`.

## Phase 3: UI & Rendering
- [ ] **Task 3.1:** Replace `.subtitle` in `web/index.html` with a scoreboard div (`#scoreboard`).
- [ ] **Task 3.2:** Implement `updateScoreboardDisplay` in `web/src/main.ts`.
  - **DoD:** Format `X: [Score] - O: [Score]`. The `[Score]` must be a **right-aligned 5-character field** (e.g., `padStart(5, ' ')`). This preserves the initial 4-space gap and ensures 1 space remains if the score reaches 4 digits.
- [ ] **Task 3.3:** Disable "winning line" and terminal state highlighting in `web/src/main.ts`.
- [ ] **Task 3.4:** Update `<meta name="description">` in `web/index.html` with new scoring rules.

## Phase 4: AI & Finalization
- [ ] **Task 4.1:** Update `makeAIMove` in `web/src/main.ts` to implement random move selection (bypassing NN and tactical rules).
- [ ] **Task 4.2:** Update game end logic in `web/src/main.ts`.
  - **Stats:** In case of a tie, neither `won` nor `lost` stats are incremented.
