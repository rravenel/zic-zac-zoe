# Tactical Heuristic AI Engine

## Status
- **State:** ✅ Complete
- **Last Updated:** 2026-04-26

## Tasks
- **Sub-Task 1: Heuristic Engine Core**
  - **Status:** complete
  - Implement `getHeuristicMove` in `web/src/ai.ts`. Includes proximity search (8-neighbors), dual-score evaluation, and 4-token threshold filtering.
- **Sub-Task 2: AI Execution Logic**
  - **Status:** complete
  - Refactor `makeAIMove` in `web/src/main.ts` to utilize the heuristic results. Implement intent-based branching: always Pass for "block", use probability for "build".
- **Sub-Task 3: Automated Logic Testing**
  - **Status:** complete
  - Create `web/src/test-heuristic.ts` and implement the 6 test scenarios (Empty Board, Threshold, Build, Block, Tie-Break, Proximity).
- **Sub-Task 4: QA & Refinement**
  - **Status:** complete
  - Final play-testing verification and cleanup of unused random fallback code if necessary.

---

## Feature Specification

### 1. Intent
The goal is to replace the current random move selection with a strategic heuristic that prioritizes high-value point awards and defensive blocks. This will make the AI a more engaging opponent and drive the board toward more complex, populated states.

### 2. Search Logic
For both players (AI and Human), the engine will evaluate potential moves based on proximity to existing tokens:
- **Scope:** Iterate through every token currently on the board.
- **Candidates:** For each token, identify all adjacent empty cells (8 directions: Horizontal, Vertical, Diagonal).
- **Consolidation:** Create a unique set of all empty cells that are adjacent to at least one piece.

### 3. Evaluation & Filtering
For each candidate cell `j` in the unique set:
1. **Score_AI(j):** Calculate the points awarded if the AI plays and claims at cell `j`.
2. **Score_Human(j):** Calculate the points awarded if the Human plays and claims at cell `j`.
3. **Threshold Filter:** Any evaluation (AI or Human) that involves **fewer than 4 tokens** (including the candidate move itself) is ignored and treated as 0 points.
4. **Move Value ($V_j$):** The heuristic value of the move is the maximum of the two filtered scores: `V_j = max(Score_AI_filtered(j), Score_Human_filtered(j))`.

### 4. Move Intent & Decision Handling
Once the highest value move `j` is selected, the AI categorizes its intent to determine its next action (Claim or Pass):

- **Intent: BLOCK**
  - **Category:** The move was selected because `Score_Human_filtered(j) >= Score_AI_filtered(j)` AND `Score_Human_filtered(j) > 0`.
  - **Rule:** **Always PASS.** The AI must never claim a blocking move, as doing so would remove the token that is obstructing the Human player's high-value line.
- **Intent: BUILD**
  - **Category:** The move was selected because `Score_AI_filtered(j) > Score_Human_filtered(j)` OR the AI used the random **Fallback** rule.
  - **Rule:** **Probabilistic Claim.** The AI uses its existing decision engine (10% to 99% scale) to decide whether to claim the points.

### 5. Selection Rules
The AI will select its move from the candidate set based on these priorities:
1. **Highest Value:** Choose the move with the highest $V_j$.
2. **Tie-Break (Human Block):** If an AI-scoring move and a Human-blocking move have the identical $V_j$, the AI **must prioritize the BLOCK intent**.
3. **Tie-Break (Random):** If multiple moves share the highest $V_j$ and the same intent, select one at random.
4. **Fallback:** If no moves meet the 4-token threshold, select a random legal move from the entire board (Categorized as **BUILD** intent).

---

## Engineering Specification

### 1. Data Structures
Update `AIResult` in `web/src/ai.ts` to include move intent:
```typescript
export type MoveIntent = "build" | "block";

export interface AIResult {
  move: number;
  intent: MoveIntent;
  guardrailWeight: number; // For future NN use
}
```

### 2. Heuristic Search (`web/src/ai.ts`)
Implement `getHeuristicMove(board: BoardState, aiPlayer: Player): AIResult | null`:
- **Step 1: Identify Candidates**
  - Iterate `board`. For every `cell !== Empty`, check its 8 neighbors.
  - Add empty neighbors to `uniqueCandidates: Set<number>`.
- **Step 2: Dual Evaluation**
  - For each `idx` in `uniqueCandidates`:
    - `dataAI = getScoringData(board, idx, aiPlayer)`
    - `dataHuman = getScoringData(board, idx, getOpponent(aiPlayer))`
    - `scoreAI = dataAI.involvedCells.length >= 4 ? dataAI.totalScore : 0`
    - `scoreHuman = dataHuman.involvedCells.length >= 4 ? dataHuman.totalScore : 0`
- **Step 3: Ranking**
  - Store candidates in an array of objects: `{ idx, scoreAI, scoreHuman, value: max(scoreAI, scoreHuman) }`.
  - Filter for `value > 0`.
- **Step 4: Selection**
  - Sort by `value` (descending).
  - If top values are tied, prioritize those where `scoreHuman >= scoreAI`.
  - From the filtered top set, select one randomly.

### 3. Integration (`web/src/main.ts`)
Refactor `makeAIMove()`:
- Call `getHeuristicMove(state.board, currentPlayer)`.
- If `null`, fall back to random move from `getLegalMoves()`.
- **Action Execution:**
  - Place token.
  - If `result.intent === "block"`, call `handlePass()` after `AI_MOVE_DELAY`.
  - If `result.intent === "build"`, use `getAIDecision(points)` to call either `handleClaim()` or `handlePass()`.

### 4. Helper Functions
Ensure `getOpponent` is available in `ai.ts` (import from `game.ts`).

### 5. Test Coverage
Create a new test file `web/src/test-heuristic.ts` to verify the AI's tactical decisions:
- **Scenario: Empty Board**
  - Verify `getHeuristicMove` returns `null` (triggering fallback).
- **Scenario: Point Threshold**
  - Setup a line of length 2 for AI. Verify the heuristic ignores the 3rd piece (fails the 4-token threshold).
- **Scenario: High Value Build**
  - Setup a line of length 3 for AI. Verify the heuristic selects the 4th piece and returns `intent: "build"`.
- **Scenario: High Value Block**
  - Setup a line of length 3 for Human. Verify the heuristic selects the blocking cell and returns `intent: "block"`.
- **Scenario: Tie-Break Priority**
  - Setup a board where AI has a 10-point move and Human has a 10-point move. Verify the heuristic selects the **BLOCK** move.
- **Scenario: Proximity Search**
  - Verify that cells non-adjacent to any piece are not included in the primary candidate pool.
