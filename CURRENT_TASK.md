# Current Task: Points-Award Cell Highlighting

## Status
- **State:** 2/3 Complete
- **Last Updated:** 2026-04-26

## Tasks
- **Sub-Task 1: Logic Refactor**
  - **Status:** complete
  - Refactor `game.ts` to use a shared `getAxisReports` core.
  - Update `calculateMoveScore` to utilize the new core.
  - Verify scoring parity via existing test suite/parity check.
- **Sub-Task 2: UI Integration**
  - **Status:** complete
  - Implement `getScoringIndices` in `game.ts`.
  - Update `GameState` in `main.ts` to track highlights.
  - Implement rendering and CSS for `.scoring-highlight`.
  - Verify persistence through final game state.
- **Sub-Task 3: QA**
  - **Status:** pending
  - Final verification of feature behavior in browser.

---

## Engineering Specification

### 1. Goal
Provide dynamic visual feedback by highlighting all board cells involved in a move's point award. This includes the move itself and all pieces forming "productive" lines (lines contributing $>0$ points), utilizing a shared logic core to ensure scoring and visualization are always in sync.

### 2. Architecture & Logic Layer (`game.ts`)
To prevent logic drift and ensure performance, the scoring engine will be refactored to use a shared internal scanner.

**Internal Data Structure**
- `AxisReport`: A private interface containing the results for a single axis scan.
  - `length: number`: Total consecutive pieces including the new move.
  - `indices: number[]`: All board indices in this continuous line.
  - `isBridge: boolean`: True if the move connected two existing segments.

**Refactored Logic Flow**
- **`getAxisReports` (Internal):** Performs the directional scans (horizontal, vertical, two diagonals). It returns an array of `AxisReport` objects.
- **`calculateMoveScore` (Public):** 
  - Iterates through `AxisReport`s.
  - Applies scoring rules ($L=1 \to 1pt$, $L=3 \to 0pt$, else $L$ pts).
  - Calculates the **Productive Multiplier** ($N$ = count of axes where points $> 0$).
  - Returns the final calculated score.
- **`getScoringIndices` (Public):**
  - Calls `getAxisReports`.
  - Filters reports to include only those where the axis score would be $> 0$ (i.e., $L \neq 3$ and $L > 1$).
  - If the move is a **Lone Tile** (score = 1), includes only the `moveIndex`.
  - Collects all indices from productive reports into a `Set<number>` for deduplication.
  - Returns a unique array of indices.

### 3. State Management (`main.ts`)
**`GameState` Update**
- Add `scoringHighlights: number[]` to the state to persist the indices of the current point award.

**Lifecycle Management**
- **Trigger:** Upon any move (Human or AI), `state.scoringHighlights` is updated using the results of `getScoringIndices`.
- **Clearance:** `state.scoringHighlights` is reset to an empty array only when `newGame()` is called.
- **Persistence:** Highlights from the current move remain visible until the next move is made. Highlights from the 36th (final) move persist through the "Game Over" state to allow the player to review the final point-scoring play.

### 4. View & Styling (`main.ts` & `style.css`)
**Rendering Logic (`renderBoard`)**
- As cells are rendered, if a cell index exists in `state.scoringHighlights`, apply the `.scoring-highlight` class.
- **Visual Precedence:** The `.scoring-highlight` class must be defined after `.last-move-x/o` in the CSS to ensure the scoring highlight's border color takes precedence.

**CSS Definition**
- **Class:** `.scoring-highlight`
- **Property:** `border-color` set to the player's primary color (Cyan for X, Yellow for O).
- **Temporal Feedback:** Apply the `crux-blink` animation (1s period, step-end) to the border. This pulsing effect provides a necessary visual distinction between a "scoring piece" and a standard occupied cell.

### 5. Function Signatures

**`game.ts`**
- `function getAxisReports(board: BoardState, moveIndex: number, player: Player): AxisReport[]` (Private/Internal)
- `export function calculateMoveScore(board: BoardState, moveIndex: number, player: Player): number`
- `export function getScoringIndices(board: BoardState, moveIndex: number, player: Player): number[]`

**`main.ts`**
- `function updateHighlights(index: number, player: Player): void` (Internal helper to update state and trigger re-render)

---

## Template
```markdown
# Current Task: [Task Name]

## Status
- **State:** 0/N Complete
- **Last Updated:** [Date]

## Tasks
- **Sub-Task 1: [Description]**
  - **Status:** pending
  - [Details]
- **Sub-Task N: QA**
  - **Status:** pending
  - Final verification of feature behavior.

---

## Engineering Specification
[Detailed Spec]
```
