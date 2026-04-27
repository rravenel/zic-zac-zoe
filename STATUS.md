# Project Status: Points Variant Prototype

## Current State
The project has successfully transitioned from a "Connect N" win/loss game to a strategic **Points Variant**. A playable prototype is live in the `web/` directory.

## Technical Context
### 1. Active Game Variant: Points
- **Grid:** 6x6.
- **Termination:** Full-board (36 moves).
- **Scoring Engine:**
  - $L=1$: 1 point.
  - $L=3$: 0 points.
  - Other lengths: $L$ points.
  - **Bridge Bonus:** 2x points if a move connects two existing segments.
  - **Productive Multiplier:** Total score is multiplied by $N$, where $N$ is the count of axes producing $>0$ points.
- **Winner:** Determined by highest cumulative score at termination.

### 2. AI Architecture
- **State:** The Neural Network and Tactical Rule layers are currently **disconnected**.
- **Behavior:** The AI selects purely random legal moves. This provides a neutral baseline for play-testing the scoring mechanics.

### 3. Visual Feedback
- **Synchronized Blinking:** The most recent move for both X and O pulses in unison using a global CSS animation clock.
- **Persistent Scoring Highlights:** All pieces involved in a move's point award are highlighted with a steady border until that player's next move.
- **Game Over Persistence:** The final move's scoring highlights remain visible through the "Game Over" state.

### 4. Developer Tools (Dev Mode Only)
- **Console Logger:** Verbose, color-coded turn snapshots in the browser console (Move #, Player, Points, ASCII board).
- **Parity Check:** Automatic validation of the TypeScript scoring engine against the Python "Golden Set" on page load.

## Handover Notes for Next Session
- **Scoring Audit:** The "Productive Multiplier" logic was recently refined to ensure non-scoring lines (length 3) do not inflate the multiplier.
- **Pending Tasks:** None. The initial build phase is 100% complete.
- **Brainstorming:** See `BRAINSTORM.md` for proposed tactical algorithms for the next AI iteration.
