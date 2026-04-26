# Zic-Zac-Zoe: Points Variant Specification

This update transforms Zic-Zac-Zoe from a "Connect N" win/loss game into a full-board strategic scoring game.

## Core Rules Changes
- **No Early Termination:** The game no longer ends when someone connects 3 or 4.
- **Victory Condition:** Play continues until all 36 cells are filled. The player with the highest cumulative score wins.
- **Turn Order:** Player X and Player O alternate moves as before.

## Scoring Mechanics
Every move is scored based on the sequences it creates or extends across four axes (Horizontal, Vertical, and two Diagonals).

### 1. Line Scoring
For each axis where the move creates a contiguous line of length $L > 1$:
- **Base Score:**
  - If $L = 3$: **0 points** (The "Danger Zone" penalty).
  - Otherwise: **$L$ points**.
- **Bridge Bonus:** If the move connects two existing segments (e.g., `X _ XX`), the Base Score for that line is **doubled**.

*Note: If a move is placed in isolation ($L = 1$ in all directions), it scores **1 point**.*

### 2. Multi-Line Multiplier
If a move contributes to $N$ distinct lines where $L > 1$, the scores are combined as follows:
- **Total Move Score = (Sum of all individual Line Scores) × N**

*This multiplier applies even if a line scores 0 points (length 3), rewarding the complexity of the intersection.*

## UI Enhancements
- **Header Update:** Remove the static rules text from the top of the interface.
- **Dynamic Scoreboard:** Implement a prominent, single-line scoreboard at the top that fits within the space vacated by the rules text.
  - **Layout:** `X:    0  -  O:    0`
  - **Spacing Logic:** The 4-space gap acts as a buffer for scores up to 4 digits (e.g., `X: 1234`), ensuring that even at maximum score, at least one space remains between the colon and the first digit to prevent layout shifting.
  - **Player X Score:** Displayed in the color of Player X's tokens.
  - **Player O Score:** Displayed in the color of Player O's tokens.

## AI & Opponent
- **Existing Model:** The game will utilize the current AlphaZero-style neural network.
- **Known Limitation:** The AI was trained on the "Connect 4/Avoid 3" win/loss condition. While it will still prioritize making lines and avoiding triples, its value function will be misaligned with the new cumulative points system.
- **AI Search Constraint:** Since early terminal states (4-in-a-row) are removed, the MCTS search depth or time-per-move must be constrained to prevent the AI from over-calculating deep into the full-board state space.
- **Future Work:** This serves as a "playable prototype" until a new model can be trained on the scoring-based rewards.

---

## Scoring Examples

| Pattern | Logic | Calculation | Total |
| :--- | :--- | :--- | :--- |
| `_` | Lone Tile | Anchor | **1** |
| `X _` | Append to 2 | $L=2$ | **2** |
| `XX _` | Append to 3 | $L=3 \rightarrow 0$ | **0** |
| `X _ X` | Bridge to 3 | $(L=3 \rightarrow 0) \times 2$ | **0** |
| `XXX _` | Append to 4 | $L=4$ | **4** |
| `X _ XX` | Bridge to 4 | $L=4 \times 2$ | **8** |
| `XX _ XX` | Bridge to 5 | $L=5 \times 2$ | **10** |
| `XXX _ XX` | Bridge to 6 | $L=6 \times 2$ | **12** |

### Multi-Line Example: The "T-Bone"
If playing `X` at the intersection of a horizontal `XX _ X` and a vertical `X _`:
- **Horizontal Line:** Bridge to 4 = **8 pts**
- **Vertical Line:** Append to 2 = **2 pts**
- **Multiplier:** $N=2$
- **Total:** $(8 + 2) \times 2 = \mathbf{20\ points}$
