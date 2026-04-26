# Brainstorming: Zic-Zac-Zoe Points Variant

## Tactical Rules Ideas

### "Greedy Local Max" Algorithm
Identify all positions that are adjacent to at least one occupied cell. For each, calculate the points award if Player X plays there, then repeat for Player O. The AI plays the highest point position (either taking its own best move or blocking the opponent's best move). 

**Tie-breaking:**
- In a tie between self-gain and blocking, prioritize self-gain.
- In a tie between multiple equal-value positions for the same side, pick one at random.

**Critique:**
This is only a 1-move lookahead. It doesn't address multi-move strategy. It also won't allow the AI to create its own gaps to fill (planning for bridges). Since the user will likely use a gap strategy, and bridges are worth more than appends, the AI might spend the whole game just blocking the user, resulting in a frustrating experience.

---

