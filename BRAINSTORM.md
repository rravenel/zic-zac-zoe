# Brainstorming: Zic-Zac-Zoe Points Variant

## Tactical Rules Ideas for Connect Version

### "Greedy Local Max" Algorithm
Identify all positions that are adjacent to at least one occupied cell. For each, calculate the points award if Player X plays there, then repeat for Player O. The AI plays the highest point position (either taking its own best move or blocking the opponent's best move). 

**Tie-breaking:**
- In a tie between self-gain and blocking, prioritize self-gain.
- In a tie between multiple equal-value positions for the same side, pick one at random.

**Critique:**
This is only a 1-move lookahead. It doesn't address multi-move strategy. It also won't allow the AI to create its own gaps to fill (planning for bridges). Since the user will likely use a gap strategy, and bridges are worth more than appends, the AI might spend the whole game just blocking the user, resulting in a frustrating experience.

---

## Tactical Rules Ideas for Claim Version

### Heuristics Ideas

**Points Per Piece**
(PPP)

For a given possible move, calculate the score, count the scoring pieces and divide.

X         X
X _  -->  X X  = 8 points, 8/3 = 2.67

X X _ X  -->  X X X X = 8 points, 8/4 = 2

X         X
X _  -->  X X = 18, 18/3 = 6
X         X

### Things we can do:

- Evaluate cells adjacent to existing pieces for points claim
  - Next move tactic
- Evaluate straight lines from existing pieces for intersection with other lines
  - 
- Evaluate straight lines between existing pieces for gap filling

**Defense**
- Evaluate all cells adjacent opponent pieces for next move points potential
  - Define a blocking threshold - don't block if it's not worth it
- Limit evaluation to straight line positions
  - Define length threshold - don't block <3

**Offense**
- Mirror defensive evaluation for own pieces
- Evaluate own gap


---