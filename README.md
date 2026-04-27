# Zic-Zac-Zoe: Claim Variant

A strategic twist on tic-tac-toe where players manage resources and points through a unique "Claim" mechanic. This project implements a neural network AI for a 6x6 variant of Zic-Zac-Zoe.

## Game Rules (Claim Variant)

- **6x6 grid**
- **Dynamic Board:** The board doesn't just fill up; cells are cleared when points are claimed.
- **The "Claim" Mechanic:** After placing a token, a player must choose to either **Claim** the points generated (clearing the contributing tokens from the board) or **Pass** (leaving the tokens but scoring 0).
- **Tactical Clearing:** Lines of 3 award 0 points, but can be claimed to clear board space.
- **X moves first.**

### Scoring Mechanics
1. **Line Length (L):**
   - $L=1$ (Isolated): **1 point**.
   - $L=3$ (Penalty): **0 points**.
   - Other lengths: **L points**.
2. **Bridge Bonus:** If a move connects two existing segments, the point value for that line is **doubled**.
3. **Productive Multiplier (N):** The total move score is multiplied by the number of axes that produced points (>0).

### Winning Conditions
The game supports multiple configurable modes (see `web/public/game_config.json`):
- **Point Cap:** First player to reach a target score wins.
- **Move Cap:** Each player has a limited number of moves; highest score wins.
- **Point Lead:** A player wins by leading their opponent by a specific margin.
- **Board Full:** Automatic claim and game end if the board fills completely.

---

## Project Structure

```
├── game.py              # Game logic and scoring engine
├── model.py             # Neural network (3-channel input)
├── train.py             # AlphaZero-style MCTS training
├── web/                 # Browser-based game UI (source)
├── web/public/game_config.json # Game mode and settings
└── docs/                # GitHub Pages deployment
```

---

## Web App

Browser-based game with retro 80s arcade styling.

### Features

- **Strategic Decision Phase**: Choose to CLAIM or PASS after every move.
- **Probabilistic AI**: Plays against a non-deterministic AI that makes rational point-claiming decisions.
- **Live Scoreboard**: Anchored to the board for clear tracking of points and remaining moves.
- **Visual Polish**: Retro animations including token "evaporation" and blinking action cues.
- **Developer Logs**: Verbose turn-by-turn breakdown in the browser console.

### Play Locally

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173` to play in the browser.

### Build and Deploy

```bash
cd web
npm run build
# Deploy by copying web/dist to the docs/ folder in the project root
```
