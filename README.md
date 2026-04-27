# Zic-Zac-Zoe: Points Variant

A strategic twist on tic-tac-toe where players bridge sequences to score high points. This project implements a neural network AI for a 6x6 variant of Zic-Zac-Zoe.

## Game Rules (Points Variant)

- **6x6 grid**
- **Full-board play:** The game ends only when all 36 cells are filled.
- **Scoring:** Points are awarded for every move based on the sequences created.
- **Victory:** The player with the highest cumulative score at the end of the game wins.
- **X moves first.**

### Scoring Mechanics
1. **Line Length (L):**
   - $L=1$ (Isolated): **1 point**.
   - $L=3$ (Penalty): **0 points**.
   - Other lengths: **L points**.
2. **Bridge Bonus:** If a move connects two existing segments (e.g., `X _ XX`), the point value for that line is **doubled**.
3. **Productive Multiplier (N):** The total move score is multiplied by the number of axes that produced points (>0).

---

## Project Structure

```
├── game.py              # Game logic and scoring engine
├── model.py             # Neural network (3-channel input)
├── train.py             # AlphaZero-style MCTS training
├── evaluate.py          # Benchmarking and ELO ratings
├── tactical_generator.py # Synthetic tactical positions
├── export_weights.py    # Export to JSON for browser
├── play.py              # Human vs AI CLI
├── checkpoints/         # Saved model weights
├── web/                 # Browser-based game UI (source)
└── docs/                # GitHub Pages deployment
```

---

## Source Files

### `game.py` - Game Logic & Scoring

| Component | Description |
|-----------|-------------|
| `Board` | 6x6 board state including cumulative player scores. |
| `Player` | Enum: EMPTY, X, O |
| `GameResult` | Enum: ONGOING, DRAW (Terminal state) |
| `calculate_move_score()` | Scoring engine implementing bridge and multiplier rules. |
| `check_result_fast()` | Optimized terminal check (triggers on full board). |

### `model.py` - Neural Network

**Architecture:**
```
Input: 6x6x3 (X positions, O positions, turn indicator)
    |
Conv 3x3 -> BatchNorm -> ReLU  (x3 layers, 64 filters)
    |
+---------------+---------------+
|  Policy Head  |  Value Head   |
|  Conv 1x1     |  Conv 1x1     |
|  Dense -> 36  |  Dense -> 1   |
|  LogSoftmax   |  Tanh         |
+---------------+---------------+
Output: 36 log probs, 1 value [-1,1]
```

*Note: The current model was trained on a win/loss variant. In the Points Variant prototype, the AI plays randomly to provide a neutral opponent for mechanic testing.*

---

## Usage

### Install Dependencies

```bash
pip install torch
```

### Play Locally (Web App)

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173` to play in the browser.

---

## Web App

Browser-based game with retro 80s arcade styling.

### Features

- **1P Mode**: Play against a random-move AI prototype.
- **2P Mode**: Local two-player.
- **Real-time Scoreboard**: Dynamic tracking of X and O points.
- **Developer Logs**: Verbose turn-by-turn scoring breakdown in the browser console.
- **Win/Loss Stats**: Persistent tracking of game outcomes in local storage.

### Build for Production

```bash
cd web
npm run build
# Output in web/dist/
```

### Deploy to GitHub Pages

```bash
cd web
npm run build
rm -rf ../docs && cp -r dist ../docs
# Commit and push - serves from /docs folder
```
