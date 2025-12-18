# Zic-Zac-Zoe AI

Neural network AI for Zic-Zac-Zoe, a tic-tac-toe variant.

## Game Rules

- 6x6 grid
- **4 in a row wins**
- **3 in a row loses** (the twist!)
- Diagonals count
- X moves first

---

## Project Structure

```
├── game.py              # Game logic and rules
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

### `game.py` - Game Logic

| Component | Description |
|-----------|-------------|
| `Board` | 6x6 board state. Immutable, creates new board on move. |
| `Player` | Enum: EMPTY, X, O |
| `GameResult` | Enum: ONGOING, X_WINS, O_WINS, DRAW |
| `check_result_fast()` | Optimized check examining only lines through last move. |

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

| Component | Description |
|-----------|-------------|
| `ZicZacNet` | CNN with policy + value heads. ~50K params. |
| `board_to_tensor()` | Convert Board to (1,3,6,6) tensor. |
| `select_move()` | Sample move from policy with temperature. |

### `train.py` - Training Pipeline

**Algorithm:** AlphaZero-style MCTS self-play with:
- Win-rate model selection (>55% to replace best)
- MuZero-style reanalyze (fresh policy targets each batch)
- Tactical injection (synthetic positions)

| Component | Description |
|-----------|-------------|
| `TrainConfig` | Hyperparameters dataclass. |
| `MCTSNode` | Tree node with neural network priors. |
| `mcts_search()` | MCTS with PUCT selection. |
| `ReplayBuffer` | FIFO buffer with uniform sampling. |
| `reanalyze_batch()` | Regenerate policy targets with current model. |

### `tactical_generator.py` - Synthetic Positions

Generates training positions for critical patterns:
- `avoid_3`: Positions where one move loses (creates 3-in-a-row)
- `complete_4`: Positions where one move wins (creates 4-in-a-row)
- `block_4`: Positions where must block opponent's winning threat

### `evaluate.py` - Benchmarking

| Component | Description |
|-----------|-------------|
| `random_player()` | Baseline: picks random legal move. |
| `heuristic_player()` | Baseline: avoids instant losses, blocks threats. |
| `evaluate_matchup()` | Play N games between two players, report win rates. |
| `compute_elo_ratings()` | Round-robin tournament -> ELO ratings. |

---

## Usage

### Install Dependencies

```bash
pip install torch
```

### Train a Model

```bash
# Basic training
python train.py --iterations 100

# Resume from checkpoint (auto-detected)
python train.py --iterations 100

# Start fresh
python train.py --iterations 100 --fresh
```

### Evaluate

```bash
# Benchmark against baselines
python evaluate.py --model checkpoints/model_best.pt

# Compare checkpoints via ELO tournament
python evaluate.py --compare "checkpoints/*.pt" --games 50
```

### Play Against AI

```bash
python play.py --model checkpoints/model_best.pt
python play.py --model checkpoints/model_best.pt --difficulty expert
```

---

## Web App

Browser-based game with retro 80s arcade styling.

### Features

- **1P Mode**: Play against neural network AI
- **2P Mode**: Local two-player
- **Difficulty Levels**: Easy, Medium, Hard, Expert
- **Play as X or O**: Choose your side (1P mode)
- **Score Tracking**: Win/loss stats persist locally
- **Checkmate Detection**: Highlights forced-win positions

### Setup

```bash
cd web
npm install
npm run dev
```

### URL Parameters

- Default: Neural network AI
- `?rules=1`: Rule-based AI (no neural network)

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
