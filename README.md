# Zic-Zac-Zoe AI

Neural network AI for Zic-Zac-Zoe, a tic-tac-toe variant.

## Game Rules

- 6×6 grid
- **4 in a row wins**
- **3 in a row loses** (the twist!)
- Diagonals count
- X moves first

---

## Project Structure

```
├── game.py         # Game logic and rules
├── model.py        # Neural network architecture
├── train.py        # Self-play training pipeline
├── evaluate.py     # Benchmarking and ELO ratings
├── play.py         # Human vs AI CLI
├── export_model.py # Export to ONNX for browser
├── checkpoints/    # Saved model weights
└── web/            # Browser-based game UI
```

---

## Source Files

### `game.py` - Game Logic

| Component | Description |
|-----------|-------------|
| `Board` | 6×6 board state. Flat list of 36 ints (0=empty, 1=X, 2=O). |
| `Player` | Enum: EMPTY, X, O |
| `GameResult` | Enum: ONGOING, X_WINS, O_WINS, DRAW |
| `check_result()` | Full board scan for win/loss. |
| `check_result_fast()` | Optimized check examining only lines through last move. |

### `model.py` - Neural Network

**Architecture:**
```
Input: 6×6×2 (one channel per player)
    ↓
Conv 3×3 → BatchNorm → ReLU  (×3 layers, 64 filters)
    ↓
┌───────────────┬───────────────┐
│  Policy Head  │  Value Head   │
│  Conv 1×1     │  Conv 1×1     │
│  Dense → 36   │  Dense → 1    │
│  Softmax      │  Tanh         │
└───────────────┴───────────────┘
Output: 36 move probs, 1 value [-1,1]
```

| Component | Description |
|-----------|-------------|
| `ZicZacNet` | CNN with policy + value heads. ~50K params. |
| `board_to_tensor()` | Convert Board to (1,2,6,6) tensor. |
| `select_move()` | Sample move from policy with temperature. |
| `get_device()` | Auto-detect CUDA/MPS/CPU. |

### `train.py` - Training Pipeline

**Algorithm:** Self-play reinforcement learning (no MCTS).

```
Loop:
  1. Self-play: generate games using current model
  2. Collect samples: (board, move, outcome)
  3. Train on samples
  4. Repeat
```

| Component | Description |
|-----------|-------------|
| `TrainConfig` | Hyperparameters dataclass. |
| `Sample` | Single training example: board state, move taken, game outcome. |
| `ReplayBuffer` | Fixed-size FIFO buffer of recent samples. |
| `play_game()` | One self-play game. Both sides use same model. |
| `train_on_samples()` | SGD update on collected data. |

**Loss functions:**
- Policy: Cross-entropy (predict the move that was played)
- Value: MSE (predict game outcome)

### `evaluate.py` - Benchmarking

| Component | Description |
|-----------|-------------|
| `random_player()` | Baseline: picks random legal move. |
| `heuristic_player()` | Baseline: avoids instant losses, blocks threats. |
| `evaluate_matchup()` | Play N games between two players, report win rates. |
| `compute_elo_ratings()` | Round-robin tournament → ELO ratings. |
| `benchmark_model()` | Test model against all baselines. |

### `play.py` - Human Interface

| Component | Description |
|-----------|-------------|
| `DIFFICULTY_SETTINGS` | Maps difficulty → temperature. |
| `get_human_move()` | Parse and validate user input. |
| `play_game()` | Interactive game loop. |

---

## Algorithms

### Self-Play Training

1. **Generate games:** Model plays against itself. Use temperature=1.0 early (exploration), lower later (exploitation).

2. **Label data:** Each position labeled with:
   - The move that was played
   - Final outcome (+1 win, -1 loss, 0 draw) from that player's perspective

3. **Train:** Policy head learns to predict winning moves. Value head learns to evaluate positions.

4. **Why it works:** Model learns from its mistakes. If a move led to a loss, that move's probability decreases. Winners' moves get reinforced.

### Skill Control

Temperature parameter in softmax move selection:
- `T → 0`: Always pick highest-probability move (strongest)
- `T = 1`: Sample proportionally (balanced)
- `T > 1`: Flatten distribution (weaker, more random)

```python
difficulty_map = {
    "easy": 2.0,
    "medium": 1.0,
    "hard": 0.5,
    "expert": 0.1
}
```

### ELO Rating

Standard chess rating system applied to AI evaluation:
- Start all players at baseline rating
- After each game, update ratings based on expected vs actual outcome
- 400 point gap ≈ 91% win probability for higher-rated player

---

## Usage

### Install Dependencies

```bash
pip install torch
```

### Train a Model

```bash
# Basic training (100 iterations)
python train.py --iterations 100

# More games per iteration (slower but better)
python train.py --iterations 200 --games 200

# Resume from checkpoint
python train.py --iterations 100 --resume checkpoints/model_iter_50.pt
```

### Evaluate

```bash
# Benchmark against baselines
python evaluate.py --model checkpoints/model_final.pt

# Compare checkpoints via ELO tournament
python evaluate.py --compare "checkpoints/*.pt" --games 50
```

### Play Against AI

```bash
# Default difficulty
python play.py --model checkpoints/model_final.pt

# Adjust difficulty
python play.py --model checkpoints/model_final.pt --difficulty easy
python play.py --model checkpoints/model_final.pt --difficulty expert

# Show AI's thinking
python play.py --model checkpoints/model_final.pt --show-thinking
```

---

## Training Tips

| Issue | Solution |
|-------|----------|
| Training too slow | Reduce `--filters` to 32 |
| Model not improving | Increase `--games` per iteration |
| Overfitting to self | Increase replay buffer size |
| Want stronger play | Train longer, use lower temperature at eval |

### Expected Training Progression

| Iteration | Expected Behavior |
|-----------|-------------------|
| 1-10 | Random play, ~50% vs random |
| 20-50 | Learns to avoid obvious losses |
| 50-100 | Beats random consistently (>80%) |
| 100-200 | Beats heuristic player |
| 200+ | Refinement, diminishing returns |

---

## Hardware Notes

**CPU only (no GPU):**
- Training is feasible but slow
- Expect ~1-2 iterations/minute
- Full training: several hours

**Apple Silicon (MPS):**
- Auto-detected via `get_device()`
- ~2-5× speedup over CPU

**NVIDIA GPU (CUDA):**
- Best performance
- ~10× speedup over CPU

---

## Limitations

- No MCTS: Relies purely on policy network. Adding MCTS would strengthen play but slow training.
- Simple architecture: Could use residual blocks, attention, etc. for marginal gains.
- No opening book: Learns everything from scratch.

---

## File Sizes

| File | Purpose | ~Size |
|------|---------|-------|
| `model_*.pt` | Model checkpoint | ~200KB |
| `model.onnx` | Browser model | ~300KB |
| Replay buffer | In-memory only | ~10-50MB |

---

## Web App

Browser-based game with retro 80s arcade styling.

### Setup

```bash
# Export trained model to ONNX
python export_model.py

# Install web dependencies
cd web
npm install

# Run dev server
npm run dev
```

### Build for Production

```bash
cd web
npm run build
# Output in web/dist/
```

### Features

- Responsive design (works on phones)
- Retro arcade aesthetic with CRT effects
- Adjustable difficulty (Easy/Medium/Hard/Pro)
- Choose to play first (X) or second (O)
- Win/lose highlighting on the board
