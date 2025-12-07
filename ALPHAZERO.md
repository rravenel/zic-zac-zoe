# AlphaZero-Style Training

Advanced training pipeline using Monte Carlo Tree Search with neural network evaluation.

## Overview

This approach replaces random self-play with MCTS-guided move selection, where the neural network both guides the search (policy) and evaluates positions (value). No random rollouts are used.

**Key difference from basic self-play:**
```
Basic:     position → neural net → sample move → play
AlphaZero: position → MCTS (guided by neural net) → best move → play
```

---

## Files

| File | Description |
|------|-------------|
| `train_alphazero.py` | AlphaZero-style training pipeline |
| `checkpoints_alphazero/` | Model checkpoints from this approach |

---

## Algorithm

### MCTS with Neural Network Evaluation

Each move selection runs N simulations of tree search:

```
1. SELECT    - Walk down tree using PUCT formula
2. EXPAND    - At leaf, create children for all legal moves
3. EVALUATE  - Neural net returns (policy, value) for leaf position
4. BACKUP    - Propagate value back up the tree
```

**No random rollouts.** The neural network directly evaluates "how good is this position?" in one forward pass, instead of playing random moves to see who wins.

### PUCT Selection Formula

At each node, select child with highest score:

```
PUCT(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Where:
  Q(s,a) = average value from previous visits (exploitation)
  P(s,a) = policy prior from neural network (guidance)
  N(s)   = parent visit count
  N(s,a) = child visit count
  c      = exploration constant (default 1.5)
```

The policy prior P(s,a) guides search toward promising moves before they're fully explored.

### Training Loop

```
For each iteration:
  1. Self-play N games using MCTS for move selection
  2. For each position, record:
     - Board state
     - MCTS visit distribution (policy target)
     - Game outcome (value target)
  3. Train neural network on collected samples
  4. Repeat
```

### Why This Works

1. **MCTS improves policy:** Even a weak network produces decent moves after search
2. **Better moves → better training data:** The network learns from MCTS-improved moves, not raw network output
3. **Self-improving cycle:** Better network → better MCTS → better training data → better network

---

## Configuration

```python
@dataclass
class TrainConfig:
    # Self-play
    games_per_iteration: int = 100
    temperature_start: float = 1.0      # Early game (exploration)
    temperature_end: float = 0.3        # Late game (exploitation)
    temperature_threshold: int = 12     # Move to switch

    # MCTS
    mcts_simulations: int = 100         # Simulations per move
    mcts_c_puct: float = 1.5            # Exploration constant
    dirichlet_alpha: float = 0.3        # Root noise
    dirichlet_epsilon: float = 0.25     # Noise weight

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs_per_iteration: int = 5
    replay_buffer_size: int = 50000

    # Model
    num_filters: int = 64
```

---

## Usage

### Train from Scratch

```bash
python train_alphazero.py --iterations 200 --fresh
```

### Resume Training

```bash
python train_alphazero.py --iterations 100
# Auto-resumes from latest checkpoint
```

### Custom Settings

```bash
# Faster iterations (fewer simulations)
python train_alphazero.py --iterations 200 --simulations 50 --games 50

# Stronger search (more simulations, slower)
python train_alphazero.py --iterations 100 --simulations 200 --games 100

# Adjust exploration
python train_alphazero.py --c-puct 2.0
```

### All Options

```bash
python train_alphazero.py --help

--iterations N      Number of training iterations
--games N           Games per iteration (default: 100)
--simulations N     MCTS simulations per move (default: 100)
--fresh             Start fresh, clearing old checkpoints
--resume PATH       Resume from specific checkpoint
--lr FLOAT          Learning rate (default: 0.001)
--filters N         Conv filter count (default: 64)
--c-puct FLOAT      PUCT exploration constant (default: 1.5)
--patience N        Early stopping patience (default: 30)
--no-early-stop     Disable early stopping
```

---

## Comparison: Basic vs AlphaZero

| Aspect | Basic Self-Play | AlphaZero MCTS |
|--------|-----------------|----------------|
| Move selection | Sample from policy | MCTS search |
| Position evaluation | Neural net only | Neural net only |
| Training signal | Raw network moves | MCTS-improved moves |
| Quality | Good | Better |
| Speed | Fast | Slower |
| Games/iteration | 100-500 | 50-100 |

### When to Use Which

**Basic (`train.py`):**
- Quick experimentation
- Limited compute
- Initial model exploration

**AlphaZero (`train_alphazero.py`):**
- Best quality training
- Final model production
- Have time/compute to spare

---

## Expected Output

```
Training on device: cpu
Using AlphaZero-style MCTS with 100 simulations per move
Neural network evaluation (no random rollouts)
Model parameters: 80,718

Starting training: iterations 1 to 100
================================================================================
Iter    1 | Games: X=12 O= 8 D= 0 | Avg len: 14.2 | P_loss: 3.5321 | V_loss: 0.9301 | Buffer:   284 | Time: 15.2s *
Iter    2 | Games: X= 9 O=11 D= 0 | Avg len: 13.8 | P_loss: 3.4126 | V_loss: 0.8842 | Buffer:   560 | Time: 14.8s *
...
```

| Column | Meaning |
|--------|---------|
| Games: X=N O=N D=N | Win counts (should be ~balanced) |
| Avg len | Average game length in moves |
| P_loss | Policy cross-entropy loss (lower = sharper) |
| V_loss | Value MSE loss (lower = more accurate) |
| Buffer | Replay buffer size |
| Time | Iteration time |
| * | New best model saved |

---

## Training Progression

| Iterations | Expected Behavior |
|------------|-------------------|
| 1-10 | Random play, losses ~3.5/0.9 |
| 10-30 | Learning basic tactics, losses dropping |
| 30-100 | Competent play, avoids obvious mistakes |
| 100-200 | Strong play, sees multi-move tactics |
| 200+ | Refinement, diminishing returns |

**Signs of healthy training:**
- X/O wins roughly balanced (not 95-5)
- Game length 10-20 moves (not 5)
- Policy loss steadily decreasing
- Value loss approaching 0.1-0.3

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| O wins 95%+ | Bug in MCTS value signs | Fixed in current version |
| Games only 5 moves | Model making 3-in-a-row | Check MCTS implementation |
| Very slow | Too many simulations | Reduce `--simulations` |
| Not improving | Stuck in local optimum | Increase `--c-puct` for more exploration |
| Memory issues | Large replay buffer | Reduce `--games` or buffer size |

---

## Implementation Notes

### Value Perspective Convention

Values are stored from each node's player perspective:
- Positive = good for player to move at that node
- Negative = bad for player to move at that node

When selecting moves, Q values are negated because the child's value is from the opponent's perspective.

### Dirichlet Noise

Added to root node priors to ensure exploration:
```python
prior = (1 - epsilon) * network_prior + epsilon * dirichlet_noise
```

This prevents the search from being too deterministic early in training.

### Temperature Schedule

High temperature early in games encourages diverse openings:
- Moves 1-12: temperature = 1.0 (sample proportionally)
- Moves 13+: temperature = 0.3 (favor best moves)

---

## Export for Web

After training, export the model for browser use:

```bash
python export_weights.py --model checkpoints_alphazero/model_best.pt --output web/public/weights.json
```

The web app uses a pure JavaScript implementation of the neural network forward pass.
