# Zic-Zac-Zoe v2 Training

Enhanced AlphaZero-style training with MuZero improvements.

## Overview

v2 builds on the base AlphaZero approach with key enhancements inspired by MuZero and lessons learned from training instability. The main improvements address the "learned then forgot" problem where the model would master tactics then regress.

**Key improvements over base AlphaZero:**

| Feature | Base AlphaZero | v2 |
|---------|----------------|-----|
| Model selection | Loss-based | Win-rate vs previous best |
| Policy targets | Fixed at game time | Reanalyzed with current model |
| Replay buffer | FIFO or prioritized | FIFO with uniform sampling |
| Board encoding | 2 channels (X, O) | 3 channels (X, O, turn indicator) |
| Training curriculum | Self-play only | Self-play + tactical injection |

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Main training pipeline with reanalyze |
| `evaluate.py` | Model evaluation and benchmarking |
| `game.py` | Game logic (shared with base) |
| `model.py` | Neural network with turn indicator channel |
| `tactical_generator.py` | Generates tactical training positions |
| `test_*.py` | Test suites for each module |
| `checkpoints/` | Model checkpoints |

---

## Algorithmic Improvements

### 1. Win-Rate Model Selection

AlphaZero doesn't use loss for model selection. Instead, candidate models must beat the previous best by a threshold.

```
Model Selection:
  1. Every eval_frequency iterations, evaluate candidate vs best
  2. If candidate wins > 55% of games, it becomes new best
  3. If no improvement after patience iterations, training stops
```

**Why this matters:**
- Loss can be misleading (policy loss drops while value loss rises)
- Win rate directly measures what we care about: game performance
- Prevents "forgetting" good play patterns

**Configuration:**
```python
win_rate_threshold: float = 0.55      # Must beat by this margin
eval_games_for_best: int = 40         # Games per evaluation
early_stopping_patience: int = 30     # Iterations without improvement
```

### 2. Reanalyze (MuZero-Style)

Policy targets are regenerated each batch using the current model's MCTS, not the stale policy from when the game was played.

```
Traditional:
  Game time: position → MCTS → policy_target → store
  Train time: load policy_target (stale!)

Reanalyze:
  Game time: position → store (no policy)
  Train time: position → MCTS (current model) → fresh policy_target
```

**Why this matters:**
- Old policy targets reflect outdated model understanding
- Current model may see tactics the old model missed
- Each position is analyzed with the best available knowledge

**Implementation:**
```python
# PositionSample: minimal storage (no policy)
@dataclass
class PositionSample:
    board_state: List[int]
    current_player: Player
    outcome: float          # Ground truth, never changes

# Reanalyze at training time
def reanalyze_batch(positions, model, device, num_simulations):
    samples = []
    for pos in positions:
        board = Board(pos.board_state)
        visit_counts = mcts_search(board, model, device, num_simulations)
        policy_target = visits_to_policy(visit_counts)  # Fresh!
        samples.append(Sample(..., policy_target=policy_target))
    return samples
```

**Configuration:**
```python
reanalyze_simulations: int = 50  # MCTS simulations for reanalyze
```

### 3. Turn Indicator Channel

The neural network receives an additional channel indicating whose turn it is.

```
Input tensor: 6x6x3
  Channel 0: X pieces (1 where X, 0 elsewhere)
  Channel 1: O pieces (1 where O, 0 elsewhere)
  Channel 2: Turn indicator (all 1s if X to move, all 0s if O to move)
```

**Why this matters:**
- Board state alone is ambiguous about who moves next
- Same position can have different values depending on whose turn
- Network can learn turn-dependent patterns

### 4. Tactical Sample Injection

Synthetic positions with known correct moves are injected into training to reinforce critical patterns.

**Pattern types:**

| Pattern | Description | Correct Action |
|---------|-------------|----------------|
| `avoid_3` | XX pattern exists | Don't play adjacent cell |
| `complete_4` | XX_X or X_XX pattern | Play the gap to win |
| `block_4` | OO_O or O_OO threat | Block the gap |

**Why this matters:**
- Self-play may not encounter rare but critical positions
- Ensures model never "forgets" fundamental tactics
- Provides stable training signal independent of self-play quality

---

## Configuration

```python
@dataclass
class TrainConfig:
    # Self-play
    games_per_iteration: int = 100
    mcts_simulations: int = 100

    # Model selection (AlphaZero-style)
    win_rate_threshold: float = 0.55
    eval_games_for_best: int = 40
    eval_frequency: int = 5           # Evaluate every N iterations
    early_stopping_patience: int = 30

    # Reanalyze (MuZero-style)
    reanalyze_simulations: int = 50

    # Tactical injection
    tactical_samples_per_iter: int = 50

    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    replay_buffer_size: int = 50000

    # Model
    num_filters: int = 64
```

---

## Usage

### Train from Scratch

```bash
python train.py --iterations 200 --fresh
```

### Resume Training

```bash
python train.py --iterations 100
# Auto-resumes from checkpoints/model_best.pt
```

### Custom Settings

```bash
# Faster training (fewer simulations)
python train.py --simulations 50 --reanalyze-sims 25

# Stricter model selection
python train.py --win-threshold 0.60 --eval-games 60

# More tactical samples
python train.py --tactical-per-iter 100
```

### Run Tests

```bash
# All tests
python test_buffer.py && python test_train.py && python test_tactical.py && python test_model.py

# Individual test files
python test_buffer.py      # ReplayBuffer and reanalyze tests
python test_train.py       # Model selection and config tests
python test_tactical.py    # Tactical generator tests
python test_model.py       # Neural network tests
```

---

## Expected Output

```
Training on device: mps
Using AlphaZero-style MCTS with 100 simulations per move
Model selection: win rate > 55% vs previous best
Reanalyze: 50 simulations for fresh policy targets

Starting training: iterations 1 to 100
================================================================================
Iter    1 | Games: X=12 O= 8 D= 0 | Avg len: 14.2 | Buffer:   284 | Time: 15.2s
Iter    5 | Evaluating vs best... 62.5% win rate. New best model! *
...
```

| Column | Meaning |
|--------|---------|
| Games: X=N O=N D=N | Self-play results (should be balanced) |
| Avg len | Average game length |
| Buffer | Replay buffer size |
| * | New best model saved |

---

## Training Progression

| Iterations | Expected Behavior |
|------------|-------------------|
| 1-10 | Random play, learning basic patterns |
| 10-30 | Avoids obvious 3-in-a-row mistakes |
| 30-100 | Beats random consistently (>80%) |
| 100-200 | Sees multi-move tactics, blocks threats |
| 200+ | Refinement, rare regressions |

**Signs of healthy training:**
- X/O wins roughly balanced in self-play
- Periodic "New best model!" messages
- Win rate against best hovering near threshold
- No long streaks without new best

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Never becomes new best | Threshold too high | Lower `--win-threshold` to 0.52 |
| Oscillating best model | Threshold too low | Raise `--win-threshold` to 0.58 |
| Slow iterations | Reanalyze overhead | Lower `--reanalyze-sims` |
| Forgetting tactics | Insufficient tactical injection | Raise `--tactical-per-iter` |
| Early stopping too soon | Patience too low | Raise `--patience` |

---

## Technical Notes

### Why Uniform Sampling?

Prioritized replay (weighting by TD error or loss) was tested but removed because:
- Priority weights add complexity and hyperparameters
- Reanalyze already provides fresh perspectives on old positions
- AlphaZero uses uniform sampling successfully
- Simpler is often better for debugging

### Replay Buffer Storage

v2 stores minimal information:
```python
PositionSample:
  board_state: List[int]    # 36 ints
  current_player: Player    # 1 byte
  outcome: float            # 4 bytes

# ~150 bytes per sample
# 50,000 samples = ~7.5 MB
```

Policy targets are not stored because they're regenerated via reanalyze.

### Value vs Policy Loss

The training loop computes both losses but only win rate matters for model selection:
- Policy loss: how well network predicts MCTS move distribution
- Value loss: how well network predicts game outcome
- Neither directly measures game-playing strength

Win rate against previous best is the ground truth metric.

---

## Comparison with Base AlphaZero

| Aspect | Base AlphaZero | v2 |
|--------|----------------|-----|
| Model saved when | Loss improves | Win rate > 55% |
| Policy stored | At game time | Regenerated each batch |
| Buffer priority | Optional | Always uniform |
| Tactical training | None | Injected samples |
| Input channels | 2 | 3 (with turn indicator) |
| Forgetting | Common problem | Addressed by design |

---

## Files Modified from Base

| File | Changes |
|------|---------|
| `model.py` | Added turn indicator channel (input now 3 channels) |
| `train.py` | Reanalyze, win-rate selection, PositionSample dataclass |
| `evaluate.py` | Added EvalResult.win_rate property |
| `tactical_generator.py` | New file for synthetic training data |
