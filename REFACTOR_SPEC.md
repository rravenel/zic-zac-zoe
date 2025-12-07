# Zic-Zac-Zoe Refactoring Specification

## Goal

Consolidate codebase by promoting v2/ code to root and removing legacy 2-channel model code. After refactoring, there should be no v1/v2 versioning split - just one clean codebase.

## Current Git State

- Branch: `main`
- Last commit: `204713a initial`
- Status: clean

---

## PART 1: ACTION PLAN

### Step 1: Archive Old Checkpoints

Create `archived_models/` directory (gitignored) to preserve best models from old training approaches.

**Actions:**
1. Create `archived_models/` directory
2. Add `archived_models/` to `.gitignore`
3. Copy best checkpoint from each old folder:
   - `checkpoints/model_best.pt` and `checkpoints/model_final.pt`
   - `checkpoints_mcts/model_best.pt` and `checkpoints_mcts/model_final.pt` (if exists)
   - `checkpoints_alphazero/model_best.pt` and `checkpoints_alphazero/model_final.pt` (if exists)
4. Create `archived_models/README.md` with:
   - Git hash: `204713a`
   - Explanation that these are 2-channel models from pre-refactor code
   - Which training script generated each (train.py, train_mcts.py, train_alphazero.py)
   - Note that original code is in git history at the above hash

### Step 2: Delete Legacy Files from Root

**Delete these files:**
- `model.py` - 2-channel CNN (replaced by v2/model.py)
- `model_v2.py` - 3-channel CNN (duplicate of v2/model.py, use v2's version)
- `train.py` - Simple self-play, no MCTS (replaced by v2/train.py)
- `train_mcts.py` - MCTS with random rollouts
- `train_alphazero.py` - AlphaZero-style MCTS
- `evaluate.py` - Basic evaluation (replaced by v2/evaluate.py)
- `export_model.py` - ONNX export (unused)
- `export_weights.py` - JSON export (replaced by v2/export_weights.py)

**Delete these directories:**
- `checkpoints/` - Old 2-channel checkpoints (after archiving best)
- `checkpoints_mcts/` - MCTS training checkpoints (after archiving best)
- `checkpoints_alphazero/` - AlphaZero training checkpoints (after archiving best)

### Step 3: Move v2/ Files to Root

**Move these files:**
- `v2/model.py` → `model.py`
- `v2/train.py` → `train.py`
- `v2/evaluate.py` → `evaluate.py`
- `v2/tactical_generator.py` → `tactical_generator.py`
- `v2/export_weights.py` → `export_weights.py`
- `v2/test_model.py` → `test_model.py`
- `v2/test_buffer.py` → `test_buffer.py`
- `v2/test_train.py` → `test_train.py`
- `v2/test_tactical.py` → `test_tactical.py`

**Move checkpoint directory:**
- `v2/checkpoints/` → `checkpoints/`

**Delete after moving:**
- `v2/archive/` - Old archived checkpoints
- `v2/README.md` - No longer needed
- `v2/` directory itself

### Step 4: Update Remaining Root Files

**Files to keep and potentially update:**

1. `game.py` - Core game logic, NO CHANGES needed (v2 imports from it)

2. `play.py` - Terminal interface for human vs AI
   - Currently imports from `model` (the old 2-channel)
   - Update imports to work with new model.py (3-channel)
   - The v2 model.py has same API: `ZicZacNet`, `select_move`, `load_model`, `get_device`, `get_policy_value`
   - Should work with minimal/no changes

3. `dedup_analysis.py` - Standalone analysis tool, NO CHANGES needed

### Step 5: Web App Changes

**File: `web/public/`**
- Delete `weights.json` (old v1 2-channel weights)
- Rename `weights_v2.json` → `weights.json`

**File: `web/src/ai.ts`**

Remove v1/v2 version detection. Currently has:
```typescript
let modelVersion: number = 1;  // 1 = 2-channel, 2 = 3-channel with turn indicator
```

Changes needed:
1. Remove `modelVersion` variable
2. Remove version detection in `loadModel()` function (lines ~328-336)
3. Hardcode 3-channel behavior in `boardToTensor()` (always use 3 channels)
4. Remove `getModelVersion()` function
5. Update console.log to not mention version

**File: `web/src/rules-ai.ts`**

Change URL param detection. Currently (line ~214-216):
```typescript
export function isRulesAI(): boolean {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("v3") === "1" || urlParams.get("model") === "v3";
}
```

Change to:
```typescript
export function isRulesAI(): boolean {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get("rules") === "1";
}
```

**File: `web/src/main.ts`**

Remove v1 loading logic. Currently (lines ~404-412):
```typescript
const urlParams = new URLSearchParams(window.location.search);
const useV1 = urlParams.get("v1") === "1" || urlParams.get("model") === "v1";

try {
  let weightsFile = "weights_v2.json";
  if (useV1) {
    weightsFile = "weights.json";
  }
```

Change to:
```typescript
try {
  const weightsFile = "weights.json";
```

Also update the iteration display text if it references version numbers.

### Step 6: Cleanup

1. Delete empty `v2/` directory
2. Verify all imports work
3. Run tests: `python -m pytest test_*.py`
4. Test web app locally

---

## PART 2: SOURCE CODE REFERENCE

### Root Directory Files

#### `game.py` (~366 lines) - KEEP UNCHANGED
Core game logic for 6x6 tic-tac-toe variant.

Key components:
- `BOARD_SIZE = 6`, `WIN_LENGTH = 4`, `LOSE_LENGTH = 3`
- `Player` enum: EMPTY, X, O
- `GameResult` enum: ONGOING, X_WINS, O_WINS, DRAW
- `Board` class: immutable board state, `make_move()`, `get_legal_moves()`, `current_player()`
- `check_result_fast(board, last_move)` - efficient game-over detection
- `index_to_coord()`, `coord_to_index()` - position conversion

#### `model.py` (~280 lines) - DELETE (2-channel, legacy)
Old 2-channel CNN. Input: (batch, 2, 6, 6) - X positions, O positions only.

#### `model_v2.py` (~310 lines) - DELETE (duplicate of v2/model.py)
3-channel CNN but file in root. Use v2/model.py instead.

#### `train.py` (~568 lines) - DELETE (simple self-play)
Basic training without MCTS. Uses temperature-based exploration only.

#### `train_mcts.py` (~710 lines) - DELETE
MCTS with random rollouts (not neural network evaluation).

#### `train_alphazero.py` (~838 lines) - DELETE
AlphaZero-style MCTS with neural network evaluation and parallel workers.
Key classes: `MCTSNode`, `TrainConfig`, `Sample`, `ReplayBuffer`

#### `evaluate.py` (~369 lines) - DELETE (replaced by v2/evaluate.py)
Basic evaluation: random player, heuristic player, model vs model, ELO ratings.

#### `play.py` (~263 lines) - KEEP, UPDATE IMPORTS
Terminal-based human vs AI interface.

Imports to verify work with new model.py:
```python
from model import ZicZacNet, select_move, load_model, get_device, get_policy_value
```
These functions exist in v2/model.py with same signatures.

Uses difficulty settings via temperature (0.1 to 2.0).

#### `export_model.py` (~69 lines) - DELETE
ONNX export, not used by web app.

#### `export_weights.py` (~66 lines) - DELETE (replaced by v2/export_weights.py)
JSON weight export for web inference.

#### `dedup_analysis.py` (~149 lines) - KEEP UNCHANGED
Standalone analysis tool for DAG deduplication ratios. No dependencies on model code.

---

### v2/ Directory Files (MOVE TO ROOT)

#### `v2/model.py` (~310 lines) → `model.py`
3-channel CNN. Input: (batch, 3, 6, 6)
- Channel 0: X positions
- Channel 1: O positions
- Channel 2: Turn indicator (1.0 if X's turn, 0.0 if O's turn)

Key components:
- `ZicZacNet(num_filters=64)` - CNN with 3 conv layers, policy head, value head
- `board_to_tensor(board, device)` - converts Board to (1, 3, 6, 6) tensor
- `boards_to_tensor(boards, device)` - batch version
- `select_move(model, board, temperature, device)` - sample move from policy
- `get_policy_value(model, board, device)` - get raw policy and value
- `save_model()`, `load_model()`, `get_device()`

#### `v2/train.py` (~1084 lines) → `train.py`
Enhanced AlphaZero training with v2 innovations.

Key components:
- `TrainConfig` dataclass (~20 hyperparameters)
- `MCTSNode` class with neural network priors
- `mcts_search()` - MCTS with PUCT selection
- `Sample` and `PositionSample` dataclasses
- `ReplayBuffer` - FIFO with uniform sampling, save/load
- `reanalyze_batch()` - MuZero-style fresh policy targets
- `self_play_games()` - parallel game generation
- `train_on_buffer()` - combines fresh + reanalyzed + tactical samples
- `train()` - main loop with win-rate model selection (>55% to replace)

v2 innovations:
1. Win-rate model selection (not loss-based)
2. Reanalyze - regenerate policy targets each batch
3. Tactical injection - synthetic positions from tactical_generator

#### `v2/evaluate.py` (~620 lines) → `evaluate.py`
Enhanced evaluation with visual output.

Key components:
- `random_player()`, `heuristic_player()`, `create_model_player()`
- `play_match()`, `evaluate_matchup()`
- `EloPlayer`, `compute_elo_ratings()`
- `benchmark_model()` - vs random, vs heuristic
- `compare_checkpoints()` - ELO tournament
- Tactical evaluation with visual board display

#### `v2/tactical_generator.py` (~457 lines) → `tactical_generator.py`
Generates synthetic tactical training positions.

Pattern types:
- `avoid_3` - positions where one move loses (creates 3-in-a-row)
- `complete_4` - positions where one move wins (creates 4-in-a-row)
- `block_4` - positions where must block opponent's winning threat

Key functions:
- `generate_lines()` - all lines of length 3+ on board
- `generate_avoid_three_sample()`, `generate_complete_four_sample()`, `generate_block_four_sample()`
- `TacticalGenerator` class - manages generation with ratios

#### `v2/export_weights.py` (~70 lines) → `export_weights.py`
JSON export for web inference.

Exports all model parameters and batch norm buffers.
Adds metadata: `_version`, `_iteration`, `_input_channels`

#### `v2/test_model.py` → `test_model.py`
Tests for model: turn indicator encoding, piece encoding, output shapes.

#### `v2/test_buffer.py` → `test_buffer.py`
Tests for ReplayBuffer and reanalyze function.

#### `v2/test_train.py` → `test_train.py`
Tests for TrainConfig, EvalResult, model selection logic.

#### `v2/test_tactical.py` → `test_tactical.py`
Tests for tactical pattern generators.

---

### Web Directory Files

#### `web/src/game.ts` (~328 lines) - NO CHANGES
TypeScript port of game rules. Matches Python game.py.

#### `web/src/ai.ts` (~595 lines) - UPDATE
Pure JavaScript neural network inference.

Current structure:
- `DIFFICULTY` object with temperature values
- `GUARDRAIL_DECAY` - difficulty-based tactical guardrail decay
- `getTacticalMask()` - generates +/-Infinity masks for win/suicide moves
- Tensor operations: `conv2d`, `batchNorm2d`, `relu`, `linear`, `flatten`, `softmax`
- `ModelWeights` interface
- `loadModel()` - fetches and parses weights JSON, detects version
- `boardToTensor()` - converts board to tensor (version-aware)
- `forward()` - full CNN forward pass
- `sampleMove()` - applies tactical mask and temperature
- `getAIMove()` - main entry point
- `getPositionValue()` - get position evaluation

**Changes needed:**
1. Remove `modelVersion` variable (line 311)
2. Remove version detection in `loadModel()` (lines 328-336)
3. Simplify `boardToTensor()` to always use 3 channels (remove conditional)
4. Remove `getModelVersion()` function (lines 342-344)

#### `web/src/rules-ai.ts` (~217 lines) - UPDATE
Rule-based AI without neural network.

Key functions:
- `createsThree()`, `createsFour()` - check if move creates 3/4 in a row
- `findThreats()` - find opponent's winning threats
- `getRulesMove()` - tactical move selection
- `isRulesAI()` - checks URL params

**Change needed:**
Update `isRulesAI()` to check `?rules=1` instead of `?v3=1` or `?model=v3`

#### `web/src/main.ts` (~438 lines) - UPDATE
Game controller and UI.

**Changes needed:**
1. Remove v1 detection logic (lines 404-412)
2. Hardcode `weightsFile = "weights.json"`
3. Remove version-related comments

#### `web/public/weights.json` - DELETE
Old 2-channel model weights.

#### `web/public/weights_v2.json` - RENAME → `weights.json`
Current 3-channel model weights.

---

## PART 3: VERIFICATION CHECKLIST

After refactoring:

1. [ ] `archived_models/` exists with README and best checkpoints
2. [ ] No `model_v2.py`, `train_mcts.py`, `train_alphazero.py` in root
3. [ ] `v2/` directory deleted
4. [ ] `python -c "from model import ZicZacNet; print('OK')"` works
5. [ ] `python -c "from train import train; print('OK')"` works
6. [ ] `python -m pytest test_*.py` passes
7. [ ] `python play.py --model checkpoints/model_final.pt` runs
8. [ ] Web app loads with `npm run dev` in web/
9. [ ] `?rules=1` enables rules-based AI
10. [ ] No "v1", "v2", "v3" strings remain in web code (except rules-ai.ts v3 → rules)
