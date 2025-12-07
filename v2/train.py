"""
Training Pipeline with AlphaZero-style MCTS for Zic-Zac-Zoe AI (v2)

Changes from v1:
- Turn indicator channel in model input
- Integrated evaluation harness (vs random, vs previous checkpoint)
- Tactical test suite for qualitative assessment
"""

import os
import re
import glob
import time
import random
import pickle
import math
import json
import multiprocessing as mp
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE
from model import (
    ZicZacNet, board_to_tensor, boards_to_tensor,
    save_model, get_device
)
from evaluate import eval_vs_random, eval_tactical, eval_model_vs_model, EvalResult
from tactical_generator import generate_tactical_samples, tactical_sample_to_training_sample, PatternType


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Self-play settings
    games_per_iteration: int = 100
    temperature_start: float = 1.0
    temperature_end: float = 0.3
    temperature_threshold: int = 12

    # MCTS settings
    mcts_simulations: int = 200
    mcts_c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Training settings
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs_per_iteration: int = 5
    replay_buffer_size: int = 50000

    # Model settings
    num_filters: int = 64

    # Tactical injection
    tactical_samples_per_type: int = 70  # Generate 70 of each pattern per iteration (~15% of training data)

    # Reanalyze (MuZero-style)
    reanalyze_simulations: int = 50  # MCTS sims for reanalyze (less than self-play)

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"

    # Evaluation
    eval_interval: int = 10           # Run evaluation every N iterations
    eval_games_vs_random: int = 50    # Games against random player
    eval_games_vs_prev: int = 20      # Games against previous best

    # Logging
    log_interval: int = 1

    # Model selection (AlphaZero-style)
    win_rate_threshold: float = 0.55    # New best if win rate > this vs previous best
    eval_games_for_best: int = 40       # Games to play for best model selection
    tactical_pass_threshold: float = 0.90  # Must pass this fraction of tactical tests
    tactical_eval_per_type: int = 10      # Number of tactical tests per pattern type

    # Early stopping
    early_stopping_patience: int = 30   # Iterations without new best before stopping

    # Parallelization
    num_workers: int = 1


# =============================================================================
# MCTS with Neural Network Evaluation
# =============================================================================

class MCTSNode:
    """Node in the MCTS tree with neural network priors."""

    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[int] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior

        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0

        self._legal_moves: Optional[List[int]] = None
        self._is_terminal: Optional[bool] = None
        self._terminal_value: Optional[float] = None

    @property
    def legal_moves(self) -> List[int]:
        if self._legal_moves is None:
            self._legal_moves = self.board.get_legal_moves()
        return self._legal_moves

    @property
    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            if self.move is None:
                self._is_terminal = False
            else:
                result = check_result_fast(self.board, self.move)
                self._is_terminal = result != GameResult.ONGOING
                if self._is_terminal:
                    current = self.board.current_player()
                    if result == GameResult.DRAW:
                        self._terminal_value = 0.0
                    elif result == GameResult.X_WINS:
                        self._terminal_value = 1.0 if current == Player.X else -1.0
                    else:
                        self._terminal_value = 1.0 if current == Player.O else -1.0
        return self._is_terminal

    @property
    def terminal_value(self) -> float:
        _ = self.is_terminal
        return self._terminal_value

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        q = -self.q_value()
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return q + u

    def best_child(self, c_puct: float) -> 'MCTSNode':
        return max(self.children.values(),
                   key=lambda n: n.puct_score(c_puct, self.visits))

    def is_expanded(self) -> bool:
        return len(self.children) > 0


def evaluate_position(board: Board, model: ZicZacNet, device: torch.device) -> Tuple[List[float], float]:
    """Evaluate position using neural network."""
    model.eval()
    with torch.no_grad():
        tensor = board_to_tensor(board, device)
        log_policy, value = model(tensor)
        policy = torch.exp(log_policy).squeeze(0).cpu().tolist()
        value = value.item()
    return policy, value


def expand_node(node: MCTSNode, policy: List[float]) -> None:
    """Expand node by creating children for all legal moves."""
    for move in node.legal_moves:
        new_board = node.board.make_move(move)
        prior = policy[move]
        child = MCTSNode(new_board, parent=node, move=move, prior=prior)
        node.children[move] = child


def add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float) -> None:
    """Add Dirichlet noise to root node priors for exploration."""
    if not node.children:
        return
    noise = [random.gammavariate(alpha, 1) for _ in node.children]
    noise_sum = sum(noise)
    noise = [n / noise_sum for n in noise]
    for i, child in enumerate(node.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]


def mcts_search(root_board: Board, model: ZicZacNet, device: torch.device,
                num_simulations: int, c_puct: float,
                dirichlet_alpha: float = 0.0, dirichlet_epsilon: float = 0.0) -> Dict[int, int]:
    """Run MCTS from given position using neural network evaluation."""
    root = MCTSNode(root_board)

    policy, _ = evaluate_position(root_board, model, device)

    legal = root.legal_moves
    masked_policy = [policy[i] if i in legal else 0.0 for i in range(len(policy))]
    policy_sum = sum(masked_policy)
    if policy_sum > 0:
        masked_policy = [p / policy_sum for p in masked_policy]
    else:
        masked_policy = [1.0 / len(legal) if i in legal else 0.0 for i in range(len(policy))]

    expand_node(root, masked_policy)

    if dirichlet_alpha > 0:
        add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        while node.is_expanded() and not node.is_terminal:
            node = node.best_child(c_puct)
            search_path.append(node)

        if node.is_terminal:
            value = node.terminal_value
        else:
            policy, value = evaluate_position(node.board, model, device)
            legal = node.legal_moves
            if legal:
                masked_policy = [policy[i] if i in legal else 0.0 for i in range(len(policy))]
                policy_sum = sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = [p / policy_sum for p in masked_policy]
                else:
                    masked_policy = [1.0 / len(legal) if i in legal else 0.0 for i in range(len(policy))]
                expand_node(node, masked_policy)

        for i, bp_node in enumerate(reversed(search_path)):
            bp_node.visits += 1
            bp_node.value_sum += value if i % 2 == 0 else -value

    return {move: child.visits for move, child in root.children.items()}


def select_move_from_visits(visit_counts: Dict[int, int], temperature: float) -> int:
    """Select a move from MCTS visit counts using temperature."""
    moves = list(visit_counts.keys())
    visits = [visit_counts[m] for m in moves]

    if temperature == 0:
        return moves[visits.index(max(visits))]

    total = sum(visits)
    if total == 0:
        return random.choice(moves)

    probs = [(v / total) ** (1 / temperature) for v in visits]
    prob_sum = sum(probs)
    probs = [p / prob_sum for p in probs]

    r = random.random()
    cumulative = 0
    for move, prob in zip(moves, probs):
        cumulative += prob
        if r < cumulative:
            return move
    return moves[-1]


def visits_to_policy(visit_counts: Dict[int, int], board_size: int = BOARD_SIZE) -> List[float]:
    """Convert visit counts to policy distribution over all moves."""
    total = sum(visit_counts.values())
    policy = [0.0] * (board_size * board_size)
    for move, visits in visit_counts.items():
        policy[move] = visits / total if total > 0 else 0.0
    return policy


# =============================================================================
# Training Sample
# =============================================================================

@dataclass
class Sample:
    """Single training sample with MCTS policy target."""
    board_state: List[int]
    current_player: Player  # Added for turn indicator
    policy_target: List[float]
    outcome: float


# =============================================================================
# Self-Play
# =============================================================================

def play_game(model: ZicZacNet, device: torch.device,
              config: TrainConfig) -> Tuple[List[Sample], GameResult]:
    """Play a single self-play game using neural network MCTS."""
    board = Board()
    history = []

    while True:
        current_player = board.current_player()

        if board.move_count() < config.temperature_threshold:
            temp = config.temperature_start
        else:
            temp = config.temperature_end

        visit_counts = mcts_search(
            board, model, device,
            config.mcts_simulations, config.mcts_c_puct,
            config.dirichlet_alpha, config.dirichlet_epsilon
        )

        policy_target = visits_to_policy(visit_counts)
        move = select_move_from_visits(visit_counts, temp)

        history.append((board.state.copy(), current_player, policy_target))

        board = board.make_move(move)

        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            break

    samples = []
    for board_state, player, policy_target in history:
        if result == GameResult.DRAW:
            outcome = 0.0
        elif (result == GameResult.X_WINS and player == Player.X) or \
             (result == GameResult.O_WINS and player == Player.O):
            outcome = 1.0
        else:
            outcome = -1.0

        samples.append(Sample(board_state, player, policy_target, outcome))

    return samples, result


def _worker_play_games(args: Tuple) -> Tuple[List[Sample], int, int, int]:
    """Worker function for parallel game playing."""
    model_state_dict, num_games, config_dict = args

    torch.set_num_threads(1)

    config = TrainConfig(**config_dict)
    model = ZicZacNet(num_filters=config.num_filters)
    model.load_state_dict(model_state_dict)
    model.eval()
    device = torch.device("cpu")

    samples = []
    x_wins = 0
    o_wins = 0
    draws = 0

    for _ in range(num_games):
        game_samples, result = play_game(model, device, config)
        samples.extend(game_samples)

        if result == GameResult.X_WINS:
            x_wins += 1
        elif result == GameResult.O_WINS:
            o_wins += 1
        else:
            draws += 1

    return samples, x_wins, o_wins, draws


def self_play_games(model: ZicZacNet, device: torch.device,
                    config: TrainConfig) -> Tuple[List[Sample], dict]:
    """Play multiple self-play games (optionally in parallel)."""
    num_workers = config.num_workers
    if num_workers == 0:
        num_workers = mp.cpu_count()

    if num_workers == 1:
        all_samples = []
        x_wins = 0
        o_wins = 0
        draws = 0
        total_moves = 0

        for _ in range(config.games_per_iteration):
            samples, result = play_game(model, device, config)
            all_samples.extend(samples)
            total_moves += len(samples)

            if result == GameResult.X_WINS:
                x_wins += 1
            elif result == GameResult.O_WINS:
                o_wins += 1
            else:
                draws += 1

        stats = {
            "x_wins": x_wins,
            "o_wins": o_wins,
            "draws": draws,
            "avg_game_length": total_moves / config.games_per_iteration,
        }
        return all_samples, stats

    # Parallel mode
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    config_dict = {
        "games_per_iteration": config.games_per_iteration,
        "temperature_start": config.temperature_start,
        "temperature_end": config.temperature_end,
        "temperature_threshold": config.temperature_threshold,
        "mcts_simulations": config.mcts_simulations,
        "mcts_c_puct": config.mcts_c_puct,
        "dirichlet_alpha": config.dirichlet_alpha,
        "dirichlet_epsilon": config.dirichlet_epsilon,
        "num_filters": config.num_filters,
    }

    games_per_worker = config.games_per_iteration // num_workers
    remainder = config.games_per_iteration % num_workers

    worker_args = []
    for i in range(num_workers):
        n_games = games_per_worker + (1 if i < remainder else 0)
        if n_games > 0:
            worker_args.append((model_state_dict, n_games, config_dict))

    ctx = mp.get_context('fork')
    with ctx.Pool(num_workers) as pool:
        results = pool.map(_worker_play_games, worker_args)

    all_samples = []
    x_wins = 0
    o_wins = 0
    draws = 0

    for samples, xw, ow, dw in results:
        all_samples.extend(samples)
        x_wins += xw
        o_wins += ow
        draws += dw

    total_moves = len(all_samples)
    stats = {
        "x_wins": x_wins,
        "o_wins": o_wins,
        "draws": draws,
        "avg_game_length": total_moves / config.games_per_iteration,
    }

    return all_samples, stats


# =============================================================================
# Replay Buffer with Reanalyze (MuZero-style)
# =============================================================================

@dataclass
class PositionSample:
    """
    Minimal sample for reanalyze: just position and outcome.
    Policy target is regenerated via MCTS during training.
    """
    board_state: List[int]
    current_player: Player
    outcome: float  # Ground truth from game result


class ReplayBuffer:
    """
    Simple FIFO replay buffer with uniform sampling.

    Following AlphaZero/MuZero: no priority weighting, just recency.
    Old samples are evicted when buffer is full (FIFO).

    Stores PositionSample (no policy) - policy is regenerated via reanalyze.
    """

    def __init__(self, max_size: int):
        self.samples: List[PositionSample] = []
        self.max_size = max_size

    def add(self, samples: List[PositionSample]) -> None:
        """Add samples to buffer, evicting oldest if needed."""
        self.samples.extend(samples)
        # FIFO eviction
        if len(self.samples) > self.max_size:
            self.samples = self.samples[-self.max_size:]

    def sample_batch(self, batch_size: int) -> List[PositionSample]:
        """Sample batch uniformly at random."""
        if len(self.samples) == 0:
            return []
        n_samples = min(batch_size, len(self.samples))
        indices = np.random.choice(len(self.samples), n_samples, replace=False)
        return [self.samples[i] for i in indices]

    def __len__(self) -> int:
        return len(self.samples)

    def save(self, path: str) -> None:
        """Save buffer to file."""
        with open(path, 'wb') as f:
            pickle.dump(self.samples, f)

    def load(self, path: str) -> None:
        """Load buffer from file."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                # Handle old format (Sample with policy_target)
                if data and hasattr(data[0], 'policy_target'):
                    self.samples = [
                        PositionSample(s.board_state, s.current_player, s.outcome)
                        for s in data
                    ]
                else:
                    self.samples = data

    def stats(self) -> dict:
        """Return buffer statistics."""
        return {'size': len(self.samples)}


# =============================================================================
# Training with Reanalyze
# =============================================================================

def _worker_reanalyze(args: Tuple) -> List[Sample]:
    """Worker function for parallel reanalyze."""
    model_state_dict, positions_data, num_simulations, c_puct, num_filters = args

    torch.set_num_threads(1)

    model = ZicZacNet(num_filters=num_filters)
    model.load_state_dict(model_state_dict)
    model.eval()
    device = torch.device("cpu")

    samples = []
    for board_state, current_player, outcome in positions_data:
        board = Board(board_state)

        visit_counts = mcts_search(
            board, model, device,
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=0,
            dirichlet_epsilon=0
        )

        policy_target = visits_to_policy(visit_counts)

        samples.append(Sample(
            board_state=board_state,
            current_player=current_player,
            policy_target=policy_target,
            outcome=outcome
        ))

    return samples


def reanalyze_batch(positions: List[PositionSample], model: ZicZacNet,
                    device: torch.device, num_simulations: int,
                    c_puct: float = 1.5, num_workers: int = 1,
                    num_filters: int = 64) -> List[Sample]:
    """
    Reanalyze positions using current model's MCTS to get fresh policy targets.

    This is the key MuZero insight: old positions get new policy targets
    based on the current model's understanding, not stale targets from
    when the position was first generated.
    """
    if num_workers <= 1:
        # Sequential version
        samples = []
        for pos in positions:
            board = Board(pos.board_state)

            visit_counts = mcts_search(
                board, model, device,
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=0,
                dirichlet_epsilon=0
            )

            policy_target = visits_to_policy(visit_counts)

            samples.append(Sample(
                board_state=pos.board_state,
                current_player=pos.current_player,
                policy_target=policy_target,
                outcome=pos.outcome
            ))
        return samples

    # Parallel version
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Convert positions to serializable format
    positions_data = [
        (pos.board_state, pos.current_player, pos.outcome)
        for pos in positions
    ]

    # Split positions among workers
    chunk_size = (len(positions_data) + num_workers - 1) // num_workers
    chunks = [
        positions_data[i:i + chunk_size]
        for i in range(0, len(positions_data), chunk_size)
    ]

    worker_args = [
        (model_state_dict, chunk, num_simulations, c_puct, num_filters)
        for chunk in chunks if chunk
    ]

    ctx = mp.get_context('fork')
    with ctx.Pool(len(worker_args)) as pool:
        results = pool.map(_worker_reanalyze, worker_args)

    # Flatten results
    samples = []
    for chunk_samples in results:
        samples.extend(chunk_samples)

    return samples


def prepare_batch(samples: List[Sample], device: torch.device):
    """Convert samples to tensors for training."""
    # Reconstruct boards with current_player for turn indicator
    boards = []
    for s in samples:
        board = Board(s.board_state)
        # Verify current player matches (for debugging)
        boards.append(board)

    states = boards_to_tensor(boards, device)

    policy_targets = torch.tensor(
        [s.policy_target for s in samples],
        dtype=torch.float32, device=device
    )

    outcomes = torch.tensor(
        [s.outcome for s in samples],
        dtype=torch.float32, device=device
    )

    return states, policy_targets, outcomes


def train_on_buffer(model: ZicZacNet, optimizer: optim.Optimizer,
                    replay_buffer: ReplayBuffer, config: TrainConfig,
                    device: torch.device,
                    fresh_samples: Optional[List[Sample]] = None,
                    tactical_samples: Optional[List[Sample]] = None) -> dict:
    """
    Train model on samples from replay buffer with reanalyze.

    - Fresh samples (from this iteration's self-play) already have good policy
      from MCTS, so we use them directly without reanalyze.
    - Old samples from buffer are reanalyzed with current model to get
      fresh policy targets (the MuZero insight).
    - Tactical samples have known-correct policy, skip reanalyze.
    """
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # Start with fresh samples (already have policy from self-play MCTS)
    training_samples = list(fresh_samples) if fresh_samples else []

    # Sample OLD positions from buffer for reanalyze
    # These benefit from being re-evaluated with the improved model
    batches_per_epoch_target = 40
    reanalyze_sample_size = min(
        len(replay_buffer),
        config.batch_size * batches_per_epoch_target
    )
    position_batch = replay_buffer.sample_batch(reanalyze_sample_size)

    if len(position_batch) >= 8:
        # Reanalyze old positions with current model
        model.eval()
        reanalyzed = reanalyze_batch(
            position_batch, model, device,
            num_simulations=config.reanalyze_simulations,
            c_puct=config.mcts_c_puct,
            num_workers=config.num_workers,
            num_filters=config.num_filters
        )
        training_samples.extend(reanalyzed)

    # Add tactical samples (known-correct policy, skip reanalyze)
    if tactical_samples:
        training_samples.extend(tactical_samples)

    if len(training_samples) < 8:
        return {"policy_loss": 0, "value_loss": 0}

    # Train multiple epochs on the combined samples
    model.train()
    for epoch in range(config.epochs_per_iteration):
        # Shuffle samples each epoch
        random.shuffle(training_samples)

        # Process in batches
        for i in range(0, len(training_samples), config.batch_size):
            batch = training_samples[i:i + config.batch_size]
            if len(batch) < 8:
                continue

            states, policy_targets, outcomes = prepare_batch(batch, device)

            log_policy, value = model(states)

            policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()
            value_loss = nn.functional.mse_loss(value.squeeze(), outcomes)

            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1

    return {
        "policy_loss": total_policy_loss / max(num_batches, 1),
        "value_loss": total_value_loss / max(num_batches, 1),
    }


# =============================================================================
# Checkpoint Management
# =============================================================================

def get_buffer_path(model_path: str) -> str:
    return model_path.replace('.pt', '_buffer.pkl')


def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
    if not os.path.exists(checkpoint_dir):
        return None, 0

    pattern = os.path.join(checkpoint_dir, "model_iter_*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0

    def get_iter_num(path: str) -> int:
        match = re.search(r'model_iter_(\d+)\.pt', path)
        return int(match.group(1)) if match else 0

    latest = max(checkpoints, key=get_iter_num)
    iteration = get_iter_num(latest)

    return latest, iteration


def export_weights_to_web(model: ZicZacNet, output_path: str, iteration: int = 0) -> None:
    """Export model weights to JSON for web inference."""
    def tensor_to_list(tensor):
        if tensor.dim() == 0:
            return tensor.item()
        return [tensor_to_list(t) for t in tensor]

    weights = {}
    for name, param in model.named_parameters():
        weights[name] = tensor_to_list(param.detach().cpu())
    for name, buf in model.named_buffers():
        weights[name] = tensor_to_list(buf.detach().cpu())

    # Add metadata
    weights["_version"] = 2
    weights["_iteration"] = iteration

    with open(output_path, "w") as f:
        json.dump(weights, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  -> Exported weights to web: {output_path} ({size_kb:.1f} KB)")


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: Optional[TrainConfig] = None, num_iterations: int = 100,
          resume_from: Optional[str] = None, auto_resume: bool = True) -> ZicZacNet:
    """Main training function with AlphaZero-style MCTS and evaluation."""
    if config is None:
        config = TrainConfig()

    device = get_device()
    print(f"Training on device: {device}")
    print(f"Model: 3-channel input (X, O, turn indicator)")
    print(f"MCTS: {config.mcts_simulations} simulations per move")
    num_workers = config.num_workers if config.num_workers > 0 else mp.cpu_count()
    if num_workers > 1:
        print(f"Parallel self-play with {num_workers} workers")

    if not auto_resume and resume_from is None:
        if os.path.exists(config.checkpoint_dir):
            import shutil
            shutil.rmtree(config.checkpoint_dir)
            print(f"Cleared checkpoint directory: {config.checkpoint_dir}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Determine starting point
    start_iteration = 0
    resume_checkpoint_path = None

    if resume_from:
        resume_checkpoint_path = resume_from
        print(f"Resuming from {resume_from}")
        model = ZicZacNet(num_filters=config.num_filters)
        model.load_state_dict(torch.load(resume_from, map_location=device))
        match = re.search(r'model_iter_(\d+)\.pt', resume_from)
        if match:
            start_iteration = int(match.group(1))
    elif auto_resume:
        latest_path, latest_iter = find_latest_checkpoint(config.checkpoint_dir)
        if latest_path:
            resume_checkpoint_path = latest_path
            print(f"Auto-resuming from {latest_path} (iteration {latest_iter})")
            model = ZicZacNet(num_filters=config.num_filters)
            model.load_state_dict(torch.load(latest_path, map_location=device))
            start_iteration = latest_iter
        else:
            print("No existing checkpoints found, starting fresh")
            model = ZicZacNet(num_filters=config.num_filters)
    else:
        model = ZicZacNet(num_filters=config.num_filters)

    model = model.to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    if resume_checkpoint_path:
        buffer_path = get_buffer_path(resume_checkpoint_path)
        if os.path.exists(buffer_path):
            replay_buffer.load(buffer_path)
            print(f"Loaded replay buffer: {len(replay_buffer)} samples")

    print(f"Reanalyze: {config.reanalyze_simulations} MCTS sims, ~{config.batch_size * 40} positions/iter")

    # Track best model for comparison (AlphaZero-style)
    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_iteration = start_iteration

    end_iteration = start_iteration + num_iterations
    print(f"\nStarting training: iterations {start_iteration + 1} to {end_iteration}")
    print(f"Model selection: win rate > {config.win_rate_threshold:.0%} vs previous best")
    print(f"Early stopping: {config.early_stopping_patience} iterations without improvement")
    print(f"Evaluation every {config.eval_interval} iterations")
    print("=" * 80)

    start_time = time.time()

    patience_counter = 0
    iteration = start_iteration

    for iteration in range(start_iteration + 1, end_iteration + 1):
        iter_start = time.time()

        # Self-play phase
        # Returns Sample objects WITH policy_target from MCTS
        fresh_samples, game_stats = self_play_games(model, device, config)

        # Generate tactical samples with known-correct policy
        tactical_training_samples = []
        if config.tactical_samples_per_type > 0:
            tactical_samples = generate_tactical_samples(
                n_per_type=config.tactical_samples_per_type,
                seed=None  # Random each time
            )
            tactical_training_samples = [
                tactical_sample_to_training_sample(ts)
                for ts in tactical_samples
            ]

        # Training phase
        # - Fresh samples: use self-play policy directly (no reanalyze)
        # - Buffer samples: reanalyze with current model
        # - Tactical samples: use known-correct policy
        train_stats = train_on_buffer(
            model, optimizer, replay_buffer, config, device,
            fresh_samples=fresh_samples,
            tactical_samples=tactical_training_samples
        )

        # NOW add fresh samples to buffer (as PositionSample for future reanalyze)
        position_samples = [
            PositionSample(s.board_state, s.current_player, s.outcome)
            for s in fresh_samples
        ]
        replay_buffer.add(position_samples)

        # Logging
        if iteration % config.log_interval == 0:
            iter_time = time.time() - iter_start
            timestamp = datetime.now().strftime("%H:%M:%S")

            print(f"[{timestamp}] Iter {iteration:4d} | "
                  f"Games: X={game_stats['x_wins']:2d} O={game_stats['o_wins']:2d} D={game_stats['draws']:2d} | "
                  f"Avg len: {game_stats['avg_game_length']:.1f} | "
                  f"P_loss: {train_stats['policy_loss']:.4f} | "
                  f"V_loss: {train_stats['value_loss']:.4f} | "
                  f"Buffer: {len(replay_buffer):5d} | "
                  f"Time: {iter_time:.1f}s")

        # Periodic evaluation and model selection
        if iteration % config.eval_interval == 0:
            print(f"\n--- Evaluation at iteration {iteration} ---")

            # Vs random
            vs_random = eval_vs_random(model, device, config.eval_games_vs_random)
            print(f"  vs Random: {vs_random}")

            # Tactical test (randomly generated positions for robustness)
            correct, total, tactical_results = eval_tactical(
                model, device, n_per_type=config.tactical_eval_per_type, seed=iteration
            )
            tactical_rate = correct / total if total > 0 else 0.0
            print(f"  Tactical: {correct}/{total} ({tactical_rate:.0%})")
            failures = [tr for tr in tactical_results if not tr.passed]
            if failures:
                for tr in failures[:2]:  # Show first 2 failures with grids
                    print(tr.render())

            # Vs current best (AlphaZero-style model selection)
            best_model = ZicZacNet(num_filters=config.num_filters)
            best_model.load_state_dict(best_model_state)
            best_model = best_model.to(device)
            vs_best = eval_model_vs_model(model, best_model, device, config.eval_games_for_best)
            print(f"  vs Best (iter {best_iteration}): {vs_best}")

            # Check if current model beats the best
            # Requires: win rate > threshold AND tactical pass rate >= tactical threshold
            wins_vs_best = vs_best.win_rate > config.win_rate_threshold
            passes_tactical = tactical_rate >= config.tactical_pass_threshold
            is_new_best = wins_vs_best and passes_tactical

            if is_new_best:
                print(f"  ** NEW BEST MODEL ** (win rate {vs_best.win_rate:.1%} > {config.win_rate_threshold:.0%}, tactical {tactical_rate:.0%})")
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_iteration = iteration
                patience_counter = 0
                # Save best model
                best_path = os.path.join(config.checkpoint_dir, "model_best.pt")
                save_model(model, best_path)
                replay_buffer.save(get_buffer_path(best_path))
                # Export to web
                web_weights_path = os.path.join(os.path.dirname(__file__), "..", "web", "public", "weights_v2.json")
                export_weights_to_web(model, web_weights_path, iteration)
            elif wins_vs_best and not passes_tactical:
                print(f"  Beats previous best but FAILED tactical gate ({tactical_rate:.0%} < {config.tactical_pass_threshold:.0%})")
                patience_counter += 1
            else:
                patience_counter += 1

            print()

        # Early stopping check
        if patience_counter >= config.early_stopping_patience:
            print(f"\n{'=' * 80}")
            print(f"Early stopping: no improvement for {patience_counter} evaluation cycles")
            print(f"Best model at iteration {best_iteration}")
            break

        # Checkpointing
        if iteration % config.checkpoint_interval == 0:
            path = os.path.join(config.checkpoint_dir, f"model_iter_{iteration}.pt")
            save_model(model, path)
            replay_buffer.save(get_buffer_path(path))
            print(f"  -> Saved checkpoint: {path}")

    # Final saves
    actual_final_iteration = iteration

    final_iter_path = os.path.join(config.checkpoint_dir, f"model_iter_{actual_final_iteration}.pt")
    if not os.path.exists(final_iter_path):
        save_model(model, final_iter_path)
        replay_buffer.save(get_buffer_path(final_iter_path))
        print(f"  -> Saved checkpoint: {final_iter_path}")

    final_path = os.path.join(config.checkpoint_dir, "model_final.pt")
    save_model(model, final_path)
    replay_buffer.save(get_buffer_path(final_path))

    # Final evaluation
    print(f"\n{'=' * 80}")
    print("Final Evaluation:")
    vs_random = eval_vs_random(model, device, 100)
    print(f"  vs Random (100 games): {vs_random}")
    correct, total, _ = eval_tactical(model, device, n_per_type=config.tactical_eval_per_type)
    print(f"  Tactical: {correct}/{total} ({correct/total:.0%})" if total > 0 else "  Tactical: 0/0")

    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"Training complete at iteration {actual_final_iteration}")
    print(f"Final model: {final_path}")
    print(f"Best model: {os.path.join(config.checkpoint_dir, 'model_best.pt')}")
    print(f"Total training time: {total_time / 60:.1f} minutes")

    return model


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Zic-Zac-Zoe AI v2 (with turn indicator)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per iteration")
    parser.add_argument("--simulations", type=int, default=200,
                        help="MCTS simulations per move")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignoring existing checkpoints")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--filters", type=int, default=64,
                        help="Number of conv filters")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (evaluation cycles)")
    parser.add_argument("--win-threshold", type=float, default=0.55,
                        help="Win rate threshold for new best model")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="PUCT exploration constant")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for self-play")
    parser.add_argument("--eval-interval", type=int, default=10,
                        help="Run evaluation every N iterations")
    parser.add_argument("--tactical-per-type", type=int, default=20,
                        help="Tactical samples per pattern type per iteration")

    args = parser.parse_args()

    config = TrainConfig(
        games_per_iteration=args.games,
        mcts_simulations=args.simulations,
        learning_rate=args.lr,
        num_filters=args.filters,
        early_stopping_patience=args.patience if not args.no_early_stop else 999999999,
        win_rate_threshold=args.win_threshold,
        mcts_c_puct=args.c_puct,
        num_workers=args.workers,
        eval_interval=args.eval_interval,
        tactical_samples_per_type=args.tactical_per_type,
    )

    train(
        config=config,
        num_iterations=args.iterations,
        resume_from=args.resume,
        auto_resume=not args.fresh,
    )
