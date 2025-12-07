"""
Training Pipeline with MCTS for Zic-Zac-Zoe AI

Uses Monte Carlo Tree Search with random rollouts to generate
higher-quality training data than pure self-play.

Training loop:
1. Self-play using MCTS for move selection
2. Train policy head to match MCTS visit distribution
3. Train value head to match game outcome
4. Repeat
"""

import os
import re
import glob
import time
import random
import pickle
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE
from model import (
    ZicZacNet, board_to_tensor, boards_to_tensor,
    save_model, get_device
)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Self-play settings
    games_per_iteration: int = 50        # Fewer games (MCTS is slower but higher quality)
    temperature_start: float = 1.0       # High temp early in game (exploration)
    temperature_end: float = 0.5         # Lower temp late in game (exploitation)
    temperature_threshold: int = 10      # Move number to switch temperatures

    # MCTS settings
    mcts_simulations: int = 50           # Simulations per move
    mcts_c: float = 1.41                 # UCB exploration constant (sqrt(2))

    # Training settings
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs_per_iteration: int = 5
    replay_buffer_size: int = 25000

    # Model settings
    num_filters: int = 64

    # Checkpointing - separate directory from non-MCTS training
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints_mcts"

    # Logging
    log_interval: int = 1

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.005


# =============================================================================
# MCTS Implementation
# =============================================================================

class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[int] = None):
        self.board = board
        self.parent = parent
        self.move = move  # Move that led to this node

        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.wins = 0.0  # From perspective of player who just moved

        # Cache legal moves and terminal status
        self._legal_moves: Optional[List[int]] = None
        self._is_terminal: Optional[bool] = None
        self._terminal_result: Optional[GameResult] = None

    @property
    def legal_moves(self) -> List[int]:
        if self._legal_moves is None:
            self._legal_moves = self.board.get_legal_moves()
        return self._legal_moves

    @property
    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            if self.move is None:
                # Root node with no move yet
                self._is_terminal = False
                self._terminal_result = GameResult.ONGOING
            else:
                result = check_result_fast(self.board, self.move)
                self._is_terminal = result != GameResult.ONGOING
                self._terminal_result = result
        return self._is_terminal

    @property
    def terminal_result(self) -> GameResult:
        _ = self.is_terminal  # Ensure computed
        return self._terminal_result

    def is_fully_expanded(self) -> bool:
        return len(self.children) == len(self.legal_moves)

    def ucb_score(self, c: float) -> float:
        """Upper Confidence Bound score."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float) -> 'MCTSNode':
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda n: n.ucb_score(c))

    def expand(self) -> 'MCTSNode':
        """Expand one unexplored move."""
        unexplored = [m for m in self.legal_moves if m not in self.children]
        move = random.choice(unexplored)
        new_board = self.board.make_move(move)
        child = MCTSNode(new_board, parent=self, move=move)
        self.children[move] = child
        return child


def random_rollout(board: Board, last_move: int) -> float:
    """
    Play random moves until game ends.
    Returns result from perspective of player who just moved (made last_move).
    """
    current_board = board
    current_last_move = last_move

    # Check if already terminal
    result = check_result_fast(current_board, current_last_move)
    if result != GameResult.ONGOING:
        # Determine who just moved (opponent of current player)
        just_moved = Player.O if current_board.current_player() == Player.X else Player.X
        return result_for_player(result, just_moved)

    # Play out randomly
    while True:
        moves = current_board.get_legal_moves()
        if not moves:
            return 0.0  # Draw

        move = random.choice(moves)
        current_board = current_board.make_move(move)
        current_last_move = move

        result = check_result_fast(current_board, move)
        if result != GameResult.ONGOING:
            # Determine who made the original last_move
            # We need result from their perspective
            original_player = Player.O if board.current_player() == Player.X else Player.X
            return result_for_player(result, original_player)


def result_for_player(result: GameResult, player: Player) -> float:
    """Convert game result to value for given player."""
    if result == GameResult.DRAW:
        return 0.0
    elif result == GameResult.X_WINS:
        return 1.0 if player == Player.X else -1.0
    else:  # O_WINS
        return 1.0 if player == Player.O else -1.0


def mcts_search(root_board: Board, num_simulations: int, c: float) -> Dict[int, int]:
    """
    Run MCTS from given position.

    Returns:
        Dictionary mapping move -> visit count
    """
    root = MCTSNode(root_board)

    for _ in range(num_simulations):
        node = root

        # Selection: traverse tree using UCB
        while not node.is_terminal and node.is_fully_expanded():
            node = node.best_child(c)

        # Expansion: add a new child if not terminal
        if not node.is_terminal and not node.is_fully_expanded():
            node = node.expand()

        # Simulation: random rollout
        if node.is_terminal:
            # Use terminal result directly
            result = node.terminal_result
            just_moved = Player.O if node.board.current_player() == Player.X else Player.X
            value = result_for_player(result, just_moved)
        else:
            value = random_rollout(node.board, node.move)

        # Backpropagation
        while node is not None:
            node.visits += 1
            # Value is from perspective of player who moved to reach node's parent
            # So we need to flip sign as we go up
            node.wins += value
            value = -value  # Flip for opponent
            node = node.parent

    # Return visit counts for root's children
    return {move: child.visits for move, child in root.children.items()}


def select_move_from_visits(visit_counts: Dict[int, int], temperature: float) -> int:
    """
    Select a move from MCTS visit counts using temperature.
    """
    moves = list(visit_counts.keys())
    visits = [visit_counts[m] for m in moves]

    if temperature == 0:
        # Greedy
        return moves[visits.index(max(visits))]

    # Apply temperature
    total = sum(visits)
    probs = [(v / total) ** (1 / temperature) for v in visits]
    prob_sum = sum(probs)
    probs = [p / prob_sum for p in probs]

    # Sample
    r = random.random()
    cumulative = 0
    for move, prob in zip(moves, probs):
        cumulative += prob
        if r < cumulative:
            return move
    return moves[-1]


def visits_to_policy(visit_counts: Dict[int, int], board_size: int = BOARD_SIZE) -> List[float]:
    """
    Convert visit counts to policy distribution over all moves.
    """
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
    board_state: List[int]      # Flat board state
    policy_target: List[float]  # MCTS visit distribution (36 values)
    outcome: float              # Game outcome: +1 win, -1 loss, 0 draw


# =============================================================================
# Self-Play with MCTS
# =============================================================================

def play_game_mcts(config: TrainConfig) -> Tuple[List[Sample], GameResult]:
    """
    Play a single self-play game using MCTS for move selection.
    """
    board = Board()
    history = []  # (board_state, policy_target, player) tuples

    last_move = None

    while True:
        current_player = board.current_player()

        # Select temperature based on move number
        if board.move_count() < config.temperature_threshold:
            temp = config.temperature_start
        else:
            temp = config.temperature_end

        # Run MCTS
        visit_counts = mcts_search(board, config.mcts_simulations, config.mcts_c)

        # Convert to policy target
        policy_target = visits_to_policy(visit_counts)

        # Select move
        move = select_move_from_visits(visit_counts, temp)

        # Record state before move
        history.append((board.state.copy(), policy_target, current_player))

        # Make move
        board = board.make_move(move)
        last_move = move

        # Check for game end
        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            break

    # Convert history to samples with outcome labels
    samples = []
    for board_state, policy_target, player in history:
        if result == GameResult.DRAW:
            outcome = 0.0
        elif (result == GameResult.X_WINS and player == Player.X) or \
             (result == GameResult.O_WINS and player == Player.O):
            outcome = 1.0
        else:
            outcome = -1.0

        samples.append(Sample(board_state, policy_target, outcome))

    return samples, result


def self_play_games_mcts(config: TrainConfig) -> Tuple[List[Sample], dict]:
    """
    Play multiple self-play games using MCTS.
    """
    all_samples = []

    x_wins = 0
    o_wins = 0
    draws = 0
    total_moves = 0

    for _ in range(config.games_per_iteration):
        samples, result = play_game_mcts(config)
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


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Fixed-size buffer storing recent training samples."""

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, samples: List[Sample]) -> None:
        self.buffer.extend(samples)

    def get_all(self) -> List[Sample]:
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str) -> None:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                samples = pickle.load(f)
                self.buffer.clear()
                self.buffer.extend(samples)


# =============================================================================
# Training
# =============================================================================

def prepare_batch(samples: List[Sample], device: torch.device):
    """Convert samples to tensors for training."""
    boards = [Board(s.board_state) for s in samples]
    states = boards_to_tensor(boards, device)

    # Policy targets: full distribution
    policy_targets = torch.tensor(
        [s.policy_target for s in samples],
        dtype=torch.float32, device=device
    )

    outcomes = torch.tensor(
        [s.outcome for s in samples],
        dtype=torch.float32, device=device
    )

    return states, policy_targets, outcomes


def train_on_samples(model: ZicZacNet, optimizer: optim.Optimizer,
                     samples: List[Sample], config: TrainConfig,
                     device: torch.device) -> dict:
    """
    Train model on collected samples.

    Uses:
    - KL divergence for policy (match MCTS distribution)
    - MSE loss for value (predict game outcome)
    """
    model.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    samples = list(samples)
    random.shuffle(samples)

    for epoch in range(config.epochs_per_iteration):
        for i in range(0, len(samples), config.batch_size):
            batch = samples[i:i + config.batch_size]
            if len(batch) < 8:
                continue

            states, policy_targets, outcomes = prepare_batch(batch, device)

            # Forward pass
            log_policy, value = model(states)

            # Policy loss: cross-entropy with soft targets
            # -sum(target * log(pred)) for each sample
            policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()

            # Value loss: MSE
            value_loss = nn.functional.mse_loss(value.squeeze(), outcomes)

            # Combined loss
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


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: Optional[TrainConfig] = None, num_iterations: int = 100,
          resume_from: Optional[str] = None, auto_resume: bool = True) -> ZicZacNet:
    """
    Main training function with MCTS.
    """
    if config is None:
        config = TrainConfig()

    device = get_device()
    print(f"Training on device: {device}")
    print(f"Using MCTS with {config.mcts_simulations} simulations per move")

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

    # Training loop
    end_iteration = start_iteration + num_iterations
    print(f"\nStarting training: iterations {start_iteration + 1} to {end_iteration}")
    print(f"Early stopping: patience={config.early_stopping_patience}, min_delta={config.early_stopping_min_delta}")
    print("=" * 70)

    start_time = time.time()

    # Early stopping state
    best_combined_loss = float('inf')
    patience_counter = 0
    best_iteration = 0
    iteration = start_iteration

    for iteration in range(start_iteration + 1, end_iteration + 1):
        iter_start = time.time()

        # Self-play phase with MCTS
        samples, game_stats = self_play_games_mcts(config)
        replay_buffer.add(samples)

        # Training phase
        if len(replay_buffer) >= config.batch_size:
            train_stats = train_on_samples(
                model, optimizer, replay_buffer.get_all(), config, device
            )
        else:
            train_stats = {"policy_loss": 0, "value_loss": 0}

        # Early stopping check
        combined_loss = train_stats['policy_loss'] + train_stats['value_loss']
        is_new_best = combined_loss < best_combined_loss - config.early_stopping_min_delta
        if is_new_best:
            best_combined_loss = combined_loss
            patience_counter = 0
            best_iteration = iteration
            best_path = os.path.join(config.checkpoint_dir, "model_best.pt")
            save_model(model, best_path)
            replay_buffer.save(get_buffer_path(best_path))
        else:
            patience_counter += 1

        # Logging
        if iteration % config.log_interval == 0:
            iter_time = time.time() - iter_start
            best_marker = " *" if is_new_best else ""

            print(f"Iter {iteration:4d} | "
                  f"Games: X={game_stats['x_wins']:2d} O={game_stats['o_wins']:2d} D={game_stats['draws']:2d} | "
                  f"Avg len: {game_stats['avg_game_length']:.1f} | "
                  f"P_loss: {train_stats['policy_loss']:.4f} | "
                  f"V_loss: {train_stats['value_loss']:.4f} | "
                  f"Buffer: {len(replay_buffer):5d} | "
                  f"Time: {iter_time:.1f}s{best_marker}")

        # Check if should stop early
        if patience_counter >= config.early_stopping_patience:
            print(f"\n{'=' * 70}")
            print(f"Early stopping triggered after {patience_counter} iterations without improvement")
            print(f"Best combined loss: {best_combined_loss:.4f} at iteration {best_iteration}")
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

    total_time = time.time() - start_time
    stopped_early = actual_final_iteration < end_iteration

    print(f"\n{'=' * 70}")
    if stopped_early:
        print(f"Training stopped early at iteration {actual_final_iteration}")
        print(f"Best model (loss={best_combined_loss:.4f}) saved at iteration {best_iteration}")
    else:
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

    parser = argparse.ArgumentParser(description="Train Zic-Zac-Zoe AI with MCTS")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=50,
                        help="Games per iteration (default: 50, less than non-MCTS)")
    parser.add_argument("--simulations", type=int, default=50,
                        help="MCTS simulations per move")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignoring existing checkpoints")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--filters", type=int, default=64,
                        help="Number of conv filters")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Minimum loss improvement")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")

    args = parser.parse_args()

    config = TrainConfig(
        games_per_iteration=args.games,
        mcts_simulations=args.simulations,
        learning_rate=args.lr,
        num_filters=args.filters,
        early_stopping_patience=args.patience if not args.no_early_stop else 999999999,
        early_stopping_min_delta=args.min_delta,
    )

    train(
        config=config,
        num_iterations=args.iterations,
        resume_from=args.resume,
        auto_resume=not args.fresh,
    )
