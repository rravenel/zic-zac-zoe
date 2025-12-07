"""
Training Pipeline for Zic-Zac-Zoe AI

Training loop:
1. Self-play: generate games using current model
2. Collect training data: (state, move, outcome) tuples
3. Train model on collected data
4. Repeat

No MCTS - uses raw policy output with temperature for exploration.
"""

import os
import re
import glob
import time
import random
import pickle
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE
from model import (
    ZicZacNet, board_to_tensor, boards_to_tensor,
    select_move, save_model, get_device
)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Self-play settings
    games_per_iteration: int = 100       # Games to play each iteration
    temperature_start: float = 1.0       # High temp early in game (exploration)
    temperature_end: float = 0.5         # Lower temp late in game (exploitation)
    temperature_threshold: int = 10      # Move number to switch temperatures

    # Training settings
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs_per_iteration: int = 5        # Training epochs per batch of games
    replay_buffer_size: int = 25000      # Max samples to keep in memory

    # Model settings
    num_filters: int = 64

    # Checkpointing
    checkpoint_interval: int = 10        # Save model every N iterations
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_interval: int = 1                # Log stats every N iterations

    # Early stopping
    early_stopping_patience: int = 20    # Stop if no improvement for N iterations
    early_stopping_min_delta: float = 0.005  # Min improvement to count as progress


# =============================================================================
# Training Sample
# =============================================================================

@dataclass
class Sample:
    """Single training sample."""
    board_state: List[int]  # Flat board state
    move: int               # Move taken (0-35)
    outcome: float          # Game outcome from this player's perspective: +1 win, -1 loss, 0 draw


# =============================================================================
# Self-Play
# =============================================================================

def play_game(model: ZicZacNet, config: TrainConfig,
              device: torch.device) -> Tuple[List[Sample], GameResult]:
    """
    Play a single self-play game.

    Args:
        model: Neural network (plays both sides)
        config: Training configuration
        device: Computation device

    Returns:
        samples: List of training samples from this game
        result: Final game result
    """
    board = Board()
    history = []  # (board_state, move, player) tuples

    # Play until game ends
    while True:
        current_player = board.current_player()

        # Select temperature based on move number
        if board.move_count() < config.temperature_threshold:
            temp = config.temperature_start
        else:
            temp = config.temperature_end

        # Get move from model
        move = select_move(model, board, temperature=temp, device=device)

        # Record state before move
        history.append((board.state.copy(), move, current_player))

        # Make move
        board = board.make_move(move)

        # Check for game end
        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            break

    # Convert history to samples with outcome labels
    samples = []
    for board_state, move, player in history:
        # Determine outcome from this player's perspective
        if result == GameResult.DRAW:
            outcome = 0.0
        elif (result == GameResult.X_WINS and player == Player.X) or \
             (result == GameResult.O_WINS and player == Player.O):
            outcome = 1.0  # Win
        else:
            outcome = -1.0  # Loss

        samples.append(Sample(board_state, move, outcome))

    return samples, result


def self_play_games(model: ZicZacNet, config: TrainConfig,
                    device: torch.device) -> Tuple[List[Sample], dict]:
    """
    Play multiple self-play games to generate training data.

    Returns:
        samples: List of all training samples
        stats: Dictionary with game statistics
    """
    model.eval()
    all_samples = []

    x_wins = 0
    o_wins = 0
    draws = 0
    total_moves = 0

    for _ in range(config.games_per_iteration):
        samples, result = play_game(model, config, device)
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
# Training
# =============================================================================

class ReplayBuffer:
    """
    Fixed-size buffer storing recent training samples.
    Old samples are discarded when buffer is full.
    """

    def __init__(self, max_size: int):
        self.buffer = deque(maxlen=max_size)

    def add(self, samples: List[Sample]) -> None:
        """Add samples to buffer."""
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> List[Sample]:
        """Randomly sample from buffer."""
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def get_all(self) -> List[Sample]:
        """Get all samples."""
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str) -> None:
        """Save buffer to disk."""
        with open(path, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: str) -> None:
        """Load buffer from disk."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                samples = pickle.load(f)
                self.buffer.clear()
                self.buffer.extend(samples)


def prepare_batch(samples: List[Sample], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert samples to tensors for training.

    Returns:
        states: (batch, 2, 6, 6) board states
        moves: (batch,) target moves
        outcomes: (batch,) target values
    """
    boards = [Board(s.board_state) for s in samples]
    states = boards_to_tensor(boards, device)
    moves = torch.tensor([s.move for s in samples], dtype=torch.long, device=device)
    outcomes = torch.tensor([s.outcome for s in samples], dtype=torch.float32, device=device)

    return states, moves, outcomes


def train_on_samples(model: ZicZacNet, optimizer: optim.Optimizer,
                     samples: List[Sample], config: TrainConfig,
                     device: torch.device) -> dict:
    """
    Train model on collected samples.

    Uses:
    - Cross-entropy loss for policy (predicting the move taken)
    - MSE loss for value (predicting game outcome)

    Returns:
        Dictionary with loss statistics
    """
    model.train()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # Shuffle samples
    samples = list(samples)
    random.shuffle(samples)

    # Train for multiple epochs
    for epoch in range(config.epochs_per_iteration):
        # Process in batches
        for i in range(0, len(samples), config.batch_size):
            batch = samples[i:i + config.batch_size]
            if len(batch) < 8:  # Skip tiny batches
                continue

            states, moves, outcomes = prepare_batch(batch, device)

            # Forward pass
            log_policy, value = model(states)

            # Policy loss: cross-entropy with the move that was played
            # This teaches the network to predict moves that led to wins
            policy_loss = nn.functional.nll_loss(log_policy, moves)

            # Value loss: MSE between predicted value and actual outcome
            value_loss = nn.functional.mse_loss(value.squeeze(), outcomes)

            # Combined loss (policy weighted more heavily)
            loss = policy_loss + 0.5 * value_loss

            # Backward pass
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
    """Get the buffer path corresponding to a model checkpoint."""
    return model_path.replace('.pt', '_buffer.pkl')


def find_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
    """
    Find the latest checkpoint in the checkpoint directory.

    Returns:
        path: Path to latest checkpoint, or None if no checkpoints exist
        iteration: Iteration number of the checkpoint (0 if none found)
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0

    # Look for model_iter_N.pt files
    pattern = os.path.join(checkpoint_dir, "model_iter_*.pt")
    checkpoints = glob.glob(pattern)

    if not checkpoints:
        return None, 0

    # Extract iteration numbers and find max
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
    Main training function.

    Args:
        config: Training configuration (uses defaults if None)
        num_iterations: Number of NEW training iterations to run
        resume_from: Path to specific checkpoint to resume from
        auto_resume: If True and resume_from is None, auto-detect latest checkpoint

    Returns:
        Trained model
    """
    if config is None:
        config = TrainConfig()

    device = get_device()
    print(f"Training on device: {device}")

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Determine starting point
    start_iteration = 0
    resume_checkpoint_path = None

    if resume_from:
        # Explicit checkpoint specified
        resume_checkpoint_path = resume_from
        print(f"Resuming from {resume_from}")
        model = ZicZacNet(num_filters=config.num_filters)
        model.load_state_dict(torch.load(resume_from, map_location=device))
        # Extract iteration number from path
        match = re.search(r'model_iter_(\d+)\.pt', resume_from)
        if match:
            start_iteration = int(match.group(1))
    elif auto_resume:
        # Auto-detect latest checkpoint
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

    # Print model info
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Replay buffer
    replay_buffer = ReplayBuffer(config.replay_buffer_size)

    # Load buffer if resuming
    if resume_checkpoint_path:
        buffer_path = get_buffer_path(resume_checkpoint_path)
        if os.path.exists(buffer_path):
            replay_buffer.load(buffer_path)
            print(f"Loaded replay buffer: {len(replay_buffer)} samples")

    # Training loop
    end_iteration = start_iteration + num_iterations
    print(f"\nStarting training: iterations {start_iteration + 1} to {end_iteration}")
    print(f"Early stopping: patience={config.early_stopping_patience}, min_delta={config.early_stopping_min_delta}")
    print("=" * 60)

    start_time = time.time()

    # Early stopping state
    best_combined_loss = float('inf')
    patience_counter = 0
    best_iteration = 0
    iteration = start_iteration  # Initialize in case loop doesn't run

    for iteration in range(start_iteration + 1, end_iteration + 1):
        iter_start = time.time()

        # ---------------------------------------------------------------------
        # Self-play phase
        # ---------------------------------------------------------------------
        samples, game_stats = self_play_games(model, config, device)
        replay_buffer.add(samples)

        # ---------------------------------------------------------------------
        # Training phase
        # ---------------------------------------------------------------------
        if len(replay_buffer) >= config.batch_size:
            train_stats = train_on_samples(
                model, optimizer, replay_buffer.get_all(), config, device
            )
        else:
            train_stats = {"policy_loss": 0, "value_loss": 0}

        # ---------------------------------------------------------------------
        # Logging
        # ---------------------------------------------------------------------
        if iteration % config.log_interval == 0:
            elapsed = time.time() - start_time
            iter_time = time.time() - iter_start

            print(f"Iter {iteration:4d} | "
                  f"Games: X={game_stats['x_wins']:2d} O={game_stats['o_wins']:2d} D={game_stats['draws']:2d} | "
                  f"Avg len: {game_stats['avg_game_length']:.1f} | "
                  f"P_loss: {train_stats['policy_loss']:.4f} | "
                  f"V_loss: {train_stats['value_loss']:.4f} | "
                  f"Buffer: {len(replay_buffer):5d} | "
                  f"Time: {iter_time:.1f}s")

        # ---------------------------------------------------------------------
        # Early stopping check
        # ---------------------------------------------------------------------
        combined_loss = train_stats['policy_loss'] + train_stats['value_loss']
        if combined_loss < best_combined_loss - config.early_stopping_min_delta:
            best_combined_loss = combined_loss
            patience_counter = 0
            best_iteration = iteration
            # Save best model
            best_path = os.path.join(config.checkpoint_dir, "model_best.pt")
            save_model(model, best_path)
            replay_buffer.save(get_buffer_path(best_path))
            print(f"  -> New best model (loss={combined_loss:.4f}): {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\n{'=' * 60}")
                print(f"Early stopping triggered after {patience_counter} iterations without improvement")
                print(f"Best combined loss: {best_combined_loss:.4f} at iteration {best_iteration}")
                break

        # ---------------------------------------------------------------------
        # Checkpointing
        # ---------------------------------------------------------------------
        if iteration % config.checkpoint_interval == 0:
            path = os.path.join(config.checkpoint_dir, f"model_iter_{iteration}.pt")
            save_model(model, path)
            replay_buffer.save(get_buffer_path(path))
            print(f"  -> Saved checkpoint: {path}")

    # Determine actual final iteration (might be early stopped)
    # Note: 'iteration' is defined by the for loop and holds last value
    actual_final_iteration = iteration

    # Save final checkpoint with iteration number
    final_iter_path = os.path.join(config.checkpoint_dir, f"model_iter_{actual_final_iteration}.pt")
    if not os.path.exists(final_iter_path):
        save_model(model, final_iter_path)
        replay_buffer.save(get_buffer_path(final_iter_path))
        print(f"  -> Saved checkpoint: {final_iter_path}")

    # Save/update model_final.pt as copy of latest
    final_path = os.path.join(config.checkpoint_dir, "model_final.pt")
    save_model(model, final_path)
    replay_buffer.save(get_buffer_path(final_path))

    total_time = time.time() - start_time
    stopped_early = actual_final_iteration < end_iteration

    print(f"\n{'=' * 60}")
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

    parser = argparse.ArgumentParser(description="Train Zic-Zac-Zoe AI")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of NEW training iterations to run")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per iteration")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to specific checkpoint to resume from")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh, ignoring existing checkpoints")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--filters", type=int, default=64,
                        help="Number of conv filters")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (iterations without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Minimum loss improvement to count as progress")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")

    args = parser.parse_args()

    config = TrainConfig(
        games_per_iteration=args.games,
        learning_rate=args.lr,
        num_filters=args.filters,
        early_stopping_patience=args.patience if not args.no_early_stop else 999999999,
        early_stopping_min_delta=args.min_delta,
    )

    train(
        config=config,
        num_iterations=args.iterations,
        resume_from=args.resume,
        auto_resume=not args.fresh
    )
