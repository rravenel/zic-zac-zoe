"""
Training Pipeline with AlphaZero-style MCTS for Zic-Zac-Zoe AI

Uses Monte Carlo Tree Search with neural network evaluation (no random rollouts).
The model's value head evaluates positions directly, and the policy head
guides which moves to explore.

This is significantly faster and produces stronger play than random rollouts.
"""

import os
import re
import glob
import time
import random
import pickle
import math
import multiprocessing as mp
from datetime import datetime
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
    games_per_iteration: int = 100       # More games (no rollouts = faster)
    temperature_start: float = 1.0       # High temp early in game (exploration)
    temperature_end: float = 0.3         # Lower temp late in game (exploitation)
    temperature_threshold: int = 12      # Move number to switch temperatures

    # MCTS settings
    mcts_simulations: int = 100          # Simulations per move
    mcts_c_puct: float = 1.5             # PUCT exploration constant
    dirichlet_alpha: float = 0.3         # Noise for root exploration
    dirichlet_epsilon: float = 0.25      # Weight of noise at root

    # Training settings
    batch_size: int = 256
    learning_rate: float = 0.001
    epochs_per_iteration: int = 5
    replay_buffer_size: int = 50000

    # Model settings
    num_filters: int = 64

    # Checkpointing - separate directory
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints_alphazero"

    # Logging
    log_interval: int = 1

    # Early stopping
    early_stopping_patience: int = 30
    early_stopping_min_delta: float = 0.005

    # Parallelization
    num_workers: int = 1                 # Number of parallel game workers (0 = auto)


# =============================================================================
# MCTS with Neural Network Evaluation
# =============================================================================

class MCTSNode:
    """Node in the MCTS tree with neural network priors."""

    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None,
                 move: Optional[int] = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move      # Move that led to this node
        self.prior = prior    # Prior probability from policy network

        self.children: Dict[int, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0  # Sum of values (from perspective of player to move)

        # Cache
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
                    # Value from perspective of current player (who is about to move)
                    # But game just ended, so the player who just moved won/lost
                    current = self.board.current_player()
                    if result == GameResult.DRAW:
                        self._terminal_value = 0.0
                    elif result == GameResult.X_WINS:
                        self._terminal_value = 1.0 if current == Player.X else -1.0
                    else:  # O_WINS
                        self._terminal_value = 1.0 if current == Player.O else -1.0
        return self._is_terminal

    @property
    def terminal_value(self) -> float:
        _ = self.is_terminal
        return self._terminal_value

    def q_value(self) -> float:
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def puct_score(self, c_puct: float, parent_visits: int) -> float:
        """
        PUCT score for node selection.
        Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Note: Q is negated because value_sum is stored from this node's
        player perspective, but parent wants moves bad for opponent.
        """
        q = -self.q_value()  # Negate: stored value is from child's perspective
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return q + u

    def best_child(self, c_puct: float) -> 'MCTSNode':
        """Select child with highest PUCT score."""
        return max(self.children.values(),
                   key=lambda n: n.puct_score(c_puct, self.visits))

    def is_expanded(self) -> bool:
        """Check if node has been expanded (children created)."""
        return len(self.children) > 0


def evaluate_position(board: Board, model: ZicZacNet, device: torch.device) -> Tuple[List[float], float]:
    """
    Evaluate position using neural network.

    Returns:
        (policy, value) where policy is probability distribution over all moves
        and value is expected outcome from current player's perspective.
    """
    model.eval()
    with torch.no_grad():
        tensor = board_to_tensor(board, device)
        log_policy, value = model(tensor)

        # Convert to probabilities (avoid numpy for compatibility)
        policy = torch.exp(log_policy).squeeze(0).cpu().tolist()
        value = value.item()

    return policy, value


def expand_node(node: MCTSNode, policy: List[float]) -> None:
    """
    Expand node by creating children for all legal moves.
    """
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
    """
    Run MCTS from given position using neural network evaluation.

    Returns:
        Dictionary mapping move -> visit count
    """
    root = MCTSNode(root_board)

    # Expand root
    policy, _ = evaluate_position(root_board, model, device)

    # Mask illegal moves and renormalize
    legal = root.legal_moves
    masked_policy = [policy[i] if i in legal else 0.0 for i in range(len(policy))]
    policy_sum = sum(masked_policy)
    if policy_sum > 0:
        masked_policy = [p / policy_sum for p in masked_policy]
    else:
        # Uniform over legal moves
        masked_policy = [1.0 / len(legal) if i in legal else 0.0 for i in range(len(policy))]

    expand_node(root, masked_policy)

    # Add exploration noise at root
    if dirichlet_alpha > 0:
        add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection: traverse tree using PUCT
        while node.is_expanded() and not node.is_terminal:
            node = node.best_child(c_puct)
            search_path.append(node)

        # Get value for leaf
        if node.is_terminal:
            value = node.terminal_value
        else:
            # Expansion and evaluation
            policy, value = evaluate_position(node.board, model, device)

            # Mask and normalize policy
            legal = node.legal_moves
            if legal:  # Only expand if there are legal moves
                masked_policy = [policy[i] if i in legal else 0.0 for i in range(len(policy))]
                policy_sum = sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = [p / policy_sum for p in masked_policy]
                else:
                    masked_policy = [1.0 / len(legal) if i in legal else 0.0 for i in range(len(policy))]

                expand_node(node, masked_policy)

            # Value is from perspective of player to move at this node
            # We need to flip as we backprop since alternating players

        # Backpropagation
        # Value is from perspective of player to move at leaf
        for i, bp_node in enumerate(reversed(search_path)):
            # Flip value for each level (alternating players)
            bp_node.visits += 1
            bp_node.value_sum += value if i % 2 == 0 else -value

    # Return visit counts for root's children
    return {move: child.visits for move, child in root.children.items()}


def select_move_from_visits(visit_counts: Dict[int, int], temperature: float) -> int:
    """Select a move from MCTS visit counts using temperature."""
    moves = list(visit_counts.keys())
    visits = [visit_counts[m] for m in moves]

    if temperature == 0:
        return moves[visits.index(max(visits))]

    # Apply temperature
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
    policy_target: List[float]
    outcome: float


# =============================================================================
# Self-Play with Neural Network MCTS
# =============================================================================

def play_game(model: ZicZacNet, device: torch.device,
              config: TrainConfig) -> Tuple[List[Sample], GameResult]:
    """Play a single self-play game using neural network MCTS."""
    board = Board()
    history = []

    while True:
        current_player = board.current_player()

        # Temperature based on move number
        if board.move_count() < config.temperature_threshold:
            temp = config.temperature_start
        else:
            temp = config.temperature_end

        # Run MCTS with neural network
        visit_counts = mcts_search(
            board, model, device,
            config.mcts_simulations, config.mcts_c_puct,
            config.dirichlet_alpha, config.dirichlet_epsilon
        )

        policy_target = visits_to_policy(visit_counts)
        move = select_move_from_visits(visit_counts, temp)

        history.append((board.state.copy(), policy_target, current_player))

        board = board.make_move(move)

        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            break

    # Convert to samples
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


def _worker_play_games(args: Tuple) -> Tuple[List[Sample], int, int, int]:
    """Worker function for parallel game playing."""
    model_state_dict, num_games, config_dict = args

    # Limit PyTorch to 1 thread per worker to avoid oversubscription
    torch.set_num_threads(1)

    # Recreate model in this process
    config = TrainConfig(**config_dict)
    model = ZicZacNet(num_filters=config.num_filters)
    model.load_state_dict(model_state_dict)
    model.eval()
    device = torch.device("cpu")  # Workers use CPU to avoid GPU contention

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

    # Sequential mode
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

    # Distribute games across workers
    games_per_worker = config.games_per_iteration // num_workers
    remainder = config.games_per_iteration % num_workers

    worker_args = []
    for i in range(num_workers):
        n_games = games_per_worker + (1 if i < remainder else 0)
        if n_games > 0:
            worker_args.append((model_state_dict, n_games, config_dict))

    # Run workers (use 'fork' to avoid re-importing modules in children)
    ctx = mp.get_context('fork')
    with ctx.Pool(num_workers) as pool:
        results = pool.map(_worker_play_games, worker_args)

    # Aggregate results
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
    """Train model on collected samples."""
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

            log_policy, value = model(states)

            # Policy loss: cross-entropy with soft targets
            policy_loss = -torch.sum(policy_targets * log_policy, dim=1).mean()

            # Value loss: MSE
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


# =============================================================================
# Main Training Loop
# =============================================================================

def train(config: Optional[TrainConfig] = None, num_iterations: int = 100,
          resume_from: Optional[str] = None, auto_resume: bool = True) -> ZicZacNet:
    """Main training function with AlphaZero-style MCTS."""
    if config is None:
        config = TrainConfig()

    device = get_device()
    print(f"Training on device: {device}")
    print(f"Using AlphaZero-style MCTS with {config.mcts_simulations} simulations per move")
    print(f"Neural network evaluation (no random rollouts)")
    num_workers = config.num_workers if config.num_workers > 0 else mp.cpu_count()
    if num_workers > 1:
        print(f"Parallel self-play with {num_workers} workers")

    # Clear checkpoint directory if starting fresh
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

    end_iteration = start_iteration + num_iterations
    print(f"\nStarting training: iterations {start_iteration + 1} to {end_iteration}")
    print(f"Early stopping: patience={config.early_stopping_patience}, min_delta={config.early_stopping_min_delta}")
    print("=" * 80)

    start_time = time.time()

    best_combined_loss = float('inf')
    patience_counter = 0
    best_iteration = 0
    iteration = start_iteration

    for iteration in range(start_iteration + 1, end_iteration + 1):
        iter_start = time.time()

        # Self-play phase with neural network MCTS
        samples, game_stats = self_play_games(model, device, config)
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
            timestamp = datetime.now().strftime("%H:%M:%S")

            print(f"[{timestamp}] Iter {iteration:4d} | "
                  f"Games: X={game_stats['x_wins']:2d} O={game_stats['o_wins']:2d} D={game_stats['draws']:2d} | "
                  f"Avg len: {game_stats['avg_game_length']:.1f} | "
                  f"P_loss: {train_stats['policy_loss']:.4f} | "
                  f"V_loss: {train_stats['value_loss']:.4f} | "
                  f"Buffer: {len(replay_buffer):5d} | "
                  f"Time: {iter_time:.1f}s{best_marker}")

        if patience_counter >= config.early_stopping_patience:
            print(f"\n{'=' * 80}")
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

    print(f"\n{'=' * 80}")
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

    parser = argparse.ArgumentParser(description="Train Zic-Zac-Zoe AI with AlphaZero-style MCTS")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per iteration")
    parser.add_argument("--simulations", type=int, default=100,
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
                        help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.005,
                        help="Minimum loss improvement")
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--c-puct", type=float, default=1.5,
                        help="PUCT exploration constant")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for self-play (0 = auto-detect CPU count)")

    args = parser.parse_args()

    config = TrainConfig(
        games_per_iteration=args.games,
        mcts_simulations=args.simulations,
        learning_rate=args.lr,
        num_filters=args.filters,
        early_stopping_patience=args.patience if not args.no_early_stop else 999999999,
        early_stopping_min_delta=args.min_delta,
        mcts_c_puct=args.c_puct,
        num_workers=args.workers,
    )

    train(
        config=config,
        num_iterations=args.iterations,
        resume_from=args.resume,
        auto_resume=not args.fresh,
    )
