"""
Evaluation and Benchmarking for Zic-Zac-Zoe AI

Provides:
- Random player baseline
- Heuristic player baseline
- Model vs model comparison
- ELO rating calculation
"""

import random
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass

import torch

from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE
from model import ZicZacNet, select_move, load_model, get_device


# =============================================================================
# Player Types (Move Selection Functions)
# =============================================================================

# A Player is a callable: (Board) -> int (move index)
PlayerFn = Callable[[Board], int]


def random_player(board: Board) -> int:
    """Select a random legal move."""
    return random.choice(board.get_legal_moves())


def heuristic_player(board: Board) -> int:
    """
    Simple heuristic player:
    1. Avoid moves that create 3-in-a-row (instant loss)
    2. Block opponent's 3-in-a-row if possible
    3. Otherwise pick randomly

    Not very sophisticated, but better than random.
    """
    legal_moves = board.get_legal_moves()
    current = board.current_player()
    opponent = current.opponent()

    safe_moves = []
    blocking_moves = []

    for move in legal_moves:
        # Simulate the move
        test_board = board.make_move(move)
        result = check_result_fast(test_board, move)

        # Check if this move loses (we made 3-in-a-row)
        if (result == GameResult.O_WINS and current == Player.X) or \
           (result == GameResult.X_WINS and current == Player.O):
            # This move loses - skip it
            continue

        # Check if this move wins (opponent made 3 previously, we got 4, etc.)
        if (result == GameResult.X_WINS and current == Player.X) or \
           (result == GameResult.O_WINS and current == Player.O):
            # Winning move - take it immediately
            return move

        safe_moves.append(move)

        # Check if this move blocks opponent's potential 3-in-a-row
        # (Simplified: just check if opponent had 2-in-a-row that we blocked)
        # This is a weak heuristic but adds some strategy

    # If we have safe moves, pick one randomly
    if safe_moves:
        return random.choice(safe_moves)

    # If all moves lose, pick randomly (we're doomed anyway)
    return random.choice(legal_moves)


def create_model_player(model: ZicZacNet, temperature: float = 0.1,
                        device: torch.device = None) -> PlayerFn:
    """
    Create a player function from a neural network model.

    Args:
        model: Trained neural network
        temperature: Lower = more deterministic
        device: Computation device

    Returns:
        Player function
    """
    def player(board: Board) -> int:
        return select_move(model, board, temperature=temperature, device=device)
    return player


# =============================================================================
# Match Play
# =============================================================================

def play_match(player_x: PlayerFn, player_o: PlayerFn) -> Tuple[GameResult, int]:
    """
    Play a single game between two players.

    Args:
        player_x: Function to select moves for X
        player_o: Function to select moves for O

    Returns:
        result: Game outcome
        num_moves: Number of moves played
    """
    board = Board()

    while True:
        # Select player based on whose turn
        if board.current_player() == Player.X:
            move = player_x(board)
        else:
            move = player_o(board)

        # Make move
        board = board.make_move(move)

        # Check result
        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            return result, board.move_count()


def evaluate_matchup(player_a: PlayerFn, player_b: PlayerFn,
                     num_games: int = 100) -> dict:
    """
    Evaluate two players against each other.
    Each player plays equal games as X and O.

    Returns:
        Dictionary with win/loss/draw stats and win rate
    """
    a_wins = 0
    b_wins = 0
    draws = 0
    total_moves = 0

    # Player A as X, Player B as O
    for _ in range(num_games // 2):
        result, moves = play_match(player_a, player_b)
        total_moves += moves
        if result == GameResult.X_WINS:
            a_wins += 1
        elif result == GameResult.O_WINS:
            b_wins += 1
        else:
            draws += 1

    # Player B as X, Player A as O
    for _ in range(num_games // 2):
        result, moves = play_match(player_b, player_a)
        total_moves += moves
        if result == GameResult.X_WINS:
            b_wins += 1
        elif result == GameResult.O_WINS:
            a_wins += 1
        else:
            draws += 1

    total_games = num_games
    return {
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / total_games,
        "b_win_rate": b_wins / total_games,
        "draw_rate": draws / total_games,
        "avg_game_length": total_moves / total_games,
    }


# =============================================================================
# ELO Rating
# =============================================================================

@dataclass
class EloPlayer:
    """Player with ELO rating."""
    name: str
    rating: float
    player_fn: PlayerFn


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def update_elo(rating: float, expected: float, actual: float, k: float = 32) -> float:
    """Update ELO rating after a game."""
    return rating + k * (actual - expected)


def compute_elo_ratings(players: List[EloPlayer], games_per_pair: int = 20) -> List[EloPlayer]:
    """
    Compute ELO ratings for a set of players via round-robin tournament.

    Args:
        players: List of players (ratings will be updated in place)
        games_per_pair: Number of games between each pair

    Returns:
        Players sorted by rating (descending)
    """
    # Round-robin tournament
    for i, player_a in enumerate(players):
        for player_b in players[i + 1:]:
            # Play games
            for _ in range(games_per_pair // 2):
                # A as X
                result, _ = play_match(player_a.player_fn, player_b.player_fn)
                if result == GameResult.X_WINS:
                    score_a, score_b = 1.0, 0.0
                elif result == GameResult.O_WINS:
                    score_a, score_b = 0.0, 1.0
                else:
                    score_a, score_b = 0.5, 0.5

                exp_a = expected_score(player_a.rating, player_b.rating)
                exp_b = expected_score(player_b.rating, player_a.rating)
                player_a.rating = update_elo(player_a.rating, exp_a, score_a)
                player_b.rating = update_elo(player_b.rating, exp_b, score_b)

                # B as X
                result, _ = play_match(player_b.player_fn, player_a.player_fn)
                if result == GameResult.X_WINS:
                    score_a, score_b = 0.0, 1.0
                elif result == GameResult.O_WINS:
                    score_a, score_b = 1.0, 0.0
                else:
                    score_a, score_b = 0.5, 0.5

                exp_a = expected_score(player_a.rating, player_b.rating)
                exp_b = expected_score(player_b.rating, player_a.rating)
                player_a.rating = update_elo(player_a.rating, exp_a, score_a)
                player_b.rating = update_elo(player_b.rating, exp_b, score_b)

    # Sort by rating
    return sorted(players, key=lambda p: p.rating, reverse=True)


# =============================================================================
# Benchmark Suite
# =============================================================================

def benchmark_model(model_path: str, num_games: int = 100) -> dict:
    """
    Run full benchmark suite on a trained model.

    Tests against:
    - Random player
    - Heuristic player

    Returns:
        Dictionary with all results
    """
    device = get_device()
    model = load_model(model_path, device)
    model.eval()

    model_player = create_model_player(model, temperature=0.1, device=device)

    results = {}

    # vs Random
    print("Evaluating vs Random player...")
    vs_random = evaluate_matchup(model_player, random_player, num_games)
    results["vs_random"] = vs_random
    print(f"  Win rate: {vs_random['a_win_rate']:.1%}")

    # vs Heuristic
    print("Evaluating vs Heuristic player...")
    vs_heuristic = evaluate_matchup(model_player, heuristic_player, num_games)
    results["vs_heuristic"] = vs_heuristic
    print(f"  Win rate: {vs_heuristic['a_win_rate']:.1%}")

    return results


def compare_checkpoints(checkpoint_paths: List[str], num_games: int = 50) -> None:
    """
    Compare multiple model checkpoints using ELO ratings.

    Args:
        checkpoint_paths: List of paths to model checkpoints
        num_games: Games per pair in tournament
    """
    device = get_device()

    # Create players
    players = []

    # Add baselines
    players.append(EloPlayer("Random", 1000, random_player))
    players.append(EloPlayer("Heuristic", 1200, heuristic_player))

    # Add models
    for path in checkpoint_paths:
        name = path.split("/")[-1].replace(".pt", "")
        model = load_model(path, device)
        model.eval()
        player_fn = create_model_player(model, temperature=0.1, device=device)
        players.append(EloPlayer(name, 1500, player_fn))

    print(f"Running ELO tournament with {len(players)} players...")
    print(f"Games per pair: {num_games}")
    print()

    # Run tournament
    players = compute_elo_ratings(players, games_per_pair=num_games)

    # Print results
    print("ELO Rankings:")
    print("-" * 40)
    for i, p in enumerate(players, 1):
        print(f"{i:2d}. {p.name:<20} {p.rating:.0f}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Evaluate Zic-Zac-Zoe models")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model to benchmark")
    parser.add_argument("--compare", type=str, default=None,
                        help="Glob pattern for checkpoints to compare (e.g., 'checkpoints/*.pt')")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games for evaluation")

    args = parser.parse_args()

    if args.model:
        print(f"Benchmarking model: {args.model}")
        print("=" * 50)
        benchmark_model(args.model, args.games)

    elif args.compare:
        paths = sorted(glob.glob(args.compare))
        if not paths:
            print(f"No checkpoints found matching: {args.compare}")
        else:
            print(f"Found {len(paths)} checkpoints")
            compare_checkpoints(paths, args.games)

    else:
        # Default: test baselines against each other
        print("Testing baseline players...")
        print("=" * 50)
        print("\nRandom vs Heuristic:")
        results = evaluate_matchup(random_player, heuristic_player, 100)
        print(f"  Random wins: {results['a_win_rate']:.1%}")
        print(f"  Heuristic wins: {results['b_win_rate']:.1%}")
        print(f"  Draws: {results['draw_rate']:.1%}")
        print(f"  Avg game length: {results['avg_game_length']:.1f}")
