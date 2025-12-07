"""
Model Evaluation for Zic-Zac-Zoe

Provides quantitative evaluation methods:
- Win rate vs random player
- Win rate vs another model (A/B testing)
- Tactical test positions
"""

import random
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import torch

from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE
from model import ZicZacNet, board_to_tensor, get_device
from tactical_generator import generate_tactical_samples, TacticalSample, PatternType


# =============================================================================
# Position Formatting Helpers
# =============================================================================

def idx_to_coord(idx: int) -> Tuple[int, int]:
    """Convert flat index to 1-indexed (row, col) tuple."""
    return (idx // BOARD_SIZE + 1, idx % BOARD_SIZE + 1)


def coord_to_idx(row: int, col: int) -> int:
    """Convert 1-indexed (row, col) to flat index."""
    return (row - 1) * BOARD_SIZE + (col - 1)


def format_move(idx: int) -> str:
    """Format a move as (row,col)."""
    r, c = idx_to_coord(idx)
    return f"({r},{c})"


def format_moves(indices: List[int], max_show: int = 5) -> str:
    """Format a list of moves, truncating if too long."""
    if len(indices) <= max_show:
        return ", ".join(format_move(i) for i in indices)
    else:
        shown = ", ".join(format_move(i) for i in indices[:max_show])
        return f"{shown}, ... ({len(indices)} total)"


def render_board(board_state: List[int], highlight_played: int = None,
                 highlight_expected: List[int] = None) -> str:
    """
    Render board as ASCII art.

    Args:
        board_state: List of 36 cell values (Player enum)
        highlight_played: Index that was played (marked with *)
        highlight_expected: Indices that were expected (marked with ?)

    Returns:
        Multi-line string showing the board
    """
    lines = []
    lines.append("    1 2 3 4 5 6")
    lines.append("   +-----------+")

    for row in range(BOARD_SIZE):
        row_str = f" {row+1} |"
        for col in range(BOARD_SIZE):
            idx = row * BOARD_SIZE + col
            cell = board_state[idx]

            if cell == Player.X:
                char = "X"
            elif cell == Player.O:
                char = "O"
            elif idx == highlight_played:
                char = "*"  # Model's move
            elif highlight_expected and idx in highlight_expected:
                char = "?"  # Expected move
            else:
                char = "."

            row_str += f" {char}"
        row_str += " |"
        lines.append(row_str)

    lines.append("   +-----------+")
    return "\n".join(lines)


# =============================================================================
# Players
# =============================================================================

def random_move(board: Board) -> int:
    """Select a random legal move."""
    moves = board.get_legal_moves()
    return random.choice(moves)


def model_move(model: ZicZacNet, board: Board, device: torch.device,
               temperature: float = 0.1) -> int:
    """Select a move using the model (low temperature = mostly greedy)."""
    model.eval()
    with torch.no_grad():
        x = board_to_tensor(board, device)
        log_policy, _ = model(x)
        policy = log_policy.exp().squeeze(0)

        # Mask illegal moves
        legal_moves = board.get_legal_moves()
        mask = torch.zeros(BOARD_SIZE * BOARD_SIZE, device=policy.device)
        for m in legal_moves:
            mask[m] = 1.0
        policy = policy * mask

        # Renormalize
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Uniform over legal
            for m in legal_moves:
                policy[m] = 1.0 / len(legal_moves)

        # Apply temperature
        if temperature == 0:
            move = policy.argmax().item()
        else:
            policy = policy.pow(1.0 / temperature)
            policy = policy / policy.sum()
            move = torch.multinomial(policy, 1).item()

    return move


# =============================================================================
# Game Playing
# =============================================================================

def play_game(player_x, player_o, verbose: bool = False) -> Tuple[GameResult, int]:
    """
    Play a game between two players.

    Args:
        player_x: Callable that takes Board and returns move index
        player_o: Callable that takes Board and returns move index
        verbose: Print board after each move

    Returns:
        (result, num_moves)
    """
    board = Board()
    num_moves = 0

    while True:
        current = board.current_player()
        player_fn = player_x if current == Player.X else player_o

        move = player_fn(board)
        board = board.make_move(move)
        num_moves += 1

        if verbose:
            print(f"\nMove {num_moves}: {current.name} plays {move}")
            print(board)

        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            if verbose:
                print(f"\nGame over: {result.name}")
            return result, num_moves

    return GameResult.DRAW, num_moves


# =============================================================================
# Evaluation: Model vs Random
# =============================================================================

@dataclass
class EvalResult:
    """Results from evaluation matches."""
    wins: int
    losses: int
    draws: int
    total_games: int
    avg_game_length: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0

    @property
    def loss_rate(self) -> float:
        return self.losses / self.total_games if self.total_games > 0 else 0.0

    def __str__(self) -> str:
        return (f"W:{self.wins} L:{self.losses} D:{self.draws} "
                f"({self.win_rate*100:.1f}% win rate, avg {self.avg_game_length:.1f} moves)")


def eval_vs_random(model: ZicZacNet, device: torch.device,
                   num_games: int = 100, temperature: float = 0.1) -> EvalResult:
    """
    Evaluate model against random player.

    Plays num_games/2 as X and num_games/2 as O.
    """
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    games_per_side = num_games // 2

    # Model plays X
    for _ in range(games_per_side):
        player_x = lambda b: model_move(model, b, device, temperature)
        player_o = random_move

        result, moves = play_game(player_x, player_o)
        total_moves += moves

        if result == GameResult.X_WINS:
            wins += 1
        elif result == GameResult.O_WINS:
            losses += 1
        else:
            draws += 1

    # Model plays O
    for _ in range(games_per_side):
        player_x = random_move
        player_o = lambda b: model_move(model, b, device, temperature)

        result, moves = play_game(player_x, player_o)
        total_moves += moves

        if result == GameResult.O_WINS:
            wins += 1
        elif result == GameResult.X_WINS:
            losses += 1
        else:
            draws += 1

    total = games_per_side * 2
    return EvalResult(
        wins=wins,
        losses=losses,
        draws=draws,
        total_games=total,
        avg_game_length=total_moves / total
    )


# =============================================================================
# Evaluation: Model vs Model
# =============================================================================

def eval_model_vs_model(model_a: ZicZacNet, model_b: ZicZacNet,
                        device: torch.device, num_games: int = 100,
                        temperature: float = 0.1) -> EvalResult:
    """
    Evaluate model_a against model_b.

    Returns results from model_a's perspective.
    Plays num_games/2 with model_a as X, num_games/2 as O.
    """
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    games_per_side = num_games // 2

    # Model A plays X
    for _ in range(games_per_side):
        player_x = lambda b: model_move(model_a, b, device, temperature)
        player_o = lambda b: model_move(model_b, b, device, temperature)

        result, moves = play_game(player_x, player_o)
        total_moves += moves

        if result == GameResult.X_WINS:
            wins += 1
        elif result == GameResult.O_WINS:
            losses += 1
        else:
            draws += 1

    # Model A plays O
    for _ in range(games_per_side):
        player_x = lambda b: model_move(model_b, b, device, temperature)
        player_o = lambda b: model_move(model_a, b, device, temperature)

        result, moves = play_game(player_x, player_o)
        total_moves += moves

        if result == GameResult.O_WINS:
            wins += 1
        elif result == GameResult.X_WINS:
            losses += 1
        else:
            draws += 1

    total = games_per_side * 2
    return EvalResult(
        wins=wins,
        losses=losses,
        draws=draws,
        total_games=total,
        avg_game_length=total_moves / total
    )


# =============================================================================
# Tactical Test Suite
# =============================================================================

@dataclass
class TacticalPosition:
    """A position with known best move(s)."""
    name: str
    board_state: List[int]
    current_player: Player
    correct_moves: List[int]  # Any of these is correct
    description: str


def create_tactical_suite() -> List[TacticalPosition]:
    """Create a suite of tactical test positions."""
    positions = []

    # Position 1: Block opponent's 3-in-a-row threat
    # O has two in a row, about to make 3 (which loses for O)
    # But X should still block to prevent O from getting 4
    # Actually in this game, O making 3 means O loses, so X doesn't need to block
    # Let's create a position where X needs to avoid making 3

    # Position 1: Don't make 3 in a row
    # X has X at 0, 1 - playing 2 would make 3 and lose
    state1 = [Player.EMPTY] * 36
    state1[0] = Player.X
    state1[1] = Player.X
    state1[6] = Player.O  # O played somewhere
    # X to move - should NOT play 2 (would make 3)
    positions.append(TacticalPosition(
        name="avoid_3_horizontal",
        board_state=state1,
        current_player=Player.X,
        correct_moves=[i for i in range(36) if i not in [0, 1, 2, 6] and state1[i] == Player.EMPTY],
        description="X has XX at (1,1),(1,2) - must not play (1,3)"
    ))

    # Position 2: Block opponent from making 4
    # O has OO_O pattern (gap at position 2) - X must block at 2 or O wins
    # Board row 1: O O _ O _ _
    state2 = [Player.EMPTY] * 36
    state2[0] = Player.O
    state2[1] = Player.O
    state2[3] = Player.O  # Gap at 2!
    # X has 3 pieces (for even move count = X's turn)
    state2[6] = Player.X
    state2[7] = Player.X
    state2[12] = Player.X
    # X to move - must play 2 to block O from making 4
    positions.append(TacticalPosition(
        name="block_4_threat",
        board_state=state2,
        current_player=Player.X,
        correct_moves=[2],
        description="O has OO_O at row 1 - X must block at (1,3)"
    ))

    # Position 3: Win by making 4
    # X has XX_X pattern (gap at position 2) - should complete it
    # Board row 1: X X _ X _ _
    state3 = [Player.EMPTY] * 36
    state3[0] = Player.X
    state3[1] = Player.X
    state3[3] = Player.X  # Gap at 2!
    state3[6] = Player.O
    state3[7] = Player.O
    state3[12] = Player.O
    # X to move - play 2 to win
    positions.append(TacticalPosition(
        name="win_with_4",
        board_state=state3,
        current_player=Player.X,
        correct_moves=[2],
        description="X has XX_X at row 1 - X should play (1,3) to win"
    ))

    # Position 4: Avoid making 3 diagonally
    state4 = [Player.EMPTY] * 36
    state4[0] = Player.X   # (0,0)
    state4[7] = Player.X   # (1,1)
    state4[6] = Player.O
    state4[12] = Player.O
    # X to move - should NOT play 14 (would make diagonal 3)
    positions.append(TacticalPosition(
        name="avoid_3_diagonal",
        board_state=state4,
        current_player=Player.X,
        correct_moves=[i for i in range(36) if i not in [0, 6, 7, 12, 14] and state4[i] == Player.EMPTY],
        description="X has diagonal at (1,1),(2,2) - must not play (3,3)"
    ))

    # Position 5: Force opponent to make 3
    # Set up a position where any O move creates 3 for O
    state5 = [Player.EMPTY] * 36
    # O has OO in multiple directions
    state5[7] = Player.O   # (1,1)
    state5[8] = Player.O   # (1,2)
    # X positions
    state5[0] = Player.X
    state5[1] = Player.X
    state5[6] = Player.X
    state5[12] = Player.X
    state5[13] = Player.X
    # If O plays 9, that's OOO horizontal - O loses
    # If O plays 14, that's OOO diagonal - O loses
    positions.append(TacticalPosition(
        name="force_opponent_3",
        board_state=state5,
        current_player=Player.O,
        correct_moves=[],  # All moves lead to loss - just check O doesn't crash
        description="O is in zugzwang - any move makes 3"
    ))

    return positions


@dataclass
class TacticalResult:
    """Result from a single tactical test."""
    name: str
    description: str
    passed: bool
    move_played: int
    correct_moves: List[int]
    board_state: List[int]

    def render(self) -> str:
        """Render the result as a visual grid with annotations."""
        status = "PASS ✓" if self.passed else "FAIL ✗"
        lines = [
            f"[{status}] {self.name}",
            f"  {self.description}",
            f"  Model played: {format_move(self.move_played)}  |  Expected: {format_moves(self.correct_moves)}",
            ""
        ]
        # Only show expected markers if there are few enough (otherwise board gets cluttered)
        show_expected = not self.passed and len(self.correct_moves) <= 6
        board_lines = render_board(
            self.board_state,
            highlight_played=self.move_played,
            highlight_expected=self.correct_moves if show_expected else None
        )
        for line in board_lines.split("\n"):
            lines.append("  " + line)
        lines.append("")
        return "\n".join(lines)


def eval_tactical(model: ZicZacNet, device: torch.device,
                  temperature: float = 0.0,
                  verbose: bool = False,
                  n_per_type: int = 10,
                  seed: int = 42) -> Tuple[int, int, List[TacticalResult]]:
    """
    Evaluate model on tactical test positions.

    Uses randomly generated tactical positions to test pattern recognition
    across different orientations and locations on the board.

    Args:
        model: The model to evaluate
        device: Torch device
        temperature: Sampling temperature (0 = greedy)
        verbose: If True, print each test result
        n_per_type: Number of samples per pattern type (avoid_3, complete_4, block_4)
        seed: Random seed for reproducible position generation

    Returns:
        (correct, total, list of TacticalResult for all tests)
    """
    # Generate random tactical positions
    tactical_samples = generate_tactical_samples(n_per_type=n_per_type, seed=seed)

    correct = 0
    results = []

    for sample in tactical_samples:
        if not sample.correct_moves:
            # Skip positions with no correct answer
            continue

        board = Board(sample.board_state)
        move = model_move(model, board, device, temperature)
        passed = move in sample.correct_moves

        if passed:
            correct += 1

        # Create description based on pattern type
        pattern_descriptions = {
            PatternType.AVOID_3: "Must not complete 3-in-a-row",
            PatternType.COMPLETE_4: "Must complete 4-in-a-row to win",
            PatternType.BLOCK_4: "Must block opponent's 4-in-a-row threat",
        }

        result = TacticalResult(
            name=sample.pattern_type.value,
            description=pattern_descriptions.get(sample.pattern_type, ""),
            passed=passed,
            move_played=move,
            correct_moves=sample.correct_moves,
            board_state=sample.board_state
        )
        results.append(result)

        if verbose:
            print(result.render())

    total = len([s for s in tactical_samples if s.correct_moves])
    return correct, total, results


# =============================================================================
# Full Evaluation Report
# =============================================================================

def full_evaluation(model: ZicZacNet, device: torch.device,
                    num_random_games: int = 100,
                    previous_model: Optional[ZicZacNet] = None,
                    num_vs_games: int = 50) -> Dict:
    """
    Run full evaluation suite.

    Returns dictionary with all results.
    """
    results = {}

    # Vs Random
    print("Evaluating vs random player...")
    vs_random = eval_vs_random(model, device, num_random_games)
    results['vs_random'] = vs_random
    print(f"  vs Random: {vs_random}")

    # Tactical suite
    print("Evaluating tactical positions...")
    correct, total, tactical_results = eval_tactical(model, device)
    results['tactical_correct'] = correct
    results['tactical_total'] = total
    results['tactical_results'] = tactical_results
    print(f"  Tactical: {correct}/{total} correct")
    print()
    for tr in tactical_results:
        print(tr.render())

    # Vs previous model
    if previous_model is not None:
        print("Evaluating vs previous model...")
        vs_prev = eval_model_vs_model(model, previous_model, device, num_vs_games)
        results['vs_previous'] = vs_prev
        print(f"  vs Previous: {vs_prev}")

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Zic-Zac-Zoe model")
    parser.add_argument("model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to model to compare against")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of games vs random")
    parser.add_argument("--vs-games", type=int, default=50,
                        help="Number of games vs comparison model")

    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    model = ZicZacNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load comparison model if specified
    prev_model = None
    if args.compare:
        print(f"Loading comparison model from {args.compare}")
        prev_model = ZicZacNet()
        prev_model.load_state_dict(torch.load(args.compare, map_location=device))
        prev_model = prev_model.to(device)
        prev_model.eval()

    # Run evaluation
    print("\n" + "=" * 60)
    results = full_evaluation(
        model, device,
        num_random_games=args.games,
        previous_model=prev_model,
        num_vs_games=args.vs_games
    )
    print("=" * 60)
