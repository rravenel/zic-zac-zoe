"""
Human vs AI Interface for Zic-Zac-Zoe

Play against a trained model with adjustable difficulty.
"""

import sys
from typing import Optional

import torch

from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE, index_to_coord
from model import ZicZacNet, select_move, load_model, get_device, get_policy_value


# =============================================================================
# Difficulty Settings
# =============================================================================

# Temperature controls how deterministic the AI plays:
# - Lower temperature = stronger, more deterministic
# - Higher temperature = weaker, more random
DIFFICULTY_SETTINGS = {
    "easy": 2.0,      # Very random moves
    "medium": 1.0,    # Balanced
    "hard": 0.5,      # Fairly deterministic
    "expert": 0.1,    # Nearly optimal
}


# =============================================================================
# Display Functions
# =============================================================================

def print_board(board: Board, last_move: Optional[int] = None) -> None:
    """
    Print the board with coordinates.
    Highlights the last move with brackets.
    """
    symbols = {Player.EMPTY: '.', Player.X: 'X', Player.O: 'O'}

    print("\n    " + " ".join(str(c) for c in range(BOARD_SIZE)))
    print("   " + "-" * (BOARD_SIZE * 2 + 1))

    for row in range(BOARD_SIZE):
        print(f" {row} |", end="")
        for col in range(BOARD_SIZE):
            idx = row * BOARD_SIZE + col
            symbol = symbols[board.get(row, col)]

            # Highlight last move
            if idx == last_move:
                print(f"[{symbol}", end="")
            elif last_move is not None and idx == last_move + 1 and col > 0:
                print(f"]{symbol}", end="")
            else:
                print(f" {symbol}", end="")
        print(" |")

    print("   " + "-" * (BOARD_SIZE * 2 + 1))


def print_ai_thinking(model: ZicZacNet, board: Board, device: torch.device) -> None:
    """Show what the AI is thinking (top move probabilities)."""
    policy, value = get_policy_value(model, board, device)

    # Get top 5 moves
    legal = board.get_legal_moves()
    legal_probs = [(i, policy[i].item()) for i in legal]
    legal_probs.sort(key=lambda x: x[1], reverse=True)

    print("\n  AI thinking...")
    print(f"  Position evaluation: {value:+.2f} (positive = good for current player)")
    print("  Top moves:")
    for i, (move, prob) in enumerate(legal_probs[:5]):
        row, col = index_to_coord(move)
        print(f"    {i+1}. ({row},{col}) - {prob:.1%}")


# =============================================================================
# Input Handling
# =============================================================================

def get_human_move(board: Board) -> int:
    """Get a valid move from the human player."""
    legal_moves = board.get_legal_moves()

    while True:
        try:
            user_input = input("\n  Your move (row col): ").strip()

            # Handle quit
            if user_input.lower() in ('q', 'quit', 'exit'):
                print("Thanks for playing!")
                sys.exit(0)

            # Parse input
            parts = user_input.replace(',', ' ').split()
            if len(parts) != 2:
                print("  Please enter row and column (e.g., '2 3' or '2,3')")
                continue

            row, col = int(parts[0]), int(parts[1])

            # Validate bounds
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                print(f"  Position must be 0-{BOARD_SIZE-1}")
                continue

            move = row * BOARD_SIZE + col

            # Validate legality
            if move not in legal_moves:
                print("  That position is already taken!")
                continue

            return move

        except ValueError:
            print("  Invalid input. Enter two numbers (row col)")


# =============================================================================
# Game Loop
# =============================================================================

def play_game(model: ZicZacNet, human_plays_x: bool, difficulty: str,
              device: torch.device, show_thinking: bool = False) -> GameResult:
    """
    Play a single game against the AI.

    Args:
        model: Trained neural network
        human_plays_x: True if human plays X (first)
        difficulty: One of 'easy', 'medium', 'hard', 'expert'
        device: Computation device
        show_thinking: Whether to show AI's move probabilities

    Returns:
        Game result
    """
    board = Board()
    temperature = DIFFICULTY_SETTINGS[difficulty]
    last_move = None

    human_player = Player.X if human_plays_x else Player.O
    ai_player = Player.O if human_plays_x else Player.X

    print(f"\n{'='*50}")
    print(f"  You are {human_player.name}. AI is {ai_player.name}.")
    print(f"  Difficulty: {difficulty} (temperature={temperature})")
    print(f"  Remember: 3 in a row LOSES, 4 in a row WINS!")
    print(f"{'='*50}")

    while True:
        print_board(board, last_move)
        current = board.current_player()

        if current == human_player:
            # Human's turn
            print(f"\n  Your turn ({human_player.name})")
            move = get_human_move(board)
        else:
            # AI's turn
            print(f"\n  AI is thinking...")
            if show_thinking:
                print_ai_thinking(model, board, device)
            move = select_move(model, board, temperature=temperature, device=device)
            row, col = index_to_coord(move)
            print(f"  AI plays: ({row}, {col})")

        # Make move
        board = board.make_move(move)
        last_move = move

        # Check for game end
        result = check_result_fast(board, move)
        if result != GameResult.ONGOING:
            print_board(board, last_move)
            return result


def main(model_path: str, difficulty: str = "medium", show_thinking: bool = False) -> None:
    """
    Main game interface.

    Args:
        model_path: Path to trained model
        difficulty: AI difficulty level
        show_thinking: Show AI's move probabilities
    """
    # Load model
    device = get_device()
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    model.eval()
    print(f"Using device: {device}")

    while True:
        # Choose sides
        print("\n" + "=" * 50)
        print("  ZIC-ZAC-ZOE")
        print("=" * 50)
        print("\nWould you like to play as X (first) or O (second)?")
        choice = input("Enter X or O (or Q to quit): ").strip().upper()

        if choice == 'Q':
            print("Thanks for playing!")
            break
        elif choice == 'X':
            human_plays_x = True
        elif choice == 'O':
            human_plays_x = False
        else:
            print("Please enter X, O, or Q")
            continue

        # Play game
        result = play_game(model, human_plays_x, difficulty, device, show_thinking)

        # Announce result
        print("\n" + "=" * 50)
        if result == GameResult.DRAW:
            print("  DRAW! The board is full.")
        elif result == GameResult.X_WINS:
            if human_plays_x:
                print("  YOU WIN! Congratulations!")
            else:
                print("  AI WINS! Better luck next time.")
        else:  # O_WINS
            if human_plays_x:
                print("  AI WINS! Better luck next time.")
            else:
                print("  YOU WIN! Congratulations!")
        print("=" * 50)

        # Play again?
        again = input("\nPlay again? (Y/N): ").strip().upper()
        if again != 'Y':
            print("Thanks for playing!")
            break


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Play Zic-Zac-Zoe against AI")
    parser.add_argument("--model", type=str, default="checkpoints/model_final.pt",
                        help="Path to trained model")
    parser.add_argument("--difficulty", type=str, default="medium",
                        choices=["easy", "medium", "hard", "expert"],
                        help="AI difficulty level")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Show AI's move probabilities")

    args = parser.parse_args()

    main(args.model, args.difficulty, args.show_thinking)
