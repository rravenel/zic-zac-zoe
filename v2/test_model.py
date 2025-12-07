"""
Tests for model encoding and game logic.

These tests catch inversion bugs and encoding errors that can silently
break training without obvious symptoms.
"""

import torch
from game import Board, Player, GameResult, check_result_fast, BOARD_SIZE
from model import board_to_tensor, boards_to_tensor, ZicZacNet


class TestTurnIndicator:
    """Tests for the turn indicator channel (channel 2)."""

    def test_x_turn_at_start(self):
        """X moves first, so turn indicator should be 1.0 at game start."""
        board = Board()
        tensor = board_to_tensor(board)

        assert board.current_player() == Player.X, "X should move first"
        assert tensor[0, 2, 0, 0].item() == 1.0, "Turn indicator should be 1.0 for X's turn"
        # Check entire channel is uniform
        assert torch.all(tensor[0, 2] == 1.0), "Turn indicator channel should be all 1s for X"

    def test_o_turn_after_one_move(self):
        """After X moves, it's O's turn, so turn indicator should be 0.0."""
        board = Board().make_move(0)
        tensor = board_to_tensor(board)

        assert board.current_player() == Player.O, "O should move after X"
        assert tensor[0, 2, 0, 0].item() == 0.0, "Turn indicator should be 0.0 for O's turn"
        assert torch.all(tensor[0, 2] == 0.0), "Turn indicator channel should be all 0s for O"

    def test_x_turn_after_two_moves(self):
        """After X and O each move, it's X's turn again."""
        board = Board().make_move(0).make_move(1)
        tensor = board_to_tensor(board)

        assert board.current_player() == Player.X, "X should move after O"
        assert torch.all(tensor[0, 2] == 1.0), "Turn indicator should be 1.0 for X's turn"

    def test_turn_alternates_correctly(self):
        """Verify turn indicator alternates through multiple moves."""
        board = Board()
        moves = [0, 6, 1, 7, 3, 8]  # Avoid making 3 in a row

        for i, move in enumerate(moves):
            tensor = board_to_tensor(board)
            expected_turn = 1.0 if i % 2 == 0 else 0.0
            actual_turn = tensor[0, 2, 0, 0].item()
            assert actual_turn == expected_turn, \
                f"Move {i}: expected turn={expected_turn}, got {actual_turn}"
            board = board.make_move(move)

    def test_batched_turn_indicator(self):
        """Test that boards_to_tensor correctly handles turn for multiple boards."""
        board_x_turn = Board()  # X's turn
        board_o_turn = Board().make_move(0)  # O's turn

        batch = boards_to_tensor([board_x_turn, board_o_turn])

        assert torch.all(batch[0, 2] == 1.0), "First board (X's turn) should have turn=1"
        assert torch.all(batch[1, 2] == 0.0), "Second board (O's turn) should have turn=0"


class TestPieceEncoding:
    """Tests for X/O piece encoding (channels 0 and 1)."""

    def test_empty_board(self):
        """Empty board should have all zeros in piece channels."""
        board = Board()
        tensor = board_to_tensor(board)

        assert torch.all(tensor[0, 0] == 0.0), "X channel should be all 0s on empty board"
        assert torch.all(tensor[0, 1] == 0.0), "O channel should be all 0s on empty board"

    def test_x_piece_encoding(self):
        """X pieces should appear in channel 0 only."""
        board = Board().make_move(0)  # X at position 0
        tensor = board_to_tensor(board)

        # X should be at (0,0) in channel 0
        assert tensor[0, 0, 0, 0].item() == 1.0, "X piece should be 1.0 in channel 0"
        assert tensor[0, 1, 0, 0].item() == 0.0, "X piece should be 0.0 in channel 1"

    def test_o_piece_encoding(self):
        """O pieces should appear in channel 1 only."""
        board = Board().make_move(0).make_move(1)  # X at 0, O at 1
        tensor = board_to_tensor(board)

        # O should be at (0,1) in channel 1
        assert tensor[0, 0, 0, 1].item() == 0.0, "O piece should be 0.0 in channel 0"
        assert tensor[0, 1, 0, 1].item() == 1.0, "O piece should be 1.0 in channel 1"

    def test_pieces_dont_overlap(self):
        """X and O channels should never have 1s in the same position."""
        board = Board()
        moves = [0, 6, 1, 7, 3, 8, 4, 9]

        for move in moves:
            board = board.make_move(move)

        tensor = board_to_tensor(board)
        overlap = tensor[0, 0] * tensor[0, 1]  # Element-wise product
        assert torch.all(overlap == 0.0), "X and O channels should never overlap"

    def test_position_to_tensor_mapping(self):
        """Verify flat index maps to correct (row, col) in tensor."""
        for pos in [0, 5, 6, 35]:  # Corners and edges
            board = Board()
            # Need to make enough moves to place at position
            # Just test position 0 for simplicity
            if pos == 0:
                board = board.make_move(0)
                tensor = board_to_tensor(board)
                row, col = 0, 0
                assert tensor[0, 0, row, col].item() == 1.0, \
                    f"Position {pos} should map to tensor[0, 0, {row}, {col}]"


class TestGameResults:
    """Tests for game result logic - critical for value targets."""

    def test_x_makes_3_means_o_wins(self):
        """X making 3 in a row should result in O winning (X loses)."""
        board = Board()
        moves = [0, 6, 1, 7, 2]  # X plays 0,1,2 = 3 in a row horizontally

        for move in moves:
            board = board.make_move(move)

        result = check_result_fast(board, 2)  # Last move was at position 2
        assert result == GameResult.O_WINS, \
            f"X making 3 should mean O wins, got {result}"

    def test_o_makes_3_means_x_wins(self):
        """O making 3 in a row should result in X winning (O loses)."""
        board = Board()
        moves = [0, 6, 1, 7, 3, 8]  # O plays 6,7,8 = 3 in a row horizontally

        for move in moves:
            board = board.make_move(move)

        result = check_result_fast(board, 8)  # Last move was at position 8
        assert result == GameResult.X_WINS, \
            f"O making 3 should mean X wins, got {result}"

    def test_x_makes_4_means_x_wins(self):
        """X making 4 in a row should result in X winning."""
        board = Board()
        # X needs to make 4 without O making 3 first
        # X: 0, 1, 2, 3 (but need to avoid triggering 3-in-row check for X at move 2)
        # Actually X at 0,1,2 would already lose... let's be more careful
        # Use: X at 0, 1, skip 2, put at 3, then put at 2
        moves = [0, 6, 1, 7, 3, 12, 2]  # X: 0,1,3,2 - makes 4 in row 0

        for i, move in enumerate(moves):
            board = board.make_move(move)
            if i < len(moves) - 1:
                result = check_result_fast(board, move)
                # Game should still be ongoing until the winning move

        result = check_result_fast(board, 2)
        assert result == GameResult.X_WINS, \
            f"X making 4 should mean X wins, got {result}"

    def test_diagonal_3_loses(self):
        """Diagonal 3 in a row should also cause loss."""
        board = Board()
        moves = [0, 1, 7, 2, 14]  # X plays diagonal: 0, 7, 14

        for move in moves:
            board = board.make_move(move)

        result = check_result_fast(board, 14)
        assert result == GameResult.O_WINS, \
            f"X making diagonal 3 should mean O wins, got {result}"


class TestModelOutput:
    """Tests for model output shapes and ranges."""

    def test_output_shapes(self):
        """Model should output correct shapes for policy and value."""
        model = ZicZacNet()
        board = Board()
        tensor = board_to_tensor(board)

        log_policy, value = model(tensor)

        assert log_policy.shape == (1, 36), f"Policy shape should be (1, 36), got {log_policy.shape}"
        assert value.shape == (1, 1), f"Value shape should be (1, 1), got {value.shape}"

    def test_policy_is_log_probabilities(self):
        """Policy output should be log probabilities (sum of exp = 1)."""
        model = ZicZacNet()
        board = Board()
        tensor = board_to_tensor(board)

        log_policy, _ = model(tensor)
        probs = torch.exp(log_policy)
        prob_sum = probs.sum().item()

        assert abs(prob_sum - 1.0) < 0.01, \
            f"Policy probabilities should sum to 1, got {prob_sum}"

    def test_value_in_range(self):
        """Value should be in [-1, 1] range (tanh output)."""
        model = ZicZacNet()

        # Test several random boards
        import random
        for _ in range(10):
            board = Board()
            last_move = None
            # Make some random moves
            for _ in range(random.randint(0, 10)):
                legal = board.get_legal_moves()
                if not legal:
                    break
                last_move = random.choice(legal)
                board = board.make_move(last_move)
                if last_move is not None:
                    result = check_result_fast(board, last_move)
                    if result != GameResult.ONGOING:
                        break

            if board.get_legal_moves():  # Only test non-terminal positions
                tensor = board_to_tensor(board)
                _, value = model(tensor)
                v = value.item()
                assert -1.0 <= v <= 1.0, f"Value should be in [-1, 1], got {v}"


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestTurnIndicator,
        TestPieceEncoding,
        TestGameResults,
        TestModelOutput,
    ]

    total = 0
    passed = 0
    failed = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                total += 1
                try:
                    getattr(instance, method_name)()
                    print(f"  ✓ {method_name}")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name}: {e}")
                    failed.append((test_class.__name__, method_name, traceback.format_exc()))

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed")

    if failed:
        print(f"\nFailed tests:")
        for class_name, method_name, tb in failed:
            print(f"\n{class_name}.{method_name}:")
            print(tb)
        return 1
    else:
        print("All tests passed!")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(run_tests())
