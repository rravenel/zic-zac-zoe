"""
Tests for Points Variant game logic.
"""

from game import Board, Player, GameResult, check_result, check_result_fast, BOARD_SIZE


class TestPointsVariant:
    """Tests for the points-based scoring and terminal logic."""

    def test_score_accumulation(self):
        """Verify that scores are accumulated correctly during moves."""
        board = Board()
        
        # X plays 0 -> Lone tile = 1 pt
        board = board.make_move(0)
        assert board.score_x == 1
        assert board.score_o == 0
        
        # O plays 6 (below X at 0) -> Lone tile for O = 1 pt
        board = board.make_move(6)
        assert board.score_x == 1
        assert board.score_o == 1
        
        # X plays 1 (next to X at 0) -> Append to 2 = 2 pts
        board = board.make_move(1)
        assert board.score_x == 1 + 2 # 3
        
        # O plays 12 (below O at 6) -> Append to 2 = 2 pts
        board = board.make_move(12)
        assert board.score_o == 1 + 2 # 3

    def test_terminal_state_only_on_full_board(self):
        """Verify that game only ends when board is full."""
        board = Board()
        # Create a line of 4 (old win condition)
        # X: 0, 1, 2
        # O: 7, 8, 9
        for m in [0, 7, 1, 8, 2, 9]:
            board = board.make_move(m)
        
        # X plays 3 to complete 4-in-a-row
        board = board.make_move(3)
        assert check_result(board) == GameResult.ONGOING
        assert check_result_fast(board, 3) == GameResult.ONGOING
        
        # Fill the board to see terminal state
        # (This is a bit slow but necessary for fidelity)
        b2 = Board()
        for i in range(BOARD_SIZE * BOARD_SIZE):
            b2.state[i] = Player.X
            b2._move_count += 1
        
        assert b2.is_full()
        assert check_result(b2) == GameResult.DRAW

    def test_board_identity_with_scores(self):
        """Verify that Board hash and equality include scores."""
        b1 = Board(score_x=10, score_o=5)
        b2 = Board(score_x=10, score_o=5)
        b3 = Board(score_x=10, score_o=6)
        b4 = Board(score_x=11, score_o=5)
        
        # Equality
        assert b1 == b2
        assert b1 != b3
        assert b1 != b4
        
        # Hashing
        assert hash(b1) == hash(b2)
        assert hash(b1) != hash(b3)
        assert hash(b1) != hash(b4)
        
        # Copying
        b1_copy = b1.copy()
        assert b1_copy == b1
        assert b1_copy.score_x == 10
        assert b1_copy.score_o == 5


class TestBasicGameLogic:
    """Basic game logic tests."""

    def test_empty_board_ongoing(self):
        """Empty board should be ongoing."""
        board = Board()
        result = check_result(board)
        assert result == GameResult.ONGOING

    def test_x_moves_first(self):
        """X should move first on empty board."""
        board = Board()
        assert board.current_player() == Player.X

    def test_alternating_turns(self):
        """Players should alternate turns."""
        board = Board()
        assert board.current_player() == Player.X
        board = board.make_move(0)
        assert board.current_player() == Player.O
        board = board.make_move(1)
        assert board.current_player() == Player.X


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestPointsVariant,
        TestBasicGameLogic,
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
