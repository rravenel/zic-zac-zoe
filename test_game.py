"""
Tests for game logic, including the 4-beats-3 rule.
"""

from game import Board, Player, GameResult, check_result, check_result_fast, BOARD_SIZE


class TestFourBeatsThree:
    """Tests for the 4-beats-3 rule: if a move creates both 4-in-a-row and 3-in-a-row, 4 wins."""

    def test_only_four_in_a_row_wins(self):
        """A move creating only 4-in-a-row should win."""
        # Set up board: X X X _ in top row, X plays at position 3
        # Position: 0 1 2 3 4 5
        #           X X X _ . .
        board = Board()
        # X plays 0, O plays 6, X plays 1, O plays 7, X plays 2, O plays 8
        moves = [0, 6, 1, 7, 2, 8]
        for m in moves:
            board = board.make_move(m)

        # Now X plays at 3 to complete 4-in-a-row
        board = board.make_move(3)

        result = check_result_fast(board, 3)
        assert result == GameResult.X_WINS, f"Expected X_WINS, got {result}"

    def test_only_three_in_a_row_loses(self):
        """A move creating only 3-in-a-row should lose."""
        # Set up board: X X _ in top row, X plays at position 2
        board = Board()
        # X plays 0, O plays 6, X plays 1, O plays 7
        moves = [0, 6, 1, 7]
        for m in moves:
            board = board.make_move(m)

        # Now X plays at 2 to complete 3-in-a-row (X loses)
        board = board.make_move(2)

        result = check_result_fast(board, 2)
        assert result == GameResult.O_WINS, f"Expected O_WINS (X loses), got {result}"

    def test_four_and_three_simultaneously_wins(self):
        """A move creating both 4-in-a-row AND 3-in-a-row should WIN (4 beats 3)."""
        # Set up a board where X playing one cell creates:
        # - 4 in a row horizontally
        # - 3 in a row vertically
        #
        # Board setup:
        #   0 1 2 3 4 5
        # 0 X X X . . .   <- X plays at 3 to make 4 horizontal
        # 1 . . . . . .
        # 2 X . . . . .   <- X already at (2,0)
        # 3 X . . . . .   <- X already at (3,0)
        # 4 . . . . . .
        # 5 . . . . . .
        #
        # When X plays at position 0 row 1 col 0 (index 6)... wait that's not right.
        # Let me rethink:
        #
        # We need X to play a cell that:
        # 1. Completes 4 horizontal: needs X X X _ pattern
        # 2. Completes 3 vertical: needs X X _ pattern in column
        #
        # Position indices:
        #   0  1  2  3  4  5
        #   6  7  8  9 10 11
        #  12 13 14 15 16 17
        #  18 19 20 21 22 23
        #  24 25 26 27 28 29
        #  30 31 32 33 34 35
        #
        # Set up:
        # Row 0: X X X _ . .  (positions 0, 1, 2)
        # Col 0: also has X at positions 12, 18 (rows 2, 3)
        #
        # X at: 0, 1, 2, 12, 18
        # When X plays at... wait, position 0 is already occupied.
        #
        # New approach:
        # Row 0: _ X X X . .  (positions 1, 2, 3)
        # Col 0: X at positions 6, 12 (rows 1, 2)
        # X plays at 0 to make:
        # - Horizontal: 0,1,2,3 = 4 in a row
        # - Vertical: 0,6,12 = 3 in a row

        state = [Player.EMPTY] * 36
        # Horizontal setup: positions 1, 2, 3 have X
        state[1] = Player.X
        state[2] = Player.X
        state[3] = Player.X
        # Vertical setup: positions 6, 12 have X
        state[6] = Player.X
        state[12] = Player.X

        # Need O pieces for turn parity (X has 5 pieces, need 5 O pieces for X's turn)
        # Place O's away from the patterns
        state[35] = Player.O
        state[34] = Player.O
        state[33] = Player.O
        state[32] = Player.O
        state[31] = Player.O

        board = Board(state)
        assert board.current_player() == Player.X, "Should be X's turn"

        # X plays at 0, creating both 4 horizontal and 3 vertical
        board = board.make_move(0)

        # 4-beats-3: X should WIN
        result = check_result_fast(board, 0)
        assert result == GameResult.X_WINS, f"4-beats-3 failed: Expected X_WINS, got {result}"

    def test_four_and_three_check_result_also_works(self):
        """The non-fast check_result should also respect 4-beats-3."""
        state = [Player.EMPTY] * 36
        # Same setup as above
        state[1] = Player.X
        state[2] = Player.X
        state[3] = Player.X
        state[6] = Player.X
        state[12] = Player.X
        state[35] = Player.O
        state[34] = Player.O
        state[33] = Player.O
        state[32] = Player.O
        state[31] = Player.O

        board = Board(state)
        board = board.make_move(0)

        # Test with check_result (not fast)
        result = check_result(board)
        assert result == GameResult.X_WINS, f"4-beats-3 (check_result) failed: Expected X_WINS, got {result}"

    def test_o_four_and_three_wins(self):
        """O creating 4 and 3 should also win."""
        state = [Player.EMPTY] * 36
        # O's horizontal: positions 1, 2, 3
        state[1] = Player.O
        state[2] = Player.O
        state[3] = Player.O
        # O's vertical: positions 6, 12
        state[6] = Player.O
        state[12] = Player.O

        # X pieces for turn parity (O has 5, X needs 6 for O's turn)
        state[35] = Player.X
        state[34] = Player.X
        state[33] = Player.X
        state[32] = Player.X
        state[31] = Player.X
        state[30] = Player.X

        board = Board(state)
        assert board.current_player() == Player.O, "Should be O's turn"

        board = board.make_move(0)

        result = check_result_fast(board, 0)
        assert result == GameResult.O_WINS, f"O 4-beats-3 failed: Expected O_WINS, got {result}"


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
        TestFourBeatsThree,
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
