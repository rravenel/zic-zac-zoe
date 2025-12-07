"""
Tests for tactical position generator.
"""

from game import Board, Player, GameResult, check_result, BOARD_SIZE
from tactical_generator import (
    generate_tactical_samples,
    generate_avoid_3,
    generate_complete_4,
    generate_block_4,
    tactical_sample_to_training_sample,
    PatternType,
    TacticalSample,
    ALL_LINES,
    count_consecutive,
)
import random


class TestLineGeneration:
    """Tests for line enumeration."""

    def test_all_lines_exist(self):
        """ALL_LINES should be populated."""
        assert len(ALL_LINES) > 0, "Should have at least one line"

    def test_lines_have_minimum_length(self):
        """All lines should have at least 4 cells."""
        for line in ALL_LINES:
            assert len(line) >= 4, f"Line {line} has fewer than 4 cells"

    def test_lines_contain_valid_indices(self):
        """All indices should be in valid range [0, 35]."""
        for line in ALL_LINES:
            for idx in line:
                assert 0 <= idx < 36, f"Invalid index {idx} in line {line}"

    def test_horizontal_lines_exist(self):
        """Should have horizontal lines."""
        # Check for a full row (indices 0-5)
        found_horizontal = False
        for line in ALL_LINES:
            if all(idx // BOARD_SIZE == 0 for idx in line) and len(line) >= 4:
                found_horizontal = True
                break
        assert found_horizontal, "Should have at least one horizontal line in row 0"

    def test_vertical_lines_exist(self):
        """Should have vertical lines."""
        # Check for a full column (indices 0, 6, 12, 18, 24, 30)
        found_vertical = False
        for line in ALL_LINES:
            if all(idx % BOARD_SIZE == 0 for idx in line) and len(line) >= 4:
                found_vertical = True
                break
        assert found_vertical, "Should have at least one vertical line in column 0"

    def test_diagonal_lines_exist(self):
        """Should have diagonal lines."""
        # Main diagonal: 0, 7, 14, 21, 28, 35
        main_diag_set = {0, 7, 14, 21, 28, 35}
        found_diagonal = False
        for line in ALL_LINES:
            if len(set(line) & main_diag_set) >= 4:
                found_diagonal = True
                break
        assert found_diagonal, "Should have at least one main diagonal line"


class TestCountConsecutive:
    """Tests for consecutive piece counting."""

    def test_empty_board(self):
        """Empty board should have 0 consecutive pieces."""
        board_state = [Player.EMPTY] * 36
        line = [0, 1, 2, 3]
        assert count_consecutive(board_state, line, Player.X) == 0

    def test_single_piece(self):
        """Single piece should count as 1."""
        board_state = [Player.EMPTY] * 36
        board_state[0] = Player.X
        line = [0, 1, 2, 3]
        assert count_consecutive(board_state, line, Player.X) == 1

    def test_three_consecutive(self):
        """Three consecutive pieces should count as 3."""
        board_state = [Player.EMPTY] * 36
        board_state[0] = Player.X
        board_state[1] = Player.X
        board_state[2] = Player.X
        line = [0, 1, 2, 3]
        assert count_consecutive(board_state, line, Player.X) == 3

    def test_gap_breaks_sequence(self):
        """Gap should break consecutive sequence."""
        board_state = [Player.EMPTY] * 36
        board_state[0] = Player.X
        board_state[1] = Player.X
        # Gap at 2
        board_state[3] = Player.X
        line = [0, 1, 2, 3]
        assert count_consecutive(board_state, line, Player.X) == 2

    def test_opponent_breaks_sequence(self):
        """Opponent piece should break consecutive sequence."""
        board_state = [Player.EMPTY] * 36
        board_state[0] = Player.X
        board_state[1] = Player.X
        board_state[2] = Player.O  # Opponent
        board_state[3] = Player.X
        line = [0, 1, 2, 3]
        assert count_consecutive(board_state, line, Player.X) == 2


class TestGenerateAvoid3:
    """Tests for avoid_3 pattern generation."""

    def test_generates_valid_sample(self):
        """Should generate a valid TacticalSample."""
        rng = random.Random(42)
        # Try multiple times since generation can fail
        sample = None
        for _ in range(100):
            sample = generate_avoid_3(rng)
            if sample is not None:
                break

        assert sample is not None, "Should generate a sample within 100 attempts"
        assert sample.pattern_type == PatternType.AVOID_3

    def test_has_two_x_pieces(self):
        """Board should have exactly 2 X pieces (the XX pattern)."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_avoid_3(rng)
            if sample is not None:
                x_count = sum(1 for c in sample.board_state if c == Player.X)
                assert x_count == 2, f"Should have 2 X pieces, got {x_count}"
                break

    def test_correct_player(self):
        """Current player should be X."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_avoid_3(rng)
            if sample is not None:
                assert sample.current_player == Player.X
                break

    def test_forbidden_move_exists(self):
        """Should have exactly one incorrect move (the forbidden cell)."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_avoid_3(rng)
            if sample is not None:
                assert len(sample.incorrect_moves) == 1, \
                    f"Should have 1 incorrect move, got {len(sample.incorrect_moves)}"
                break

    def test_game_not_over(self):
        """Generated position should have game ongoing."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_avoid_3(rng)
            if sample is not None:
                board = Board(sample.board_state)
                result = check_result(board)
                assert result == GameResult.ONGOING, \
                    f"Game should be ongoing, got {result}"
                break


class TestGenerateComplete4:
    """Tests for complete_4 pattern generation."""

    def test_generates_valid_sample(self):
        """Should generate a valid TacticalSample."""
        rng = random.Random(42)
        sample = None
        for _ in range(100):
            sample = generate_complete_4(rng)
            if sample is not None:
                break

        assert sample is not None, "Should generate a sample within 100 attempts"
        assert sample.pattern_type == PatternType.COMPLETE_4

    def test_has_three_x_pieces_in_pattern(self):
        """Board should have X pieces forming XX_X or X_XX pattern."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_complete_4(rng)
            if sample is not None:
                x_count = sum(1 for c in sample.board_state if c == Player.X)
                assert x_count == 3, f"Should have 3 X pieces, got {x_count}"
                break

    def test_exactly_one_correct_move(self):
        """Should have exactly one correct move (the gap)."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_complete_4(rng)
            if sample is not None:
                assert len(sample.correct_moves) == 1, \
                    f"Should have 1 correct move, got {len(sample.correct_moves)}"
                break

    def test_correct_move_is_empty(self):
        """The correct move should be on an empty cell."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_complete_4(rng)
            if sample is not None:
                correct = sample.correct_moves[0]
                assert sample.board_state[correct] == Player.EMPTY, \
                    "Correct move should be on empty cell"
                break

    def test_outcome_is_win(self):
        """Outcome should be 1.0 (win)."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_complete_4(rng)
            if sample is not None:
                assert sample.outcome == 1.0, f"Outcome should be 1.0, got {sample.outcome}"
                break


class TestGenerateBlock4:
    """Tests for block_4 pattern generation."""

    def test_generates_valid_sample(self):
        """Should generate a valid TacticalSample."""
        rng = random.Random(42)
        sample = None
        for _ in range(100):
            sample = generate_block_4(rng)
            if sample is not None:
                break

        assert sample is not None, "Should generate a sample within 100 attempts"
        assert sample.pattern_type == PatternType.BLOCK_4

    def test_has_three_o_pieces_in_pattern(self):
        """Board should have O pieces forming OO_O or O_OO pattern."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_block_4(rng)
            if sample is not None:
                o_count = sum(1 for c in sample.board_state if c == Player.O)
                assert o_count == 3, f"Should have 3 O pieces, got {o_count}"
                break

    def test_current_player_is_x(self):
        """X should be the one to block."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_block_4(rng)
            if sample is not None:
                assert sample.current_player == Player.X
                break

    def test_exactly_one_correct_move(self):
        """Should have exactly one correct move (the blocking cell)."""
        rng = random.Random(42)
        for _ in range(50):
            sample = generate_block_4(rng)
            if sample is not None:
                assert len(sample.correct_moves) == 1, \
                    f"Should have 1 correct move, got {len(sample.correct_moves)}"
                break


class TestGenerateTacticalSamples:
    """Tests for bulk sample generation."""

    def test_generates_samples_per_type(self):
        """Should generate requested number of samples per type."""
        samples = generate_tactical_samples(n_per_type=5, seed=42)

        # Count by type
        counts = {t: 0 for t in PatternType}
        for s in samples:
            counts[s.pattern_type] += 1

        for pattern_type, count in counts.items():
            assert count == 5, \
                f"Should have 5 {pattern_type.value} samples, got {count}"

    def test_reproducible_with_seed(self):
        """Same seed should produce same samples."""
        samples1 = generate_tactical_samples(n_per_type=3, seed=123)
        samples2 = generate_tactical_samples(n_per_type=3, seed=123)

        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1.board_state == s2.board_state
            assert s1.correct_moves == s2.correct_moves

    def test_different_seeds_produce_different_samples(self):
        """Different seeds should produce different samples."""
        samples1 = generate_tactical_samples(n_per_type=3, seed=1)
        samples2 = generate_tactical_samples(n_per_type=3, seed=2)

        # At least some should be different
        different = False
        for s1, s2 in zip(samples1, samples2):
            if s1.board_state != s2.board_state:
                different = True
                break
        assert different, "Different seeds should produce different samples"


class TestTacticalToTraining:
    """Tests for conversion to training samples."""

    def test_converts_correctly(self):
        """Should convert TacticalSample to training Sample."""
        tactical = TacticalSample(
            board_state=[Player.EMPTY] * 36,
            current_player=Player.X,
            pattern_type=PatternType.COMPLETE_4,
            correct_moves=[15],
            incorrect_moves=[i for i in range(36) if i != 15],
            outcome=1.0
        )

        training = tactical_sample_to_training_sample(tactical)

        assert training.board_state == tactical.board_state
        assert training.current_player == tactical.current_player
        assert training.outcome == tactical.outcome

    def test_policy_target_sums_to_one(self):
        """Policy target should sum to 1.0."""
        tactical = TacticalSample(
            board_state=[Player.EMPTY] * 36,
            current_player=Player.X,
            pattern_type=PatternType.COMPLETE_4,
            correct_moves=[15],
            incorrect_moves=[i for i in range(36) if i != 15],
            outcome=1.0
        )

        training = tactical_sample_to_training_sample(tactical)
        policy_sum = sum(training.policy_target)

        assert abs(policy_sum - 1.0) < 0.001, \
            f"Policy target should sum to 1.0, got {policy_sum}"

    def test_policy_target_concentrated_on_correct_moves(self):
        """Policy target should only have mass on correct moves."""
        tactical = TacticalSample(
            board_state=[Player.EMPTY] * 36,
            current_player=Player.X,
            pattern_type=PatternType.COMPLETE_4,
            correct_moves=[10, 20],  # Two correct moves
            incorrect_moves=[i for i in range(36) if i not in [10, 20]],
            outcome=1.0
        )

        training = tactical_sample_to_training_sample(tactical)

        # Correct moves should have equal probability
        assert training.policy_target[10] == 0.5
        assert training.policy_target[20] == 0.5

        # Incorrect moves should have 0 probability
        for i in range(36):
            if i not in [10, 20]:
                assert training.policy_target[i] == 0.0


class TestBoardValidity:
    """Tests that generated boards are valid game states."""

    def test_generated_boards_are_valid(self):
        """All generated boards should be valid game states."""
        samples = generate_tactical_samples(n_per_type=10, seed=42)

        for sample in samples:
            # Check valid piece counts
            x_count = sum(1 for c in sample.board_state if c == Player.X)
            o_count = sum(1 for c in sample.board_state if c == Player.O)

            # X moves first, so x_count should be >= o_count
            # and x_count - o_count should be 0 or 1
            diff = x_count - o_count
            assert diff in [0, 1], \
                f"Invalid piece counts: X={x_count}, O={o_count}"

    def test_no_premature_wins_or_losses(self):
        """Generated positions should not already have 3 or 4 in a row."""
        samples = generate_tactical_samples(n_per_type=10, seed=42)

        for sample in samples:
            board = Board(sample.board_state)
            result = check_result(board)
            assert result == GameResult.ONGOING, \
                f"Game should be ongoing, got {result} for pattern {sample.pattern_type}"


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestLineGeneration,
        TestCountConsecutive,
        TestGenerateAvoid3,
        TestGenerateComplete4,
        TestGenerateBlock4,
        TestGenerateTacticalSamples,
        TestTacticalToTraining,
        TestBoardValidity,
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
