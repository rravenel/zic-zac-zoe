"""
Tests for training configuration and model selection logic.
"""

import torch
from game import Player
from model import ZicZacNet
from train import TrainConfig, Sample
from evaluate import EvalResult


class TestTrainConfig:
    """Tests for TrainConfig with AlphaZero-style model selection."""

    def test_default_win_rate_threshold(self):
        """Default win rate threshold should be 55%."""
        config = TrainConfig()
        assert config.win_rate_threshold == 0.55

    def test_default_eval_games_for_best(self):
        """Default games for best model evaluation."""
        config = TrainConfig()
        assert config.eval_games_for_best == 40

    def test_custom_win_rate_threshold(self):
        """Custom win rate threshold should be settable."""
        config = TrainConfig(win_rate_threshold=0.60)
        assert config.win_rate_threshold == 0.60

    def test_patience_default(self):
        """Early stopping patience should default to 30."""
        config = TrainConfig()
        assert config.early_stopping_patience == 30

    def test_tactical_pass_threshold_default(self):
        """Default tactical pass threshold should be 90%."""
        config = TrainConfig()
        assert config.tactical_pass_threshold == 0.90

    def test_tactical_eval_per_type_default(self):
        """Default tactical tests per pattern type should be 10."""
        config = TrainConfig()
        assert config.tactical_eval_per_type == 10

    def test_custom_tactical_pass_threshold(self):
        """Custom tactical pass threshold should be settable."""
        config = TrainConfig(tactical_pass_threshold=0.90)
        assert config.tactical_pass_threshold == 0.90


class TestEvalResult:
    """Tests for EvalResult win rate calculation."""

    def test_win_rate_all_wins(self):
        """100% win rate when all games are wins."""
        result = EvalResult(wins=10, losses=0, draws=0, total_games=10, avg_game_length=10.0)
        assert result.win_rate == 1.0

    def test_win_rate_all_losses(self):
        """0% win rate when all games are losses."""
        result = EvalResult(wins=0, losses=10, draws=0, total_games=10, avg_game_length=10.0)
        assert result.win_rate == 0.0

    def test_win_rate_mixed(self):
        """Win rate calculation with mixed results."""
        result = EvalResult(wins=6, losses=4, draws=0, total_games=10, avg_game_length=10.0)
        assert result.win_rate == 0.6

    def test_win_rate_with_draws(self):
        """Win rate only counts wins, not draws."""
        result = EvalResult(wins=5, losses=3, draws=2, total_games=10, avg_game_length=10.0)
        assert result.win_rate == 0.5

    def test_win_rate_threshold_comparison(self):
        """Win rate can be compared against threshold."""
        result = EvalResult(wins=6, losses=4, draws=0, total_games=10, avg_game_length=10.0)
        threshold = 0.55
        assert result.win_rate > threshold  # 0.6 > 0.55

        result2 = EvalResult(wins=5, losses=5, draws=0, total_games=10, avg_game_length=10.0)
        assert not (result2.win_rate > threshold)  # 0.5 not > 0.55


class TestModelSelection:
    """Tests for AlphaZero-style model selection logic."""

    def test_new_best_when_above_threshold(self):
        """Model becomes new best when win rate exceeds threshold."""
        config = TrainConfig(win_rate_threshold=0.55)

        # Simulate evaluation result: 24 wins out of 40 = 60%
        vs_best = EvalResult(wins=24, losses=16, draws=0, total_games=40, avg_game_length=10.0)
        is_new_best = vs_best.win_rate > config.win_rate_threshold

        assert is_new_best, f"Win rate {vs_best.win_rate:.2f} should beat threshold {config.win_rate_threshold}"

    def test_not_new_best_when_below_threshold(self):
        """Model does not become new best when win rate below threshold."""
        config = TrainConfig(win_rate_threshold=0.55)

        vs_best = EvalResult(wins=20, losses=20, draws=0, total_games=40, avg_game_length=10.0)
        is_new_best = vs_best.win_rate > config.win_rate_threshold

        assert not is_new_best, f"Win rate {vs_best.win_rate:.2f} should not beat threshold {config.win_rate_threshold}"

    def test_not_new_best_at_exact_threshold(self):
        """Model does not become new best at exactly the threshold (need to beat it)."""
        config = TrainConfig(win_rate_threshold=0.55)

        # 22 wins out of 40 = 0.55 exactly
        vs_best = EvalResult(wins=22, losses=18, draws=0, total_games=40, avg_game_length=10.0)
        is_new_best = vs_best.win_rate > config.win_rate_threshold

        assert not is_new_best, "Exactly at threshold should not count as new best"

    def test_patience_counter_logic(self):
        """Patience counter increments when no improvement, resets on new best."""
        config = TrainConfig(win_rate_threshold=0.55, early_stopping_patience=3)

        patience_counter = 0

        # No improvement (50% win rate)
        vs_best = EvalResult(wins=20, losses=20, draws=0, total_games=40, avg_game_length=10.0)
        if vs_best.win_rate > config.win_rate_threshold:
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 1

        # No improvement again
        if vs_best.win_rate > config.win_rate_threshold:
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 2

        # Improvement! (62.5% win rate)
        vs_best = EvalResult(wins=25, losses=15, draws=0, total_games=40, avg_game_length=10.0)
        if vs_best.win_rate > config.win_rate_threshold:
            patience_counter = 0
        else:
            patience_counter += 1
        assert patience_counter == 0, "Patience should reset on new best"

    def test_early_stopping_trigger(self):
        """Early stopping triggers after patience exhausted."""
        config = TrainConfig(win_rate_threshold=0.55, early_stopping_patience=3)

        patience_counter = 3
        should_stop = patience_counter >= config.early_stopping_patience

        assert should_stop, "Should trigger early stopping when patience exhausted"

    def test_new_best_requires_tactical_pass(self):
        """Model must pass tactical gate AND beat win rate threshold."""
        config = TrainConfig(win_rate_threshold=0.55, tactical_pass_threshold=0.90)

        # High win rate but failed tactical
        vs_best = EvalResult(wins=30, losses=10, draws=0, total_games=40, avg_game_length=10.0)
        tactical_rate = 0.85  # Below 90% threshold

        wins_vs_best = vs_best.win_rate > config.win_rate_threshold
        passes_tactical = tactical_rate >= config.tactical_pass_threshold
        is_new_best = wins_vs_best and passes_tactical

        assert wins_vs_best, "Should beat win rate threshold"
        assert not passes_tactical, "Should fail tactical threshold"
        assert not is_new_best, "Should NOT be new best without tactical pass"

    def test_new_best_with_both_gates_passed(self):
        """Model becomes new best when both win rate and tactical pass."""
        config = TrainConfig(win_rate_threshold=0.55, tactical_pass_threshold=0.90)

        vs_best = EvalResult(wins=30, losses=10, draws=0, total_games=40, avg_game_length=10.0)
        tactical_rate = 1.0  # 100% tactical

        wins_vs_best = vs_best.win_rate > config.win_rate_threshold
        passes_tactical = tactical_rate >= config.tactical_pass_threshold
        is_new_best = wins_vs_best and passes_tactical

        assert is_new_best, "Should be new best with both gates passed"

    def test_not_new_best_with_low_win_rate_but_good_tactical(self):
        """Model with good tactical but low win rate is not new best."""
        config = TrainConfig(win_rate_threshold=0.55, tactical_pass_threshold=0.90)

        vs_best = EvalResult(wins=20, losses=20, draws=0, total_games=40, avg_game_length=10.0)
        tactical_rate = 1.0  # Perfect tactical

        wins_vs_best = vs_best.win_rate > config.win_rate_threshold
        passes_tactical = tactical_rate >= config.tactical_pass_threshold
        is_new_best = wins_vs_best and passes_tactical

        assert not wins_vs_best, "Should not beat win rate threshold"
        assert passes_tactical, "Should pass tactical threshold"
        assert not is_new_best, "Should NOT be new best without win rate"


class TestSample:
    """Tests for training Sample dataclass."""

    def test_sample_creation(self):
        """Sample can be created with required fields."""
        sample = Sample(
            board_state=[0] * 36,
            current_player=Player.X,
            policy_target=[1/36] * 36,
            outcome=1.0
        )
        assert sample.outcome == 1.0
        assert sample.current_player == Player.X


def run_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestTrainConfig,
        TestEvalResult,
        TestModelSelection,
        TestSample,
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
