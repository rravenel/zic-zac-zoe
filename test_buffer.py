"""
Tests for ReplayBuffer and Reanalyze.
"""

import tempfile
import os
import torch
from game import Board, Player
from model import ZicZacNet
from train import (
    ReplayBuffer,
    PositionSample,
    reanalyze_batch,
    Sample,
)


def make_position_sample(outcome: float, board_fill: int = 0) -> PositionSample:
    """Helper to create a PositionSample with required fields."""
    return PositionSample(
        board_state=[board_fill] * 36,
        current_player=Player.X,
        outcome=outcome
    )


class TestReplayBuffer:
    """Tests for the simple FIFO replay buffer."""

    def test_add_and_len(self):
        """Basic add and length check."""
        buffer = ReplayBuffer(max_size=100)
        assert len(buffer) == 0

        samples = [make_position_sample(1.0), make_position_sample(-1.0)]
        buffer.add(samples)
        assert len(buffer) == 2

    def test_fifo_eviction(self):
        """FIFO eviction should remove oldest samples."""
        buffer = ReplayBuffer(max_size=3)

        # Add 3 samples with different outcomes to identify them
        for i in range(3):
            buffer.add([make_position_sample(float(i), board_fill=i)])

        assert len(buffer) == 3
        outcomes = [s.outcome for s in buffer.samples]
        assert outcomes == [0.0, 1.0, 2.0]

        # Add one more - should evict the oldest (0.0)
        buffer.add([make_position_sample(99.0, board_fill=99)])

        assert len(buffer) == 3
        outcomes = [s.outcome for s in buffer.samples]
        assert 0.0 not in outcomes, "Oldest sample should be evicted"
        assert 99.0 in outcomes, "New sample should be present"

    def test_uniform_sampling(self):
        """Sampling should be approximately uniform."""
        buffer = ReplayBuffer(max_size=100)

        # Add 10 samples with distinct outcomes
        samples = [make_position_sample(float(i), board_fill=i) for i in range(10)]
        buffer.add(samples)

        # Sample many times and count
        counts = {i: 0 for i in range(10)}
        n_samples = 1000
        for _ in range(n_samples):
            batch = buffer.sample_batch(1)
            counts[int(batch[0].outcome)] += 1

        # Each should be sampled roughly 100 times (10% of 1000)
        # Allow wide margin for randomness
        for i, count in counts.items():
            assert 50 < count < 150, f"Sample {i} count {count} too far from expected 100"

    def test_sample_batch_size(self):
        """sample_batch should return requested batch size."""
        buffer = ReplayBuffer(max_size=100)

        samples = [make_position_sample(float(i), board_fill=i) for i in range(10)]
        buffer.add(samples)

        batch = buffer.sample_batch(5)
        assert len(batch) == 5

        # Request more than available
        batch = buffer.sample_batch(20)
        assert len(batch) == 10  # Should return all available

    def test_save_and_load(self):
        """Buffer should save and load correctly."""
        buffer = ReplayBuffer(max_size=100)
        samples = [make_position_sample(float(i), board_fill=i) for i in range(5)]
        buffer.add(samples)

        # Save
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        buffer.save(path)

        # Load into new buffer
        buffer2 = ReplayBuffer(max_size=100)
        buffer2.load(path)

        assert len(buffer2) == 5
        assert buffer2.samples[0].outcome == buffer.samples[0].outcome

        os.unlink(path)

    def test_stats(self):
        """Stats should return correct values."""
        buffer = ReplayBuffer(max_size=100)

        samples = [make_position_sample(float(i), board_fill=i) for i in range(3)]
        buffer.add(samples)

        stats = buffer.stats()
        assert stats['size'] == 3


class TestReanalyze:
    """Tests for the reanalyze function."""

    def test_reanalyze_returns_samples(self):
        """reanalyze_batch should return Sample objects."""
        model = ZicZacNet()
        device = torch.device('cpu')

        positions = [
            PositionSample(board_state=[0]*36, current_player=Player.X, outcome=1.0),
            PositionSample(board_state=[0]*36, current_player=Player.X, outcome=-1.0),
        ]

        samples = reanalyze_batch(positions, model, device, num_simulations=10)

        assert len(samples) == 2
        assert all(isinstance(s, Sample) for s in samples)

    def test_reanalyze_preserves_outcome(self):
        """Reanalyze should preserve the original outcome (ground truth)."""
        model = ZicZacNet()
        device = torch.device('cpu')

        positions = [
            PositionSample(board_state=[0]*36, current_player=Player.X, outcome=1.0),
            PositionSample(board_state=[0]*36, current_player=Player.X, outcome=-1.0),
        ]

        samples = reanalyze_batch(positions, model, device, num_simulations=10)

        assert samples[0].outcome == 1.0
        assert samples[1].outcome == -1.0

    def test_reanalyze_generates_policy(self):
        """Reanalyze should generate valid policy targets."""
        model = ZicZacNet()
        device = torch.device('cpu')

        positions = [
            PositionSample(board_state=[0]*36, current_player=Player.X, outcome=0.0),
        ]

        samples = reanalyze_batch(positions, model, device, num_simulations=10)

        # Policy should sum to 1
        policy_sum = sum(samples[0].policy_target)
        assert abs(policy_sum - 1.0) < 0.01, f"Policy should sum to 1, got {policy_sum}"

        # Policy should have 36 elements
        assert len(samples[0].policy_target) == 36

    def test_reanalyze_different_models_give_different_policy(self):
        """Different models should produce different policy targets."""
        device = torch.device('cpu')

        # Two different random models
        model1 = ZicZacNet()
        model2 = ZicZacNet()

        positions = [
            PositionSample(board_state=[0]*36, current_player=Player.X, outcome=0.0),
        ]

        samples1 = reanalyze_batch(positions, model1, device, num_simulations=20)
        samples2 = reanalyze_batch(positions, model2, device, num_simulations=20)

        # Policies should likely be different (very unlikely to be exactly same)
        policy1 = samples1[0].policy_target
        policy2 = samples2[0].policy_target

        # At least some values should differ
        differences = sum(abs(p1 - p2) for p1, p2 in zip(policy1, policy2))
        assert differences > 0.01, "Different models should give different policies"


class TestPositionSample:
    """Tests for PositionSample dataclass."""

    def test_position_sample_creation(self):
        """PositionSample can be created with required fields."""
        sample = PositionSample(
            board_state=[0] * 36,
            current_player=Player.X,
            outcome=1.0
        )
        assert sample.outcome == 1.0
        assert sample.current_player == Player.X
        assert len(sample.board_state) == 36


def run_tests():
    """Run all tests."""
    import traceback

    test_classes = [
        TestReplayBuffer,
        TestReanalyze,
        TestPositionSample,
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
