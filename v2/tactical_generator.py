"""
Tactical Position Generator

Generates random tactical positions for training and evaluation.
Patterns are placed at random orientations and locations on the board.
"""

import random
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from game import Board, Player, BOARD_SIZE, check_result


# =============================================================================
# Line Definitions
# =============================================================================

def get_all_lines() -> List[List[int]]:
    """
    Get all lines of length 4+ on the 6x6 board.
    Returns list of lines, where each line is a list of cell indices.
    """
    lines = []

    # Horizontal lines
    for row in range(BOARD_SIZE):
        for start_col in range(BOARD_SIZE - 3):  # Need at least 4 cells
            line = [row * BOARD_SIZE + col for col in range(start_col, min(start_col + 6, BOARD_SIZE))]
            if len(line) >= 4:
                lines.append(line)

    # Vertical lines
    for col in range(BOARD_SIZE):
        for start_row in range(BOARD_SIZE - 3):
            line = [row * BOARD_SIZE + col for row in range(start_row, min(start_row + 6, BOARD_SIZE))]
            if len(line) >= 4:
                lines.append(line)

    # Diagonal down-right (\)
    for start_row in range(BOARD_SIZE - 3):
        for start_col in range(BOARD_SIZE - 3):
            line = []
            r, c = start_row, start_col
            while r < BOARD_SIZE and c < BOARD_SIZE:
                line.append(r * BOARD_SIZE + c)
                r += 1
                c += 1
            if len(line) >= 4:
                lines.append(line)

    # Diagonal up-right (/)
    for start_row in range(3, BOARD_SIZE):
        for start_col in range(BOARD_SIZE - 3):
            line = []
            r, c = start_row, start_col
            while r >= 0 and c < BOARD_SIZE:
                line.append(r * BOARD_SIZE + c)
                r -= 1
                c += 1
            if len(line) >= 4:
                lines.append(line)

    return lines


# Pre-compute all lines
ALL_LINES = get_all_lines()


# =============================================================================
# Pattern Types
# =============================================================================

class PatternType(Enum):
    AVOID_3 = "avoid_3"      # Don't complete your own 3-in-a-row
    COMPLETE_4 = "complete_4"  # Complete your own 4-in-a-row (XX_X pattern)
    BLOCK_4 = "block_4"      # Block opponent's 4-in-a-row


@dataclass
class TacticalSample:
    """A generated tactical position with known correct action."""
    board_state: List[int]
    current_player: Player
    pattern_type: PatternType
    correct_moves: List[int]    # Moves that are correct
    incorrect_moves: List[int]  # Moves that are definitely wrong
    outcome: float              # Expected outcome if played correctly


# =============================================================================
# Board Validation
# =============================================================================

def count_consecutive(board_state: List[int], line: List[int], player: Player) -> int:
    """Count maximum consecutive pieces of player in a line."""
    max_count = 0
    current_count = 0

    for idx in line:
        if board_state[idx] == player:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0

    return max_count


def has_other_tactical_conditions(board_state: List[int],
                                   pattern_line: List[int],
                                   current_player: Player) -> bool:
    """
    Check if the board has any tactical conditions outside the pattern line.

    Returns True if there are other threats that could confuse the test.
    """
    opponent = Player.O if current_player == Player.X else Player.X
    pattern_set = set(pattern_line)

    for line in ALL_LINES:
        # Skip if this line overlaps with our pattern line
        if set(line) & pattern_set:
            continue

        # Check for any 2+ consecutive pieces (potential threats)
        if count_consecutive(board_state, line, current_player) >= 2:
            return True
        if count_consecutive(board_state, line, opponent) >= 2:
            return True

    return False


def find_gap_in_pattern(board_state: List[int], indices: List[int], player: Player) -> Optional[int]:
    """
    Find the gap position in a XX_X or X_XX pattern.
    Returns the gap index, or None if no valid pattern.
    """
    # indices should be 4 consecutive positions in a line
    # We want exactly 3 pieces with 1 gap
    pieces = [board_state[i] == player for i in indices]
    empties = [board_state[i] == Player.EMPTY for i in indices]

    if sum(pieces) == 3 and sum(empties) == 1:
        for i, idx in enumerate(indices):
            if board_state[idx] == Player.EMPTY:
                return idx
    return None


# =============================================================================
# Pattern Generators
# =============================================================================

def generate_avoid_3(rng: random.Random) -> Optional[TacticalSample]:
    """
    Generate an avoid_3 position.

    Player has XX in a line. Must not play the adjacent cell to make XXX.
    """
    # Pick a random line
    line = rng.choice(ALL_LINES)
    if len(line) < 3:
        return None

    # Pick a starting position for XX pattern (need room for 3rd)
    max_start = len(line) - 3
    if max_start < 0:
        return None
    start = rng.randint(0, max_start)

    # Positions: XX_ (the _ is the forbidden move)
    pos1, pos2, forbidden = line[start], line[start + 1], line[start + 2]

    # Current player is X, X has 2 pieces
    current_player = Player.X
    board_state = [Player.EMPTY] * (BOARD_SIZE * BOARD_SIZE)
    board_state[pos1] = Player.X
    board_state[pos2] = Player.X

    # Add O pieces for turn parity (X has 2, need O to have 1 for X's turn)
    # Pick a random empty cell not in the pattern line
    pattern_cells = set(line)
    available = [i for i in range(36) if i not in pattern_cells and board_state[i] == Player.EMPTY]
    if not available:
        return None

    o_pos = rng.choice(available)
    board_state[o_pos] = Player.O

    # Verify no other tactical conditions
    if has_other_tactical_conditions(board_state, line, current_player):
        return None

    # Verify the board is valid (game not over)
    board = Board(board_state)
    if check_result(board) != board.GameResult.ONGOING if hasattr(board, 'GameResult') else True:
        pass  # Check using imported GameResult

    from game import GameResult
    if check_result(board) != GameResult.ONGOING:
        return None

    # Correct moves: anything except the forbidden cell
    all_empty = [i for i in range(36) if board_state[i] == Player.EMPTY]
    correct_moves = [i for i in all_empty if i != forbidden]
    incorrect_moves = [forbidden]

    return TacticalSample(
        board_state=board_state,
        current_player=current_player,
        pattern_type=PatternType.AVOID_3,
        correct_moves=correct_moves,
        incorrect_moves=incorrect_moves,
        outcome=0.0  # Neutral - just don't lose
    )


def generate_complete_4(rng: random.Random) -> Optional[TacticalSample]:
    """
    Generate a complete_4 position.

    Player has XX_X or X_XX pattern. Must play the gap to win.
    """
    # Pick a random line with at least 4 cells
    line = rng.choice([l for l in ALL_LINES if len(l) >= 4])

    # Pick 4 consecutive positions
    max_start = len(line) - 4
    if max_start < 0:
        return None
    start = rng.randint(0, max_start)

    four_cells = line[start:start + 4]

    # Choose gap position (index 1 or 2 for X_XX or XX_X patterns)
    gap_idx = rng.choice([1, 2])

    current_player = Player.X
    board_state = [Player.EMPTY] * 36

    # Place X pieces with gap
    for i, cell in enumerate(four_cells):
        if i != gap_idx:
            board_state[cell] = Player.X

    gap_cell = four_cells[gap_idx]

    # Add O pieces for turn parity (X has 3, need O to have 3 for X's turn)
    pattern_cells = set(line)
    available = [i for i in range(36) if i not in pattern_cells and board_state[i] == Player.EMPTY]

    if len(available) < 3:
        return None

    o_positions = rng.sample(available, 3)
    for pos in o_positions:
        board_state[pos] = Player.O

    # Verify no other tactical conditions
    if has_other_tactical_conditions(board_state, line, current_player):
        return None

    # Verify game is ongoing
    from game import GameResult
    board = Board(board_state)
    if check_result(board) != GameResult.ONGOING:
        return None

    # Verify it's X's turn
    if board.current_player() != Player.X:
        return None

    correct_moves = [gap_cell]
    all_empty = [i for i in range(36) if board_state[i] == Player.EMPTY]
    incorrect_moves = [i for i in all_empty if i != gap_cell]

    return TacticalSample(
        board_state=board_state,
        current_player=current_player,
        pattern_type=PatternType.COMPLETE_4,
        correct_moves=correct_moves,
        incorrect_moves=incorrect_moves,
        outcome=1.0  # Win
    )


def generate_block_4(rng: random.Random) -> Optional[TacticalSample]:
    """
    Generate a block_4 position.

    Opponent has OO_O or O_OO pattern. Player must block the gap.
    """
    # Pick a random line with at least 4 cells
    line = rng.choice([l for l in ALL_LINES if len(l) >= 4])

    # Pick 4 consecutive positions
    max_start = len(line) - 4
    if max_start < 0:
        return None
    start = rng.randint(0, max_start)

    four_cells = line[start:start + 4]

    # Choose gap position
    gap_idx = rng.choice([1, 2])

    current_player = Player.X  # X must block O's threat
    board_state = [Player.EMPTY] * 36

    # Place O pieces with gap
    for i, cell in enumerate(four_cells):
        if i != gap_idx:
            board_state[cell] = Player.O

    gap_cell = four_cells[gap_idx]

    # Add X pieces for turn parity (O has 3, need X to have 3 for X's turn)
    pattern_cells = set(line)
    available = [i for i in range(36) if i not in pattern_cells and board_state[i] == Player.EMPTY]

    if len(available) < 3:
        return None

    x_positions = rng.sample(available, 3)
    for pos in x_positions:
        board_state[pos] = Player.X

    # Verify no other tactical conditions
    if has_other_tactical_conditions(board_state, line, current_player):
        return None

    # Verify game is ongoing
    from game import GameResult
    board = Board(board_state)
    if check_result(board) != GameResult.ONGOING:
        return None

    # Verify it's X's turn
    if board.current_player() != Player.X:
        return None

    correct_moves = [gap_cell]
    all_empty = [i for i in range(36) if board_state[i] == Player.EMPTY]
    incorrect_moves = [i for i in all_empty if i != gap_cell]

    return TacticalSample(
        board_state=board_state,
        current_player=current_player,
        pattern_type=PatternType.BLOCK_4,
        correct_moves=correct_moves,
        incorrect_moves=incorrect_moves,
        outcome=0.0  # Neutral - just block the threat
    )


# =============================================================================
# Main Generator
# =============================================================================

def generate_tactical_samples(n_per_type: int = 20,
                              seed: Optional[int] = None) -> List[TacticalSample]:
    """
    Generate a batch of tactical samples.

    Args:
        n_per_type: Number of samples to generate per pattern type
        seed: Random seed for reproducibility

    Returns:
        List of TacticalSample objects
    """
    rng = random.Random(seed)
    samples = []

    generators = {
        PatternType.AVOID_3: generate_avoid_3,
        PatternType.COMPLETE_4: generate_complete_4,
        PatternType.BLOCK_4: generate_block_4,
    }

    for pattern_type, generator in generators.items():
        count = 0
        attempts = 0
        max_attempts = n_per_type * 20  # Allow many attempts due to validation

        while count < n_per_type and attempts < max_attempts:
            attempts += 1
            sample = generator(rng)
            if sample is not None:
                samples.append(sample)
                count += 1

    return samples


def tactical_sample_to_training_sample(tactical: TacticalSample):
    """
    Convert a TacticalSample to a training Sample for the replay buffer.

    The policy target is set to encourage the correct move(s).
    """
    from train import Sample

    # Create policy target: uniform over correct moves
    policy_target = [0.0] * 36
    if tactical.correct_moves:
        prob = 1.0 / len(tactical.correct_moves)
        for move in tactical.correct_moves:
            policy_target[move] = prob

    return Sample(
        board_state=tactical.board_state,
        current_player=tactical.current_player,
        policy_target=policy_target,
        outcome=tactical.outcome
    )


# =============================================================================
# Testing / Demo
# =============================================================================

if __name__ == "__main__":
    # Generate some samples and display them
    samples = generate_tactical_samples(n_per_type=3, seed=42)

    print(f"Generated {len(samples)} tactical samples:\n")

    for sample in samples:
        print(f"Pattern: {sample.pattern_type.value}")
        print(f"Current player: {sample.current_player.name}")
        print(f"Correct moves: {sample.correct_moves}")
        print(f"Incorrect moves: {sample.incorrect_moves[:5]}..." if len(sample.incorrect_moves) > 5 else f"Incorrect moves: {sample.incorrect_moves}")

        # Render board
        print("Board:")
        for row in range(BOARD_SIZE):
            line = "  "
            for col in range(BOARD_SIZE):
                idx = row * BOARD_SIZE + col
                cell = sample.board_state[idx]
                if cell == Player.X:
                    char = "X"
                elif cell == Player.O:
                    char = "O"
                elif idx in sample.correct_moves:
                    char = "?"
                else:
                    char = "."
                line += char + " "
            print(line)
        print()
