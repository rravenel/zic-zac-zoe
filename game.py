"""
Zic-Zac-Zoe Game Logic

Rules:
- 6x6 grid
- 4 in a row wins
- 3 in a row loses (this is the twist!)
- Diagonals count
- X moves first
"""

from typing import List, Tuple, Optional
from enum import IntEnum

# =============================================================================
# Constants
# =============================================================================

BOARD_SIZE = 6
WIN_LENGTH = 4   # 4 in a row wins
LOSE_LENGTH = 3  # 3 in a row loses


# =============================================================================
# Enums
# =============================================================================

class Player(IntEnum):
    """Represents a cell state or current player."""
    EMPTY = 0
    X = 1
    O = 2

    def opponent(self) -> 'Player':
        """Return the opposing player."""
        if self == Player.X:
            return Player.O
        elif self == Player.O:
            return Player.X
        return Player.EMPTY


class GameResult(IntEnum):
    """Possible game outcomes."""
    ONGOING = 0
    X_WINS = 1   # X got 4 in a row, or O got 3 in a row
    O_WINS = 2   # O got 4 in a row, or X got 3 in a row
    DRAW = 3     # Board full, no winner (rare given the 3-in-a-row rule)


# =============================================================================
# Board Class
# =============================================================================

class Board:
    """
    6x6 game board for Zic-Zac-Zoe.

    Internal representation: list of 36 integers (0=empty, 1=X, 2=O)
    Position mapping: index = row * 6 + col
    """

    def __init__(self, state: Optional[List[int]] = None):
        """Initialize board, optionally from existing state."""
        if state is not None:
            self.state = state.copy()
        else:
            self.state = [Player.EMPTY] * (BOARD_SIZE * BOARD_SIZE)
        self._move_count = sum(1 for cell in self.state if cell != Player.EMPTY)

    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(self.state)
        new_board._move_count = self._move_count
        return new_board

    # -------------------------------------------------------------------------
    # Accessors
    # -------------------------------------------------------------------------

    def get(self, row: int, col: int) -> Player:
        """Get cell value at (row, col)."""
        return Player(self.state[row * BOARD_SIZE + col])

    def get_flat(self, index: int) -> Player:
        """Get cell value by flat index (0-35)."""
        return Player(self.state[index])

    def is_empty(self, row: int, col: int) -> bool:
        """Check if cell at (row, col) is empty."""
        return self.state[row * BOARD_SIZE + col] == Player.EMPTY

    def is_empty_flat(self, index: int) -> bool:
        """Check if cell at flat index is empty."""
        return self.state[index] == Player.EMPTY

    # -------------------------------------------------------------------------
    # Game State
    # -------------------------------------------------------------------------

    def move_count(self) -> int:
        """Return number of moves made so far."""
        return self._move_count

    def current_player(self) -> Player:
        """Return whose turn it is. X moves first."""
        return Player.X if self._move_count % 2 == 0 else Player.O

    def get_legal_moves(self) -> List[int]:
        """Return list of legal move indices (empty cells)."""
        return [i for i, cell in enumerate(self.state) if cell == Player.EMPTY]

    def is_full(self) -> bool:
        """Check if board is completely filled."""
        return self._move_count == BOARD_SIZE * BOARD_SIZE

    # -------------------------------------------------------------------------
    # Making Moves
    # -------------------------------------------------------------------------

    def make_move(self, index: int) -> 'Board':
        """
        Make a move at the given flat index.
        Returns a new Board (does not modify self).
        """
        if self.state[index] != Player.EMPTY:
            raise ValueError(f"Position {index} is not empty")

        new_board = self.copy()
        new_board.state[index] = self.current_player()
        new_board._move_count += 1
        return new_board

    def make_move_inplace(self, index: int) -> None:
        """Make a move in place (modifies self). Use for performance."""
        if self.state[index] != Player.EMPTY:
            raise ValueError(f"Position {index} is not empty")

        self.state[index] = self.current_player()
        self._move_count += 1

    # -------------------------------------------------------------------------
    # Hashing (for deduplication)
    # -------------------------------------------------------------------------

    def hash(self) -> int:
        """Return hash of board state."""
        return hash(tuple(self.state))

    def __hash__(self) -> int:
        return self.hash()

    def __eq__(self, other: 'Board') -> bool:
        return self.state == other.state

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    def __str__(self) -> str:
        """Pretty print the board."""
        symbols = {Player.EMPTY: '.', Player.X: 'X', Player.O: 'O'}
        lines = []
        lines.append("  " + " ".join(str(c) for c in range(BOARD_SIZE)))
        for row in range(BOARD_SIZE):
            cells = [symbols[self.get(row, col)] for col in range(BOARD_SIZE)]
            lines.append(f"{row} " + " ".join(cells))
        return "\n".join(lines)


# =============================================================================
# Win/Loss Detection
# =============================================================================

# Precompute all lines (rows, cols, diagonals) that could form 3 or 4 in a row.
# Each line is a list of flat indices.

def _generate_lines() -> List[List[int]]:
    """Generate all lines of length >= 3 on the board."""
    lines = []

    # Horizontal lines
    for row in range(BOARD_SIZE):
        for start_col in range(BOARD_SIZE - 2):  # Need at least 3
            # Lines of length 3, 4, 5, 6
            for length in range(3, BOARD_SIZE - start_col + 1):
                line = [row * BOARD_SIZE + start_col + i for i in range(length)]
                lines.append(line)

    # Vertical lines
    for col in range(BOARD_SIZE):
        for start_row in range(BOARD_SIZE - 2):
            for length in range(3, BOARD_SIZE - start_row + 1):
                line = [(start_row + i) * BOARD_SIZE + col for i in range(length)]
                lines.append(line)

    # Diagonal lines (top-left to bottom-right)
    for start_row in range(BOARD_SIZE):
        for start_col in range(BOARD_SIZE):
            max_len = min(BOARD_SIZE - start_row, BOARD_SIZE - start_col)
            for length in range(3, max_len + 1):
                line = [(start_row + i) * BOARD_SIZE + (start_col + i) for i in range(length)]
                lines.append(line)

    # Diagonal lines (top-right to bottom-left)
    for start_row in range(BOARD_SIZE):
        for start_col in range(BOARD_SIZE):
            max_len = min(BOARD_SIZE - start_row, start_col + 1)
            for length in range(3, max_len + 1):
                line = [(start_row + i) * BOARD_SIZE + (start_col - i) for i in range(length)]
                lines.append(line)

    return lines

# Cache all lines at module load
ALL_LINES = _generate_lines()


def check_result(board: Board) -> GameResult:
    """
    Check if the game has ended.

    Returns:
        GameResult.X_WINS if X wins (X got 4, or O got exactly 3)
        GameResult.O_WINS if O wins (O got 4, or X got exactly 3)
        GameResult.DRAW if board is full with no winner
        GameResult.ONGOING if game continues
    """
    x_max = 0  # Max consecutive X pieces in any line
    o_max = 0  # Max consecutive O pieces in any line

    # Check each possible line for consecutive pieces
    for line in ALL_LINES:
        # Count consecutive pieces in this line
        x_count = 0
        o_count = 0

        # Check if entire line is same player
        first = board.get_flat(line[0])
        if first != Player.EMPTY:
            all_same = all(board.get_flat(i) == first for i in line)
            if all_same:
                if first == Player.X:
                    x_count = len(line)
                else:
                    o_count = len(line)

        x_max = max(x_max, x_count)
        o_max = max(o_max, o_count)

    # Check for exactly 3 (loss) - takes priority over 4+ check
    # because if you make a move that creates both 3 and 4,
    # the 3 was created first (you lose)
    if x_max == LOSE_LENGTH:
        return GameResult.O_WINS  # X loses = O wins
    if o_max == LOSE_LENGTH:
        return GameResult.X_WINS  # O loses = X wins

    # Check for 4+ (win)
    if x_max >= WIN_LENGTH:
        return GameResult.X_WINS
    if o_max >= WIN_LENGTH:
        return GameResult.O_WINS

    # Check for draw (full board)
    if board.is_full():
        return GameResult.DRAW

    return GameResult.ONGOING


def check_result_fast(board: Board, last_move: int) -> GameResult:
    """
    Optimized result check - only examines lines containing last_move.
    Use this after making a move for better performance.
    """
    last_player = Player.X if board.move_count() % 2 == 1 else Player.O
    last_row = last_move // BOARD_SIZE
    last_col = last_move % BOARD_SIZE

    max_consecutive = 0

    # Check all 4 directions from the last move
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for d_row, d_col in directions:
        count = 1  # Count the piece itself

        # Count in positive direction
        r, c = last_row + d_row, last_col + d_col
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board.get(r, c) == last_player:
                count += 1
                r += d_row
                c += d_col
            else:
                break

        # Count in negative direction
        r, c = last_row - d_row, last_col - d_col
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            if board.get(r, c) == last_player:
                count += 1
                r -= d_row
                c -= d_col
            else:
                break

        max_consecutive = max(max_consecutive, count)

    # Check loss condition first (exactly 3)
    if max_consecutive == LOSE_LENGTH:
        if last_player == Player.X:
            return GameResult.O_WINS
        else:
            return GameResult.X_WINS

    # Check win condition (4+)
    if max_consecutive >= WIN_LENGTH:
        if last_player == Player.X:
            return GameResult.X_WINS
        else:
            return GameResult.O_WINS

    # Check draw
    if board.is_full():
        return GameResult.DRAW

    return GameResult.ONGOING


# =============================================================================
# Utility Functions
# =============================================================================

def index_to_coord(index: int) -> Tuple[int, int]:
    """Convert flat index to (row, col)."""
    return index // BOARD_SIZE, index % BOARD_SIZE


def coord_to_index(row: int, col: int) -> int:
    """Convert (row, col) to flat index."""
    return row * BOARD_SIZE + col


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check
    board = Board()
    print("Empty board:")
    print(board)
    print(f"\nCurrent player: {board.current_player().name}")
    print(f"Legal moves: {len(board.get_legal_moves())}")

    # Test 3-in-a-row loss
    board = Board()
    moves = [0, 6, 1, 7, 2]  # X plays 0,1,2 (top row) = 3 in a row = X loses
    for m in moves:
        board = board.make_move(m)
    print("\nAfter X plays 3 in a row:")
    print(board)
    print(f"Result: {check_result(board).name}")
