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

    def __init__(self, state: Optional[List[int]] = None, score_x: int = 0, score_o: int = 0):
        """Initialize board, optionally from existing state."""
        if state is not None:
            self.state = state.copy()
        else:
            self.state = [Player.EMPTY] * (BOARD_SIZE * BOARD_SIZE)
        self._move_count = sum(1 for cell in self.state if cell != Player.EMPTY)
        self.score_x = score_x
        self.score_o = score_o

    def copy(self) -> 'Board':
        """Create a deep copy of the board."""
        new_board = Board(self.state, self.score_x, self.score_o)
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

        player = self.current_player()
        # Calculate score for this move BEFORE applying it to the board
        move_score = calculate_move_score(self, index, player)

        new_board = self.copy()
        new_board.state[index] = player
        new_board._move_count += 1

        if player == Player.X:
            new_board.score_x += move_score
        else:
            new_board.score_o += move_score

        return new_board

    def make_move_inplace(self, index: int) -> None:
        """Make a move in place (modifies self). Use for performance."""
        if self.state[index] != Player.EMPTY:
            raise ValueError(f"Position {index} is not empty")

        player = self.current_player()
        # Calculate score for this move BEFORE applying it to the board
        move_score = calculate_move_score(self, index, player)

        self.state[index] = player
        self._move_count += 1

        if player == Player.X:
            self.score_x += move_score
        else:
            self.score_o += move_score

    # -------------------------------------------------------------------------
    # Hashing (for deduplication)
    # -------------------------------------------------------------------------

    def hash(self) -> int:
        """Return hash of board state, including scores."""
        return hash((tuple(self.state), self.score_x, self.score_o))

    def __hash__(self) -> int:
        return self.hash()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return False
        return (self.state == other.state and
                self.score_x == other.score_x and
                self.score_o == other.score_o)

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
# Scoring Engine
# =============================================================================

def calculate_move_score(board: 'Board', move_index: int, player: Player) -> int:
    """
    Calculate the score for a specific move based on the points variant rules.

    Rules:
    - Base score is the new length L (L=3 -> 0).
    - Bridge Bonus: 2x Base Score if move connects two existing segments.
    - Multi-Line Multiplier N: (Sum of line scores) * N (where N = count of axes with L > 1).
    - Lone Tile: If N=0 (no neighbors), the move scores exactly 1 point.
    """
    row, col = move_index // BOARD_SIZE, move_index % BOARD_SIZE
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    line_scores = []

    for dr, dc in directions:
        # Scan in negative direction
        l1 = 0
        r, c = row - dr, col - dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board.get(r, c) == player:
            l1 += 1
            r -= dr
            c -= dc

        # Scan in positive direction
        l2 = 0
        r, c = row + dr, col + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board.get(r, c) == player:
            l2 += 1
            r += dr
            c += dc

        total_l = l1 + l2 + 1

        if total_l > 1:
            # Base Score logic
            base = 0 if total_l == 3 else total_l

            # Bridge Bonus: 2x if move has neighbors on both sides of this axis
            if l1 > 0 and l2 > 0:
                base *= 2

            line_scores.append(base)

    # Multiplier N: count of axes that actually scored points (> 0)
    # This prevents lines of length 3 from contributing to the multiplier.
    n_productive = sum(1 for s in line_scores if s > 0)

    if n_productive == 0:
        # Check if this was a Lone Tile or a Trap
        if len(line_scores) == 0:
            return 1  # Lone tile rule (no neighbors)
        else:
            return 0  # The Trap (all neighbors were lines of 3)

    return sum(line_scores) * n_productive


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
        GameResult.DRAW if board is full
        GameResult.ONGOING if game continues
    """
    # In the points variant, the game only ends when the board is full.
    if board.is_full():
        return GameResult.DRAW

    return GameResult.ONGOING


def check_result_fast(board: Board, last_move: int) -> GameResult:
    """
    Optimized result check.
    In the points variant, only the board occupancy matters for termination.
    """
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
