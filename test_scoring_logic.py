
from game import Board, Player, calculate_move_score

def test_scoring_examples():
    """Verify all examples from FEATURE_SPEC.md"""
    
    def setup_board(indices, player=Player.X):
        b = Board()
        for idx in indices:
            b.state[idx] = player
        return b

    # (existing_indices, move_to_make, expected_score, description)
    test_cases = [
        ([], 0, 1, "Lone Tile"),
        ([1], 0, 2, "Append to 2"),
        ([1, 2], 0, 0, "Append to 3"),
        ([1, 3], 2, 0, "Bridge to 3 (1_1)"),
        ([1, 2, 3], 0, 4, "Append to 4"),
        ([1, 3, 4], 2, 8, "Bridge to 4 (1_2)"),
        ([1, 2, 4, 5], 3, 10, "Bridge to 5 (2_2)"),
        ([0, 2, 3, 4, 5], 1, 12, "Bridge to 6 (1_4)"),
        
        # Multi-line: The "T-Bone"
        # Horizontal: XX _ X (Indices 0, 1, _, 3) -> Bridge to 4 (8 pts)
        # Vertical: X _ (Indices _, 8) -> Append to 2 (2 pts)
        # Intersection at index 2 (row 0, col 2)
        ([0, 1, 3, 8], 2, 20, "T-Bone (8+2)*2"),
        
        # Productive Multiplier Rule: 4-in-a-row + 3-in-a-row
        # Horiz: XX _ X (Indices 0, 1, _, 3) -> Bridge to 4 (8 pts)
        # Vert: XX _ (Indices 8, 14) -> Append to 3 (0 pts)
        # Multiplier should be 1 (only the horizontal line scored points)
        # Expected: (8 + 0) * 1 = 8
        # Wait, if Horizontal is Bridge-to-4 (8 pts) and Vertical is Append-to-3 (0 pts).
        # Old logic: (8 + 0) * 2 = 16
        # New logic: (8 + 0) * 1 = 8
        ([0, 1, 3, 8, 14], 2, 8, "Productive Multiplier (8+0)*1"),
    ]
    
    for existing, move, expected, desc in test_cases:
        b = setup_board(existing)
        score = calculate_move_score(b, move, Player.X)
        print(f"Testing {desc}: Expected {expected}, got {score}")
        assert score == expected, f"FAILED {desc}: expected {expected}, got {score}"

def test_3x3_death_trap():
    """Test playing in the center of a 3x3 square (4 lines of 3-in-a-row)"""
    # X . X
    # . _ .
    # X . X
    # Neighbors to form lines of 3 through (1,1) [index 7]:
    # Horiz: (1,0) and (1,2) -> indices 6 and 8
    # Vert: (0,1) and (2,1) -> indices 1 and 13
    # Diag1: (0,0) and (2,2) -> indices 0 and 14
    # Diag2: (0,2) and (2,0) -> indices 2 and 12
    
    existing = [6, 8, 1, 13, 0, 14, 2, 12]
    b = Board()
    for idx in existing:
        b.state[idx] = Player.X
        
    # Playing at 7 should create 4 lines of length 3 (each scores 0)
    # Total = (0 + 0 + 0 + 0) * 4 = 0
    score = calculate_move_score(b, 7, Player.X)
    print(f"Testing 3x3 Death Trap: Expected 0, got {score}")
    assert score == 0

if __name__ == "__main__":
    test_scoring_examples()
    test_3x3_death_trap()
    print("ALL PYTHON SCORING TESTS PASSED!")
