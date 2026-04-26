
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
        ([1, 3, 4, 5, 6], 2, 12, "Bridge to 6 (1_4)"),
        
        # Multi-line: The "T-Bone"
        # Horizontal: XX _ X (Indices 0, 1, _, 3) -> Bridge to 4 (8 pts)
        # Vertical: X _ (Indices _, 8) -> Append to 2 (2 pts)
        # Intersection at index 2 (row 0, col 2)
        # Indices:
        # Row 0: 0, 1, 2, 3, 4, 5
        # Row 1: 6, 7, 8, 9, 10, 11
        ([0, 1, 3, 8], 2, 20, "T-Bone (8+2)*2"),
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
    # Middle is index 7 (row 1, col 1)
    # Neighbors at 0, 2, 6, 8 (not helpful for lines of 3 through 7)
    # Wait, 3-in-a-row through middle (1,1) would be:
    # Horiz: (1,0) and (1,2) -> indices 6 and 8
    # Vert: (0,1) and (2,1) -> indices 1 and 13
    # Diag1: (0,0) and (2,2) -> indices 0 and 14
    # Diag2: (0,2) and (2,0) -> indices 2 and 12
    
    # Let's use indices:
    # 0 1 2
    # 6 7 8
    # 12 13 14
    existing = [6, 8, 1, 13, 0, 14, 2, 12]
    b = Board()
    for idx in existing:
        b.state[idx] = Player.X
        
    # Playing at 7 should create 4 lines of length 3 (each scores 0)
    # Total = (0 + 0 + 0 + 0) * 4 = 0
    score = calculate_move_score(b, 7, Player.X)
    print(f"Testing 3x3 Death Trap: Expected 0, got {score}")
    assert score == 0

def test_multi_line_with_bridge():
    """Test move that is a bridge in one direction and append in another"""
    # Horiz: X _ X (Indices 0, 2) -> Bridge to 3 (0 pts)
    # Vert: X X _ (Indices 8, 14) -> Append to 3 (0 pts)
    # Result: (0 + 0) * 2 = 0
    b = Board()
    b.state[0] = Player.X
    b.state[2] = Player.X
    b.state[8] = Player.X
    b.state[14] = Player.X
    
    score = calculate_move_score(b, 1, Player.X) # Move at (0,1)
    # Wait, index 1: Horiz 0_2 (L=3, bridge), Vert 1 (L=1).
    # N = 1 (only horiz has L > 1).
    # Score = 0 * 1 = 0.
    assert score == 0
    
    # Try indices 0, 2, 7, 13 for move at 1
    # Row 0: 0, 1, 2
    # Row 1: 6, 7, 8
    # Row 2: 12, 13, 14
    # Move at 1 (0,1):
    # Horiz: 0_2 (L=3, bridge) -> 0 pts
    # Vert: 1-7-13 (L=3, append) -> 0 pts
    # N = 2. Total = (0+0)*2 = 0.
    b = Board()
    for idx in [0, 2, 7, 13]:
        b.state[idx] = Player.X
    score = calculate_move_score(b, 1, Player.X)
    assert score == 0

if __name__ == "__main__":
    test_scoring_examples()
    test_3x3_death_trap()
    test_multi_line_with_bridge()
    print("ALL PYTHON SCORING TESTS PASSED!")
