"""
Calculate theoretical deduplication ratio for trie -> DAG conversion.

At depth D:
- Trie nodes = P(36, D) = 36!/(36-D)!  (permutations - order matters)
- Unique board states = C(36, D) × player_assignments
  where player_assignments accounts for which pieces are X vs O

For a game where X moves first:
- At depth D, there are ceil(D/2) X pieces and floor(D/2) O pieces
- Unique states = C(36, ceil(D/2)) × C(36-ceil(D/2), floor(D/2))
                = C(36, D) × C(D, floor(D/2))
"""

from math import comb, perm, factorial

def unique_states_at_depth(d: int, board_size: int = 36) -> int:
    """
    Calculate unique board states at depth d.

    We need to:
    1. Choose d positions from 36 for pieces: C(36, d)
    2. Assign X to ceil(d/2) of them: C(d, ceil(d/2))
    """
    if d == 0:
        return 1

    x_count = (d + 1) // 2  # X moves first, so has one more on odd depths
    o_count = d // 2

    # Choose positions for all pieces, then assign X to some of them
    positions = comb(board_size, d)
    assignments = comb(d, x_count)

    return positions * assignments


def trie_nodes_at_depth(d: int, board_size: int = 36) -> int:
    """Trie nodes at depth d = P(36, d) permutations."""
    if d == 0:
        return 1
    return perm(board_size, d)


def analyze_deduplication():
    print("DEDUPLICATION ANALYSIS: TRIE -> DAG")
    print("=" * 70)
    print()
    print(f"{'Depth':<6} {'Trie Nodes':<20} {'Unique States':<20} {'Dedup Ratio':<15} {'DAG Size'}")
    print("-" * 70)

    cumulative_trie = 0
    cumulative_dag = 0

    for d in range(21):
        trie_nodes = trie_nodes_at_depth(d)
        unique = unique_states_at_depth(d)

        cumulative_trie += trie_nodes
        cumulative_dag += unique

        # Deduplication ratio: how many trie nodes map to one unique state
        ratio = trie_nodes / unique if unique > 0 else 0

        print(f"{d:<6} {trie_nodes:<20,} {unique:<20,} {ratio:<15.1f}")

    print("-" * 70)
    print()
    print("CUMULATIVE TOTALS:")
    print(f"  Trie (no dedup):  {cumulative_trie:,} nodes")
    print(f"  DAG (with dedup): {cumulative_dag:,} nodes")
    print(f"  Reduction factor: {cumulative_trie / cumulative_dag:.1f}x")
    print()

    # Memory estimates
    bytes_per_node = 16
    trie_gb = (cumulative_trie * bytes_per_node) / (1024**3)
    dag_gb = (cumulative_dag * bytes_per_node) / (1024**3)

    print("MEMORY ESTIMATES (at 16 bytes/node):")
    if trie_gb > 1024:
        print(f"  Trie: {trie_gb/1024:.2f} TB")
    else:
        print(f"  Trie: {trie_gb:.2f} GB")

    if dag_gb > 1024:
        print(f"  DAG:  {dag_gb/1024:.2f} TB")
    elif dag_gb > 1:
        print(f"  DAG:  {dag_gb:.2f} GB")
    else:
        print(f"  DAG:  {dag_gb*1024:.2f} MB")


def full_dag_size():
    """Calculate total unique states across all depths."""
    print("\n" + "=" * 70)
    print("CUMULATIVE DAG SIZE BY MAX DEPTH")
    print("=" * 70)
    print()
    print(f"{'Max Depth':<10} {'Cumulative States':<25} {'Memory':<15} {'Feasible?'}")
    print("-" * 70)

    total_states = 0
    bytes_per_node = 16

    for d in range(37):  # 0 to 36 moves
        states = unique_states_at_depth(d)
        total_states += states

        total_bytes = total_states * bytes_per_node
        gb = total_bytes / (1024**3)

        if gb < 1:
            mem_str = f"{gb*1024:.1f} MB"
        elif gb < 1024:
            mem_str = f"{gb:.1f} GB"
        elif gb < 1024**2:
            mem_str = f"{gb/1024:.1f} TB"
        else:
            mem_str = f"{gb/(1024**2):.1f} PB"

        # Feasibility thresholds
        if gb < 100:
            feasible = "✓ Easy"
        elif gb < 1000:
            feasible = "~ Possible"
        elif gb < 10000:
            feasible = "✗ Hard"
        else:
            feasible = "✗ No"

        if d <= 15 or d >= 33:
            print(f"{d:<10} {total_states:<25,} {mem_str:<15} {feasible}")
        elif d == 16:
            print("...")

    print()
    print("CONCLUSION:")
    print("- Depth 10: ~78B states, 1.1 TB  - challenging but possible")
    print("- Depth 12: ~1.4T states, 20 TB  - requires serious hardware")
    print("- Depth 15+: infeasible without game termination pruning")
    print()
    print("NEXT STEP: Need game logic to measure actual termination rates")


if __name__ == "__main__":
    analyze_deduplication()
    full_dag_size()
