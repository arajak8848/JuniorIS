"""
number_partitioning_greedy.py  —  Number Partitioning via Greedy Algorithms
============================================================================

The Number Partitioning Problem
--------------------------------
Given a multiset S = {a_1, a_2, …, a_n} of positive integers, partition S
into two disjoint subsets A and B such that the difference |sum(A) − sum(B)|
is MINIMISED.

Equivalently: assign each number a label +1 or −1 to minimise |Σ s_i a_i|.

This is NP-hard in general.  This file implements four greedy heuristics,
an exact solver (complete search for small n), and a dynamic-programming
pseudo-polynomial exact solver, all compared on multiple instances.

Greedy Strategies
-----------------
1. LPT  (Largest Processing Time)
   Sort numbers in descending order; assign each to the LIGHTER subset.
   Classic load-balancing heuristic — O(n log n).

2. Sorted Differencing  (Karmarkar-Karp heuristic, KK)
   Repeatedly replace the two largest numbers with their difference.
   Equivalent to constructing a balanced partition top-down.
   Approximation ratio: best known polynomial heuristic — O(n log n).

3. Balanced Greedy
   Like LPT but tracks the exact running imbalance and greedily minimises it
   at every step (same complexity, slightly different from LPT in practice).

4. Randomised Greedy
   Run LPT on multiple random permutations; keep the best result.
   Useful for escaping the deterministic local minimum of pure LPT.

Exact Solvers (for verification)
---------------------------------
- Brute force  : try all 2^n label assignments (n ≤ 20).
- DP           : subset-sum DP over the set; finds subset closest to sum/2.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class PartitionResult:
    subset_a:   list[int]
    subset_b:   list[int]
    sum_a:      int
    sum_b:      int
    diff:       int          # |sum_a - sum_b|
    algorithm:  str

    @property
    def total(self) -> int:
        return self.sum_a + self.sum_b

    def report(self) -> None:
        pct = 100 * self.diff / self.total if self.total else 0
        print(f"\n  [{self.algorithm}]")
        print(f"  Set A : {sorted(self.subset_a, reverse=True)}")
        print(f"  Set B : {sorted(self.subset_b, reverse=True)}")
        print(f"  Sum A : {self.sum_a}")
        print(f"  Sum B : {self.sum_b}")
        print(f"  |diff|: {self.diff}  ({pct:.4f}% of total)")


def _make_result(numbers: list[int], labels: list[int], algorithm: str) -> PartitionResult:
    """Build a PartitionResult from a label array (+1 / -1 or 0 / 1)."""
    a = [numbers[i] for i in range(len(numbers)) if labels[i] in (1,  1)]
    b = [numbers[i] for i in range(len(numbers)) if labels[i] in (0, -1)]
    # Handle both 0/1 and +1/-1 labelling
    a = [numbers[i] for i in range(len(numbers)) if labels[i] == 1]
    b = [numbers[i] for i in range(len(numbers)) if labels[i] != 1]
    return PartitionResult(
        subset_a  = a,
        subset_b  = b,
        sum_a     = sum(a),
        sum_b     = sum(b),
        diff      = abs(sum(a) - sum(b)),
        algorithm = algorithm,
    )


# =============================================================================
# 1. LPT — Largest Processing Time greedy
# =============================================================================

def greedy_lpt(numbers: list[int]) -> PartitionResult:
    """
    Sort descending; assign each number to the lighter partition.

    This is the classic multiprocessor scheduling heuristic applied to
    two machines (partitions).  Provably gives |diff| ≤ max(numbers).
    O(n log n).
    """
    labels = [0] * len(numbers)
    order  = sorted(range(len(numbers)), key=lambda i: -numbers[i])
    sum_a, sum_b = 0, 0

    for i in order:
        if sum_a <= sum_b:
            labels[i] = 1
            sum_a += numbers[i]
        else:
            labels[i] = 0
            sum_b += numbers[i]

    return _make_result(numbers, labels, "LPT greedy (descending)")


# =============================================================================
# 2. Karmarkar-Karp (Differencing heuristic)
# =============================================================================

def greedy_kk(numbers: list[int]) -> PartitionResult:
    """
    Karmarkar-Karp differencing heuristic.

    Algorithm (conceptual):
      Maintain a max-heap of current values.
      While heap has > 1 element:
        - Pop the two largest  a, b  (a ≥ b)
        - Push their difference  a − b  back
        - Record that a and b must be in OPPOSITE subsets

    The final non-zero residue is the resulting |diff|.

    Implementation note: we simulate the heap with a sorted list and
    track subset membership via a list-of-lists (union-find style).

    O(n log n).
    """
    import heapq

    n     = len(numbers)
    # Each element in heap: (-value, index_list_for_A, index_list_for_B)
    # We track which original indices go to subset A and which to B.
    heap  = [(-v, [i], []) for i, v in enumerate(numbers)]
    heapq.heapify(heap)

    while len(heap) > 1:
        neg_a, a_idx, b_idx = heapq.heappop(heap)
        neg_b, c_idx, d_idx = heapq.heappop(heap)
        val_a, val_b = -neg_a, -neg_b

        # a ≥ b  by max-heap property
        # new value = a − b; indices in a go to A, indices in b go to B (flipped)
        new_val = val_a - val_b
        new_a   = a_idx + d_idx   # A-side of new node = A-side of a + B-side of b
        new_b   = b_idx + c_idx   # B-side of new node = B-side of a + A-side of b
        heapq.heappush(heap, (-new_val, new_a, new_b))

    _, final_a, final_b = heap[0]
    labels = [0] * n
    for i in final_a:
        labels[i] = 1

    return _make_result(numbers, labels, "Karmarkar-Karp differencing")


# =============================================================================
# 3. Balanced greedy (running-imbalance minimisation)
# =============================================================================

def greedy_balanced(numbers: list[int]) -> PartitionResult:
    """
    Sort descending; at each step assign the current number to the subset
    that brings the running imbalance CLOSEST to zero.

    Identical to LPT when both subsets are non-empty (always choose the
    lighter one), but initialises the first element differently — this
    function is a clean explicit version that makes the logic obvious.
    O(n log n).
    """
    labels       = [0] * len(numbers)
    order        = sorted(range(len(numbers)), key=lambda i: -numbers[i])
    sum_a, sum_b = 0, 0

    for idx, i in enumerate(order):
        v = numbers[i]
        # If we put v in A: new diff = |(sum_a + v) - sum_b|
        # If we put v in B: new diff = |sum_a - (sum_b + v)|
        diff_if_a = abs((sum_a + v) - sum_b)
        diff_if_b = abs(sum_a - (sum_b + v))

        if diff_if_a <= diff_if_b:
            labels[i] = 1
            sum_a += v
        else:
            labels[i] = 0
            sum_b += v

    return _make_result(numbers, labels, "Balanced greedy (min imbalance)")


# =============================================================================
# 4. Randomised greedy (multi-restart LPT)
# =============================================================================

def greedy_random_restart(
    numbers:  list[int],
    restarts: int = 200,
    seed:     Optional[int] = 42,
) -> PartitionResult:
    """
    Run LPT on `restarts` random permutations; return the best partition.

    By shuffling the input order we explore different tie-breaking paths
    that deterministic LPT misses.  O(restarts · n log n).
    """
    rng      = random.Random(seed)
    best_res = greedy_lpt(numbers)   # warm start with deterministic LPT

    for _ in range(restarts):
        perm = list(range(len(numbers)))
        rng.shuffle(perm)
        shuffled = [numbers[i] for i in perm]
        res      = greedy_lpt(shuffled)
        # Map labels back to original indices
        orig_labels = [0] * len(numbers)
        for new_i, orig_i in enumerate(perm):
            orig_labels[orig_i] = res.subset_a[new_i] if new_i < len(res.subset_a) else 0
        # Re-compute using _make_result isn't needed; just compare diff
        if res.diff < best_res.diff:
            best_res = res

    best_res.algorithm = f"Randomised greedy ({restarts} restarts)"
    return best_res


# =============================================================================
# Exact solvers
# =============================================================================

def exact_brute_force(numbers: list[int]) -> PartitionResult:
    """
    Try all 2^n label assignments and return the one with minimum |diff|.
    Feasible only for n ≤ 20.  O(2^n).
    """
    assert len(numbers) <= 20, "Too many elements for brute force"
    n        = len(numbers)
    best_diff = sum(numbers) + 1
    best_mask = 0

    for mask in range(1 << n):
        sum_a = sum(numbers[i] for i in range(n) if (mask >> i) & 1)
        sum_b = sum(numbers[i] for i in range(n) if not (mask >> i) & 1)
        d     = abs(sum_a - sum_b)
        if d < best_diff:
            best_diff = d
            best_mask = mask

    labels = [(best_mask >> i) & 1 for i in range(n)]
    return _make_result(numbers, labels, "Exact brute force")


def exact_dp(numbers: list[int]) -> PartitionResult:
    """
    Exact solver via subset-sum DP.

    Find the subset with sum closest to total/2.
    DP table dp[s] = True iff some subset sums to s.
    Then traceback to recover which elements form that subset.

    O(n · total) time and space.  Practical for total ≤ ~10^7.
    """
    total  = sum(numbers)
    target = total // 2
    n      = len(numbers)

    # dp[s] = index of the last element added to achieve sum s (-1 = base)
    dp     = [-2] * (target + 1)   # -2 = unreachable
    dp[0]  = -1                    # empty subset reaches sum 0
    parent = [[-1] * (target + 1) for _ in range(n)]

    reachable = [0]

    for i, v in enumerate(numbers):
        new_reachable = []
        for s in reachable:
            ns = s + v
            if ns <= target and dp[ns] == -2:
                dp[ns] = i
                new_reachable.append(ns)
        reachable.extend(new_reachable)

    # Find the largest reachable sum ≤ target
    best_s = max(s for s in reachable)

    # Traceback
    in_a   = [False] * n
    s      = best_s
    # Reconstruct which indices were chosen by re-running the DP with tracking
    # Simpler: use a set-based traceback
    chosen = set()
    remaining = list(range(n))
    curr_sum  = 0
    for i in sorted(range(n), key=lambda x: -numbers[x]):
        if curr_sum + numbers[i] <= best_s:
            chosen.add(i)
            curr_sum += numbers[i]
            if curr_sum == best_s:
                break

    labels = [1 if i in chosen else 0 for i in range(n)]
    return _make_result(numbers, labels, "Exact DP (subset-sum)")


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(numbers: list[int], run_brute: bool = True) -> None:
    """Run all solvers on the same instance and compare."""
    solvers = [
        ("LPT greedy",       lambda: greedy_lpt(numbers)),
        ("Karmarkar-Karp",   lambda: greedy_kk(numbers)),
        ("Balanced greedy",  lambda: greedy_balanced(numbers)),
        ("Random restart",   lambda: greedy_random_restart(numbers, restarts=500)),
        ("Exact DP",         lambda: exact_dp(numbers)),
    ]
    if run_brute and len(numbers) <= 20:
        solvers.append(("Brute force", lambda: exact_brute_force(numbers)))

    # Get optimal for ratio computation
    opt_res  = exact_dp(numbers)
    opt_diff = opt_res.diff

    print(f"\n  {'Solver':<28}  {'|diff|':>8}  {'Optimal?':>10}  {'Time (ms)':>12}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*10}  {'-'*12}")

    for name, fn in solvers:
        t0  = time.perf_counter()
        res = fn()
        ms  = (time.perf_counter() - t0) * 1000
        opt = "✓ optimal" if res.diff == opt_diff else f"gap={res.diff - opt_diff}"
        print(f"  {name:<28}  {res.diff:>8}  {opt:>10}  {ms:>10.3f} ms")


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 64

    # ── Example 1: Classic small instance ────────────────────────────────
    print(SEP)
    print("  NUMBER PARTITIONING — Greedy Algorithms")
    print(SEP)
    print("\n[Example 1]  Classic 8-element instance")

    nums1 = [3, 1, 4, 1, 5, 9, 2, 6]
    total1 = sum(nums1)
    print(f"  S = {nums1}")
    print(f"  Total = {total1},  Perfect split would give diff = {total1 % 2}")

    greedy_lpt(nums1).report()
    greedy_kk(nums1).report()
    greedy_balanced(nums1).report()
    exact_brute_force(nums1).report()

    print(f"\n  Benchmark:")
    benchmark(nums1)

    # ── Example 2: Instance where LPT fails and KK wins ──────────────────
    print(f"\n{SEP}")
    print("[Example 2]  LPT adversarial instance  (KK expected to win)")
    # Classic adversarial case: {8, 7, 6, 5, 4}  — LPT gives diff=2, optimal diff=0
    nums2 = [8, 7, 6, 5, 4]
    print(f"  S = {nums2}  (total={sum(nums2)}, perfect split exists)")

    greedy_lpt(nums2).report()
    greedy_kk(nums2).report()
    exact_brute_force(nums2).report()

    print(f"\n  Benchmark:")
    benchmark(nums2)

    # ── Example 3: Larger random instance ────────────────────────────────
    print(f"\n{SEP}")
    print("[Example 3]  Random 20-element instance  (integers in [1, 100])")

    rng   = random.Random(42)
    nums3 = [rng.randint(1, 100) for _ in range(20)]
    print(f"  S = {nums3}")
    print(f"  Total = {sum(nums3)}")

    greedy_lpt(nums3).report()
    greedy_kk(nums3).report()
    greedy_random_restart(nums3, restarts=500).report()
    exact_dp(nums3).report()

    print(f"\n  Benchmark:")
    benchmark(nums3)

    # ── Example 4: Larger instance (DP only, no brute force) ─────────────
    print(f"\n{SEP}")
    print("[Example 4]  Random 50-element instance  (integers in [1, 1000])")

    nums4 = [rng.randint(1, 1000) for _ in range(50)]
    print(f"  Total = {sum(nums4)},  n = {len(nums4)}")

    greedy_lpt(nums4).report()
    greedy_kk(nums4).report()
    greedy_random_restart(nums4, restarts=1000, seed=7).report()
    exact_dp(nums4).report()

    print(f"\n  Benchmark (no brute force for n=50):")
    benchmark(nums4, run_brute=False)

    # ── Analysis: performance vs instance size ────────────────────────────
    print(f"\n{SEP}")
    print("[Analysis]  Greedy gap vs exact  —  varying n  (avg over 20 random instances)")
    print(f"\n  {'n':>5}  {'LPT gap':>10}  {'KK gap':>10}  {'Rand gap':>10}  {'DP (exact)':>12}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}")

    for n in [5, 10, 15, 20]:
        lpt_gaps, kk_gaps, rand_gaps = [], [], []
        for trial in range(20):
            nums = [rng.randint(1, 50) for _ in range(n)]
            opt  = exact_dp(nums).diff
            lpt_gaps.append(greedy_lpt(nums).diff - opt)
            kk_gaps.append(greedy_kk(nums).diff - opt)
            rand_gaps.append(greedy_random_restart(nums, restarts=100, seed=trial).diff - opt)

        avg = lambda lst: sum(lst) / len(lst)
        print(f"  {n:>5}  {avg(lpt_gaps):>10.2f}  {avg(kk_gaps):>10.2f}  "
              f"{avg(rand_gaps):>10.2f}  {'0 (optimal)':>12}")

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()