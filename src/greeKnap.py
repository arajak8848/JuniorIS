"""
knapsack_greedy.py  —  0/1 and Fractional Knapsack via Greedy Algorithm
========================================================================

The Knapsack Problem: given n items each with a weight and value, and a
knapsack with capacity W, select items to maximise total value without
exceeding W.

Two variants are solved here:

  Fractional Knapsack  — items can be taken in fractions.
                         Greedy is OPTIMAL: sort by value/weight ratio,
                         take greedily, split the last item if needed.

  0/1 Knapsack         — items must be taken whole or not at all.
                         Greedy is a HEURISTIC (not guaranteed optimal),
                         but fast and often near-optimal in practice.
                         A brute-force exact solver is included for
                         comparison on small instances.

Run:
    python knapsack_greedy.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations


# =============================================================================
# Data model
# =============================================================================

@dataclass
class Item:
    name:   str
    weight: float
    value:  float

    @property
    def ratio(self) -> float:
        """Value-to-weight ratio — the greedy key."""
        return self.value / self.weight if self.weight > 0 else float("inf")

    def __repr__(self) -> str:
        return (f"Item({self.name!r}, w={self.weight:.2f}, "
                f"v={self.value:.2f}, r={self.ratio:.3f})")


@dataclass
class KnapsackResult:
    """Outcome of a knapsack solver."""
    selected:    list[tuple[Item, float]]  # (item, fraction_taken)  fraction=1.0 for 0/1
    total_value: float
    total_weight: float
    algorithm:   str

    def report(self, capacity: float) -> None:
        print(f"\n  [{self.algorithm}]")
        print(f"  {'Item':<18} {'Weight':>8} {'Fraction':>10} {'Value':>10}")
        print(f"  {'-'*18}  {'-'*8}  {'-'*10}  {'-'*10}")
        for item, frac in self.selected:
            taken_w = item.weight * frac
            taken_v = item.value  * frac
            frac_str = f"{frac:.4f}" if frac < 1.0 else "  whole"
            print(f"  {item.name:<18} {taken_w:>8.2f} {frac_str:>10} {taken_v:>10.2f}")
        print(f"  {'':18}  {'--------':>8}  {'':>10}  {'--------':>10}")
        print(f"  {'TOTAL':<18} {self.total_weight:>8.2f} {'':>10} {self.total_value:>10.2f}")
        print(f"  Capacity used: {self.total_weight:.2f} / {capacity:.2f}  "
              f"({100*self.total_weight/capacity:.1f}%)")


# =============================================================================
# Greedy — Fractional Knapsack  (provably optimal)
# =============================================================================

def greedy_fractional(items: list[Item], capacity: float) -> KnapsackResult:
    """
    Optimal greedy solution for the FRACTIONAL knapsack.

    Strategy
    --------
    Sort items by value/weight ratio (descending).
    Take each item in full until the knapsack is full;
    take a fraction of the next item to fill remaining space.

    Complexity: O(n log n)  dominated by sorting.
    """
    sorted_items = sorted(items, key=lambda it: it.ratio, reverse=True)

    selected:     list[tuple[Item, float]] = []
    remaining     = capacity
    total_value   = 0.0

    for item in sorted_items:
        if remaining <= 0:
            break
        frac = min(1.0, remaining / item.weight)
        selected.append((item, frac))
        remaining    -= item.weight * frac
        total_value  += item.value  * frac

    return KnapsackResult(
        selected     = selected,
        total_value  = total_value,
        total_weight = capacity - remaining,
        algorithm    = "Greedy — Fractional (optimal)",
    )


# =============================================================================
# Greedy — 0/1 Knapsack  (heuristic, three strategies)
# =============================================================================

def _greedy_01_by(
    items: list[Item],
    capacity: float,
    key,
    label: str,
) -> KnapsackResult:
    """Generic 0/1 greedy: sort by `key`, take items that still fit."""
    sorted_items = sorted(items, key=key, reverse=True)
    selected:   list[tuple[Item, float]] = []
    remaining   = capacity
    total_value = 0.0

    for item in sorted_items:
        if item.weight <= remaining:
            selected.append((item, 1.0))
            remaining    -= item.weight
            total_value  += item.value

    return KnapsackResult(
        selected     = selected,
        total_value  = total_value,
        total_weight = capacity - remaining,
        algorithm    = label,
    )


def greedy_01_by_ratio(items: list[Item], capacity: float) -> KnapsackResult:
    """0/1 greedy sorted by value/weight ratio (most common heuristic)."""
    return _greedy_01_by(
        items, capacity,
        key=lambda it: it.ratio,
        label="Greedy 0/1 — by ratio (heuristic)",
    )


def greedy_01_by_value(items: list[Item], capacity: float) -> KnapsackResult:
    """0/1 greedy sorted by pure value."""
    return _greedy_01_by(
        items, capacity,
        key=lambda it: it.value,
        label="Greedy 0/1 — by value (heuristic)",
    )


def greedy_01_by_weight(items: list[Item], capacity: float) -> KnapsackResult:
    """0/1 greedy sorted by weight ascending (lightest-first)."""
    return _greedy_01_by(
        items, capacity,
        key=lambda it: -it.weight,   # ascending = reverse=True on negated
        label="Greedy 0/1 — by weight asc (heuristic)",
    )


def greedy_01_best_of_three(items: list[Item], capacity: float) -> KnapsackResult:
    """
    Run all three 0/1 greedy strategies and return the best.
    This is a simple but effective meta-heuristic.
    """
    candidates = [
        greedy_01_by_ratio(items, capacity),
        greedy_01_by_value(items, capacity),
        greedy_01_by_weight(items, capacity),
    ]
    best = max(candidates, key=lambda r: r.total_value)
    best.algorithm = f"Greedy 0/1 — best-of-3  (winner: {best.algorithm.split('—')[1].strip()})"
    return best


# =============================================================================
# Exact 0/1 solver  (brute force, for verification on small instances)
# =============================================================================

def exact_01(items: list[Item], capacity: float) -> KnapsackResult:
    """
    Exact 0/1 knapsack by exhaustive enumeration.
    Feasible only for n <= ~20.

    Complexity: O(2^n)
    """
    assert len(items) <= 20, "Too many items for brute force"
    n           = len(items)
    best_value  = 0.0
    best_subset: list[Item] = []

    for r in range(n + 1):
        for subset in combinations(items, r):
            w = sum(it.weight for it in subset)
            v = sum(it.value  for it in subset)
            if w <= capacity and v > best_value:
                best_value  = v
                best_subset = list(subset)

    return KnapsackResult(
        selected     = [(it, 1.0) for it in best_subset],
        total_value  = best_value,
        total_weight = sum(it.weight for it in best_subset),
        algorithm    = "Exact 0/1 — brute force (optimal)",
    )


# =============================================================================
# Dynamic-programming exact solver  (faster exact, for larger instances)
# =============================================================================

def dp_01(items: list[Item], capacity: float, precision: int = 2) -> KnapsackResult:
    """
    Exact 0/1 knapsack via dynamic programming.
    Weights are rounded to `precision` decimal places and scaled to integers.

    Complexity: O(n * W_int)  where W_int = capacity * 10^precision
    """
    scale    = 10 ** precision
    W        = int(round(capacity * scale))
    weights  = [int(round(it.weight * scale)) for it in items]
    values   = [it.value for it in items]
    n        = len(items)

    # dp[j] = best value achievable with capacity j
    dp   = [0.0] * (W + 1)
    keep = [[False] * (W + 1) for _ in range(n)]   # traceback table

    for i in range(n):
        wi, vi = weights[i], values[i]
        for j in range(W, wi - 1, -1):
            candidate = dp[j - wi] + vi
            if candidate > dp[j]:
                dp[j]      = candidate
                keep[i][j] = True

    # Traceback
    selected: list[tuple[Item, float]] = []
    j = W
    for i in range(n - 1, -1, -1):
        if keep[i][j]:
            selected.append((items[i], 1.0))
            j -= weights[i]
    selected.reverse()

    total_w = sum(items[i].weight for items[i], _ in selected)
    return KnapsackResult(
        selected     = selected,
        total_value  = dp[W],
        total_weight = total_w,
        algorithm    = "DP 0/1 — exact (optimal)",
    )


# =============================================================================
# Benchmark helper
# =============================================================================

def benchmark(
    items: list[Item],
    capacity: float,
    run_exact: bool = True,
) -> None:
    """Run all solvers on the same instance and print a comparison table."""

    solvers = [
        ("Greedy Fractional",  lambda: greedy_fractional(items, capacity)),
        ("Greedy 0/1 ratio",   lambda: greedy_01_by_ratio(items, capacity)),
        ("Greedy 0/1 value",   lambda: greedy_01_by_value(items, capacity)),
        ("Greedy 0/1 weight",  lambda: greedy_01_by_weight(items, capacity)),
        ("Greedy 0/1 best-3",  lambda: greedy_01_best_of_three(items, capacity)),
        ("DP exact",           lambda: dp_01(items, capacity)),
    ]
    if run_exact and len(items) <= 20:
        solvers.append(("Brute force",  lambda: exact_01(items, capacity)))

    print(f"\n  {'Solver':<30}  {'Value':>10}  {'Weight':>10}  {'Time (ms)':>12}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*12}")

    dp_val = None
    for name, fn in solvers:
        t0   = time.perf_counter()
        res  = fn()
        ms   = (time.perf_counter() - t0) * 1000
        flag = ""
        if "DP" in name or "Brute" in name:
            dp_val = res.total_value
        elif dp_val is not None:
            gap  = (dp_val - res.total_value) / dp_val * 100 if dp_val > 0 else 0
            flag = f"  gap={gap:.1f}%" if gap > 0.001 else "  ✓ optimal"
        print(f"  {name:<30}  {res.total_value:>10.2f}  "
              f"{res.total_weight:>10.2f}  {ms:>10.3f} ms{flag}")


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 64

    # ── Example 1: Classic textbook instance ─────────────────────────────
    print(SEP)
    print("  KNAPSACK PROBLEM  —  Greedy Algorithm")
    print(SEP)
    print("\n[Example 1]  Classic 6-item instance  (capacity = 10 kg)")

    items1 = [
        Item("Gold bar",     weight=5.0,  value=80.0),
        Item("Silver coins", weight=3.0,  value=40.0),
        Item("Laptop",       weight=4.0,  value=60.0),
        Item("Camera",       weight=2.0,  value=30.0),
        Item("Watch",        weight=1.0,  value=20.0),
        Item("Necklace",     weight=1.5,  value=25.0),
    ]
    capacity1 = 10.0

    print("\n  Items (sorted by value/weight ratio):")
    for it in sorted(items1, key=lambda x: x.ratio, reverse=True):
        print(f"    {it.name:<18}  w={it.weight:.1f}  v={it.value:.1f}  "
              f"ratio={it.ratio:.3f}")

    r_frac = greedy_fractional(items1, capacity1)
    r_frac.report(capacity1)

    r_01 = greedy_01_best_of_three(items1, capacity1)
    r_01.report(capacity1)

    r_dp  = dp_01(items1, capacity1)
    r_dp.report(capacity1)

    print(f"\n  Benchmark (capacity={capacity1}):")
    benchmark(items1, capacity1)

    # ── Example 2: Larger random instance ────────────────────────────────
    print(f"\n{SEP}")
    print("[Example 2]  Random 15-item instance  (capacity = 50 kg)")

    import random
    rng = random.Random(42)
    items2 = [
        Item(f"item_{i:02d}",
             weight=round(rng.uniform(1.0, 15.0), 2),
             value =round(rng.uniform(5.0, 100.0), 2))
        for i in range(15)
    ]
    capacity2 = 50.0

    print("\n  Items:")
    print(f"  {'Name':<12} {'Weight':>8} {'Value':>8} {'Ratio':>8}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    for it in sorted(items2, key=lambda x: x.ratio, reverse=True):
        print(f"  {it.name:<12} {it.weight:>8.2f} {it.value:>8.2f} {it.ratio:>8.3f}")

    r2_frac = greedy_fractional(items2, capacity2)
    r2_frac.report(capacity2)

    r2_01 = greedy_01_best_of_three(items2, capacity2)
    r2_01.report(capacity2)

    print(f"\n  Benchmark (capacity={capacity2}):")
    benchmark(items2, capacity2)

    # ── Example 3: Edge case — all items too heavy except one ─────────────
    print(f"\n{SEP}")
    print("[Example 3]  Edge case — tight capacity")

    items3 = [
        Item("Huge rock",   weight=99.0, value=1000.0),
        Item("Small gem",   weight=2.0,  value=50.0),
        Item("Tiny crystal",weight=1.0,  value=30.0),
        Item("Dust",        weight=0.5,  value=10.0),
    ]
    capacity3 = 3.0

    greedy_fractional(items3, capacity3).report(capacity3)
    greedy_01_best_of_three(items3, capacity3).report(capacity3)
    dp_01(items3, capacity3).report(capacity3)

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()