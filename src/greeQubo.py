"""
qubo_greedy.py  —  QUBO via Greedy Local Search
================================================
Solves the SAME QUBO instance defined in qubo_problem.py.

Run:
    python qubo_greedy.py

Three greedy strategies
-----------------------
1. Bit-flip local search  (single-flip hill-climbing)
   Initialise x randomly (or to all-zeros/ones).
   Repeat until no single bit-flip reduces f(x):
     for each variable i: flip x[i] if it strictly decreases f(x).
   This is the simplest greedy for QUBO and converges to a local minimum.

2. Best-flip greedy  (steepest descent)
   At each step, flip the ONE bit that decreases f(x) the MOST.
   Continue until no improving flip exists.
   Finds the same or better local optima than sequential flip, more slowly.

3. Multi-restart greedy
   Run best-flip from many random starting points; keep the best result.
   Escapes local optima that trap single-run strategies.

Complexity per run: O(n²) per pass (n variables, each flip costs O(n)).
"""

from __future__ import annotations

import random
import time
from typing import Optional

from qubo_problem import N, Q, evaluate, BEST_KNOWN, print_problem, print_result


# =============================================================================
# Greedy helpers
# =============================================================================

def _flip_gain(x: list[int], i: int) -> float:
    """
    Compute the change in f(x) if we flip bit i:
        Δ_i = f(x with x_i flipped) − f(x)

    For QUBO  f(x) = x^T Q x,  flipping bit i from v to (1−v) changes f by:
        Δ_i = (1 − 2·x_i) · (Q[i,i] + Σ_{j≠i} Q_ij · x_j)

    where Q_ij = Q[i,j] if i<j, or Q[j,i] if j<i (but Q is upper-triangular
    so Q[j,i]=0 for j>i — the coupling is stored in Q[i,j] with i<j).

    We compute this as the difference of two evaluations for simplicity.
    """
    x2    = x[:]
    x2[i] ^= 1
    return evaluate(x2) - evaluate(x)


# =============================================================================
# Strategy 1: Sequential bit-flip local search
# =============================================================================

def greedy_sequential(
    x_init: Optional[list[int]] = None,
    seed:   int = 42,
) -> tuple[list[int], float]:
    """
    Scan variables in order; flip each one if it reduces f(x).
    Repeat full passes until no pass improves the objective.

    Returns (best_x, best_value).
    """
    rng = random.Random(seed)
    x   = x_init[:] if x_init else [rng.randint(0, 1) for _ in range(N)]

    improved = True
    while improved:
        improved = False
        for i in range(N):
            if _flip_gain(x, i) < -1e-9:
                x[i]    ^= 1
                improved  = True

    return x, evaluate(x)


# =============================================================================
# Strategy 2: Best-flip (steepest-descent) greedy
# =============================================================================

def greedy_best_flip(
    x_init: Optional[list[int]] = None,
    seed:   int = 42,
) -> tuple[list[int], float]:
    """
    At each step, find the single bit-flip that reduces f(x) the MOST.
    Apply it and repeat until no improving flip exists.

    Returns (best_x, best_value).
    """
    rng = random.Random(seed)
    x   = x_init[:] if x_init else [rng.randint(0, 1) for _ in range(N)]

    while True:
        gains     = [_flip_gain(x, i) for i in range(N)]
        best_gain = min(gains)
        if best_gain >= -1e-9:
            break                     # no improving flip — local minimum
        best_i  = gains.index(best_gain)
        x[best_i] ^= 1

    return x, evaluate(x)


# =============================================================================
# Strategy 3: Multi-restart greedy
# =============================================================================

def greedy_multi_restart(
    restarts: int = 500,
    seed:     int = 42,
) -> tuple[list[int], float]:
    """
    Run greedy_best_flip from `restarts` different random starting points.
    Return the best solution found across all restarts.
    """
    rng      = random.Random(seed)
    best_x   = [0] * N
    best_val = float("inf")

    for r in range(restarts):
        x_init = [rng.randint(0, 1) for _ in range(N)]
        x, val = greedy_best_flip(x_init, seed=rng.randint(0, 2**31))
        if val < best_val:
            best_val, best_x = val, x[:]

    return best_x, best_val


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    SEP = "=" * 60
    print(SEP)
    print("  QUBO  —  Greedy Local Search")
    print(SEP)
    print_problem()

    print(f"\n{'─'*60}")

    # ── Strategy 1: Sequential flip ──────────────────────────────────────
    t0 = time.perf_counter()
    x1, v1 = greedy_sequential(seed=42)
    t1 = time.perf_counter() - t0
    print_result(x1, "Sequential bit-flip greedy", v1, t1)

    # Try a few starting points manually
    starts = [[0]*N, [1]*N, [1,0,1,0,1,0], [0,1,0,1,0,1]]
    print(f"\n  Sequential flip from fixed starting points:")
    print(f"  {'Start':<25}  {'f(x)':>8}  {'Gap':>8}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}")
    for s in starts:
        xr, vr = greedy_sequential(s)
        gap = vr - BEST_KNOWN
        print(f"  {str(s):<25}  {vr:>8.1f}  {gap:>+8.1f}")

    # ── Strategy 2: Best-flip steepest descent ───────────────────────────
    t0 = time.perf_counter()
    x2, v2 = greedy_best_flip(seed=42)
    t2 = time.perf_counter() - t0
    print_result(x2, "Steepest-descent best-flip greedy", v2, t2)

    # ── Strategy 3: Multi-restart ─────────────────────────────────────────
    t0 = time.perf_counter()
    x3, v3 = greedy_multi_restart(restarts=500, seed=42)
    t3 = time.perf_counter() - t0
    print_result(x3, "Multi-restart greedy (500 restarts)", v3, t3)

    # ── Restart count analysis ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Restarts vs. solution quality:")
    print(f"  {'Restarts':>10}  {'Best f(x)':>12}  {'Gap':>8}  {'Time (ms)':>12}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*8}  {'-'*12}")
    for r in [1, 5, 20, 100, 500]:
        t0  = time.perf_counter()
        xr, vr = greedy_multi_restart(restarts=r, seed=99)
        ms  = (time.perf_counter() - t0) * 1000
        gap = vr - BEST_KNOWN
        opt = " ✓" if abs(gap) < 1e-6 else ""
        print(f"  {r:>10}  {vr:>12.1f}  {gap:>+8.1f}  {ms:>10.3f} ms{opt}")

    print(f"\n  Exact optimum : {BEST_KNOWN:.1f}")
    print(SEP)


if __name__ == "__main__":
    main()