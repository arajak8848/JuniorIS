"""
qubo_problem.py  —  Shared QUBO Problem Definition
====================================================
Imported by all four solver files:
    qubo_greedy.py  |  qubo_genetic.py  |  qubo_qaoa_qiskit.py  |  qubo_grover_qiskit.py

The QUBO Problem (Quadratic Unconstrained Binary Optimisation)
--------------------------------------------------------------
Minimise  f(x) = x^T Q x = Σ_i Q[i,i] x_i  +  Σ_{i<j} Q[i,j] x_i x_j

where  x ∈ {0,1}^n  and  Q ∈ ℝ^{n×n}  is an upper-triangular matrix.

  - Diagonal entries  Q[i,i]  are the linear (bias) terms.
  - Off-diagonal entries  Q[i,j]  (i < j) are the quadratic (coupling) terms.
  - QUBO is NP-hard in general and unifies many combinatorial problems
    (Max-Cut, Number Partitioning, MAX-SAT, k-colouring, TSP, …).

This module defines:
  N          — number of binary variables  (6)
  Q          — the upper-triangular QUBO matrix  (fixed seed)
  evaluate() — compute f(x) for a given assignment
  brute_force_optimum() — exact solution by exhaustive search
  print_result()        — pretty-print a solution
  BEST_KNOWN            — the exact optimum (populated at import time)
"""

from __future__ import annotations
import numpy as np

# ── Problem parameters ────────────────────────────────────────────────────────

N    = 6          # number of binary variables  (keep ≤ 8 for Grover tractability)
SEED = 42         # fixed seed — every solver file sees IDENTICAL Q

rng  = np.random.default_rng(SEED)

# Upper-triangular Q with integer entries in [-7, 7]
# (mixed signs make the landscape interesting — neither all-zeros nor
#  all-ones is trivially optimal)
_raw = rng.integers(-7, 8, size=(N, N))
Q    = np.triu(_raw).astype(float)

# ── Core function ─────────────────────────────────────────────────────────────

def evaluate(x: list[int] | np.ndarray) -> float:
    """
    Compute the QUBO objective  f(x) = x^T Q x.

    Parameters
    ----------
    x : length-N binary vector  (0 or 1)

    Returns
    -------
    float  —  the QUBO value (lower = better, since we MINIMISE)
    """
    xv = np.asarray(x, dtype=float)
    return float(xv @ Q @ xv)


# ── Exact solver ──────────────────────────────────────────────────────────────

def brute_force_optimum() -> tuple[list[int], float]:
    """
    Exhaustive search over all 2^N assignments.
    Returns (best_x, best_value).
    """
    best_val = float("inf")
    best_x   = [0] * N
    for mask in range(1 << N):
        x   = [(mask >> i) & 1 for i in range(N)]
        val = evaluate(x)
        if val < best_val:
            best_val, best_x = val, x
    return best_x, best_val


# Pre-compute at import time so every file can reference it
BEST_X, BEST_KNOWN = brute_force_optimum()


# ── Display helpers ───────────────────────────────────────────────────────────

def print_problem() -> None:
    """Print the QUBO matrix and key statistics."""
    print(f"\n  QUBO instance  (N={N}, seed={SEED})")
    print(f"  Q matrix (upper-triangular):")
    for row in Q:
        print("    " + "  ".join(f"{v:5.0f}" for v in row))

    all_vals = [evaluate([(m >> i) & 1 for i in range(N)]) for m in range(1 << N)]
    print(f"\n  Objective range : [{min(all_vals):.1f}, {max(all_vals):.1f}]")
    print(f"  Exact optimum   : {BEST_KNOWN:.1f}  at  x = {BEST_X}")


def print_result(
    x:         list[int],
    algorithm: str,
    val:       float | None = None,
    elapsed:   float | None = None,
) -> None:
    """Pretty-print a single solution."""
    if val is None:
        val = evaluate(x)
    gap     = val - BEST_KNOWN
    opt_str = "✓ OPTIMAL" if abs(gap) < 1e-6 else f"gap = {gap:+.1f}"
    t_str   = f"  ({elapsed:.3f}s)" if elapsed is not None else ""
    print(f"\n  [{algorithm}]{t_str}")
    print(f"  x       = {x}")
    print(f"  f(x)    = {val:.1f}")
    print(f"  Optimum = {BEST_KNOWN:.1f}  —  {opt_str}")