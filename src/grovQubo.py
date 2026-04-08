"""
qubo_grover_qiskit.py  —  QUBO via Grover's Algorithm using IBM Qiskit
=======================================================================
Solves the SAME QUBO instance defined in qubo_problem.py.

Install:  pip install qiskit qiskit-aer scipy
Run:      python qubo_grover_qiskit.py

Algorithm: Grover's Adaptive Search for QUBO minimisation
----------------------------------------------------------
We iteratively tighten a threshold t, using Grover's algorithm to find
assignments with f(x) < t:

  1. Warm-start: run greedy to get initial best value → threshold t
  2. Build oracle O_t: applies phase −1 to all x where f(x) < t
  3. Run Grover(O_t) with r = 1, 2, …, max_iters iterations; measure
  4. If a sample has f(x) < t: update best, set t ← f(x), go to 2
  5. Stop when no improvement — return best found

Oracle Construction
-------------------
Register layout:
  var_q   [0 … N−1]      N qubits  — the assignment x
  val_q   [N … N+s−1]    s qubits  — unsigned QUBO value accumulator
  anc_q   [N+s]          1 qubit   — scratch ancilla for quadratic terms

QUBO value computation (all integer Q, shifted to non-negative):
  f_shifted(x) = f(x) − f_min_bound  ≥ 0  for all x

  For each term in f:
    • Linear  Q[i,i] x_i:
        controlled_add_const(var_q[i], val_q, Q[i,i])

    • Quadratic  Q[i,j] x_i x_j  (i < j):
        Toffoli(var_q[i], var_q[j], anc_q)      ← anc = x_i AND x_j
        controlled_add_const(anc_q, val_q, Q[i,j])
        Toffoli(var_q[i], var_q[j], anc_q)      ← uncompute anc

Steps inside the oracle (val_q starts and ends at |0⟩):
  compute_qubo_value → mark_lt_threshold → uncompute_qubo_value

Controlled constant addition (same as number_partitioning_grover):
  Decompose const into set bits; for each bit b:
    Carry propagation MSB→b: MCX([ctrl, val[b..i-1]], val[i])
    CX(ctrl, val[b])
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qubo_problem import N, Q, evaluate, BEST_KNOWN, print_problem, print_result


# =============================================================================
# Prepare integer QUBO and shift to non-negative values
# =============================================================================

# Round Q to nearest integer (it was generated as integers already)
Q_INT = np.round(Q).astype(int)

def _compute_shift() -> int:
    """
    Compute an offset so that  f_shifted(x) = f(x) + SHIFT ≥ 0  for all x.
    SHIFT = − (sum of all negative Q terms when they all contribute).
    """
    shift = 0
    for i in range(N):
        if Q_INT[i, i] < 0:
            shift -= Q_INT[i, i]
        for j in range(i + 1, N):
            if Q_INT[i, j] < 0:
                shift -= Q_INT[i, j]   # negative term removed by subtracting
    return shift

SHIFT = _compute_shift()

def evaluate_shifted(x: list[int]) -> int:
    """Compute f_shifted(x) = f(x) + SHIFT  (always non-negative)."""
    val = SHIFT
    for i in range(N):
        if x[i] == 1:
            val += Q_INT[i, i]
            for j in range(i + 1, N):
                if x[j] == 1:
                    val += Q_INT[i, j]
    return int(val)

# Maximum possible shifted QUBO value (all positive terms, all x=1)
MAX_SHIFTED = SHIFT + sum(
    Q_INT[i, j]
    for i in range(N) for j in range(i, N)
    if Q_INT[i, j] > 0
)
S_BITS = max(1, math.ceil(math.log2(MAX_SHIFTED + 2)))   # bits for val_q


# =============================================================================
# Controlled constant addition (identical to number_partitioning_grover)
# =============================================================================

def _ctrl_add_power2(qc: QuantumCircuit, ctrl: int, reg: list[int], bit_b: int) -> None:
    s = len(reg)
    for i in range(s - 1, bit_b, -1):
        qc.mcx([ctrl] + reg[bit_b:i], reg[i])
    qc.cx(ctrl, reg[bit_b])


def _ctrl_sub_power2(qc: QuantumCircuit, ctrl: int, reg: list[int], bit_b: int) -> None:
    s = len(reg)
    qc.cx(ctrl, reg[bit_b])
    for i in range(bit_b + 1, s):
        qc.mcx([ctrl] + reg[bit_b:i], reg[i])


def controlled_add_const(qc: QuantumCircuit, ctrl: int, reg: list[int], const: int) -> None:
    """If ctrl=1, add integer constant `const` to register reg (reg[0]=LSB)."""
    if const == 0:
        return
    s = len(reg)
    for b in range(s):
        if (const >> b) & 1:
            _ctrl_add_power2(qc, ctrl, reg, b)


def controlled_sub_const(qc: QuantumCircuit, ctrl: int, reg: list[int], const: int) -> None:
    """Inverse of controlled_add_const."""
    if const == 0:
        return
    s = len(reg)
    for b in range(s - 1, -1, -1):
        if (const >> b) & 1:
            _ctrl_sub_power2(qc, ctrl, reg, b)


# =============================================================================
# QUBO value computation circuit
# =============================================================================

def compute_qubo_value(
    qc:    QuantumCircuit,
    var_q: list[int],
    val_q: list[int],
    anc:   int,
) -> None:
    """
    Compute val_q ← f_shifted(x)  using x stored in var_q.

    Linear terms  : controlled_add(var_q[i], val_q, Q_INT[i,i] + shift_portion)
    Quadratic terms: Toffoli(i, j, anc); controlled_add(anc, val_q, Q_INT[i,j]);
                     Toffoli(i, j, anc) to uncompute anc

    We add the SHIFT as an unconditional constant via X gates as a simpler
    alternative: initialise val_q to SHIFT directly.
    """
    # Add the shift as an unconditional constant to val_q
    shift_remaining = SHIFT
    for b in range(S_BITS):
        if (shift_remaining >> b) & 1:
            qc.x(val_q[b])

    # Linear terms
    for i in range(N):
        v = Q_INT[i, i]
        if v != 0:
            controlled_add_const(qc, var_q[i], val_q, v)

    # Quadratic terms (use ancilla qubit to compute x_i AND x_j)
    for i in range(N):
        for j in range(i + 1, N):
            v = Q_INT[i, j]
            if v != 0:
                qc.ccx(var_q[i], var_q[j], anc)      # anc = x_i AND x_j
                controlled_add_const(qc, anc, val_q, v)
                qc.ccx(var_q[i], var_q[j], anc)      # uncompute anc


def uncompute_qubo_value(
    qc:    QuantumCircuit,
    var_q: list[int],
    val_q: list[int],
    anc:   int,
) -> None:
    """Reverse of compute_qubo_value (restores val_q to |0⟩)."""
    # Reverse quadratic terms (MSB to LSB pair order, operations reversed)
    for i in range(N - 1, -1, -1):
        for j in range(N - 1, i, -1):
            v = Q_INT[i, j]
            if v != 0:
                qc.ccx(var_q[i], var_q[j], anc)
                controlled_sub_const(qc, anc, val_q, v)
                qc.ccx(var_q[i], var_q[j], anc)

    # Reverse linear terms
    for i in range(N - 1, -1, -1):
        v = Q_INT[i, i]
        if v != 0:
            controlled_sub_const(qc, var_q[i], val_q, v)

    # Uncompute shift (flip same bits back)
    shift_remaining = SHIFT
    for b in range(S_BITS):
        if (shift_remaining >> b) & 1:
            qc.x(val_q[b])


# =============================================================================
# Phase marking: mark val_q < threshold_shifted
# =============================================================================

def mark_lt_threshold(
    qc:        QuantumCircuit,
    val_q:     list[int],
    threshold: int,        # shifted threshold (= original_threshold + SHIFT)
) -> None:
    """
    Apply phase −1 to all basis states of val_q representing values in [0, threshold−1].

    For each target value v in [0, threshold):
      1. Flip bits of val_q that are 0 in v
      2. MCZ on all val_q bits  (H · MCX · H on last qubit)
      3. Unflip
    """
    s = len(val_q)
    if threshold <= 0:
        return

    for val in range(0, min(threshold, 1 << s)):
        bits     = [(val >> b) & 1 for b in range(s)]
        zero_idx = [val_q[b] for b in range(s) if bits[b] == 0]

        for q in zero_idx:
            qc.x(q)
        if s == 1:
            qc.z(val_q[0])
        else:
            qc.h(val_q[-1])
            qc.mcx(val_q[:-1], val_q[-1])
            qc.h(val_q[-1])
        for q in zero_idx:
            qc.x(q)


# =============================================================================
# Full oracle + diffuser
# =============================================================================

def build_oracle(
    qc:        QuantumCircuit,
    var_q:     list[int],
    val_q:     list[int],
    anc:       int,
    threshold: int,        # ORIGINAL (unshifted) threshold
) -> None:
    """
    Oracle: phase −1 for assignments with f(x) < threshold.
    f_shifted(x) < threshold  ⟺  f_shifted(x) ∈ [0, threshold + SHIFT − 1]
    So mark val_q values in [0, threshold + SHIFT).
    """
    threshold_shifted = threshold + SHIFT
    compute_qubo_value(qc, var_q, val_q, anc)
    mark_lt_threshold(qc, val_q, threshold_shifted)
    uncompute_qubo_value(qc, var_q, val_q, anc)


def build_diffuser(qc: QuantumCircuit, var_q: list[int]) -> None:
    """Standard Grover diffuser: 2|s⟩⟨s|−I on var_q."""
    qc.h(var_q)
    qc.x(var_q)
    qc.h(var_q[-1])
    qc.mcx(var_q[:-1], var_q[-1])
    qc.h(var_q[-1])
    qc.x(var_q)
    qc.h(var_q)


# =============================================================================
# Inlined greedy warm-start (no dependency on qubo_greedy.py)
# =============================================================================

def _greedy_warmstart(seed: int = 42) -> tuple[list[int], float]:
    """
    Simple best-flip greedy: at each step flip the bit that reduces f(x) most.
    Used to initialise the Grover threshold without importing qubo_greedy.py.
    """
    import random
    rng = random.Random(seed)
    x   = [rng.randint(0, 1) for _ in range(N)]

    while True:
        gains = []
        for i in range(N):
            x[i] ^= 1
            gains.append(evaluate(x) - evaluate([x[j] ^ (1 if j == i else 0)
                                                  for j in range(N)]))
            x[i] ^= 1  # revert

        # recompute gains cleanly
        cur = evaluate(x)
        best_gain, best_i = 0.0, -1
        for i in range(N):
            x[i] ^= 1
            gain = evaluate(x) - cur
            x[i] ^= 1
            if gain < best_gain:
                best_gain, best_i = gain, i

        if best_i == -1:
            break
        x[best_i] ^= 1

    return x, evaluate(x)


# =============================================================================
# Config + Solver
# =============================================================================

@dataclass
class GroverConfig:
    max_iters: int = 5
    shots:     int = 4096
    seed:      int = 42


class GroverQUBO:
    """
    Grover's Adaptive Search solver for QUBO minimisation.

    Register layout:
      var_q  [0 … N−1]          N qubits  — assignment x
      val_q  [N … N+s−1]        s qubits  — QUBO value register
      anc    [N+s]               1 qubit   — ancilla for quadratic terms
    Total: N + s + 1  qubits
    """

    def __init__(self, config: GroverConfig = GroverConfig()) -> None:
        self.config   = config
        self.backend  = AerSimulator()

        self.n_qubits = N + S_BITS + 1
        self.var_q    = list(range(N))
        self.val_q    = list(range(N, N + S_BITS))
        self.anc      = N + S_BITS

        # Greedy warm-start (inlined — no dependency on qubo_greedy.py)
        init_x, init_val = _greedy_warmstart()
        self.best_x   = init_x
        self.best_val = init_val
        self.search_log: list[dict] = []

    # ── Single Grover run ────────────────────────────────────────────────────

    def _run_grover(self, threshold: int, n_iters: int) -> tuple[list[int], float]:
        """Run a Grover circuit; return (best_x, best_val) found in samples."""
        qc = QuantumCircuit(self.n_qubits, N)

        for q in self.var_q:
            qc.h(q)

        for _ in range(n_iters):
            build_oracle(qc, self.var_q, self.val_q, self.anc, threshold)
            build_diffuser(qc, self.var_q)

        for i, q in enumerate(self.var_q):
            qc.measure(q, i)

        tc     = transpile(qc, self.backend, optimization_level=1)
        counts = self.backend.run(tc, shots=self.config.shots).result().get_counts()

        best_val, best_x = float("inf"), [0] * N
        for bs in counts:
            x   = [int(b) for b in reversed(bs)]
            val = evaluate(x)
            if val < best_val:
                best_val, best_x = val, x

        return best_x, best_val

    # ── Adaptive search ──────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> list[int]:
        """
        Grover Adaptive Search: iteratively lower the threshold until
        no further improvement can be found.
        """
        cfg       = self.config
        threshold = int(self.best_val)   # start just below greedy quality
        t0        = time.perf_counter()

        if verbose:
            print(f"Grover QUBO  |  N={N}  s={S_BITS}  qubits={self.n_qubits}")
            print(f"  SHIFT={SHIFT}  MAX_SHIFTED={MAX_SHIFTED}")
            print(f"  Greedy warm-start: f(x)={self.best_val:.1f}  "
                  f"(optimum={BEST_KNOWN:.1f})")
            print(f"\n{'Threshold':>10}  {'Iters':>6}  {'Found f(x)':>12}  "
                  f"{'Improved':>10}  {'Time(s)':>9}")
            print("─" * 54)

        while threshold > BEST_KNOWN:
            improved = False

            for n_iters in range(1, cfg.max_iters + 1):
                x, val   = self._run_grover(threshold, n_iters)
                elapsed  = time.perf_counter() - t0
                improved_now = val < threshold

                self.search_log.append(dict(
                    threshold=threshold, iters=n_iters,
                    val=val, time=elapsed,
                ))

                if verbose:
                    mark = "  ← better!" if improved_now else ""
                    print(f"{threshold:>10}  {n_iters:>6}  {val:>12.1f}  "
                          f"{'YES' if improved_now else 'no':>10}  "
                          f"{elapsed:>9.3f}{mark}")

                if improved_now:
                    self.best_x   = x
                    self.best_val = val
                    threshold     = int(val)
                    improved      = True
                    break

            if not improved:
                break

        elapsed = time.perf_counter() - t0
        if verbose:
            print("─" * 54)
            print(f"\n  Best f(x) : {self.best_val:.1f}")
            print(f"  Time      : {elapsed:.3f} s")

        return self.best_x


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    SEP = "=" * 60
    print(SEP)
    print("  QUBO  —  Grover's Algorithm + Qiskit Aer")
    print(SEP)
    print_problem()
    print(f"\n  Integer Q (rounded):")
    for row in Q_INT:
        print("    " + "  ".join(f"{v:4d}" for v in row))
    print(f"  SHIFT = {SHIFT}  (added so f_shifted ≥ 0 always)")
    print(f"  MAX_SHIFTED = {MAX_SHIFTED}")
    print(f"  Val register: {S_BITS} qubits")
    print(f"  Total qubits: {N} + {S_BITS} + 1 = {N + S_BITS + 1}")
    print(f"\n{'─'*60}")

    cfg    = GroverConfig(max_iters=5, shots=4096, seed=42)
    solver = GroverQUBO(cfg)

    t0   = time.perf_counter()
    best = solver.optimise(verbose=True)
    t1   = time.perf_counter() - t0

    print_result(best, "Grover Adaptive Search", solver.best_val, t1)

    # ── Iteration sweep ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Iteration count vs. probability of finding optimal")
    print(f"  Oracle marks f(x) < {int(BEST_KNOWN)+1}  (i.e. f(x) = {BEST_KNOWN:.1f})")
    print(f"\n  {'Iters':>6}  {'Best f(x)':>10}  {'Opt shots':>12}  {'Opt %':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}")

    tmp = GroverQUBO(GroverConfig(shots=2048, seed=99))
    for n_iters in range(1, 6):
        qc = QuantumCircuit(tmp.n_qubits, N)
        for q in tmp.var_q:
            qc.h(q)
        for _ in range(n_iters):
            build_oracle(qc, tmp.var_q, tmp.val_q, tmp.anc, int(BEST_KNOWN) + 1)
            build_diffuser(qc, tmp.var_q)
        for i, q in enumerate(tmp.var_q):
            qc.measure(q, i)

        tc     = transpile(qc, tmp.backend, optimization_level=1)
        counts = tmp.backend.run(tc, shots=2048).result().get_counts()

        best_d    = float("inf")
        opt_shots = 0
        for bs, cnt in counts.items():
            x   = [int(b) for b in reversed(bs)]
            val = evaluate(x)
            if val < best_d:
                best_d = val
            if abs(val - BEST_KNOWN) < 1e-6:
                opt_shots += cnt

        pct = 100 * opt_shots / 2048
        print(f"  {n_iters:>6}  {best_d:>10.1f}  "
              f"{opt_shots:>8}/2048  {pct:>7.1f}%")

    print(f"\n  Exact optimum : {BEST_KNOWN:.1f}")
    print(SEP)


if __name__ == "__main__":
    main()