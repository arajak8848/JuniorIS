"""
number_partitioning_grover_qiskit.py
— Number Partitioning via Grover's Algorithm using IBM Qiskit
=============================================================

Install dependencies (once):
    pip install qiskit qiskit-aer scipy

Then run:
    python number_partitioning_grover_qiskit.py

The Number Partitioning Problem
--------------------------------
Given S = {a_1, …, a_n} of positive integers, assign each element a label
x_i ∈ {0, 1}  (0 → set B,  1 → set A) to minimise  |Σ a_i x_i − Σ a_i (1−x_i)|
= |2·sum_A − total|  where  sum_A = Σ_{x_i=1} a_i.

Grover's Adaptive Search
------------------------
We use Grover's Algorithm as a threshold search:

  1. Initialise best_diff from a classical greedy run.
  2. Set threshold = best_diff  (we seek diff STRICTLY LESS THAN this).
  3. Build an oracle O_t that applies phase −1 to all assignments x where
       |2·sum_A(x) − total| < threshold
     i.e.,  sum_A(x)  lies in the interval  [lo_t, hi_t]  where:
       lo_t = ceil((total − threshold + 1) / 2)
       hi_t = floor((total + threshold − 1) / 2)
  4. Run Grover(O_t) with r = 1, 2, … iterations; measure.
  5. If any sample has diff < threshold: update best, tighten threshold, go to 3.
  6. Stop when no improvement is found at the current threshold.

Quantum Oracle Construction
----------------------------
Register layout:
  var_q   [0 … n−1]       n qubits  — the partition labels x_i
  sum_q   [n … n+s−1]     s qubits  — holds sum_A  (s = ⌈log₂(total+1)⌉)

Steps inside the oracle (sum_q starts and ends at |0⟩):
  1. compute_sum   : for each i, if x_i=1, add a_i to sum_q
  2. mark_range    : apply phase −1 to sum_q values in [lo_t, hi_t]
  3. uncompute_sum : reverse step 1 to restore sum_q to |0⟩

Controlled constant addition  (core arithmetic primitive)
----------------------------------------------------------
To add constant c to register reg (s bits, reg[0]=LSB), controlled on ctrl:

  Decompose c = Σ_b 2^b · bit_b(c)
  For each set bit b (LSB first), add 2^b to reg controlled on ctrl:

    # Carry propagation for adding 2^b:
    for i in range(s-1, b, -1):
        MCX([ctrl, reg[b], reg[b+1], …, reg[i-1]], reg[i])   # propagate carry
    CX(ctrl, reg[b])                                          # flip bit b

  Uncompute (controlled subtract c) applies the same gates in REVERSE:
    CX(ctrl, reg[b])
    for i in range(b+1, s):
        MCX([ctrl, reg[b], …, reg[i-1]], reg[i])

Phase marking of an interval [lo, hi]
--------------------------------------
For each integer value v in [lo, hi]:
  1. Flip bits of sum_q that are 0 in v's binary representation
  2. Apply MCZ on all s sum bits  (phase-flips |1…1⟩, hence |v⟩)
  3. Unflip

This is the same technique used in the Grover MAX-SAT oracle.

Qubit count  (why instances must stay small)
---------------------------------------------
  total qubits = n + s = n + ⌈log₂(Σ a_i + 1)⌉

  For n=6, a_i ≤ 15  →  total ≤ 90  →  s = 7  →  13 qubits  (manageable)
  For n=8, a_i ≤ 20  →  total ≤ 160  →  s = 8  →  16 qubits  (feasible)
  Beyond n≈10, circuit depth explodes; use a real quantum device.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# =============================================================================
# Result dataclass
# =============================================================================

@dataclass
class PartitionResult:
    subset_a:    list[int]
    subset_b:    list[int]
    sum_a:       int
    sum_b:       int
    diff:        int
    algorithm:   str

    @property
    def total(self) -> int:
        return self.sum_a + self.sum_b

    def report(self, indent: str = "  ") -> None:
        pct = 100 * self.diff / self.total if self.total else 0.0
        print(f"\n{indent}[{self.algorithm}]")
        print(f"{indent}  Set A : {sorted(self.subset_a, reverse=True)}")
        print(f"{indent}  Set B : {sorted(self.subset_b, reverse=True)}")
        print(f"{indent}  Sum A : {self.sum_a}")
        print(f"{indent}  Sum B : {self.sum_b}")
        print(f"{indent}  |diff|: {self.diff}  ({pct:.4f}% of total)")


def _make_result(
    numbers:   list[int],
    labels:    list[int],   # 1 → set A,  0 → set B
    algorithm: str,
) -> PartitionResult:
    a = [numbers[i] for i in range(len(numbers)) if labels[i] == 1]
    b = [numbers[i] for i in range(len(numbers)) if labels[i] == 0]
    return PartitionResult(
        subset_a  = a,
        subset_b  = b,
        sum_a     = sum(a),
        sum_b     = sum(b),
        diff      = abs(sum(a) - sum(b)),
        algorithm = algorithm,
    )


# =============================================================================
# Quantum arithmetic: controlled constant addition / subtraction
# =============================================================================

def _ctrl_add_power2(
    qc:    QuantumCircuit,
    ctrl:  int,
    reg:   list[int],
    bit_b: int,
) -> None:
    """
    Controlled addition of 2^bit_b to register reg (reg[0] = LSB).

    Carry-propagation circuit (MSB first to avoid overwriting carry bits):
      for i in range(s-1, bit_b, -1):
          MCX([ctrl, reg[bit_b], …, reg[i-1]], reg[i])
      CX(ctrl, reg[bit_b])
    """
    s = len(reg)
    for i in range(s - 1, bit_b, -1):
        controls = [ctrl] + reg[bit_b:i]
        qc.mcx(controls, reg[i])
    qc.cx(ctrl, reg[bit_b])


def _ctrl_sub_power2(
    qc:    QuantumCircuit,
    ctrl:  int,
    reg:   list[int],
    bit_b: int,
) -> None:
    """
    Controlled subtraction of 2^bit_b from register reg.
    Reverse of _ctrl_add_power2.
    """
    s = len(reg)
    qc.cx(ctrl, reg[bit_b])
    for i in range(bit_b + 1, s):
        controls = [ctrl] + reg[bit_b:i]
        qc.mcx(controls, reg[i])


def controlled_add_const(
    qc:    QuantumCircuit,
    ctrl:  int,
    reg:   list[int],
    const: int,
) -> None:
    """
    If ctrl = 1, add integer constant `const` to the unsigned integer
    encoded in `reg` (reg[0] = LSB).

    Decompose const into set bits and add each power-of-2 contribution.
    Process bits from LSB to MSB so carry propagation is independent.
    """
    s = len(reg)
    for b in range(s):
        if (const >> b) & 1:
            _ctrl_add_power2(qc, ctrl, reg, b)


def controlled_sub_const(
    qc:    QuantumCircuit,
    ctrl:  int,
    reg:   list[int],
    const: int,
) -> None:
    """
    Inverse of controlled_add_const  (uncomputes the addition).
    Process bits from MSB to LSB (reverse order of addition).
    """
    s = len(reg)
    for b in range(s - 1, -1, -1):
        if (const >> b) & 1:
            _ctrl_sub_power2(qc, ctrl, reg, b)


# =============================================================================
# Oracle: compute weighted sum, mark interval, uncompute
# =============================================================================

def compute_sum(
    qc:      QuantumCircuit,
    var_q:   list[int],
    sum_q:   list[int],
    numbers: list[int],
) -> None:
    """
    Compute  sum_q ← Σ_{i : x_i=1} numbers[i]
    by adding numbers[i] to sum_q controlled on var_q[i].
    """
    for i, a in enumerate(numbers):
        controlled_add_const(qc, var_q[i], sum_q, a)


def uncompute_sum(
    qc:      QuantumCircuit,
    var_q:   list[int],
    sum_q:   list[int],
    numbers: list[int],
) -> None:
    """
    Uncompute  sum_q → |0⟩  (reverse of compute_sum).
    Subtract in reverse order to restore exactly.
    """
    for i in range(len(numbers) - 1, -1, -1):
        controlled_sub_const(qc, var_q[i], sum_q, numbers[i])


def mark_range(
    qc:    QuantumCircuit,
    sum_q: list[int],
    lo:    int,
    hi:    int,
) -> None:
    """
    Apply phase −1 to all basis states of sum_q representing integers in [lo, hi].

    For each target value v ∈ [lo, hi]:
      1. X-flip qubits that correspond to 0-bits in v  (maps |v⟩ → |1…1⟩)
      2. MCZ on all sum_q bits  (implemented as H · MCX · H on the last qubit)
      3. Unflip

    Because different values v map to ORTHOGONAL basis states, the MCZ
    applications compose independently over a superposition.
    """
    s       = len(sum_q)
    max_val = 1 << s

    lo = max(0, lo)
    hi = min(max_val - 1, hi)
    if lo > hi:
        return   # empty interval — nothing to mark

    for val in range(lo, hi + 1):
        # Bits of val (LSB first)
        bits     = [(val >> b) & 1 for b in range(s)]
        zero_idx = [sum_q[b] for b in range(s) if bits[b] == 0]

        for q in zero_idx:
            qc.x(q)

        # MCZ via H–MCX–H on last qubit
        if s == 1:
            qc.z(sum_q[0])
        else:
            qc.h(sum_q[-1])
            qc.mcx(sum_q[:-1], sum_q[-1])
            qc.h(sum_q[-1])

        for q in zero_idx:
            qc.x(q)


def build_oracle(
    qc:        QuantumCircuit,
    var_q:     list[int],
    sum_q:     list[int],
    numbers:   list[int],
    threshold: int,
    total:     int,
) -> None:
    """
    Full Grover oracle for Number Partitioning.

    Marks assignments x where  |2·sum_A(x) − total| < threshold,
    i.e., sum_A ∈ [lo_t, hi_t]:
      lo_t = ceil((total − threshold + 1) / 2)
      hi_t = floor((total + threshold − 1) / 2)

    Steps (sum_q starts and ends at |0⟩):
      compute_sum → mark_range → uncompute_sum
    """
    # Compute the target interval for sum_A
    lo_t = math.ceil((total - threshold + 1) / 2)
    hi_t = math.floor((total + threshold - 1) / 2)

    compute_sum(qc, var_q, sum_q, numbers)
    mark_range(qc, sum_q, lo_t, hi_t)
    uncompute_sum(qc, var_q, sum_q, numbers)


def build_diffuser(qc: QuantumCircuit, var_q: list[int]) -> None:
    """
    Grover diffuser (inversion about the mean) on var_q.
    Implements  2|s⟩⟨s| − I  where  |s⟩ = H^⊗n|0⟩:
      H^n → X^n → MCZ → X^n → H^n
    """
    qc.h(var_q)
    qc.x(var_q)
    qc.h(var_q[-1])
    qc.mcx(var_q[:-1], var_q[-1])
    qc.h(var_q[-1])
    qc.x(var_q)
    qc.h(var_q)


# =============================================================================
# Config + Solver
# =============================================================================

@dataclass
class GroverConfig:
    max_iters:  int  = 6       # max Grover iterations per threshold level
    shots:      int  = 4096    # measurement shots per circuit run
    seed:       int  = 42


class GroverPartition:
    """
    Grover's Adaptive Search solver for the Number Partitioning problem.

    Usage
    -----
        solver = GroverPartition(numbers, GroverConfig())
        result = solver.optimise()
        result.report()

    Register layout
    ---------------
        var_q  [0 … n−1]     n qubits  —  partition labels
        sum_q  [n … n+s−1]   s qubits  —  weighted sum accumulator
    Total: n + s qubits  (s = ⌈log₂(total + 1)⌉)
    """

    def __init__(
        self,
        numbers: list[int],
        config:  GroverConfig = GroverConfig(),
    ) -> None:
        self.numbers = numbers
        self.config  = config
        self.backend = AerSimulator()

        n             = len(numbers)
        total         = sum(numbers)
        s             = max(1, math.ceil(math.log2(total + 2)))  # +2 for safety

        self.n        = n
        self.total    = total
        self.s        = s
        self.n_qubits = n + s

        self.var_q  = list(range(n))
        self.sum_q  = list(range(n, n + s))

        # Warm-start: greedy LPT gives the initial upper bound for diff
        self.best_result  = _greedy_lpt(numbers)
        self.search_log:  list[dict] = []

    # ── Single Grover circuit ────────────────────────────────────────────────

    def _run_grover(
        self, threshold: int, n_iters: int
    ) -> tuple[list[int], int]:
        """
        Build and run a Grover circuit marking assignments with diff < threshold.

        Returns (best_labels, best_diff) across all measurement outcomes.
        """
        qc = QuantumCircuit(self.n_qubits, self.n)

        # Uniform superposition
        for q in self.var_q:
            qc.h(q)

        # Grover iterations
        for _ in range(n_iters):
            build_oracle(
                qc, self.var_q, self.sum_q,
                self.numbers, threshold, self.total,
            )
            build_diffuser(qc, self.var_q)

        # Measure only the variable qubits
        for i, q in enumerate(self.var_q):
            qc.measure(q, i)

        transpiled = transpile(qc, self.backend, optimization_level=1)
        counts     = (self.backend
                      .run(transpiled, shots=self.config.shots)
                      .result()
                      .get_counts())

        # Decode all samples; keep the best
        best_diff, best_labels = self.total + 1, [0] * self.n
        for bs in counts:
            labels = [int(b) for b in reversed(bs)]   # qubit 0 = rightmost
            sa     = sum(self.numbers[i] for i in range(self.n) if labels[i] == 1)
            diff   = abs(2 * sa - self.total)
            if diff < best_diff:
                best_diff, best_labels = diff, labels

        return best_labels, best_diff

    # ── Adaptive search ──────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> PartitionResult:
        """
        Run Grover's Adaptive Search and return the best partition found.

        Algorithm
        ---------
        threshold ← initial greedy diff
        while threshold > 0:
          for n_iters = 1, 2, …, max_iters:
            run Grover with n_iters iterations, oracle marks diff < threshold
            if any sample has diff < threshold:
              update best, set threshold ← new diff, break inner loop
          else:
            break outer loop (no improvement possible at this level)
        """
        cfg       = self.config
        t0        = time.perf_counter()
        threshold = self.best_result.diff   # start at greedy quality

        if verbose:
            print(f"Grover Partition  |  n={self.n}  total={self.total}  "
                  f"s={self.s}  qubits={self.n_qubits}")
            print(f"Greedy warm-start : diff = {threshold}")
            print(f"Optimal possible  : diff = {self.total % 2}  "
                  f"(total is {'even' if self.total % 2 == 0 else 'odd'})")
            print(f"\n{'Threshold':>10}  {'Iters':>6}  {'Found diff':>11}  "
                  f"{'Improved':>10}  {'Time(s)':>9}")
            print("-" * 54)

        while threshold > 0:
            improved = False

            for n_iters in range(1, cfg.max_iters + 1):
                labels, diff = self._run_grover(threshold, n_iters)
                elapsed      = time.perf_counter() - t0

                entry = dict(threshold=threshold, iters=n_iters,
                             diff=diff, time=elapsed)
                self.search_log.append(entry)

                if verbose:
                    mark = "  ← improved!" if diff < threshold else ""
                    print(f"{threshold:>10}  {n_iters:>6}  {diff:>11}  "
                          f"{'YES' if diff < threshold else 'no':>10}  "
                          f"{elapsed:>9.3f}{mark}")

                if diff < threshold:
                    self.best_result = _make_result(
                        self.numbers, labels,
                        f"Grover GAS (p={n_iters}, threshold={threshold})"
                    )
                    threshold = diff
                    improved  = True
                    break

            if not improved:
                break  # cannot beat current threshold

            if threshold == 0:
                if verbose:
                    print("  ✓ Perfect partition found (diff = 0)!")
                break

        elapsed = time.perf_counter() - t0
        self.best_result.algorithm = (
            f"Grover Adaptive Search  (final diff={self.best_result.diff})"
        )

        if verbose:
            print("-" * 54)
            print(f"\n  Best diff : {self.best_result.diff}")
            print(f"  Time      : {elapsed:.3f} s")

        return self.best_result


# =============================================================================
# Classical baselines (self-contained, no imports from greedy file)
# =============================================================================

def _greedy_lpt(numbers: list[int]) -> PartitionResult:
    """LPT greedy: sort descending, assign each to lighter subset."""
    labels       = [0] * len(numbers)
    order        = sorted(range(len(numbers)), key=lambda i: -numbers[i])
    sum_a, sum_b = 0, 0
    for i in order:
        if sum_a <= sum_b:
            labels[i] = 1;  sum_a += numbers[i]
        else:
            labels[i] = 0;  sum_b += numbers[i]
    return _make_result(numbers, labels, "LPT greedy")


def _greedy_kk(numbers: list[int]) -> PartitionResult:
    """Karmarkar-Karp differencing heuristic."""
    import heapq
    n    = len(numbers)
    heap = [(-v, [i], []) for i, v in enumerate(numbers)]
    heapq.heapify(heap)
    while len(heap) > 1:
        neg_a, a_idx, b_idx = heapq.heappop(heap)
        neg_b, c_idx, d_idx = heapq.heappop(heap)
        heapq.heappush(heap, (
            -(-neg_a - (-neg_b)),
            a_idx + d_idx,
            b_idx + c_idx,
        ))
    _, final_a, _ = heap[0]
    labels = [0] * n
    for i in final_a:
        labels[i] = 1
    return _make_result(numbers, labels, "Karmarkar-Karp")


def exact_brute_force(numbers: list[int]) -> PartitionResult:
    """Exact by exhaustive search (n ≤ 20)."""
    assert len(numbers) <= 20
    n = len(numbers)
    best_diff, best_mask = sum(numbers) + 1, 0
    for mask in range(1 << n):
        sa = sum(numbers[i] for i in range(n) if (mask >> i) & 1)
        sb = sum(numbers[i] for i in range(n) if not (mask >> i) & 1)
        d  = abs(sa - sb)
        if d < best_diff:
            best_diff, best_mask = d, mask
    labels = [(best_mask >> i) & 1 for i in range(n)]
    return _make_result(numbers, labels, "Exact brute force")


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 64

    print(SEP)
    print("  NUMBER PARTITIONING — Grover's Algorithm + Qiskit Aer")
    print(SEP)

    # ── Example 1: Perfect partition exists (diff = 0) ───────────────────
    print("\n[Example 1]  Perfect partition  (n=6, diff=0 achievable)")
    nums1  = [8, 7, 6, 5, 4, 2]     # total=32, A={8,7,1}? Let's see
    total1 = sum(nums1)
    print(f"  S = {nums1}  |  total = {total1}  |  "
          f"min possible diff = {total1 % 2}")
    print(f"  Qubits needed: {len(nums1)} var + "
          f"{max(1, math.ceil(math.log2(total1+2)))} sum = "
          f"{len(nums1) + max(1, math.ceil(math.log2(total1+2)))}")

    lpt1 = _greedy_lpt(nums1);  lpt1.report()
    kk1  = _greedy_kk(nums1);   kk1.report()
    bf1  = exact_brute_force(nums1); bf1.report()

    print(f"\n  Running Grover's Adaptive Search ...")
    cfg1  = GroverConfig(max_iters=6, shots=4096, seed=42)
    sol1  = GroverPartition(nums1, cfg1)
    res1  = sol1.optimise(verbose=True)
    res1.report()

    print(f"\n  Summary for Example 1:")
    print(f"  {'Solver':<35}  {'|diff|':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    print(f"  {'LPT greedy':<35}  {lpt1.diff:>8}")
    print(f"  {'Karmarkar-Karp':<35}  {kk1.diff:>8}")
    print(f"  {'Exact brute force':<35}  {bf1.diff:>8}")
    print(f"  {'Grover Adaptive Search':<35}  {res1.diff:>8}")

    # ── Example 2: Adversarial for greedy  ───────────────────────────────
    print(f"\n{SEP}")
    print("[Example 2]  Greedy-adversarial instance  (n=5)")
    nums2  = [6, 5, 4, 3, 2]    # total=20, perfect split at {6,4}={10} vs {5,3,2}={10}
    total2 = sum(nums2)
    print(f"  S = {nums2}  |  total = {total2}  |  min diff = {total2 % 2}")

    lpt2 = _greedy_lpt(nums2);  lpt2.report()
    kk2  = _greedy_kk(nums2);   kk2.report()
    bf2  = exact_brute_force(nums2); bf2.report()

    print(f"\n  Running Grover's Adaptive Search ...")
    cfg2 = GroverConfig(max_iters=6, shots=4096, seed=0)
    sol2 = GroverPartition(nums2, cfg2)
    res2 = sol2.optimise(verbose=True)
    res2.report()

    print(f"\n  Summary for Example 2:")
    print(f"  {'Solver':<35}  {'|diff|':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    print(f"  {'LPT greedy':<35}  {lpt2.diff:>8}")
    print(f"  {'Karmarkar-Karp':<35}  {kk2.diff:>8}")
    print(f"  {'Exact brute force':<35}  {bf2.diff:>8}")
    print(f"  {'Grover Adaptive Search':<35}  {res2.diff:>8}")

    # ── Example 3: Random instance (n=6) ─────────────────────────────────
    print(f"\n{SEP}")
    print("[Example 3]  Random instance  (n=6, values in [1, 15])")

    rng   = random.Random(7)
    nums3 = [rng.randint(1, 15) for _ in range(6)]
    total3 = sum(nums3)
    print(f"  S = {nums3}  |  total = {total3}  |  min diff = {total3 % 2}")

    lpt3 = _greedy_lpt(nums3);  lpt3.report()
    kk3  = _greedy_kk(nums3);   kk3.report()
    bf3  = exact_brute_force(nums3); bf3.report()

    print(f"\n  Running Grover's Adaptive Search ...")
    cfg3 = GroverConfig(max_iters=8, shots=8192, seed=42)
    sol3 = GroverPartition(nums3, cfg3)
    res3 = sol3.optimise(verbose=True)
    res3.report()

    print(f"\n  Summary for Example 3:")
    print(f"  {'Solver':<35}  {'|diff|':>8}")
    print(f"  {'-'*35}  {'-'*8}")
    print(f"  {'LPT greedy':<35}  {lpt3.diff:>8}")
    print(f"  {'Karmarkar-Karp':<35}  {kk3.diff:>8}")
    print(f"  {'Exact brute force':<35}  {bf3.diff:>8}")
    print(f"  {'Grover Adaptive Search':<35}  {res3.diff:>8}")

    # ── Iteration sweep: probability of finding optimal ───────────────────
    print(f"\n{SEP}")
    print("[Analysis]  Grover iteration count vs. probability of finding optimal")
    print(f"  Instance: {nums1}  (Example 1)")
    print(f"  Oracle marks diff = 0  (sum_A = {sum(nums1)//2})")

    target_diff = bf1.diff   # = 0

    print(f"\n  {'Iters':>6}  {'Best diff':>10}  {'Opt shots':>12}  {'Opt %':>8}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*8}")

    for n_iters in range(1, 7):
        solver_tmp = GroverPartition(nums1, GroverConfig(shots=2048, seed=99))

        # Build and run the circuit directly
        qc = QuantumCircuit(solver_tmp.n_qubits, solver_tmp.n)
        for q in solver_tmp.var_q:
            qc.h(q)
        for _ in range(n_iters):
            build_oracle(
                qc, solver_tmp.var_q, solver_tmp.sum_q,
                nums1, target_diff + 1, sum(nums1),
            )
            build_diffuser(qc, solver_tmp.var_q)
        for i, q in enumerate(solver_tmp.var_q):
            qc.measure(q, i)

        tc     = transpile(qc, solver_tmp.backend, optimization_level=1)
        counts = solver_tmp.backend.run(tc, shots=2048).result().get_counts()

        opt_shots = 0
        best_d    = sum(nums1) + 1
        for bs, cnt in counts.items():
            labels = [int(b) for b in reversed(bs)]
            sa     = sum(nums1[i] for i in range(solver_tmp.n) if labels[i] == 1)
            d      = abs(2 * sa - sum(nums1))
            if d < best_d:
                best_d = d
            if d <= target_diff:
                opt_shots += cnt

        pct = 100 * opt_shots / 2048
        print(f"  {n_iters:>6}  {best_d:>10}  {opt_shots:>8}/2048  {pct:>7.1f}%")

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()