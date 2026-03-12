"""
max_sat_grover_qiskit.py  —  MAX-SAT via Grover's Algorithm using IBM Qiskit
=============================================================================

Install dependencies (once):
    pip install qiskit qiskit-aer scipy

Then run:
    python max_sat_grover_qiskit.py

The MAX-SAT Problem
-------------------
Given a Boolean formula in CNF with n variables and m clauses, find the
truth assignment that MAXIMISES the number of satisfied clauses.

Why Grover's for MAX-SAT?
-------------------------
Grover's algorithm provides a quadratic speedup for UNSTRUCTURED SEARCH:
given a black-box oracle that marks "good" states, Grover finds one in
O(√(N/M)) oracle calls vs O(N/M) classically (N = 2^n, M = #solutions).

For MAX-SAT we use GROVER'S ADAPTIVE SEARCH (GAS):
  1. Start with threshold  t  = initial greedy value + 1
  2. Build an oracle that marks assignments satisfying ≥ t clauses
  3. Run Grover's with r = 1, 2, … iterations and measure
  4. If a marked assignment is found, update  t ← fitness + 1  and repeat
  5. Stop when no improvement is found — best found so far is the answer

QUBO / Ising encoding is NOT needed here; the oracle is built directly
from the CNF structure using standard quantum gates.

Quantum Oracle Construction
----------------------------
For a threshold t, the oracle applies phase −1 to assignments x with
  (# satisfied clauses in x) ≥ t

Register layout  (total: n + m + s qubits)
  var_qreg    [0 … n−1]         search space  (the n Boolean variables)
  clause_qreg [n … n+m−1]       one ancilla per clause
  sum_qreg    [n+m … n+m+s−1]   s = ⌈log₂(m+1)⌉ bits for the clause count

Oracle steps
  1. compute_clause_ancillas   : clause_qreg[j] ← 1 iff clause j satisfied
  2. quantum_count             : sum_qreg ← number of 1s in clause_qreg
  3. mark_geq_threshold        : phase −1 if sum_qreg ≥ t  (MCZ per value)
  4. uncompute_quantum_count   : restore sum_qreg to |0⟩
  5. uncompute_clause_ancillas : restore clause_qreg to |0⟩

Clause ancilla trick
  Clause (l₁ ∨ l₂ ∨ l₃) is UNSATISFIED iff all literals are false.
  To detect this, flip positive-literal variables, run MCX to set ancilla
  if all are 1 (= clause unsatisfied), then flip ancilla  →  ancilla=1
  iff clause IS satisfied.  Uncompute the variable flips.

Quantum counter
  For each clause ancilla c_j, apply a CONTROLLED INCREMENT on sum_qreg.
  Uncompute with CONTROLLED DECREMENT in reverse order.

  Controlled increment of an s-bit register reg  (reg[0] = LSB):
    for i in s-1 downto 1: MCX([ctrl, reg[0..i-1]], reg[i])
    CX(ctrl, reg[0])
  Controlled decrement (reverse):
    CX(ctrl, reg[0])
    for i in 1 to s-1:     MCX([ctrl, reg[0..i-1]], reg[i])

Phase marking (no flag ancilla needed)
  For each integer value v ∈ [t, 2^s):
    • flip sum bits that are 0 in v's binary expansion
    • apply MCZ on all sum bits  (flips phase of |1…1⟩, hence of |v⟩)
    • unflip
  Since different values of sum_qreg are orthogonal, these operations
  compose correctly without interference.

Grover Diffuser
  Standard inversion-about-average on var_qreg:
    H^n  X^n  MCZ  X^n  H^n
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# =============================================================================
# CNF Formula
# =============================================================================

@dataclass
class CNF:
    """Propositional formula in Conjunctive Normal Form (DIMACS convention)."""

    n_vars:  int
    clauses: list[list[int]]   # positive int → x_i TRUE; negative → x_i FALSE

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def evaluate(self, assignment: list[int]) -> int:
        """Return the number of clauses satisfied by assignment (0-indexed)."""
        count = 0
        for clause in self.clauses:
            for lit in clause:
                var = abs(lit) - 1
                if (lit > 0 and assignment[var] == 1) or \
                   (lit < 0 and assignment[var] == 0):
                    count += 1
                    break
        return count

    def is_sat(self, assignment: list[int]) -> bool:
        return self.evaluate(assignment) == self.n_clauses

    @staticmethod
    def random_3sat(n: int, m: int, seed: Optional[int] = None) -> "CNF":
        """Random 3-SAT with n variables and m clauses."""
        rng = random.Random(seed)
        clauses = []
        for _ in range(m):
            vars_ = rng.sample(range(1, n + 1), min(3, n))
            clauses.append([v if rng.random() < 0.5 else -v for v in vars_])
        return CNF(n_vars=n, clauses=clauses)

    def __repr__(self) -> str:
        return f"CNF(vars={self.n_vars}, clauses={self.n_clauses})"


# =============================================================================
# Oracle building blocks
# =============================================================================

def controlled_increment(qc: QuantumCircuit, ctrl: int, reg: list[int]) -> None:
    """
    If qubit ctrl = 1, add 1 to the unsigned integer encoded in reg.
    reg[0] = LSB,  reg[-1] = MSB.

    Circuit (MSB → LSB order to propagate carry correctly):
      for i = s-1 downto 1:  MCX([ctrl, reg[0..i-1]], reg[i])
      CX(ctrl, reg[0])
    """
    s = len(reg)
    for i in range(s - 1, 0, -1):
        qc.mcx([ctrl] + reg[:i], reg[i])
    qc.cx(ctrl, reg[0])


def controlled_decrement(qc: QuantumCircuit, ctrl: int, reg: list[int]) -> None:
    """
    Reverse of controlled_increment.  If ctrl = 1, subtract 1 from reg.

    Apply the gates of controlled_increment in REVERSE order
    (all gates are self-inverse so reversing the sequence inverts the op):
      CX(ctrl, reg[0])
      for i = 1 to s-1:  MCX([ctrl, reg[0..i-1]], reg[i])
    """
    s = len(reg)
    qc.cx(ctrl, reg[0])
    for i in range(1, s):
        qc.mcx([ctrl] + reg[:i], reg[i])


def compute_clause_ancillas(
    qc: QuantumCircuit,
    var_q: list[int],
    clause_q: list[int],
    clauses: list[list[int]],
) -> None:
    """
    Set clause_q[j] = 1  iff  clause j is satisfied by the current var_q state.

    For clause (l₁ ∨ l₂ ∨ … ∨ lₖ):
      A clause is UNSATISFIED only when ALL literals are false, i.e. when:
        • positive literal +i  is false  →  x_i = 0
        • negative literal −i  is false  →  x_i = 1

      Algorithm:
        1. Flip var_q[i] for every POSITIVE literal +i  (maps "false" → 1)
        2. MCX(all clause vars → clause_anc[j])  ← fires when clause UNSATISFIED
        3. X(clause_anc[j])                       ← invert: now 1 = SATISFIED
        4. Unflip the same vars from step 1
    """
    for j, clause in enumerate(clauses):
        pos_idx  = [abs(lit) - 1 for lit in clause if lit > 0]   # vars to flip
        all_idx  = [abs(lit) - 1 for lit in clause]               # all vars

        for v in pos_idx:
            qc.x(var_q[v])
        qc.mcx([var_q[v] for v in all_idx], clause_q[j])
        qc.x(clause_q[j])
        for v in pos_idx:
            qc.x(var_q[v])


def uncompute_clause_ancillas(
    qc: QuantumCircuit,
    var_q: list[int],
    clause_q: list[int],
    clauses: list[list[int]],
) -> None:
    """
    Uncompute clause ancillas (return to |0⟩).

    The inverse of compute_clause_ancillas swaps the order of the X(ancilla)
    and MCX within each clause block (since X·MCX ≠ MCX·X in general):
        1. Flip pos vars
        2. X(ancilla)         ← reversed vs. compute
        3. MCX                ← reversed vs. compute
        4. Unflip pos vars
    """
    for j, clause in enumerate(clauses):
        pos_idx = [abs(lit) - 1 for lit in clause if lit > 0]
        all_idx = [abs(lit) - 1 for lit in clause]

        for v in pos_idx:
            qc.x(var_q[v])
        qc.x(clause_q[j])                                        # swapped
        qc.mcx([var_q[v] for v in all_idx], clause_q[j])         # swapped
        for v in pos_idx:
            qc.x(var_q[v])


def quantum_count(
    qc: QuantumCircuit,
    clause_q: list[int],
    sum_q: list[int],
) -> None:
    """
    Count the number of 1s in clause_q into sum_q.
    Applies controlled_increment once per clause ancilla qubit.
    """
    for cq in clause_q:
        controlled_increment(qc, cq, sum_q)


def uncompute_quantum_count(
    qc: QuantumCircuit,
    clause_q: list[int],
    sum_q: list[int],
) -> None:
    """
    Uncompute quantum_count  (reverse order, decrement instead of increment).
    """
    for cq in reversed(clause_q):
        controlled_decrement(qc, cq, sum_q)


def mark_geq_threshold(
    qc: QuantumCircuit,
    sum_q: list[int],
    threshold: int,
) -> None:
    """
    Apply phase −1 to all basis states of sum_q that represent values ≥ threshold.

    For each integer v in [threshold, 2^s):
      1. Flip bits of sum_q that are 0 in v  (maps |v⟩ → |1…1⟩)
      2. MCZ on all sum_q bits               (flips phase of |1…1⟩)
      3. Unflip
    Since each v maps to a DIFFERENT basis state, the MCZ applications
    are independent and compose correctly on a superposition.

    MCZ is implemented as  H · MCX · H  on the last sum qubit.
    """
    s       = len(sum_q)
    max_val = 1 << s

    for val in range(threshold, max_val):
        bits     = [(val >> b) & 1 for b in range(s)]
        zero_idx = [sum_q[b] for b, bit in enumerate(bits) if bit == 0]

        for q in zero_idx:
            qc.x(q)

        # Multi-controlled Z: flip phase of |1…1⟩ in sum_q
        if s == 1:
            qc.z(sum_q[0])
        else:
            qc.h(sum_q[-1])
            qc.mcx(sum_q[:-1], sum_q[-1])
            qc.h(sum_q[-1])

        for q in zero_idx:
            qc.x(q)


def build_oracle(
    qc: QuantumCircuit,
    var_q: list[int],
    clause_q: list[int],
    sum_q: list[int],
    clauses: list[list[int]],
    threshold: int,
) -> None:
    """
    Full Grover oracle: apply phase −1 to var assignments satisfying ≥ threshold clauses.

    Steps (ancillas start and end in |0⟩):
      compute clause ancillas → count into sum → phase mark → uncount → uncompute ancillas
    """
    compute_clause_ancillas(qc, var_q, clause_q, clauses)
    quantum_count(qc, clause_q, sum_q)
    mark_geq_threshold(qc, sum_q, threshold)          # phase kick — no uncompute needed
    uncompute_quantum_count(qc, clause_q, sum_q)
    uncompute_clause_ancillas(qc, var_q, clause_q, clauses)


def build_diffuser(qc: QuantumCircuit, var_q: list[int]) -> None:
    """
    Grover diffuser (inversion about the mean) on var_q.
    Implements  2|s⟩⟨s| − I  where  |s⟩ = H^⊗n|0⟩:
      H^n → X^n → MCZ → X^n → H^n
    """
    n = len(var_q)
    qc.h(var_q)
    qc.x(var_q)
    # MCZ = H on last qubit, MCX, H on last qubit
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
    """Hyper-parameters for the Grover MAX-SAT solver."""
    max_iters:  int  = 8       # maximum Grover iterations per threshold level
    shots:      int  = 4096    # measurement shots per circuit run
    seed:       int  = 42


class GroverMaxSAT:
    """
    Grover's Adaptive Search solver for MAX-SAT using Qiskit Aer.

    Usage
    -----
        solver = GroverMaxSAT(formula, GroverConfig())
        best_assignment = solver.optimise()
        solver.report()

    Register layout
    ---------------
        var_q    : n qubits   (the search space)
        clause_q : m qubits   (one ancilla per clause)
        sum_q    : s qubits   (s = ceil(log2(m+1)), the clause counter)
    Total: n + m + s qubits
    """

    def __init__(self, formula: CNF, config: GroverConfig = GroverConfig()) -> None:
        self.formula = formula
        self.config  = config
        self.backend = AerSimulator()

        n = formula.n_vars
        m = formula.n_clauses
        s = max(1, math.ceil(math.log2(m + 1)))

        self.n = n
        self.m = m
        self.s = s
        self.total_qubits = n + m + s

        self.var_q    = list(range(n))
        self.clause_q = list(range(n, n + m))
        self.sum_q    = list(range(n + m, n + m + s))

        # Results (set after optimise())
        self.best_assignment: list[int] = [1] * n
        self.best_fitness:    int       = formula.evaluate([1] * n)
        self.search_log:      list[dict] = []

    # ── Single Grover run ────────────────────────────────────────────────────

    def _run_grover(self, threshold: int, n_iters: int) -> tuple[list[int], int]:
        """
        Build and execute a Grover circuit with n_iters iterations
        and an oracle marking assignments satisfying >= threshold clauses.

        Returns
        -------
        (best_assignment, best_fitness) across all measurement outcomes.
        """
        qc = QuantumCircuit(self.total_qubits, self.n)

        # Uniform superposition over search variables
        for q in self.var_q:
            qc.h(q)

        # Alternate oracle + diffuser
        for _ in range(n_iters):
            build_oracle(
                qc,
                self.var_q, self.clause_q, self.sum_q,
                self.formula.clauses,
                threshold,
            )
            build_diffuser(qc, self.var_q)

        # Measure only the variable qubits
        for i, q in enumerate(self.var_q):
            qc.measure(q, i)

        transpiled = transpile(qc, self.backend, optimization_level=1)
        counts     = self.backend.run(transpiled, shots=self.config.shots).result().get_counts()

        # Find best assignment across all sampled bitstrings
        best_val,  best_asgn = -1, [0] * self.n
        for bs, _ in counts.items():
            # Qiskit: rightmost character = qubit 0 (LSB)
            asgn = [int(b) for b in reversed(bs)]
            val  = self.formula.evaluate(asgn)
            if val > best_val:
                best_val, best_asgn = val, asgn

        return best_asgn, best_val

    # ── Adaptive search ──────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> list[int]:
        """
        Run Grover's Adaptive Search and return the best assignment found.

        For each threshold level:
          • Try n_iters = 1, 2, …, max_iters Grover iterations.
          • Accept the first run whose best sample beats the threshold.
          • Advance threshold ← best_fitness + 1 and repeat.
          • Stop when no run beats the current threshold.
        """
        cfg = self.config
        n, m = self.n, self.m

        # Theoretical upper bound for iterations (M=1 case: sqrt(N))
        iter_cap = max(cfg.max_iters, round(math.pi / 4 * math.sqrt(2 ** n)))

        threshold = self.best_fitness + 1
        t0        = time.perf_counter()

        if verbose:
            print(f"Grover MAX-SAT  |  n={n}  m={m}  s={self.s}  "
                  f"total_qubits={self.total_qubits}")
            print(f"Initial lower bound : {self.best_fitness} / {m}")
            print(f"Iter cap per level  : {iter_cap}")
            print(f"\n{'Threshold':>10}  {'Iters':>6}  {'Found':>8}  "
                  f"{'Improved':>10}  {'Time(s)':>9}")
            print("-" * 52)

        while threshold <= m:
            improved = False

            for n_iters in range(1, iter_cap + 1):
                asgn, fitness = self._run_grover(threshold, n_iters)
                elapsed       = time.perf_counter() - t0

                entry = dict(threshold=threshold, iters=n_iters,
                             fitness=fitness, time=elapsed)
                self.search_log.append(entry)

                if verbose:
                    mark = "  ← better!" if fitness >= threshold else ""
                    print(f"{threshold:>10}  {n_iters:>6}  {fitness:>8}  "
                          f"{'YES' if fitness >= threshold else 'no':>10}  "
                          f"{elapsed:>9.3f}{mark}")

                if fitness >= threshold:
                    self.best_assignment = asgn
                    self.best_fitness    = fitness
                    threshold            = fitness + 1
                    improved             = True
                    break

            if not improved:
                break   # Grover couldn't beat the current threshold

        if verbose:
            elapsed = time.perf_counter() - t0
            pct     = 100 * self.best_fitness / m
            print("-" * 52)
            print(f"\n  Best : {self.best_fitness} / {m} clauses  ({pct:.2f}%)")
            print(f"  SAT  : {'YES ✓' if self.best_fitness == m else 'NO'}")
            print(f"  Time : {elapsed:.3f} s")

        return self.best_assignment

    # ── Report ───────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print a human-readable summary of the best solution found."""
        asgn  = self.best_assignment
        n, m  = self.n, self.m
        pct   = 100 * self.best_fitness / m

        unsat = [
            c for c in self.formula.clauses
            if not any(
                (lit > 0 and asgn[abs(lit) - 1] == 1) or
                (lit < 0 and asgn[abs(lit) - 1] == 0)
                for lit in c
            )
        ]

        print("\n=== Best Assignment ===")
        print(f"  Satisfied : {self.best_fitness} / {m}  ({pct:.2f}%)")
        print(f"  Fully SAT : {'YES ✓' if not unsat else 'NO'}")
        row = []
        for i in range(n):
            row.append(f"x{i+1}={'T' if asgn[i] else 'F'}")
            if len(row) == 10:
                print("  " + "  ".join(row)); row = []
        if row:
            print("  " + "  ".join(row))
        if unsat:
            print(f"\n  Unsatisfied clauses ({len(unsat)}):")
            for c in unsat[:5]:
                lits = " ∨ ".join(
                    (f"x{abs(l)}" if l > 0 else f"¬x{abs(l)}") for l in c
                )
                print(f"    ({lits})")
            if len(unsat) > 5:
                print(f"    … and {len(unsat)-5} more")


# =============================================================================
# Baselines
# =============================================================================

def exact_max_sat(formula: CNF) -> tuple[list[int], int]:
    """Exact MAX-SAT by exhaustive search (n ≤ 20)."""
    assert formula.n_vars <= 20
    best_val, best_asgn = 0, [0] * formula.n_vars
    for mask in range(1 << formula.n_vars):
        asgn = [(mask >> i) & 1 for i in range(formula.n_vars)]
        val  = formula.evaluate(asgn)
        if val > best_val:
            best_val, best_asgn = val, asgn
    return best_asgn, best_val


def greedy_max_sat(formula: CNF) -> tuple[list[int], int]:
    """Greedy MAX-SAT: pick each variable assignment that satisfies the most clauses."""
    asgn = [0] * formula.n_vars
    for i in range(formula.n_vars):
        val0 = formula.evaluate(asgn)
        asgn[i] = 1
        if formula.evaluate(asgn) < val0:
            asgn[i] = 0
    return asgn, formula.evaluate(asgn)


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 64

    # ── Example 1: Hand-crafted 4-variable instance ───────────────────────
    print(SEP)
    print("  MAX-SAT — Grover's Algorithm + Qiskit Aer")
    print(SEP)
    print("\n[Example 1]  Hand-crafted 4-variable, 8-clause instance")
    print("  (x1∨x2∨¬x3) ∧ (¬x1∨x3) ∧ (x2∨¬x4) ∧ (¬x2∨x3∨x4)")
    print("  ∧ (x1∨¬x4) ∧ (¬x3∨x4) ∧ (x1∨x2∨x4) ∧ (¬x1∨¬x2∨x3)")

    f1 = CNF(n_vars=4, clauses=[
        [ 1,  2, -3],
        [-1,  3],
        [ 2, -4],
        [-2,  3,  4],
        [ 1, -4],
        [-3,  4],
        [ 1,  2,  4],
        [-1, -2,  3],
    ])
    print(f"\n  {f1}  |  total qubits: {f1.n_vars + f1.n_clauses + max(1, math.ceil(math.log2(f1.n_clauses+1)))}")

    cfg1  = GroverConfig(max_iters=6, shots=4096, seed=42)
    sol1  = GroverMaxSAT(f1, cfg1)
    sol1.optimise(verbose=True)
    sol1.report()

    _, exact1   = exact_max_sat(f1)
    _, greedy1  = greedy_max_sat(f1)
    ratio1      = sol1.best_fitness / exact1 if exact1 > 0 else 0.0
    print(f"\n  {'Solver':<25}  {'Satisfied':>10}  {'/ m':>6}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*6}")
    print(f"  {'Exact (brute force)':<25}  {exact1:>10}  {f1.n_clauses:>6}")
    print(f"  {'Greedy baseline':<25}  {greedy1:>10}  {f1.n_clauses:>6}")
    print(f"  {'Grover GAS':<25}  {sol1.best_fitness:>10}  {f1.n_clauses:>6}")
    print(f"\n  Approximation ratio : {ratio1:.4f}")

    # ── Example 2: Random 3-SAT (5 variables) ────────────────────────────
    print(f"\n{SEP}")
    print("[Example 2]  Random 3-SAT  (n=5 vars, m=12 clauses)")

    f2 = CNF.random_3sat(n=5, m=12, seed=7)
    print(f"\n  {f2}  |  total qubits: {f2.n_vars + f2.n_clauses + max(1, math.ceil(math.log2(f2.n_clauses+1)))}")
    print("  Clauses:")
    for i, c in enumerate(f2.clauses):
        lits = " ∨ ".join(f"x{abs(l)}" if l > 0 else f"¬x{abs(l)}" for l in c)
        print(f"    C{i+1}: ({lits})")

    cfg2 = GroverConfig(max_iters=6, shots=8192, seed=42)
    sol2 = GroverMaxSAT(f2, cfg2)
    sol2.optimise(verbose=True)
    sol2.report()

    _, exact2  = exact_max_sat(f2)
    _, greedy2 = greedy_max_sat(f2)
    ratio2     = sol2.best_fitness / exact2 if exact2 > 0 else 0.0
    print(f"\n  {'Solver':<25}  {'Satisfied':>10}  {'/ m':>6}")
    print(f"  {'-'*25}  {'-'*10}  {'-'*6}")
    print(f"  {'Exact (brute force)':<25}  {exact2:>10}  {f2.n_clauses:>6}")
    print(f"  {'Greedy baseline':<25}  {greedy2:>10}  {f2.n_clauses:>6}")
    print(f"  {'Grover GAS':<25}  {sol2.best_fitness:>10}  {f2.n_clauses:>6}")
    print(f"\n  Approximation ratio : {ratio2:.4f}")

    # ── Example 3: Satisfiable instance — Grover should find SAT ─────────
    print(f"\n{SEP}")
    print("[Example 3]  Satisfiable 3-SAT  (n=5 vars, m=10 clauses)")
    print("  Known satisfying assignment: x1=T x2=F x3=T x4=T x5=F")

    # Build around a known solution so SAT is guaranteed
    solution = [1, 0, 1, 1, 0]
    f3_clauses = []
    rng = random.Random(99)
    while len(f3_clauses) < 10:
        vars_ = rng.sample(range(1, 6), 3)
        # Always include the known solution as a satisfying option
        clause = []
        for v in vars_:
            # With prob 0.5, include the literal that the known solution satisfies
            if rng.random() < 0.5:
                clause.append(v if solution[v-1] == 1 else -v)
            else:
                clause.append(-v if solution[v-1] == 1 else v)
        if any(
            (lit > 0 and solution[abs(lit)-1] == 1) or
            (lit < 0 and solution[abs(lit)-1] == 0)
            for lit in clause
        ):
            f3_clauses.append(clause)
    f3 = CNF(n_vars=5, clauses=f3_clauses)
    assert f3.evaluate(solution) == f3.n_clauses, "Instance must be satisfiable"

    print(f"\n  {f3}  |  total qubits: {f3.n_vars + f3.n_clauses + max(1, math.ceil(math.log2(f3.n_clauses+1)))}")

    cfg3 = GroverConfig(max_iters=8, shots=8192, seed=0)
    sol3 = GroverMaxSAT(f3, cfg3)
    sol3.optimise(verbose=True)
    sol3.report()

    print(f"\n  Known solution value : {f3.evaluate(solution)} / {f3.n_clauses}")
    print(f"  Grover found         : {sol3.best_fitness} / {f3.n_clauses}")

    # ── Iteration sweep analysis ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("[Analysis]  Effect of Grover iteration count on Example 1")
    print(f"  Oracle marks assignments satisfying ≥ {exact1} / {f1.n_clauses} clauses")
    print(f"\n  {'Iters':>6}  {'Best fitness':>14}  {'Shots hitting opt':>20}")
    print(f"  {'-'*6}  {'-'*14}  {'-'*20}")

    for n_iters in [1, 2, 3, 4, 5]:
        # Build and run just the oracle-marking circuit (no adaptive search)
        solver_tmp = GroverMaxSAT(f1, GroverConfig(shots=2048, seed=42))
        asgn_tmp, fit_tmp = solver_tmp._run_grover(exact1, n_iters)

        # Re-run to get counts
        qc_tmp = QuantumCircuit(solver_tmp.total_qubits, solver_tmp.n)
        for q in solver_tmp.var_q:
            qc_tmp.h(q)
        for _ in range(n_iters):
            build_oracle(qc_tmp, solver_tmp.var_q, solver_tmp.clause_q,
                         solver_tmp.sum_q, f1.clauses, exact1)
            build_diffuser(qc_tmp, solver_tmp.var_q)
        for i, q in enumerate(solver_tmp.var_q):
            qc_tmp.measure(q, i)
        tc = transpile(qc_tmp, solver_tmp.backend, optimization_level=1)
        raw_counts = solver_tmp.backend.run(tc, shots=2048).result().get_counts()

        opt_shots = sum(
            cnt for bs, cnt in raw_counts.items()
            if f1.evaluate([int(b) for b in reversed(bs)]) == exact1
        )
        pct_opt = 100 * opt_shots / 2048
        print(f"  {n_iters:>6}  {fit_tmp:>14}  {opt_shots:>10} / 2048  ({pct_opt:.1f}%)")

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()