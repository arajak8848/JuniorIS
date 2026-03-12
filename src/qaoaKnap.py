"""
knapsack_qaoa_qiskit.py  —  0/1 Knapsack Problem via QAOA using IBM Qiskit
===========================================================================

Install dependencies (once):
    pip install qiskit qiskit-aer scipy

Then run:
    python knapsack_qaoa_qiskit.py

Problem
-------
Given n items (each with weight w_i and value v_i) and a knapsack of
capacity W, choose a subset to maximise total value without exceeding W.

QUBO Encoding
-------------
We introduce:
  - n binary item variables  x_i  ∈ {0,1}  (1 = take the item)
  - K slack variables        y_k  ∈ {0,1}  encoding the unused capacity
    in binary:  s = Σ_k 2^k · y_k,  s ∈ {0, …, W_int}

The knapsack constraint  Σ w_i x_i ≤ W  becomes the equality
    Σ w_i x_i + Σ 2^k y_k = W_int
after integer-scaling weights and capacity.

The full QUBO objective to MINIMISE is:

    H_QUBO = −A · Σ v_i x_i          ← maximise value  (A = normalised)
           + P · (Σ c_j x_j − W)²    ← enforce capacity constraint

where c_j = w_j for item qubits and c_{n+k} = 2^k for slack qubits,
and P is a penalty large enough that no infeasible solution beats any
feasible one.

Ising / QAOA Mapping
--------------------
Each binary variable x_i is mapped to a qubit via  x_i = (1 − Z_i)/2.
Substituting into H_QUBO yields the Ising Hamiltonian

    H_C = Σ_i  h_i Z_i  +  Σ_{i<j} J_ij Z_i Z_j  +  const

The QAOA circuit then implements p layers of
    U_C(γ) = exp(−iγ H_C)   via RZ and RZZ gates
    U_B(β) = exp(−iβ H_B)   via RX gates on every qubit
starting from  |s⟩ = H^⊗n |0…0⟩.

The 2p angles are optimised classically (COBYLA) to minimise ⟨H_C⟩.
The best feasible bitstring found across all samples is returned.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator


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
        return self.value / self.weight if self.weight > 0 else float("inf")


@dataclass
class KnapsackResult:
    selected:     list[Item]
    total_value:  float
    total_weight: float
    algorithm:    str
    feasible:     bool = True

    def report(self, capacity: float) -> None:
        status = "feasible" if self.feasible else "INFEASIBLE"
        print(f"\n  [{self.algorithm}]  ({status})")
        print(f"  {'Item':<18} {'Weight':>8} {'Value':>8}")
        print(f"  {'-'*18}  {'-'*8}  {'-'*8}")
        for it in self.selected:
            print(f"  {it.name:<18} {it.weight:>8.2f} {it.value:>8.2f}")
        print(f"  {'':18}  {'--------':>8}  {'--------':>8}")
        print(f"  {'TOTAL':<18} {self.total_weight:>8.2f} {self.total_value:>8.2f}")
        pct = 100 * self.total_weight / capacity if capacity > 0 else 0
        print(f"  Capacity used: {self.total_weight:.2f} / {capacity:.2f}  ({pct:.1f}%)")


# =============================================================================
# QUBO builder
# =============================================================================

def build_qubo(
    items:    list[Item],
    capacity: float,
    penalty:  float,
    scale:    int = 10,
) -> tuple[np.ndarray, int, int]:
    """
    Construct the QUBO matrix Q for the 0/1 knapsack problem.

    The QUBO is over (n + K) binary variables:
      - x_0 … x_{n-1}   item variables
      - y_0 … y_{K-1}   slack variables (binary-encoded unused capacity)

    Objective (minimisation):
        H = −A·Σ v_i x_i  +  P·(Σ c_j q_j − W_int)²

    where c_j = w_j (scaled integer) for items, c_{n+k} = 2^k for slack,
    and A is chosen so that penalty terms dominate infeasible solutions.

    Returns
    -------
    Q      : (n+K, n+K) upper-triangular QUBO matrix
    n_item : number of item qubits
    n_slack: number of slack qubits
    """
    n = len(items)

    # Integer-scale weights and capacity to avoid floating-point QUBO issues
    W_int = int(round(capacity * scale))
    w_int = [int(round(it.weight * scale)) for it in items]

    # Number of slack bits to encode integers in [0, W_int]
    K = max(1, math.floor(math.log2(W_int)) + 1) if W_int > 0 else 1

    total = n + K  # total number of qubits

    # Coefficients in the constraint: c_j q_j  (items then slack bits)
    c = w_int + [2**k for k in range(K)]

    # Normalise values to the same scale as the integer weights
    # so the penalty and objective are commensurate
    v_scaled = [it.value * scale for it in items]
    A = 1.0  # objective coefficient (penalty dominates by construction)

    Q = np.zeros((total, total))

    # ── Objective: −A · Σ v_i x_i ────────────────────────────────────────
    for i in range(n):
        Q[i, i] -= A * v_scaled[i]

    # ── Penalty: P · (Σ c_j q_j − W_int)² ───────────────────────────────
    # Expanding: P · [Σ_j c_j² q_j  +  2·Σ_{j<k} c_j c_k q_j q_k
    #                 − 2·W·Σ_j c_j q_j  +  W²]
    # Since q² = q for binary variables the linear term absorbs the c²:
    #   diagonal Q[j,j] += P · (c_j² − 2·W·c_j)
    #   off-diag Q[j,k] += P · 2·c_j·c_k   (j < k, upper-triangular)
    for j in range(total):
        Q[j, j] += penalty * (c[j] ** 2 - 2 * W_int * c[j])
        for k in range(j + 1, total):
            Q[j, k] += penalty * 2 * c[j] * c[k]

    return Q, n, K


# =============================================================================
# QUBO  →  Ising Hamiltonian coefficients
# =============================================================================

def qubo_to_ising(
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert an upper-triangular QUBO  H = x^T Q x  to Ising form.

        H_C = Σ_i h_i Z_i  +  Σ_{i<j} J_ij Z_i Z_j  +  offset

    via the substitution  x_i = (1 − Z_i) / 2.

    Returns
    -------
    h      : (N,)     linear Ising coefficients
    J      : (N, N)   upper-triangular quadratic coefficients
    offset : float    constant energy offset (not needed for optimisation)
    """
    N      = Q.shape[0]
    h      = np.zeros(N)
    J      = np.zeros((N, N))
    offset = 0.0

    # Diagonal QUBO terms: Q[i,i] · x_i  →  Q[i,i]/2·(1 − Z_i)
    for i in range(N):
        h[i]    -= Q[i, i] / 2.0
        offset  += Q[i, i] / 2.0

    # Off-diagonal QUBO terms (upper): Q[i,j] · x_i x_j
    # (1−Z_i)/2 · (1−Z_j)/2 = (1 − Z_i − Z_j + Z_i Z_j) / 4
    for i in range(N):
        for j in range(i + 1, N):
            if Q[i, j] == 0:
                continue
            J[i, j] += Q[i, j] / 4.0
            h[i]    -= Q[i, j] / 4.0
            h[j]    -= Q[i, j] / 4.0
            offset  += Q[i, j] / 4.0

    return h, J, offset


# =============================================================================
# QAOA circuit builder (Qiskit)
# =============================================================================

def build_qaoa_circuit(
    h: np.ndarray,
    J: np.ndarray,
    p: int,
) -> QuantumCircuit:
    """
    Build a parameterised Qiskit QAOA circuit for a generic Ising Hamiltonian.

    Gates used
    ----------
    U_C(γ):
      RZ(2·γ·h_i, i)         for each non-zero linear term
      RZZ(2·γ·J_ij, i, j)    for each non-zero quadratic term
    U_B(β):
      RX(2·β, q)              for every qubit

    Parameters are ParameterVectors 'gamma' (length p) and 'beta' (length p).
    """
    N     = len(h)
    gamma = ParameterVector("gamma", p)
    beta  = ParameterVector("beta",  p)

    qc = QuantumCircuit(N, N)

    # Initial state: uniform superposition
    qc.h(range(N))

    for layer in range(p):
        # ── Cost unitary U_C(γ) ──────────────────────────────────────────
        for i in range(N):
            if h[i] != 0.0:
                qc.rz(2.0 * gamma[layer] * h[i], i)

        for i in range(N):
            for j in range(i + 1, N):
                if J[i, j] != 0.0:
                    qc.rzz(2.0 * gamma[layer] * J[i, j], i, j)

        # ── Mixer unitary U_B(β) ─────────────────────────────────────────
        for q in range(N):
            qc.rx(2.0 * beta[layer], q)

    qc.measure(range(N), range(N))
    return qc


# =============================================================================
# Config + solver
# =============================================================================

@dataclass
class QAOAConfig:
    p:          int   = 2         # QAOA circuit depth
    shots:      int   = 8192      # measurement shots per evaluation
    max_iter:   int   = 300       # COBYLA iteration budget
    n_restarts: int   = 5         # random restarts
    penalty:    float = 0.0       # 0 = auto-compute
    weight_scale: int = 10        # integer scaling factor for weights
    seed:       int   = 42


class QAOAKnapsack:
    """
    Hybrid quantum-classical 0/1 Knapsack solver using QAOA + Qiskit Aer.

    Usage
    -----
        solver = QAOAKnapsack(items, capacity, QAOAConfig(p=2))
        result = solver.optimise()
        result.report(capacity)
    """

    def __init__(
        self,
        items:    list[Item],
        capacity: float,
        config:   QAOAConfig = QAOAConfig(),
    ) -> None:
        self.items    = items
        self.capacity = capacity
        self.config   = config
        self.rng      = np.random.default_rng(config.seed)
        self.backend  = AerSimulator()

        # Auto-compute penalty if not supplied
        # Rule: P must exceed the maximum total value so that any infeasible
        # solution is penalised more than the best feasible solution gains.
        if config.penalty == 0.0:
            max_value  = sum(it.value for it in items) * config.weight_scale
            self.penalty = max_value + 1.0
        else:
            self.penalty = config.penalty

        # Build QUBO → Ising → circuit
        Q, self.n_item, self.n_slack = build_qubo(
            items, capacity, self.penalty, config.weight_scale
        )
        self.h, self.J, self.offset = qubo_to_ising(Q)
        self.n_qubits = len(self.h)

        self._qc         = build_qaoa_circuit(self.h, self.J, config.p)
        self._transpiled = transpile(self._qc, self.backend, optimization_level=1)

        # Results (set after optimise())
        self.best_params:      np.ndarray  = np.array([])
        self.best_expectation: float       = np.inf
        self.best_result:      KnapsackResult | None = None

    # ── Circuit execution ────────────────────────────────────────────────────

    def _run(self, params: np.ndarray) -> dict[str, int]:
        p     = self.config.p
        gamma = params[:p]
        beta  = params[p:]

        binding: dict = {}
        for param in self._transpiled.parameters:
            name     = param.name            # "gamma[k]" or "beta[k]"
            vec, idx = name.rstrip("]").split("[")
            idx      = int(idx)
            binding[param] = float(gamma[idx] if vec == "gamma" else beta[idx])

        bound = self._transpiled.assign_parameters(binding)
        job   = self.backend.run(bound, shots=self.config.shots)
        return job.result().get_counts()

    # ── Energy of a single bitstring ─────────────────────────────────────────

    def _ising_energy(self, bits: list[int]) -> float:
        """Compute ⟨H_C⟩ for a fixed bitstring (Z_i = 1−2·x_i)."""
        z   = np.array([1 - 2 * b for b in bits], dtype=float)
        eng = float(np.dot(self.h, z))
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                eng += self.J[i, j] * z[i] * z[j]
        return eng

    # ── Decode a bitstring to a knapsack solution ────────────────────────────

    def _decode(self, bitstring: str) -> KnapsackResult:
        """
        Parse a Qiskit bitstring (qubit 0 = rightmost char) into a
        KnapsackResult, checking feasibility.
        """
        bits       = [int(b) for b in reversed(bitstring)]
        item_bits  = bits[: self.n_item]

        selected   = [it for it, x in zip(self.items, item_bits) if x == 1]
        total_w    = sum(it.weight for it in selected)
        total_v    = sum(it.value  for it in selected)
        feasible   = total_w <= self.capacity + 1e-6

        return KnapsackResult(
            selected     = selected,
            total_value  = total_v,
            total_weight = total_w,
            algorithm    = f"QAOA (p={self.config.p})",
            feasible     = feasible,
        )

    # ── Expectation value ────────────────────────────────────────────────────

    def _expectation(self, counts: dict[str, int]) -> float:
        total = sum(counts.values())
        return sum(
            (cnt / total) * self._ising_energy(
                [int(b) for b in reversed(bs)]
            )
            for bs, cnt in counts.items()
        )

    # ── Objective ────────────────────────────────────────────────────────────

    def _objective(self, params: np.ndarray) -> float:
        counts = self._run(params)
        return self._expectation(counts)   # minimise Ising energy

    # ── Single optimisation run ──────────────────────────────────────────────

    def _run_once(self, init: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(
            self._objective, init,
            method=self.config.method if hasattr(self.config, "method") else "COBYLA",
            options={"maxiter": self.config.max_iter, "rhobeg": 0.5},
        )
        return result.x, result.fun

    # ── Full optimisation ────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> KnapsackResult:
        """
        Optimise QAOA angles and return the best feasible solution found.

        Steps
        -----
        1. For each restart, randomly initialise (γ, β) and run COBYLA.
        2. Keep the parameter vector with the lowest Ising energy ⟨H_C⟩.
        3. Sample the optimised state (high-shot run).
        4. Among all sampled bitstrings, return the feasible one with the
           highest knapsack value.
        """
        cfg = self.config
        p   = cfg.p
        t0  = time.perf_counter()

        if verbose:
            print(f"QAOA Knapsack  |  qubits={self.n_qubits} "
                  f"({self.n_item} items + {self.n_slack} slack)  "
                  f"|  p={p}  |  restarts={cfg.n_restarts}")
            print(f"{'Restart':>8}  {'⟨H_C⟩':>12}  {'Time (s)':>10}")
            print("-" * 36)

        best_energy = np.inf
        best_params = None

        for restart in range(cfg.n_restarts):
            init = np.concatenate([
                self.rng.uniform(0,           np.pi,     p),
                self.rng.uniform(0, np.pi / 2,           p),
            ])
            params, energy = self._run_once(init)
            elapsed = time.perf_counter() - t0

            if verbose:
                print(f"{restart+1:>8}  {energy:>12.4f}  {elapsed:>10.3f}")

            if energy < best_energy:
                best_energy = energy
                best_params = params

        self.best_params      = best_params
        self.best_expectation = best_energy

        # Final high-shot sampling
        final_counts = self._run(best_params)

        # Pick the feasible solution with the highest value
        best_result: KnapsackResult | None = None
        for bs in final_counts:
            r = self._decode(bs)
            if r.feasible:
                if best_result is None or r.total_value > best_result.total_value:
                    best_result = r

        # Fallback: if no feasible solution sampled, return the least-infeasible
        if best_result is None:
            if verbose:
                print("  WARNING: no feasible solution found in samples — "
                      "try increasing penalty or shots.")
            best_result = max(
                (self._decode(bs) for bs in final_counts),
                key=lambda r: r.total_value - 1e6 * max(0, r.total_weight - self.capacity),
            )

        self.best_result = best_result

        if verbose:
            elapsed = time.perf_counter() - t0
            print("-" * 36)
            print(f"\n  Best ⟨H_C⟩        : {self.best_expectation:.4f}")
            print(f"  Best value found  : {best_result.total_value:.2f}")
            print(f"  Total weight      : {best_result.total_weight:.2f} "
                  f"/ {self.capacity:.2f}")
            print(f"  Feasible          : {best_result.feasible}")
            print(f"  Penalty used      : {self.penalty:.2f}")
            print(f"  Total time        : {elapsed:.2f} s")

        return best_result

    def top_feasible_samples(self, top_k: int = 8) -> None:
        """Sample the optimised circuit and display the top-k feasible bitstrings."""
        if not len(self.best_params):
            print("  Run optimise() first.")
            return

        counts = self._run(self.best_params)
        total  = sum(counts.values())
        W      = self.n_qubits

        # Collect feasible results
        feasible = []
        for bs, cnt in counts.items():
            r = self._decode(bs)
            if r.feasible:
                feasible.append((r, cnt / total, bs))

        feasible.sort(key=lambda x: (-x[0].total_value, -x[1]))
        feasible = feasible[:top_k]

        if not feasible:
            print("  No feasible solutions in sample.")
            return

        bar_w   = 24
        max_cnt = feasible[0][1]
        print(f"\n  {'Bitstring':<{W+2}}  {'Value':>7}  {'Weight':>7}  "
              f"{'Prob':>7}  Bar")
        print(f"  {'-'*(W+2)}  {'-------':>7}  {'-------':>7}  "
              f"{'-------':>7}  {'-'*bar_w}")
        for r, prob, bs in feasible:
            bar = "#" * int(bar_w * prob / max_cnt)
            print(f"  {bs:<{W+2}}  {r.total_value:>7.2f}  "
                  f"{r.total_weight:>7.2f}  {prob:>7.4f}  {bar}")


# =============================================================================
# Baselines
# =============================================================================

def exact_01(items: list[Item], capacity: float) -> KnapsackResult:
    """Brute-force exact 0/1 knapsack (n ≤ 20)."""
    assert len(items) <= 20
    best_v, best_s = 0.0, []
    for r in range(len(items) + 1):
        for subset in combinations(items, r):
            w = sum(it.weight for it in subset)
            v = sum(it.value  for it in subset)
            if w <= capacity and v > best_v:
                best_v, best_s = v, list(subset)
    return KnapsackResult(
        selected     = best_s,
        total_value  = best_v,
        total_weight = sum(it.weight for it in best_s),
        algorithm    = "Exact brute-force (optimal)",
    )


def greedy_01(items: list[Item], capacity: float) -> KnapsackResult:
    """Greedy 0/1 knapsack by value/weight ratio."""
    selected, remaining, total_v = [], capacity, 0.0
    for it in sorted(items, key=lambda x: x.ratio, reverse=True):
        if it.weight <= remaining:
            selected.append(it)
            remaining -= it.weight
            total_v   += it.value
    return KnapsackResult(
        selected     = selected,
        total_value  = total_v,
        total_weight = capacity - remaining,
        algorithm    = "Greedy 0/1 (heuristic)",
    )


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 64

    # ── Example 1: Small 5-item instance (QAOA tractable) ────────────────
    print(SEP)
    print("  0/1 KNAPSACK  —  QAOA + Qiskit Aer")
    print(SEP)
    print("\n[Example 1]  5-item instance  (capacity = 10 kg)")

    items1 = [
        Item("Gold bar",     weight=5.0, value=80.0),
        Item("Laptop",       weight=4.0, value=60.0),
        Item("Camera",       weight=2.0, value=30.0),
        Item("Watch",        weight=1.0, value=20.0),
        Item("Necklace",     weight=1.5, value=25.0),
    ]
    cap1 = 10.0

    print("\n  Items:")
    print(f"  {'Name':<16} {'Weight':>8} {'Value':>8} {'Ratio':>8}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*8}  {'-'*8}")
    for it in sorted(items1, key=lambda x: x.ratio, reverse=True):
        print(f"  {it.name:<16} {it.weight:>8.1f} {it.value:>8.1f} {it.ratio:>8.3f}")

    cfg1  = QAOAConfig(p=2, shots=8192, n_restarts=5, seed=42)
    sol1  = QAOAKnapsack(items1, cap1, cfg1)
    res1  = sol1.optimise(verbose=True)
    res1.report(cap1)

    print("\n  Top feasible samples:")
    sol1.top_feasible_samples(top_k=6)

    # Compare with baselines
    exact1  = exact_01(items1, cap1)
    greedy1 = greedy_01(items1, cap1)
    ratio1  = res1.total_value / exact1.total_value if exact1.total_value > 0 else 0

    print(f"\n  {'Solver':<30}  {'Value':>8}  {'Weight':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
    print(f"  {'Exact (brute force)':<30}  {exact1.total_value:>8.2f}  "
          f"{exact1.total_weight:>8.2f}")
    print(f"  {'Greedy (ratio)':<30}  {greedy1.total_value:>8.2f}  "
          f"{greedy1.total_weight:>8.2f}")
    print(f"  {f'QAOA (p={cfg1.p})':<30}  {res1.total_value:>8.2f}  "
          f"{res1.total_weight:>8.2f}")
    print(f"\n  Approximation ratio (QAOA / exact) : {ratio1:.4f}")

    # ── Example 2: Slightly larger instance, deeper circuit ──────────────
    print(f"\n{SEP}")
    print("[Example 2]  6-item instance  (capacity = 15 kg,  p=3)")

    items2 = [
        Item("Diamond",     weight=1.0,  value=90.0),
        Item("Gold bar",    weight=5.0,  value=80.0),
        Item("Laptop",      weight=4.0,  value=60.0),
        Item("Silver",      weight=3.0,  value=40.0),
        Item("Camera",      weight=2.0,  value=30.0),
        Item("Watch",       weight=1.0,  value=20.0),
    ]
    cap2 = 15.0

    cfg2  = QAOAConfig(p=3, shots=16384, n_restarts=6, seed=7)
    sol2  = QAOAKnapsack(items2, cap2, cfg2)
    res2  = sol2.optimise(verbose=True)
    res2.report(cap2)

    exact2  = exact_01(items2, cap2)
    greedy2 = greedy_01(items2, cap2)
    ratio2  = res2.total_value / exact2.total_value if exact2.total_value > 0 else 0

    print(f"\n  {'Solver':<30}  {'Value':>8}  {'Weight':>8}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
    print(f"  {'Exact (brute force)':<30}  {exact2.total_value:>8.2f}  "
          f"{exact2.total_weight:>8.2f}")
    print(f"  {'Greedy (ratio)':<30}  {greedy2.total_value:>8.2f}  "
          f"{greedy2.total_weight:>8.2f}")
    print(f"  {f'QAOA (p={cfg2.p})':<30}  {res2.total_value:>8.2f}  "
          f"{res2.total_weight:>8.2f}")
    print(f"\n  Approximation ratio (QAOA / exact) : {ratio2:.4f}")

    # ── Depth sweep: p = 1 … 4 on Example 1 ─────────────────────────────
    print(f"\n{SEP}")
    print("[Analysis]  QAOA depth p=1…4 on Example 1")
    print(f"  {'p':>3}  {'⟨H_C⟩':>12}  {'Value':>8}  {'Feasible':>9}  {'Ratio':>8}")
    print(f"  {'-'*3}  {'-'*12}  {'-'*8}  {'-'*9}  {'-'*8}")

    opt_val = exact1.total_value
    for pv in [1, 2, 3, 4]:
        s = QAOAKnapsack(
            items1, cap1,
            QAOAConfig(p=pv, shots=8192, n_restarts=4, seed=99),
        )
        r = s.optimise(verbose=False)
        ratio = r.total_value / opt_val if opt_val > 0 else 0
        print(f"  {pv:>3}  {s.best_expectation:>12.4f}  {r.total_value:>8.2f}  "
              f"{'yes' if r.feasible else 'NO':>9}  {ratio:>8.4f}")

    print(f"\n  Exact optimum: {opt_val:.2f}")
    print(SEP)
    print("Done.")


if __name__ == "__main__":
    main()