"""
qubo_qaoa_qiskit.py  —  QUBO via QAOA using IBM Qiskit
=======================================================
Solves the SAME QUBO instance defined in qubo_problem.py.

Install:  pip install qiskit qiskit-aer scipy
Run:      python qubo_qaoa_qiskit.py

QUBO → Ising Mapping
---------------------
Substitute  x_i = (1 − Z_i) / 2  into  f(x) = x^T Q x:

  x_i x_j = (1−Z_i)(1−Z_j)/4 = (1 − Z_i − Z_j + Z_i Z_j) / 4
  x_i     = (1 − Z_i) / 2

Collecting terms:

  f(x) = C  +  Σ_i h_i Z_i  +  Σ_{i<j} J_ij Z_i Z_j

where:
  J_ij  = Q[i,j] / 4                                  (i < j)
  h_i   = Q[i,i]/2 + Σ_{j>i} Q[i,j]/4  +  Σ_{j<i} Q[j,i]/4
  C     = constant (not needed for optimisation)

We MINIMISE f, so we MINIMISE the Ising Hamiltonian H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j.

QAOA Circuit  (p layers)
  |ψ₀⟩ = H^⊗N |0…0⟩
  for each layer ℓ:
    U_C(γ_ℓ) = exp(−i γ_ℓ H_C)
             = Π_i RZ(2 γ h_i, i)  ·  Π_{i<j} RZZ(2 γ J_ij, i, j)
    U_B(β_ℓ) = exp(−i β_ℓ H_B)
             = Π_i RX(2 β, i)

The 2p parameters (γ, β) are optimised by COBYLA to minimise ⟨H_C⟩.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

from qubo_problem import N, Q, evaluate, BEST_KNOWN, print_problem, print_result


# =============================================================================
# QUBO → Ising coefficients
# =============================================================================

def qubo_to_ising(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert upper-triangular QUBO matrix to Ising (h, J, offset).

    Returns
    -------
    h      : (N,)   linear Z coefficients
    J      : (N,N)  upper-triangular ZZ coefficients
    offset : float  constant energy term
    """
    n      = Q.shape[0]
    h      = np.zeros(n)
    J      = np.zeros((n, n))
    offset = 0.0

    # Diagonal: Q[i,i] x_i = Q[i,i]/2 (1 − Z_i)
    for i in range(n):
        h[i]    -= Q[i, i] / 2.0
        offset  += Q[i, i] / 2.0

    # Off-diagonal: Q[i,j] x_i x_j = Q[i,j]/4 (1 − Z_i − Z_j + Z_i Z_j)
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] == 0:
                continue
            J[i, j] += Q[i, j] / 4.0
            h[i]    -= Q[i, j] / 4.0
            h[j]    -= Q[i, j] / 4.0
            offset  += Q[i, j] / 4.0

    return h, J, offset


# =============================================================================
# QAOA circuit
# =============================================================================

def build_qaoa_circuit(h: np.ndarray, J: np.ndarray, p: int) -> QuantumCircuit:
    """
    Build parameterised QAOA circuit for the Ising Hamiltonian.

    Gates:
      U_C(γ): RZ(2γhᵢ, i) per qubit; RZZ(2γJᵢⱼ, i, j) per coupling
      U_B(β): RX(2β, i) on every qubit
    """
    n     = len(h)
    gamma = ParameterVector("gamma", p)
    beta  = ParameterVector("beta",  p)
    qc    = QuantumCircuit(n, n)

    qc.h(range(n))   # uniform superposition

    for layer in range(p):
        # Cost unitary U_C(γ)
        for i in range(n):
            if h[i] != 0.0:
                qc.rz(2.0 * gamma[layer] * h[i], i)
        for i in range(n):
            for j in range(i + 1, n):
                if J[i, j] != 0.0:
                    qc.rzz(2.0 * gamma[layer] * J[i, j], i, j)
        # Mixer unitary U_B(β)
        for i in range(n):
            qc.rx(2.0 * beta[layer], i)

    qc.measure(range(n), range(n))
    return qc


# =============================================================================
# Config + Solver
# =============================================================================

@dataclass
class QAOAConfig:
    p:          int   = 3
    shots:      int   = 8192
    max_iter:   int   = 400
    n_restarts: int   = 6
    seed:       int   = 42


class QAOA_QUBO:
    """QAOA solver for QUBO minimisation via Qiskit Aer."""

    def __init__(self, config: QAOAConfig = QAOAConfig()) -> None:
        self.config  = config
        self.rng     = np.random.default_rng(config.seed)
        self.backend = AerSimulator()

        self.h, self.J, self.offset = qubo_to_ising(Q)
        self._qc         = build_qaoa_circuit(self.h, self.J, config.p)
        self._transpiled = transpile(self._qc, self.backend, optimization_level=1)

        self.best_params:      np.ndarray = np.array([])
        self.best_expectation: float      = float("inf")
        self.best_x:           list[int]  = [0] * N
        self.best_val:         float      = float("inf")

    # ── Circuit execution ────────────────────────────────────────────────────

    def _run(self, params: np.ndarray) -> dict[str, int]:
        p     = self.config.p
        gamma = params[:p]
        beta  = params[p:]
        binding: dict = {}
        for param in self._transpiled.parameters:
            name     = param.name
            vec, idx = name.rstrip("]").split("[")
            idx      = int(idx)
            binding[param] = float(gamma[idx] if vec == "gamma" else beta[idx])
        bound = self._transpiled.assign_parameters(binding)
        return self.backend.run(bound, shots=self.config.shots).result().get_counts()

    # ── Expectation value ⟨H_C⟩ ─────────────────────────────────────────────

    def _ising_energy(self, x: list[int]) -> float:
        """Energy of the Ising Hamiltonian for a given bitstring (Z = 1−2x)."""
        z   = np.array([1 - 2 * b for b in x], dtype=float)
        eng = float(np.dot(self.h, z))
        for i in range(N):
            for j in range(i + 1, N):
                eng += self.J[i, j] * z[i] * z[j]
        return eng

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
        return self._expectation(self._run(params))   # minimise ⟨H_C⟩

    def _run_once(self, init: np.ndarray) -> tuple[np.ndarray, float]:
        res = minimize(
            self._objective, init, method="COBYLA",
            options={"maxiter": self.config.max_iter, "rhobeg": 0.5},
        )
        return res.x, res.fun

    # ── Optimise ─────────────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> list[int]:
        cfg = self.config
        p   = cfg.p
        t0  = time.perf_counter()

        if verbose:
            print(f"QAOA  p={p}  restarts={cfg.n_restarts}  shots={cfg.shots}")
            print(f"{'Restart':>8}  {'⟨H_C⟩':>12}  {'Time(s)':>10}")
            print("─" * 36)

        best_e, best_p = float("inf"), None

        for restart in range(cfg.n_restarts):
            init = np.concatenate([
                self.rng.uniform(0,           np.pi,     p),
                self.rng.uniform(0, np.pi / 2,           p),
            ])
            params, eng = self._run_once(init)
            if verbose:
                print(f"{restart+1:>8}  {eng:>12.4f}  "
                      f"{time.perf_counter()-t0:>10.3f}")
            if eng < best_e:
                best_e, best_p = eng, params

        self.best_params      = best_p
        self.best_expectation = best_e

        # Final sample: pick best QUBO assignment found
        counts = self._run(best_p)
        best_val, best_x = float("inf"), [0] * N
        for bs in counts:
            x   = [int(b) for b in reversed(bs)]
            val = evaluate(x)
            if val < best_val:
                best_val, best_x = val, x

        self.best_x   = best_x
        self.best_val = best_val

        if verbose:
            elapsed = time.perf_counter() - t0
            print("─" * 36)
            print(f"\n  Best ⟨H_C⟩ : {self.best_expectation:.4f}")
            print(f"  Best f(x)  : {self.best_val:.2f}")
            print(f"  Time       : {elapsed:.3f} s")

        return best_x


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    SEP = "=" * 60
    print(SEP)
    print("  QUBO  —  QAOA + Qiskit Aer")
    print(SEP)
    print_problem()
    print(f"\n  Ising coefficients:")
    h, J, offset = qubo_to_ising(Q)
    print(f"  h (linear Z)      = {np.round(h, 3).tolist()}")
    print(f"  J (ZZ couplings, upper tri):")
    for row in np.round(J, 3):
        print(f"    {row.tolist()}")
    print(f"  offset            = {offset:.3f}")
    print(f"\n{'─'*60}")

    # ── Standard run ─────────────────────────────────────────────────────
    cfg  = QAOAConfig(p=3, shots=8192, n_restarts=6, seed=42)
    qaoa = QAOA_QUBO(cfg)

    t0   = time.perf_counter()
    best = qaoa.optimise(verbose=True)
    t1   = time.perf_counter() - t0

    print_result(best, f"QAOA (p={cfg.p})", qaoa.best_val, t1)

    # ── Depth sweep p = 1 … 4 ────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Depth sweep  p = 1 … 4  (restarts=5):")
    print(f"  {'p':>4}  {'⟨H_C⟩':>10}  {'f(x)':>8}  {'Gap':>8}  {'Time(s)':>10}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}")

    for pv in [1, 2, 3, 4]:
        t0   = time.perf_counter()
        q_   = QAOA_QUBO(QAOAConfig(p=pv, shots=8192, n_restarts=5, seed=99))
        bx_  = q_.optimise(verbose=False)
        bv_  = q_.best_val
        gap  = bv_ - BEST_KNOWN
        opt  = " ✓" if abs(gap) < 1e-6 else ""
        print(f"  {pv:>4}  {q_.best_expectation:>10.4f}  {bv_:>8.2f}  "
              f"{gap:>+8.2f}  {time.perf_counter()-t0:>10.3f}{opt}")

    print(f"\n  Exact optimum : {BEST_KNOWN:.2f}")
    print(SEP)


if __name__ == "__main__":
    main()