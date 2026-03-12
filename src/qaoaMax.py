"""
max_cut_qaoa_qiskit.py  —  Max-Cut via QAOA using IBM Qiskit
=============================================================

Install dependencies (once):
    pip install qiskit qiskit-aer scipy

Then run:
    python max_cut_qaoa_qiskit.py

What this file contains
-----------------------
  Graph            — weighted undirected graph + cut-value helper
  build_qaoa()     — constructs the parameterised Qiskit QAOA circuit
  QAOAConfig       — dataclass of hyper-parameters
  QAOAMaxCut       — hybrid quantum-classical solver (optimise + sample)
  brute_force()    — exact solver for small graphs (verification)
  greedy()         — fast greedy baseline
  main()           — two worked examples + depth-sweep analysis

Algorithm overview
------------------
QAOA (Farhi, Goldstone, Gutmann 2014) encodes Max-Cut as an Ising cost
Hamiltonian  H_C = ½ Σ_{(u,v)∈E} w_uv (I − Z_u Z_v)  and prepares the
variational state

    |ψ(γ,β)⟩ = U_B(β_p) U_C(γ_p) … U_B(β_1) U_C(γ_1) |s⟩

where |s⟩ = H^⊗n|0⟩,  U_C(γ) = Π_e RZZ(2γw_e),  U_B(β) = Π_q RX(2β).
The 2p angles are optimised classically (COBYLA) to maximise ⟨H_C⟩.
The optimised circuit is then sampled; the highest-cut bitstring is
the approximate Max-Cut solution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

# ── Qiskit imports ────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit_aer import AerSimulator


# =============================================================================
# Graph
# =============================================================================

@dataclass
class Graph:
    """Weighted undirected graph."""

    n_vertices: int
    edges: list[tuple[int, int, float]] = field(default_factory=list)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        self.edges.append((u, v, float(weight)))

    def cut_value(self, assignment: list[int] | np.ndarray) -> float:
        """Total weight of edges whose endpoints lie in different partitions."""
        return float(
            sum(w for u, v, w in self.edges if assignment[u] != assignment[v])
        )

    @staticmethod
    def random_graph(n: int, m: int, seed: int | None = None) -> "Graph":
        """Random graph with n vertices and m edges (random weights in [1, 10])."""
        rng = np.random.default_rng(seed)
        g = Graph(n_vertices=n)
        possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        chosen = rng.choice(len(possible), size=min(m, len(possible)), replace=False)
        for idx in chosen:
            u, v = possible[idx]
            g.add_edge(u, v, round(float(rng.uniform(1.0, 10.0)), 2))
        return g


# =============================================================================
# QAOA circuit builder (Qiskit)
# =============================================================================

def build_qaoa(graph: Graph, p: int) -> QuantumCircuit:
    """Return a parameterised Qiskit QAOA circuit for Max-Cut.

    Parameters
    ----------
    graph : Graph
    p     : int   number of QAOA layers

    Returns
    -------
    QuantumCircuit with ParameterVector 'gamma' (length p)
                   and ParameterVector 'beta'  (length p)

    Circuit structure
    -----------------
      H^⊗n
      for layer in 0..p-1:
        for each edge (u, v, w):   RZZ(2 * gamma[layer] * w, u, v)
        for each qubit q:          RX (2 * beta[layer],       q)
      measure all
    """
    n      = graph.n_vertices
    gamma  = ParameterVector("gamma", p)
    beta   = ParameterVector("beta",  p)

    qc = QuantumCircuit(n, n)

    # |s⟩ = H^⊗n |0…0⟩
    qc.h(range(n))

    for layer in range(p):
        # ── Cost unitary U_C(γ) ──────────────────────────────────────────
        # RZZ(θ) = exp(−i θ/2 · Z⊗Z)
        # For Ising term w(I − ZuZv)/2 we need exp(−i γ w ZuZv)
        # which is RZZ(2 γ w) up to a global phase.
        for u, v, w in graph.edges:
            qc.rzz(2.0 * gamma[layer] * w, u, v)

        # ── Mixer unitary U_B(β) ─────────────────────────────────────────
        for q in range(n):
            qc.rx(2.0 * beta[layer], q)

    qc.measure(range(n), range(n))
    return qc


# =============================================================================
# Expectation value from Qiskit counts
# =============================================================================

def expectation_from_counts(
    counts: dict[str, int], graph: Graph, shots: int
) -> float:
    """Estimate ⟨H_C⟩ = Σ_x P(x) · cut(x) from measurement counts."""
    exp_val = 0.0
    for bitstring, count in counts.items():
        # Qiskit bit-string order: rightmost char = qubit 0
        assignment = [int(b) for b in reversed(bitstring)]
        exp_val   += (count / shots) * graph.cut_value(assignment)
    return exp_val


# =============================================================================
# Config + solver
# =============================================================================

@dataclass
class QAOAConfig:
    """Hyper-parameters for the QAOA solver."""
    p:          int   = 2         # circuit depth
    shots:      int   = 8192      # measurement shots per evaluation
    max_iter:   int   = 500       # classical optimiser iterations
    n_restarts: int   = 5         # random restarts (best result kept)
    method:     str   = "COBYLA"  # scipy optimiser
    seed:       int   = 42


class QAOAMaxCut:
    """Hybrid quantum-classical Max-Cut solver using QAOA + Qiskit Aer.

    Usage
    -----
        solver = QAOAMaxCut(graph, QAOAConfig(p=2, n_restarts=6))
        best_bitstring = solver.optimise()
        solver.report()
    """

    def __init__(self, graph: Graph, config: QAOAConfig = QAOAConfig()) -> None:
        self.graph   = graph
        self.config  = config
        self.rng     = np.random.default_rng(config.seed)
        self.backend = AerSimulator()

        # Parameterised circuit (compiled once)
        self._qc         = build_qaoa(graph, config.p)
        self._transpiled = transpile(self._qc, self.backend, optimization_level=1)

        # Results (populated after optimise())
        self.best_params:      np.ndarray = np.array([])
        self.best_expectation: float      = 0.0
        self.best_bitstring:   str        = ""
        self.best_cut:         float      = 0.0

    # ── Circuit execution ────────────────────────────────────────────────────

    def _run(self, params: np.ndarray) -> dict[str, int]:
        """Bind angles to the circuit, run on Aer, return raw counts."""
        p     = self.config.p
        gamma = params[:p]
        beta  = params[p:]

        # qc.parameters returns a ParameterView sorted alphabetically by name.
        # ParameterVector names sort as: beta[0], beta[1], …, gamma[0], gamma[1], …
        # We build the binding by parsing each parameter's name directly.
        binding: dict = {}
        for param in self._transpiled.parameters:
            name = param.name          # e.g. "gamma[0]", "beta[1]"
            vec_name, idx_str = name.rstrip("]").split("[")
            idx = int(idx_str)
            binding[param] = float(gamma[idx] if vec_name == "gamma" else beta[idx])

        bound_qc = self._transpiled.assign_parameters(binding)
        job      = self.backend.run(bound_qc, shots=self.config.shots)
        return job.result().get_counts()

    # ── Objective ────────────────────────────────────────────────────────────

    def _objective(self, params: np.ndarray) -> float:
        counts = self._run(params)
        return -expectation_from_counts(counts, self.graph, self.config.shots)

    # ── Single optimisation run ──────────────────────────────────────────────

    def _run_once(self, init: np.ndarray) -> tuple[np.ndarray, float]:
        result = minimize(
            self._objective,
            init,
            method=self.config.method,
            options={"maxiter": self.config.max_iter, "rhobeg": 0.5},
        )
        return result.x, -result.fun

    # ── Public interface ─────────────────────────────────────────────────────

    def optimise(self, verbose: bool = True) -> str:
        """Optimise QAOA angles; return the best bitstring found.

        Steps
        -----
        1. For each restart, randomly initialise (γ, β) and run COBYLA.
        2. Keep the parameter vector that achieved the highest ⟨H_C⟩.
        3. Take a large final sample from the optimised state.
        4. Return the bitstring with the highest cut value across all samples.
        """
        cfg = self.config
        p   = cfg.p
        t0  = time.perf_counter()

        if verbose:
            print(f"QAOA  p={p}  restarts={cfg.n_restarts}  "
                  f"shots={cfg.shots}  method={cfg.method}")
            print(f"{'Restart':>8}  {'⟨H_C⟩':>12}  {'Time (s)':>10}")
            print("-" * 36)

        best_exp, best_params = -np.inf, None

        for restart in range(cfg.n_restarts):
            init = np.concatenate([
                self.rng.uniform(0,           np.pi,     p),
                self.rng.uniform(0, np.pi / 2,           p),
            ])
            params, exp = self._run_once(init)

            if verbose:
                elapsed = time.perf_counter() - t0
                print(f"{restart+1:>8}  {exp:>12.4f}  {elapsed:>10.3f}")

            if exp > best_exp:
                best_exp, best_params = exp, params

        self.best_params      = best_params
        self.best_expectation = best_exp

        # Final high-shot sampling of the optimised state
        final_counts = self._run(best_params)

        best_cut, best_bs = -1.0, ""
        for bs, _ in final_counts.items():
            assignment = [int(b) for b in reversed(bs)]
            val = self.graph.cut_value(assignment)
            if val > best_cut:
                best_cut, best_bs = val, bs

        self.best_cut       = best_cut
        self.best_bitstring = best_bs

        if verbose:
            elapsed = time.perf_counter() - t0
            p_ = cfg.p
            print("-" * 36)
            print(f"\n  Best ⟨H_C⟩       : {self.best_expectation:.4f}")
            print(f"  Best sampled cut : {self.best_cut:.2f}")
            print(f"  Best bitstring   : {self.best_bitstring}")
            print(f"  Optimal gamma    : "
                  f"{np.round(best_params[:p_], 4).tolist()}")
            print(f"  Optimal beta     : "
                  f"{np.round(best_params[p_:], 4).tolist()}")
            print(f"  Total time       : {elapsed:.2f} s")

        return best_bs

    def report(self) -> None:
        """Print a human-readable summary of the best partition found."""
        assignment = [int(b) for b in reversed(self.best_bitstring)]
        set_s  = [v for v, a in enumerate(assignment) if a == 0]
        set_sb = [v for v, a in enumerate(assignment) if a == 1]
        cut_edges = [
            (u, v, w) for u, v, w in self.graph.edges
            if assignment[u] != assignment[v]
        ]

        print("\n=== Best Solution ===")
        print(f"  Cut value : {self.best_cut:.2f}")
        print(f"  Set S     : {set_s}")
        print(f"  Set S-bar : {set_sb}")
        print(f"  Cut edges ({len(cut_edges)}):")
        for u, v, w in cut_edges:
            print(f"    ({u} - {v})  weight = {w:.2f}")

    def print_top_bitstrings(self, top_k: int = 10) -> None:
        """Sample and display the top-k most frequent bitstrings with cut values."""
        counts = self._run(self.best_params)
        ranked = sorted(counts.items(), key=lambda x: -x[1])[:top_k]
        total  = sum(counts.values())
        W      = 28
        n      = self.graph.n_vertices

        print(f"\n  {'Bitstring':<{n+2}}  {'Cut':>7}  {'Prob':>7}  Bar")
        print(f"  {'-'*(n+2)}  {'-------':>7}  {'-------':>7}  {'-'*W}")
        max_cnt = ranked[0][1] if ranked else 1
        for bs, cnt in ranked:
            assignment = [int(b) for b in reversed(bs)]
            cut  = self.graph.cut_value(assignment)
            prob = cnt / total
            bar  = "#" * int(W * cnt / max_cnt)
            print(f"  {bs:<{n+2}}  {cut:>7.2f}  {prob:>7.4f}  {bar}")


# =============================================================================
# Baselines
# =============================================================================

def brute_force(graph: Graph) -> tuple[list[int], float]:
    """Exact Max-Cut by exhaustive search (feasible for n <= ~20)."""
    assert graph.n_vertices <= 20, "Too many vertices for brute force"
    best_val, best_asgn = -1.0, [0] * graph.n_vertices
    for mask in range(1 << graph.n_vertices):
        asgn = [(mask >> v) & 1 for v in range(graph.n_vertices)]
        val  = graph.cut_value(asgn)
        if val > best_val:
            best_val, best_asgn = val, asgn
    return best_asgn, best_val


def greedy(graph: Graph) -> tuple[list[int], float]:
    """Simple greedy heuristic: assign each vertex to maximise current cut."""
    asgn = [0] * graph.n_vertices
    for v in range(graph.n_vertices):
        val0 = graph.cut_value(asgn)
        asgn[v] = 1
        if graph.cut_value(asgn) < val0:
            asgn[v] = 0
    return asgn, graph.cut_value(asgn)


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 62

    # ── Example 1: Petersen-like graph (10 vertices) ─────────────────────
    print(SEP)
    print("  MAX-CUT  —  QAOA + Qiskit Aer")
    print(SEP)
    print("\n[Example 1]  Petersen-like graph  (10 vertices, 19 edges)")

    g1 = Graph(n_vertices=10)
    for u, v, w in [
        (0,1,3),(0,4,2),(1,2,4),(1,6,1),(2,3,2),(2,7,5),
        (3,4,3),(3,8,2),(4,9,4),(5,6,3),(5,9,2),(6,7,4),
        (7,8,1),(8,9,3),(0,5,2),(1,7,3),(2,9,1),(3,6,4),(4,8,2),
    ]:
        g1.add_edge(u, v, w)

    cfg1 = QAOAConfig(p=2, shots=8192, n_restarts=6, seed=42)
    sol1 = QAOAMaxCut(g1, cfg1)
    sol1.optimise(verbose=True)
    sol1.report()

    _, opt1 = brute_force(g1)
    _, gr1  = greedy(g1)
    ratio1  = sol1.best_cut / opt1
    print(f"\n  Brute-force optimum  : {opt1:.2f}")
    print(f"  Greedy baseline      : {gr1:.2f}")
    print(f"  QAOA solution        : {sol1.best_cut:.2f}")
    print(f"  Approximation ratio  : {ratio1:.4f}")

    print("\n  Top bitstrings in optimised state:")
    sol1.print_top_bitstrings(top_k=8)

    # ── Example 2: Random graph (12 vertices, p=3) ────────────────────────
    print(f"\n{SEP}")
    print("[Example 2]  Random graph  (12 vertices, 25 edges,  p=3)")

    g2   = Graph.random_graph(n=12, m=25, seed=7)
    cfg2 = QAOAConfig(p=3, shots=16384, n_restarts=8, seed=0)
    sol2 = QAOAMaxCut(g2, cfg2)
    sol2.optimise(verbose=True)
    sol2.report()

    _, opt2 = brute_force(g2)
    _, gr2  = greedy(g2)
    ratio2  = sol2.best_cut / opt2
    print(f"\n  Brute-force optimum  : {opt2:.2f}")
    print(f"  Greedy baseline      : {gr2:.2f}")
    print(f"  QAOA solution        : {sol2.best_cut:.2f}")
    print(f"  Approximation ratio  : {ratio2:.4f}")

    # ── Depth sweep ───────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("[Analysis]  QAOA depth p = 1…4 on Example 2 graph")
    print(f"  {'p':>3}  {'<H_C>':>10}  {'Best cut':>10}  {'Ratio':>8}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*10}  {'-'*8}")

    for pv in [1, 2, 3, 4]:
        s = QAOAMaxCut(g2, QAOAConfig(p=pv, shots=8192, n_restarts=5, seed=99))
        s.optimise(verbose=False)
        print(f"  {pv:>3}  {s.best_expectation:>10.4f}  "
              f"{s.best_cut:>10.2f}  {s.best_cut/opt2:>8.4f}")

    print(f"\n  Brute-force optimum: {opt2:.2f}")
    print(SEP)
    print("Done.")


if __name__ == "__main__":
    main()