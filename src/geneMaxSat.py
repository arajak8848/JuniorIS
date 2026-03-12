"""
max_sat_genetic.py  —  MAX-SAT (Search) via Genetic Algorithm
=============================================================

Install dependencies (none beyond stdlib):
    python max_sat_genetic.py

The MAX-SAT Problem
-------------------
Given a Boolean formula in Conjunctive Normal Form (CNF):

    F = C_1 ∧ C_2 ∧ … ∧ C_m

where each clause C_j is a disjunction of literals (a variable x_i or
its negation ¬x_i), find a truth assignment to the n variables that
MAXIMISES the number of satisfied clauses.

  - SAT  (decision)  : does a satisfying assignment exist?
  - MAX-SAT (search) : find the assignment satisfying the MOST clauses.

MAX-SAT is NP-hard.  We use a Genetic Algorithm (GA) as a metaheuristic.

Representation
--------------
Each individual is a binary string of length n:
    gene[i] = 1  →  variable x_{i+1} is TRUE
    gene[i] = 0  →  variable x_{i+1} is FALSE

Fitness = number of satisfied clauses  (integer in [0, m])

GA Operators
------------
  Selection  : Tournament selection (k candidates, best wins)
  Crossover  : Uniform crossover (each gene independently from either parent)
  Mutation   : Bit-flip with per-gene probability p_mut
  Elitism    : Top-e individuals survive unchanged each generation

CNF Input Format
----------------
Clauses are lists of non-zero integers (DIMACS convention):
    positive literal  +i  →  x_i  must be TRUE
    negative literal  −i  →  x_i  must be FALSE
e.g. clause [1, -3, 5] means  (x1 ∨ ¬x3 ∨ x5)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional


# =============================================================================
# CNF Formula
# =============================================================================

@dataclass
class CNF:
    """A propositional formula in Conjunctive Normal Form."""

    n_vars:   int                          # number of Boolean variables
    clauses:  list[list[int]]              # each clause: list of DIMACS literals

    @property
    def n_clauses(self) -> int:
        return len(self.clauses)

    def evaluate(self, assignment: list[int]) -> int:
        """Count the number of clauses satisfied by `assignment`.

        assignment[i] = 1 means variable x_{i+1} is TRUE (0-indexed).
        """
        satisfied = 0
        for clause in self.clauses:
            for lit in clause:
                var = abs(lit) - 1          # 0-indexed
                val = assignment[var]
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    satisfied += 1
                    break                   # clause is satisfied; move on
        return satisfied

    def is_satisfied(self, assignment: list[int]) -> bool:
        return self.evaluate(assignment) == self.n_clauses

    # ── Convenience constructors ─────────────────────────────────────────────

    @staticmethod
    def random_3sat(
        n: int,
        m: int,
        seed: Optional[int] = None,
    ) -> "CNF":
        """
        Generate a random 3-SAT instance with n variables and m clauses.
        Each clause is a uniformly random choice of 3 distinct variables,
        each negated independently with probability 0.5.
        """
        rng = random.Random(seed)
        clauses: list[list[int]] = []
        for _ in range(m):
            vars_ = rng.sample(range(1, n + 1), 3)
            clause = [v if rng.random() < 0.5 else -v for v in vars_]
            clauses.append(clause)
        return CNF(n_vars=n, clauses=clauses)

    @staticmethod
    def random_ksat(
        n: int,
        m: int,
        k: int = 3,
        seed: Optional[int] = None,
    ) -> "CNF":
        """Random k-SAT instance."""
        rng = random.Random(seed)
        clauses: list[list[int]] = []
        for _ in range(m):
            vars_ = rng.sample(range(1, n + 1), min(k, n))
            clause = [v if rng.random() < 0.5 else -v for v in vars_]
            clauses.append(clause)
        return CNF(n_vars=n, clauses=clauses)

    @staticmethod
    def from_dimacs(text: str) -> "CNF":
        """
        Parse a DIMACS CNF string.

        Example
        -------
        p cnf 3 2
        1 -3 0
        2 3 -1 0
        """
        clauses: list[list[int]] = []
        n_vars = 0
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts  = line.split()
                n_vars = int(parts[2])
                continue
            lits = [int(x) for x in line.split() if x != "0"]
            if lits:
                clauses.append(lits)
        return CNF(n_vars=n_vars, clauses=clauses)

    def to_dimacs(self) -> str:
        """Serialise formula to DIMACS CNF format."""
        lines = [f"p cnf {self.n_vars} {self.n_clauses}"]
        for clause in self.clauses:
            lines.append(" ".join(map(str, clause)) + " 0")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CNF(vars={self.n_vars}, clauses={self.n_clauses})"


# =============================================================================
# Genetic Algorithm
# =============================================================================

@dataclass
class GAConfig:
    """Hyper-parameters for the MAX-SAT genetic algorithm."""
    pop_size:     int   = 200     # population size
    generations:  int   = 500     # number of generations
    p_crossover:  float = 0.85    # probability of crossover
    p_mutation:   float = 0.02    # per-gene mutation probability
    tournament_k: int   = 5       # tournament selection size
    elite_count:  int   = 10      # elitism: top-k survivors per generation
    seed:  Optional[int] = 42


class GeneticMaxSAT:
    """
    Genetic Algorithm solver for MAX-SAT.

    Usage
    -----
        solver = GeneticMaxSAT(formula, GAConfig())
        best   = solver.evolve()
        solver.report()
    """

    def __init__(self, formula: CNF, config: GAConfig = GAConfig()) -> None:
        self.formula = formula
        self.config  = config
        self.rng     = random.Random(config.seed)

        self.population:      list[list[int]] = []
        self.fitnesses:       list[int]        = []
        self.best_individual: list[int]        = []
        self.best_fitness:    int              = 0
        self.history:         list[int]        = []   # best fitness per generation
        self.avg_history:     list[float]      = []   # avg  fitness per generation

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _random_individual(self) -> list[int]:
        return [self.rng.randint(0, 1) for _ in range(self.formula.n_vars)]

    def _init_population(self) -> None:
        self.population = [self._random_individual() for _ in range(self.config.pop_size)]
        self.fitnesses  = [self.formula.evaluate(ind) for ind in self.population]

    # ── Selection ────────────────────────────────────────────────────────────

    def _tournament(self) -> list[int]:
        """Return a copy of the tournament winner (highest fitness)."""
        k       = self.config.tournament_k
        indices = self.rng.sample(range(self.config.pop_size), k)
        winner  = max(indices, key=lambda i: self.fitnesses[i])
        return self.population[winner][:]

    # ── Crossover ────────────────────────────────────────────────────────────

    def _crossover(
        self, p1: list[int], p2: list[int]
    ) -> tuple[list[int], list[int]]:
        """Uniform crossover: each bit drawn independently from either parent."""
        if self.rng.random() > self.config.p_crossover:
            return p1[:], p2[:]
        n     = self.formula.n_vars
        mask  = [self.rng.randint(0, 1) for _ in range(n)]
        c1    = [p1[i] if mask[i] == 0 else p2[i] for i in range(n)]
        c2    = [p2[i] if mask[i] == 0 else p1[i] for i in range(n)]
        return c1, c2

    # ── Mutation ─────────────────────────────────────────────────────────────

    def _mutate(self, ind: list[int]) -> list[int]:
        """Flip each bit with probability p_mutation."""
        p = self.config.p_mutation
        return [b ^ 1 if self.rng.random() < p else b for b in ind]

    # ── Clause-directed repair (optional local search step) ──────────────────

    def _repair(self, ind: list[int]) -> list[int]:
        """
        One pass of clause-directed local search:
        For each unsatisfied clause, flip the variable that increases fitness
        the most (or do nothing if no improvement exists).
        This is a single WalkSAT-style step applied as a post-crossover repair.
        """
        formula = self.formula
        current = formula.evaluate(ind)
        improved = ind[:]

        unsatisfied = [
            clause for clause in formula.clauses
            if not any(
                (lit > 0 and improved[abs(lit) - 1] == 1) or
                (lit < 0 and improved[abs(lit) - 1] == 0)
                for lit in clause
            )
        ]

        for clause in unsatisfied:
            best_gain = 0
            best_var  = -1
            for lit in clause:
                var = abs(lit) - 1
                improved[var] ^= 1
                new_val = formula.evaluate(improved)
                gain    = new_val - current
                improved[var] ^= 1   # revert
                if gain > best_gain:
                    best_gain = gain
                    best_var  = var
            if best_var >= 0:
                improved[best_var] ^= 1
                current += best_gain

        return improved

    # ── Main evolution loop ───────────────────────────────────────────────────

    def evolve(
        self,
        verbose:      bool = True,
        use_repair:   bool = True,
        print_every:  int  = 100,
    ) -> list[int]:
        """
        Run the genetic algorithm and return the best assignment found.

        Parameters
        ----------
        verbose     : print progress table
        use_repair  : apply one WalkSAT repair step after crossover/mutation
        print_every : print a row every N generations

        Returns
        -------
        best_individual : list[int] of length n_vars
        """
        cfg = self.config
        self._init_population()

        # Seed best
        best_idx          = max(range(cfg.pop_size), key=lambda i: self.fitnesses[i])
        self.best_fitness = self.fitnesses[best_idx]
        self.best_individual = self.population[best_idx][:]

        if verbose:
            m = self.formula.n_clauses
            print(f"{'Gen':>6}  {'Best':>8}  {'Avg':>8}  {'Best%':>7}  {'Time(s)':>8}")
            print("-" * 46)

        t0 = time.perf_counter()

        for gen in range(1, cfg.generations + 1):

            # ── Elitism ──────────────────────────────────────────────────
            elite_idx  = sorted(
                range(cfg.pop_size),
                key=lambda i: self.fitnesses[i],
                reverse=True,
            )[: cfg.elite_count]
            new_pop = [self.population[i][:] for i in elite_idx]

            # ── Fill new population ───────────────────────────────────────
            while len(new_pop) < cfg.pop_size:
                p1 = self._tournament()
                p2 = self._tournament()
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                if use_repair:
                    c1 = self._repair(c1)
                    c2 = self._repair(c2)
                new_pop.append(c1)
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(c2)

            self.population = new_pop
            self.fitnesses  = [self.formula.evaluate(ind) for ind in self.population]

            # ── Track best ───────────────────────────────────────────────
            best_idx  = max(range(cfg.pop_size), key=lambda i: self.fitnesses[i])
            gen_best  = self.fitnesses[best_idx]
            gen_avg   = sum(self.fitnesses) / cfg.pop_size

            self.history.append(gen_best)
            self.avg_history.append(gen_avg)

            if gen_best > self.best_fitness:
                self.best_fitness    = gen_best
                self.best_individual = self.population[best_idx][:]

            # Early exit: all clauses satisfied
            if self.best_fitness == self.formula.n_clauses:
                if verbose:
                    elapsed = time.perf_counter() - t0
                    pct     = 100 * gen_best / self.formula.n_clauses
                    print(f"{gen:>6}  {gen_best:>8}  {gen_avg:>8.2f}  "
                          f"{pct:>6.2f}%  {elapsed:>8.3f}")
                    print(f"\n  ✓ Full satisfying assignment found at generation {gen}!")
                break

            if verbose and (gen % print_every == 0 or gen == 1):
                elapsed = time.perf_counter() - t0
                pct     = 100 * gen_best / self.formula.n_clauses
                print(f"{gen:>6}  {gen_best:>8}  {gen_avg:>8.2f}  "
                      f"{pct:>6.2f}%  {elapsed:>8.3f}")

        if verbose:
            elapsed = time.perf_counter() - t0
            print("-" * 46)
            pct = 100 * self.best_fitness / self.formula.n_clauses
            print(f"\n  Best   : {self.best_fitness} / {self.formula.n_clauses} "
                  f"clauses  ({pct:.2f}%)")
            print(f"  Time   : {elapsed:.3f} s")

        return self.best_individual

    # ── Report ───────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print a human-readable summary of the best solution found."""
        asgn   = self.best_individual
        n      = self.formula.n_vars
        m      = self.formula.n_clauses
        pct    = 100 * self.best_fitness / m

        print("\n=== Best Assignment ===")
        print(f"  Satisfied : {self.best_fitness} / {m}  ({pct:.2f}%)")
        print(f"  Fully SAT : {'YES ✓' if self.best_fitness == m else 'NO'}")

        # Variable assignments
        row = []
        for i in range(n):
            val = "T" if asgn[i] == 1 else "F"
            row.append(f"x{i+1}={val}")
            if len(row) == 10:
                print("  " + "  ".join(row))
                row = []
        if row:
            print("  " + "  ".join(row))

        # Unsatisfied clauses
        unsat = [
            c for c in self.formula.clauses
            if not any(
                (lit > 0 and asgn[abs(lit) - 1] == 1) or
                (lit < 0 and asgn[abs(lit) - 1] == 0)
                for lit in c
            )
        ]
        if unsat:
            print(f"\n  Unsatisfied clauses ({len(unsat)}):")
            for c in unsat[:10]:
                lits = " ∨ ".join(
                    f"x{abs(l)}" if l > 0 else f"¬x{abs(l)}" for l in c
                )
                print(f"    ({lits})")
            if len(unsat) > 10:
                print(f"    … and {len(unsat)-10} more")


# =============================================================================
# Exact solver (brute force, small instances only)
# =============================================================================

def exact_max_sat(formula: CNF) -> tuple[list[int], int]:
    """
    Exact MAX-SAT by exhaustive search over all 2^n assignments.
    Feasible only for n ≤ ~20.
    """
    assert formula.n_vars <= 20, "Too many variables for brute force"
    n = formula.n_vars
    best_val,  best_asgn = 0, [0] * n

    for mask in range(1 << n):
        asgn = [(mask >> i) & 1 for i in range(n)]
        val  = formula.evaluate(asgn)
        if val > best_val:
            best_val, best_asgn = val, asgn

    return best_asgn, best_val


# =============================================================================
# Random-restart local search baseline
# =============================================================================

def random_walk_max_sat(
    formula: CNF,
    restarts: int = 50,
    steps:    int = 1000,
    p_random: float = 0.3,
    seed:     Optional[int] = 0,
) -> tuple[list[int], int]:
    """
    WalkSAT-style random-walk local search baseline.

    At each step:
    - Pick an unsatisfied clause at random.
    - With probability p_random: flip a random literal in the clause.
    - Otherwise: flip the literal that maximises the number of satisfied clauses.
    """
    rng      = random.Random(seed)
    n        = formula.n_vars
    best_val = 0
    best_asgn: list[int] = []

    for _ in range(restarts):
        asgn    = [rng.randint(0, 1) for _ in range(n)]
        current = formula.evaluate(asgn)

        if current > best_val:
            best_val, best_asgn = current, asgn[:]

        for __ in range(steps):
            if current == formula.n_clauses:
                break

            # Collect unsatisfied clauses
            unsat = [
                clause for clause in formula.clauses
                if not any(
                    (lit > 0 and asgn[abs(lit) - 1] == 1) or
                    (lit < 0 and asgn[abs(lit) - 1] == 0)
                    for lit in clause
                )
            ]
            if not unsat:
                break

            clause = rng.choice(unsat)

            if rng.random() < p_random:
                # Random flip (noise step — avoids local optima)
                lit = rng.choice(clause)
                asgn[abs(lit) - 1] ^= 1
            else:
                # Greedy flip: pick variable that satisfies the most clauses
                best_gain, best_var = -1, -1
                for lit in clause:
                    var = abs(lit) - 1
                    asgn[var] ^= 1
                    gain = formula.evaluate(asgn) - current
                    asgn[var] ^= 1
                    if gain > best_gain:
                        best_gain, best_var = gain, var
                if best_var >= 0:
                    asgn[best_var] ^= 1

            current = formula.evaluate(asgn)
            if current > best_val:
                best_val, best_asgn = current, asgn[:]

    return best_asgn, best_val


# =============================================================================
# Main demo
# =============================================================================

def main() -> None:
    SEP = "=" * 62

    # ── Example 1: Tiny satisfiable instance (exact verification) ────────
    print(SEP)
    print("  MAX-SAT — Genetic Algorithm")
    print(SEP)
    print("\n[Example 1]  Hand-crafted 5-variable instance")
    print("  Formula: (x1∨x2∨¬x3) ∧ (¬x1∨x3) ∧ (x2∨¬x4) ∧")
    print("           (¬x2∨x4∨x5) ∧ (x1∨¬x5) ∧ (¬x3∨x4) ∧ (x2∨x3∨¬x5)")

    f1 = CNF(n_vars=5, clauses=[
        [1,  2, -3],
        [-1, 3],
        [2, -4],
        [-2, 4, 5],
        [1, -5],
        [-3, 4],
        [2,  3, -5],
    ])

    cfg1 = GAConfig(pop_size=50, generations=100, p_mutation=0.05, seed=1)
    ga1  = GeneticMaxSAT(f1, cfg1)
    ga1.evolve(verbose=True, print_every=25)
    ga1.report()

    _, exact1 = exact_max_sat(f1)
    print(f"\n  Exact optimum : {exact1} / {f1.n_clauses}")
    print(f"  GA solution   : {ga1.best_fitness} / {f1.n_clauses}")

    # ── Example 2: Random 3-SAT near phase transition (ratio m/n ≈ 4.27) ─
    print(f"\n{SEP}")
    print("[Example 2]  Random 3-SAT  (n=20 vars, m=85 clauses — phase transition)")

    f2 = CNF.random_3sat(n=20, m=85, seed=42)
    print(f"  {f2}")

    cfg2 = GAConfig(
        pop_size=150, generations=400, p_mutation=0.02,
        tournament_k=7, elite_count=8, seed=42,
    )
    ga2 = GeneticMaxSAT(f2, cfg2)
    ga2.evolve(verbose=True, use_repair=True, print_every=100)
    ga2.report()

    _, walk2 = random_walk_max_sat(f2, restarts=50, steps=2000, seed=0)
    print(f"\n  WalkSAT baseline : {walk2} / {f2.n_clauses}")
    print(f"  GA solution      : {ga2.best_fitness} / {f2.n_clauses}")

    # ── Example 3: Harder 3-SAT (n=50) ───────────────────────────────────
    print(f"\n{SEP}")
    print("[Example 3]  Random 3-SAT  (n=50 vars, m=215 clauses)")

    f3 = CNF.random_3sat(n=50, m=215, seed=7)
    print(f"  {f3}")

    cfg3 = GAConfig(
        pop_size=300, generations=600, p_mutation=0.015,
        tournament_k=5, elite_count=15, seed=7,
    )
    ga3 = GeneticMaxSAT(f3, cfg3)
    ga3.evolve(verbose=True, use_repair=True, print_every=150)
    ga3.report()

    _, walk3 = random_walk_max_sat(f3, restarts=100, steps=3000, seed=0)
    pct_ga   = 100 * ga3.best_fitness   / f3.n_clauses
    pct_walk = 100 * walk3 / f3.n_clauses
    print(f"\n  WalkSAT baseline : {walk3} / {f3.n_clauses}  ({pct_walk:.1f}%)")
    print(f"  GA solution      : {ga3.best_fitness} / {f3.n_clauses}  ({pct_ga:.1f}%)")

    # ── Example 4: Weighted MAX-SAT feel — k-SAT clause length sweep ─────
    print(f"\n{SEP}")
    print("[Analysis]  Effect of clause length k on GA performance")
    print(f"  (n=30 vars, m=120 clauses, 1 run each)")
    print(f"\n  {'k':>4}  {'Best':>8}  {'/ m':>6}  {'%':>7}  {'vs WalkSAT':>12}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*12}")

    for k in [2, 3, 4, 5]:
        fk  = CNF.random_ksat(n=30, m=120, k=k, seed=99)
        cfk = GAConfig(pop_size=120, generations=300, p_mutation=0.02, seed=99)
        gak = GeneticMaxSAT(fk, cfk)
        gak.evolve(verbose=False)

        _, wk  = random_walk_max_sat(fk, restarts=30, steps=1000, seed=0)
        pct    = 100 * gak.best_fitness / fk.n_clauses
        delta  = gak.best_fitness - wk
        sign   = f"+{delta}" if delta >= 0 else str(delta)
        print(f"  {k:>4}  {gak.best_fitness:>8}  {fk.n_clauses:>6}  "
              f"{pct:>6.2f}%  {sign:>12}")

    print(f"\n{SEP}")
    print("Done.")


if __name__ == "__main__":
    main()