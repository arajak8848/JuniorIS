"""
qubo_genetic.py  —  QUBO via Genetic Algorithm
===============================================
Solves the SAME QUBO instance defined in qubo_problem.py.

Run:
    python qubo_genetic.py

Representation
--------------
Each individual is a binary vector x ∈ {0,1}^N.
Fitness = −f(x) = −x^T Q x   (negated so we MAXIMISE fitness = MINIMISE QUBO).

GA Operators
------------
  Selection  : Tournament (k=5)
  Crossover  : Uniform crossover (each gene drawn from either parent)
  Mutation   : Per-gene bit-flip at probability p_mut
  Elitism    : Top-e individuals copied unchanged each generation
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

from qubo_problem import N, Q, evaluate, BEST_KNOWN, print_problem, print_result


# =============================================================================
# Config
# =============================================================================

@dataclass
class GAConfig:
    pop_size:     int   = 200
    generations:  int   = 500
    p_crossover:  float = 0.85
    p_mutation:   float = 0.05   # slightly higher than usual — small N
    tournament_k: int   = 5
    elite_count:  int   = 10
    seed:         int   = 42


# =============================================================================
# Genetic Algorithm
# =============================================================================

class GeneticQUBO:
    """Genetic Algorithm solver for QUBO minimisation."""

    def __init__(self, config: GAConfig = GAConfig()) -> None:
        self.config = config
        self.rng    = random.Random(config.seed)

        self.population:      list[list[int]] = []
        self.fitnesses:       list[float]     = []   # negated QUBO value
        self.best_x:          list[int]       = []
        self.best_val:        float           = float("inf")
        self.history:         list[float]     = []   # best QUBO value per gen

    # ── Initialisation ───────────────────────────────────────────────────────

    def _random_ind(self) -> list[int]:
        return [self.rng.randint(0, 1) for _ in range(N)]

    def _init_pop(self) -> None:
        self.population = [self._random_ind() for _ in range(self.config.pop_size)]
        self.fitnesses  = [-evaluate(ind) for ind in self.population]

    # ── Operators ────────────────────────────────────────────────────────────

    def _tournament(self) -> list[int]:
        k      = self.config.tournament_k
        idx    = self.rng.sample(range(self.config.pop_size), k)
        winner = max(idx, key=lambda i: self.fitnesses[i])
        return self.population[winner][:]

    def _crossover(self, p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
        if self.rng.random() > self.config.p_crossover:
            return p1[:], p2[:]
        mask = [self.rng.randint(0, 1) for _ in range(N)]
        c1   = [p1[i] if mask[i] == 0 else p2[i] for i in range(N)]
        c2   = [p2[i] if mask[i] == 0 else p1[i] for i in range(N)]
        return c1, c2

    def _mutate(self, ind: list[int]) -> list[int]:
        p = self.config.p_mutation
        return [b ^ 1 if self.rng.random() < p else b for b in ind]

    # ── Main loop ────────────────────────────────────────────────────────────

    def evolve(self, verbose: bool = True, print_every: int = 100) -> list[int]:
        cfg = self.config
        self._init_pop()

        best_idx      = max(range(cfg.pop_size), key=lambda i: self.fitnesses[i])
        self.best_val = -self.fitnesses[best_idx]
        self.best_x   = self.population[best_idx][:]

        if verbose:
            print(f"{'Gen':>6}  {'Best f(x)':>12}  {'Avg f(x)':>12}  "
                  f"{'Gap':>8}  {'Time(s)':>9}")
            print("─" * 54)

        t0 = time.perf_counter()

        for gen in range(1, cfg.generations + 1):

            # Elitism
            elite = sorted(range(cfg.pop_size),
                           key=lambda i: self.fitnesses[i],
                           reverse=True)[:cfg.elite_count]
            new_pop = [self.population[i][:] for i in elite]

            # Fill population
            while len(new_pop) < cfg.pop_size:
                c1, c2 = self._crossover(self._tournament(), self._tournament())
                new_pop.append(self._mutate(c1))
                if len(new_pop) < cfg.pop_size:
                    new_pop.append(self._mutate(c2))

            self.population = new_pop
            self.fitnesses  = [-evaluate(ind) for ind in self.population]

            best_idx  = max(range(cfg.pop_size), key=lambda i: self.fitnesses[i])
            gen_best  = -self.fitnesses[best_idx]
            gen_avg   = -sum(self.fitnesses) / cfg.pop_size
            self.history.append(gen_best)

            if gen_best < self.best_val:
                self.best_val = gen_best
                self.best_x   = self.population[best_idx][:]

            if verbose and (gen % print_every == 0 or gen == 1):
                gap     = self.best_val - BEST_KNOWN
                elapsed = time.perf_counter() - t0
                print(f"{gen:>6}  {gen_best:>12.2f}  {gen_avg:>12.2f}  "
                      f"{gap:>+8.2f}  {elapsed:>9.3f}")

        if verbose:
            elapsed = time.perf_counter() - t0
            print("─" * 54)
            print(f"\n  Best f(x) = {self.best_val:.2f}  "
                  f"(optimum = {BEST_KNOWN:.2f}, gap = {self.best_val - BEST_KNOWN:+.2f})")
            print(f"  Time      = {elapsed:.3f} s")

        return self.best_x


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    SEP = "=" * 60
    print(SEP)
    print("  QUBO  —  Genetic Algorithm")
    print(SEP)
    print_problem()
    print(f"\n{'─'*60}")

    # ── Standard run ─────────────────────────────────────────────────────
    cfg = GAConfig(pop_size=200, generations=500, p_mutation=0.05, seed=42)
    ga  = GeneticQUBO(cfg)

    t0   = time.perf_counter()
    best = ga.evolve(verbose=True, print_every=100)
    t1   = time.perf_counter() - t0

    print_result(best, "Genetic Algorithm", ga.best_val, t1)

    # ── Hyperparameter sweep ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Hyperparameter sweep  (pop × generations):")
    print(f"  {'Config':<28}  {'Best f(x)':>10}  {'Gap':>8}  {'Time (ms)':>12}")
    print(f"  {'-'*28}  {'-'*10}  {'-'*8}  {'-'*12}")

    for pop, gens, pmut in [
        (50,  200, 0.05),
        (100, 300, 0.05),
        (200, 500, 0.05),
        (200, 500, 0.10),
        (300, 800, 0.03),
    ]:
        t0  = time.perf_counter()
        ga_ = GeneticQUBO(GAConfig(pop_size=pop, generations=gens,
                                   p_mutation=pmut, seed=42))
        bx_ = ga_.evolve(verbose=False)
        bv_ = ga_.best_val
        ms  = (time.perf_counter() - t0) * 1000
        gap = bv_ - BEST_KNOWN
        opt = " ✓" if abs(gap) < 1e-6 else ""
        label = f"pop={pop}, gen={gens}, pmut={pmut}"
        print(f"  {label:<28}  {bv_:>10.2f}  {gap:>+8.2f}  {ms:>10.1f} ms{opt}")

    print(f"\n  Exact optimum : {BEST_KNOWN:.2f}")
    print(SEP)


if __name__ == "__main__":
    main()