"""
Max-Cut Problem — Genetic Algorithm Solution
=============================================
The Max-Cut problem: Given an undirected weighted graph G = (V, E),
partition the vertices into two disjoint sets S and S̄ to maximize
the total weight of edges crossing the partition.

This is NP-hard, so we use a Genetic Algorithm (GA) as a metaheuristic.

Representation
--------------
Each individual is a binary string of length |V|.
  0 → vertex belongs to set S
  1 → vertex belongs to set S̄

Fitness = total weight of edges (u, v) where assignment[u] ≠ assignment[v]

Operators
---------
  Selection   : Tournament selection
  Crossover   : Uniform crossover (each gene copied from either parent)
  Mutation    : Bit-flip with probability p_mut per gene
  Elitism     : Top k individuals survive unchanged each generation
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Graph representation
# ---------------------------------------------------------------------------

@dataclass
class Graph:
    n_vertices: int
    edges: list[tuple[int, int, float]] = field(default_factory=list)   # (u, v, weight)

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        assert 0 <= u < self.n_vertices and 0 <= v < self.n_vertices
        self.edges.append((u, v, weight))

    def cut_value(self, assignment: list[int]) -> float:
        """Return the total weight of edges crossing the cut."""
        return sum(
            w for u, v, w in self.edges if assignment[u] != assignment[v]
        )

    @staticmethod
    def random_graph(n: int, m: int, seed: Optional[int] = None) -> "Graph":
        """Generate a random graph with n vertices and m edges."""
        rng = random.Random(seed)
        g = Graph(n_vertices=n)
        possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        for u, v in rng.sample(possible, min(m, len(possible))):
            g.add_edge(u, v, weight=round(rng.uniform(1.0, 10.0), 2))
        return g


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    pop_size:        int   = 100      # population size
    generations:     int   = 300      # number of generations
    p_crossover:     float = 0.85     # crossover probability
    p_mutation:      float = 0.02     # per-gene mutation probability
    tournament_k:    int   = 5        # tournament selection size
    elite_count:     int   = 5        # elitism — top k survivors
    seed:            Optional[int] = 42


class GeneticMaxCut:
    def __init__(self, graph: Graph, config: GAConfig = GAConfig()):
        self.graph  = graph
        self.config = config
        self.rng    = random.Random(config.seed)
        self.n      = graph.n_vertices

        self.population: list[list[int]] = []
        self.fitnesses:  list[float]     = []
        self.best_individual: list[int]  = []
        self.best_fitness: float         = 0.0
        self.history: list[float]        = []   # best fitness per generation

    # --- Initialisation -------------------------------------------------

    def _random_individual(self) -> list[int]:
        return [self.rng.randint(0, 1) for _ in range(self.n)]

    def _init_population(self) -> None:
        self.population = [self._random_individual() for _ in range(self.config.pop_size)]
        self.fitnesses  = [self.graph.cut_value(ind) for ind in self.population]

    # --- Selection -------------------------------------------------------

    def _tournament_select(self) -> list[int]:
        """Return winner of a random k-tournament (higher fitness wins)."""
        k        = self.config.tournament_k
        indices  = self.rng.sample(range(self.config.pop_size), k)
        winner   = max(indices, key=lambda i: self.fitnesses[i])
        return self.population[winner][:]   # copy

    # --- Crossover -------------------------------------------------------

    def _uniform_crossover(self, p1: list[int], p2: list[int]) -> tuple[list[int], list[int]]:
        """Uniform crossover: each gene is taken from either parent with equal probability."""
        if self.rng.random() > self.config.p_crossover:
            return p1[:], p2[:]
        mask = [self.rng.randint(0, 1) for _ in range(self.n)]
        child1 = [p1[i] if mask[i] == 0 else p2[i] for i in range(self.n)]
        child2 = [p2[i] if mask[i] == 0 else p1[i] for i in range(self.n)]
        return child1, child2

    # --- Mutation --------------------------------------------------------

    def _mutate(self, individual: list[int]) -> list[int]:
        """Flip each bit with probability p_mutation."""
        p = self.config.p_mutation
        return [gene ^ 1 if self.rng.random() < p else gene for gene in individual]

    # --- Main loop -------------------------------------------------------

    def evolve(self, verbose: bool = True) -> list[int]:
        cfg = self.config
        self._init_population()

        # Track initial best
        best_idx          = max(range(cfg.pop_size), key=lambda i: self.fitnesses[i])
        self.best_fitness = self.fitnesses[best_idx]
        self.best_individual = self.population[best_idx][:]

        if verbose:
            print(f"{'Gen':>6}  {'Best':>12}  {'Avg':>12}  {'Time (s)':>10}")
            print("-" * 48)

        t0 = time.perf_counter()

        for gen in range(1, cfg.generations + 1):
            # Elitism: keep top-k
            elite_indices = sorted(
                range(cfg.pop_size), key=lambda i: self.fitnesses[i], reverse=True
            )[: cfg.elite_count]
            new_population = [self.population[i][:] for i in elite_indices]

            # Fill rest via selection → crossover → mutation
            while len(new_population) < cfg.pop_size:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._uniform_crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < cfg.pop_size:
                    new_population.append(self._mutate(c2))

            self.population = new_population
            self.fitnesses  = [self.graph.cut_value(ind) for ind in self.population]

            # Update best
            best_idx     = max(range(cfg.pop_size), key=lambda i: self.fitnesses[i])
            gen_best     = self.fitnesses[best_idx]
            gen_avg      = sum(self.fitnesses) / cfg.pop_size
            self.history.append(gen_best)

            if gen_best > self.best_fitness:
                self.best_fitness    = gen_best
                self.best_individual = self.population[best_idx][:]

            if verbose and (gen % 50 == 0 or gen == 1):
                elapsed = time.perf_counter() - t0
                print(f"{gen:>6}  {gen_best:>12.2f}  {gen_avg:>12.2f}  {elapsed:>10.3f}")

        if verbose:
            elapsed = time.perf_counter() - t0
            print("-" * 48)
            print(f"\n✓ Best cut value : {self.best_fitness:.2f}")
            print(f"  Total time     : {elapsed:.3f} s")

        return self.best_individual

    def report(self) -> None:
        """Print a human-readable summary of the best solution found."""
        assignment = self.best_individual
        set_s  = [v for v in range(self.n) if assignment[v] == 0]
        set_sb = [v for v in range(self.n) if assignment[v] == 1]
        cut_edges = [(u, v, w) for u, v, w in self.graph.edges if assignment[u] != assignment[v]]

        print("\n=== Best Solution ===")
        print(f"  Cut value : {self.best_fitness:.2f}")
        print(f"  Set S     : {set_s}")
        print(f"  Set S̄     : {set_sb}")
        print(f"  Cut edges ({len(cut_edges)}):")
        for u, v, w in cut_edges:
            print(f"    ({u} — {v})  weight={w:.2f}")


# ---------------------------------------------------------------------------
# Greedy baseline (for comparison)
# ---------------------------------------------------------------------------

def greedy_max_cut(graph: Graph) -> tuple[list[int], float]:
    """
    Simple greedy heuristic: assign each vertex to the partition that
    maximises the current cut value.
    """
    assignment = [0] * graph.n_vertices
    for v in range(graph.n_vertices):
        val0 = graph.cut_value(assignment)
        assignment[v] = 1
        val1 = graph.cut_value(assignment)
        if val0 >= val1:
            assignment[v] = 0   # revert if no improvement
    return assignment, graph.cut_value(assignment)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MAX-CUT — Genetic Algorithm Demo")
    print("=" * 60)

    # ── Small hand-crafted example ──────────────────────────────────────
    print("\n[Example 1] Petersen-like graph (10 vertices)")
    g_small = Graph(n_vertices=10)
    edges_small = [
        (0,1,3),(0,4,2),(1,2,4),(1,6,1),(2,3,2),(2,7,5),
        (3,4,3),(3,8,2),(4,9,4),(5,6,3),(5,9,2),(6,7,4),
        (7,8,1),(8,9,3),(0,5,2),(1,7,3),(2,9,1),(3,6,4),(4,8,2),
    ]
    for u, v, w in edges_small:
        g_small.add_edge(u, v, w)

    cfg_small = GAConfig(pop_size=60, generations=150, p_mutation=0.03, seed=7)
    ga_small  = GeneticMaxCut(g_small, cfg_small)
    ga_small.evolve(verbose=True)
    ga_small.report()

    _, greedy_val = greedy_max_cut(g_small)
    print(f"\n  Greedy baseline : {greedy_val:.2f}")
    print(f"  GA solution     : {ga_small.best_fitness:.2f}")

    # ── Larger random graph ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Example 2] Random graph (50 vertices, 200 edges)")
    g_large = Graph.random_graph(n=50, m=200, seed=99)

    cfg_large = GAConfig(
        pop_size=150, generations=500, p_mutation=0.015,
        tournament_k=7, elite_count=8, seed=42,
    )
    ga_large = GeneticMaxCut(g_large, cfg_large)
    ga_large.evolve(verbose=True)

    _, greedy_val_l = greedy_max_cut(g_large)
    print(f"\n  Greedy baseline : {greedy_val_l:.2f}")
    print(f"  GA solution     : {ga_large.best_fitness:.2f}")
    improvement = 100 * (ga_large.best_fitness - greedy_val_l) / max(greedy_val_l, 1)