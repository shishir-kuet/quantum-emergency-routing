"""Classical baselines -- the yardstick every quantum result is measured against.

Three roles, and it is worth keeping them straight:

**Ground truth.**
:mod:`~qroute.classical.bruteforce` (exact QUBO ground state),
:mod:`~qroute.classical.dijkstra` (exact shortest path) and the exact modes of
:mod:`~qroute.classical.tsp` and :mod:`~qroute.classical.assignment` give
certified optima.

Only these may be used as the denominator of an approximation ratio.

**Fair competition.**
:mod:`~qroute.classical.annealing` is the metaheuristic QAOA is genuinely
competing with: same QUBO, same anytime behaviour, no structural hints.

**Encoding verification.**
Every solver here decodes its answer through the formulation that built the
QUBO. A classical optimum that decodes as infeasible, or whose QUBO energy
disagrees with its own cost, is a bug in the encoding.

All solvers return a :class:`~qroute.classical.base.ClassicalResult`, whose
``optimal`` flag records whether the answer is provably optimal.
"""

from __future__ import annotations

from .annealing import (
    AnnealingResult,
    solve_formulation_annealing,
    solve_qubo_annealing,
)

from .assignment import (
    greedy_assignment,
    hungarian_assignment,
    solve_assignment,
)

from .base import (
    ClassicalResult,
    Stopwatch,
)

from .bruteforce import (
    SpectrumResult,
    enumerate_assignments,
    solve_formulation_bruteforce,
    solve_qubo_bruteforce,
)

from .dijkstra import (
    dijkstra_path,
    solve_path_problem,
    solve_shortest_path,
)

from .tsp import (
    TSP_METHODS,
    brute_force_tour,
    nearest_neighbour_tour,
    or_tools_tour,
    solve_tsp,
    two_opt_tour,
)


# ---------------------------------------------------------------------------
# Compatibility aliases used by qroute.pipeline
# ---------------------------------------------------------------------------

# Rung 1 exact shortest-path reference
solve_dijkstra = solve_shortest_path

# Exact QUBO baseline
solve_bruteforce = solve_formulation_bruteforce

# Simulated-annealing QUBO baseline
solve_simulated_annealing = solve_formulation_annealing


__all__ = [
    # shared
    "ClassicalResult",
    "Stopwatch",

    # exact QUBO
    "SpectrumResult",
    "enumerate_assignments",
    "solve_qubo_bruteforce",
    "solve_formulation_bruteforce",
    "solve_bruteforce",

    # annealing
    "AnnealingResult",
    "solve_qubo_annealing",
    "solve_formulation_annealing",
    "solve_simulated_annealing",

    # rung 1
    "dijkstra_path",
    "solve_path_problem",
    "solve_shortest_path",
    "solve_dijkstra",

    # rung 2
    "TSP_METHODS",
    "brute_force_tour",
    "nearest_neighbour_tour",
    "two_opt_tour",
    "or_tools_tour",
    "solve_tsp",

    # rung 3
    "hungarian_assignment",
    "greedy_assignment",
    "solve_assignment",
]