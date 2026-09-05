r"""Classical tour solvers: exact, greedy, local search, and OR-Tools.

Four solvers, because at these sizes they tell four different stories:

* :func:`brute_force_tour` -- exact, :math:`(n-1)!` tours. At :math:`n = 6` that
  is 120 tours and takes microseconds, so on the instances a simulator can hold,
  the "hard" problem is trivially solvable. Saying that out loud is more useful
  than hiding it: the interest is in whether QAOA *reproduces* the optimum, not
  in beating a method that already has it.
* :func:`nearest_neighbour_tour` -- the greedy baseline. Fast, and typically
  10-25% above optimal, which makes it a fair yardstick for "did the quantum
  result beat naive".
* :func:`two_opt_tour` -- greedy plus local search, the realistic industrial
  baseline. This is the one QAOA actually has to beat to be interesting.
* :func:`or_tools_tour` -- Google's routing solver, optional. Import is guarded
  so the package works without it.

A note on 2-opt and one-way streets
-----------------------------------
The textbook 2-opt gain formula assumes a symmetric cost matrix, because
reversing a tour segment leaves the reversed part's internal cost unchanged. On a
real road network with one-way streets that is false: the reversed segment is
driven the other way, at a different cost. So :func:`two_opt_tour` re-evaluates
the whole candidate tour instead of applying the gain shortcut. That is
:math:`O(n)` per move rather than :math:`O(1)`, which at :math:`n \le 10` costs
nothing and buys correctness on asymmetric instances.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.instance import RoutingInstance
from ..exceptions import MissingDependencyError, SolverError
from ..logging_utils import get_logger
from ..qubo.tsp import TSPFormulation
from .base import ClassicalResult, Stopwatch

__all__ = [
    "brute_force_tour",
    "nearest_neighbour_tour",
    "two_opt_tour",
    "or_tools_tour",
    "solve_tsp",
    "TSP_METHODS",
]

_LOG = get_logger(__name__)

#: Above this, ``(n-1)!`` enumeration stops being instant.
MAX_EXACT_N = 10

TSP_METHODS = ("auto", "bruteforce", "nearest_neighbour", "two_opt", "ortools")


def _depot_of(instance: RoutingInstance) -> int:
    return int(instance.depot)


def brute_force_tour(instance: RoutingInstance) -> Tuple[Tuple[int, ...], float, float]:
    """Exact optimal tour by enumerating every permutation of the non-depot stops.

    Returns ``(order, cost, seconds)``. All :math:`(n-1)!` orders are tried, not
    :math:`(n-1)!/2`: halving assumes a symmetric matrix, and on a one-way street
    network a tour and its reverse genuinely cost different amounts.
    """
    n = instance.n
    if n > MAX_EXACT_N:
        raise SolverError(
            f"Exact enumeration over {n} locations means {n - 1}! tours; the limit "
            f"is {MAX_EXACT_N}. Use two_opt instead."
        )
    depot = _depot_of(instance)
    others = [index for index in range(n) if index != depot]

    best_order: Tuple[int, ...] = ()
    best_cost = float("inf")
    with Stopwatch() as timer:
        for permutation in permutations(others):
            order = (depot,) + permutation
            cost = instance.tour_cost(order, closed=True)
            if cost < best_cost:
                best_cost = cost
                best_order = order
    return best_order, float(best_cost), timer.seconds


def nearest_neighbour_tour(
    instance: RoutingInstance, *, start: Optional[int] = None
) -> Tuple[Tuple[int, ...], float, float]:
    """Greedy tour: always drive to the nearest unvisited stop."""
    depot = _depot_of(instance) if start is None else int(start)
    remaining = {index for index in range(instance.n) if index != depot}
    order: List[int] = [depot]

    with Stopwatch() as timer:
        current = depot
        while remaining:
            # `here` is bound as a default argument rather than captured, so the
            # key function cannot pick up a later value of `current`.
            nearest = min(
                remaining,
                key=lambda index, here=current: instance.cost_between(here, index),
            )
            order.append(nearest)
            remaining.discard(nearest)
            current = nearest
    return tuple(order), instance.tour_cost(order, closed=True), timer.seconds


def two_opt_tour(
    instance: RoutingInstance,
    *,
    initial: Optional[Sequence[int]] = None,
    max_passes: int = 100,
) -> Tuple[Tuple[int, ...], float, float]:
    """2-opt local search from a starting tour (nearest-neighbour by default).

    Segment reversals are evaluated by full re-costing so the move is valid on
    asymmetric matrices -- see the module docstring. The depot stays at position
    0 throughout, since rotating a closed tour cannot change its cost.
    """
    depot = _depot_of(instance)
    if initial is None:
        order = list(nearest_neighbour_tour(instance)[0])
    else:
        order = [int(index) for index in initial]
        if sorted(order) != list(range(instance.n)):
            raise SolverError("Initial tour must be a permutation of all locations")
        if order[0] != depot:
            pivot = order.index(depot)
            order = order[pivot:] + order[:pivot]

    best_cost = instance.tour_cost(order, closed=True)
    passes = 0

    with Stopwatch() as timer:
        improved = True
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(1, len(order) - 1):
                for j in range(i + 1, len(order)):
                    candidate = order[:i] + order[i : j + 1][::-1] + order[j + 1 :]
                    cost = instance.tour_cost(candidate, closed=True)
                    if cost < best_cost - 1e-12:
                        order, best_cost = candidate, cost
                        improved = True

    return tuple(order), float(best_cost), timer.seconds


def or_tools_tour(
    instance: RoutingInstance,
    *,
    time_limit_s: int = 5,
    scale: float = 1000.0,
) -> Tuple[Tuple[int, ...], float, float]:
    """Solve with Google OR-Tools, if it is installed.

    OR-Tools works in integers, so costs are multiplied by *scale* and rounded;
    at the default that preserves millisecond resolution on travel times. The
    returned cost is recomputed from the original float matrix, so the rounding
    never leaks into reported results.
    """
    try:  # pragma: no cover - exercised only when the optional dep is present
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("ortools", "the OR-Tools TSP baseline") from exc

    n = instance.n
    depot = _depot_of(instance)
    integer_costs = np.rint(instance.cost * scale).astype(np.int64)

    with Stopwatch() as timer:
        manager = pywrapcp.RoutingIndexManager(n, 1, depot)
        routing = pywrapcp.RoutingModel(manager)

        def transit(from_index: int, to_index: int) -> int:
            return int(
                integer_costs[
                    manager.IndexToNode(from_index), manager.IndexToNode(to_index)
                ]
            )

        transit_callback = routing.RegisterTransitCallback(transit)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

        parameters = pywrapcp.DefaultRoutingSearchParameters()
        parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        parameters.time_limit.FromSeconds(int(time_limit_s))

        assignment = routing.SolveWithParameters(parameters)
        if assignment is None:
            raise SolverError("OR-Tools found no solution")

        order: List[int] = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            order.append(int(manager.IndexToNode(index)))
            index = assignment.Value(routing.NextVar(index))

    return tuple(order), instance.tour_cost(order, closed=True), timer.seconds


def solve_tsp(
    formulation: TSPFormulation,
    *,
    method: str = "auto",
    **kwargs: Any,
) -> ClassicalResult:
    """Solve rung 2 classically and decode through *formulation*.

    ``method="auto"`` picks exact enumeration when the instance is small enough
    and 2-opt otherwise, so ``optimal`` is set truthfully in both cases.
    """
    if method not in TSP_METHODS:
        raise SolverError(f"method must be one of {TSP_METHODS}, got {method!r}")

    instance = formulation.instance
    chosen = method
    if chosen == "auto":
        chosen = "bruteforce" if instance.n <= MAX_EXACT_N else "two_opt"
        _LOG.debug("method='auto' resolved to %r for n=%d", chosen, instance.n)

    if chosen == "bruteforce":
        order, cost, seconds = brute_force_tour(instance)
        exact = True
    elif chosen == "nearest_neighbour":
        order, cost, seconds = nearest_neighbour_tour(instance, **kwargs)
        exact = False
    elif chosen == "two_opt":
        order, cost, seconds = two_opt_tour(instance, **kwargs)
        exact = False
    elif chosen == "ortools":
        order, cost, seconds = or_tools_tour(instance, **kwargs)
        # OR-Tools is exact only if it proves optimality, which we do not query.
        exact = False
    else:  # pragma: no cover - guarded above
        raise SolverError(f"Unhandled method {chosen!r}")

    solution = formulation.decode(formulation.encode_route(order), qiskit_order=False)
    details: Dict[str, Any] = {
        "method": chosen,
        "order": order,
        "tour_cost": cost,
        "asymmetry": instance.asymmetry(),
    }
    if solution.objective is not None and abs(solution.objective - cost) > 1e-6:
        # Two independent cost computations disagreeing means encode/decode has
        # lost information. Surface it loudly rather than picking one.
        _LOG.error(
            "Tour cost mismatch: solver says %.6f, decoded solution says %.6f",
            cost,
            solution.objective,
        )
        details["cost_mismatch"] = True

    return ClassicalResult(
        solver=f"tsp:{chosen}",
        objective=solution.objective if solution.objective is not None else cost,
        seconds=seconds,
        solution=solution,
        optimal=exact,
        details=details,
    )
