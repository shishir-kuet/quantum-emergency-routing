"""Classical shortest path: Dijkstra, and the sanity check it enables.

Dijkstra solves rung 1 exactly in near-linear time. That is not a defeat for the
quantum side, it is the *point*: because the classical optimum is known with
certainty, this rung becomes a test harness. Encode the Dijkstra path, evaluate
it on the QUBO, and it must come out feasible and at the ground-state energy. If
it doesn't, the bug is in the formulation, not the optimiser -- and finding that
out here is far cheaper than finding it out after a hardware run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import networkx as nx

from ..exceptions import SolverError
from ..logging_utils import get_logger
from ..qubo.shortest_path import PathProblem, ShortestPathFormulation
from .base import ClassicalResult, Stopwatch

__all__ = [
    "dijkstra_path",
    "solve_path_problem",
    "solve_shortest_path",
]

_LOG = get_logger(__name__)


def dijkstra_path(
    graph: nx.Graph,
    source: int,
    target: int,
    *,
    weight: str = "travel_time",
) -> Tuple[Tuple[int, ...], float, float]:
    """Cheapest path through the **full** graph as ``(path, cost, seconds)``.

    Note the *weight* argument is passed straight to NetworkX, which silently
    treats a missing attribute as 1. Run
    :func:`qroute.data.graph_io.validate_edge_weight` on the graph first; on a
    travel-time graph an unweighted edge becomes a one-second shortcut across the
    city and the answer is quietly nonsense.
    """
    if source not in graph or target not in graph:
        raise SolverError(f"Both {source} and {target} must be nodes of the graph")
    try:
        with Stopwatch() as timer:
            cost, path = nx.single_source_dijkstra(
                graph, source, target=target, weight=weight
            )
    except nx.NetworkXNoPath as exc:
        raise SolverError(f"No path from {source} to {target}") from exc
    return tuple(int(node) for node in path), float(cost), timer.seconds


def solve_path_problem(
    problem: PathProblem,
) -> Tuple[Tuple[int, ...], float, float]:
    """Cheapest path using only the problem's candidate segments.

    Restricting Dijkstra to the candidate set answers "what is the best the QUBO
    could possibly do", which is the right comparison for a quantum solver that
    only ever sees those segments.
    """
    graph = nx.DiGraph()
    for u, v, cost in problem.edges:
        graph.add_edge(u, v, cost=cost)
    if problem.source not in graph or problem.target not in graph:
        raise SolverError("Problem endpoints are missing from the candidate segments")
    return dijkstra_path(graph, problem.source, problem.target, weight="cost")


def solve_shortest_path(
    formulation: ShortestPathFormulation,
    *,
    full_graph: Optional[nx.Graph] = None,
) -> ClassicalResult:
    """Solve rung 1 exactly and cross-check the encoding.

    Pass *full_graph* to additionally verify that the candidate set really did
    capture the city-wide optimum -- it should, since the candidates are built
    from the k shortest paths, but confirming it costs one Dijkstra run and
    protects against a mis-built :class:`~qroute.qubo.shortest_path.PathProblem`.
    """
    problem = formulation.problem
    path, cost, seconds = solve_path_problem(problem)

    solution = formulation.decode(formulation.encode_path(path), qiskit_order=False)
    details: Dict[str, Any] = {
        "n_segments": len(path) - 1,
        "path": path,
        "candidate_cost": cost,
    }

    if not solution.feasible:
        # The exact classical answer must decode as feasible. If it doesn't, the
        # formulation's constraints are wrong -- shout rather than report a number.
        _LOG.error(
            "The Dijkstra path decoded as INFEASIBLE (%s). This is an encoding bug, "
            "not a solver result.",
            "; ".join(solution.violations),
        )
        details["encoding_error"] = list(solution.violations)

    if full_graph is not None:
        try:
            global_path, global_cost, _ = dijkstra_path(
                full_graph, problem.source, problem.target, weight=problem.weight
            )
        except SolverError:
            _LOG.warning("Could not verify against the full graph: no path found")
        else:
            details["global_cost"] = global_cost
            details["global_path"] = global_path
            details["candidates_contain_optimum"] = bool(
                abs(global_cost - cost) < 1e-6
            )
            if not details["candidates_contain_optimum"]:
                _LOG.warning(
                    "Candidate segments miss the city-wide optimum: %.3f vs %.3f. "
                    "Increase k_paths or max_edges.",
                    cost,
                    global_cost,
                )

    return ClassicalResult(
        solver="dijkstra",
        objective=solution.objective if solution.feasible else cost,
        seconds=seconds,
        solution=solution,
        optimal=True,
        details=details,
    )
