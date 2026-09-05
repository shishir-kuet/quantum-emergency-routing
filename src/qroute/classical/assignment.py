r"""Classical dispatch: the Hungarian algorithm, and its honest limits.

When there are as many ambulances as incidents, rung 3 *is* the linear assignment
problem, and :func:`scipy.optimize.linear_sum_assignment` solves it optimally in
:math:`O(n^3)`. That makes it the ground truth for
``vehicle_mode="exactly_one"``: QAOA's answer must match it exactly.

Fewer ambulances than incidents
-------------------------------
:func:`hungarian_assignment` handles that by **replicating** each vehicle
:math:`\lceil C/V \rceil` times before running the assignment, which yields the
minimum total response time subject to a per-vehicle cap. It is exact for *that*
problem -- but note carefully that it is a different problem from the one the QUBO
encodes in ``vehicle_mode="balance"``, where the load term is soft and can be
paid off by a large enough travel-time saving. So:

* ``exactly_one`` (V = C): Hungarian is the exact optimum. ``optimal=True``.
* ``balance`` (V < C): Hungarian is a strong reference but a *different*
  objective, so it is reported with ``optimal=False`` and the two objectives are
  both recorded. Do not call the ratio between them an approximation ratio.

Getting this distinction wrong is the easiest way to publish a quantum result
that looks better or worse than it is.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..exceptions import SolverError
from ..logging_utils import get_logger
from ..qubo.assignment import AssignmentFormulation
from .base import ClassicalResult, Stopwatch

__all__ = ["hungarian_assignment", "greedy_assignment", "solve_assignment"]

_LOG = get_logger(__name__)


def hungarian_assignment(
    costs: np.ndarray, *, capacity: Optional[int] = None
) -> Tuple[Dict[int, int], float, float]:
    """Minimum-total-cost assignment of columns (incidents) to rows (vehicles).

    Parameters
    ----------
    costs:
        A ``(V, C)`` matrix of response times.
    capacity:
        Maximum incidents per vehicle. Defaults to ``ceil(C / V)``, the smallest
        cap for which a solution exists.

    Returns
    -------
    ``({column: row}, total_cost, seconds)`` with local (0-based) indices.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover - scipy is a hard dependency
        raise SolverError("scipy is required for the Hungarian baseline") from exc

    matrix = np.asarray(costs, dtype=float)
    if matrix.ndim != 2:
        raise SolverError(f"costs must be 2-D, got shape {matrix.shape}")
    n_vehicles, n_incidents = matrix.shape
    if n_vehicles == 0 or n_incidents == 0:
        raise SolverError("costs must have at least one vehicle and one incident")

    cap = int(capacity) if capacity is not None else ceil(n_incidents / n_vehicles)
    if cap * n_vehicles < n_incidents:
        raise SolverError(
            f"capacity {cap} x {n_vehicles} vehicle(s) cannot cover {n_incidents} "
            f"incident(s)"
        )

    # Replicate each vehicle `cap` times so a rectangular assignment can give one
    # vehicle several incidents while still forbidding more than `cap`.
    expanded = np.repeat(matrix, cap, axis=0)
    owner = np.repeat(np.arange(n_vehicles), cap)

    with Stopwatch() as timer:
        rows, columns = linear_sum_assignment(expanded)

    assignments = {int(column): int(owner[row]) for row, column in zip(rows, columns)}
    total = float(expanded[rows, columns].sum())
    return assignments, total, timer.seconds


def greedy_assignment(costs: np.ndarray) -> Tuple[Dict[int, int], float, float]:
    """Assign each incident to its nearest vehicle, ignoring load entirely.

    Kept as the "what a dispatcher does under pressure" baseline, and because the
    pile-up it produces is the clearest illustration of why the QUBO needs a
    vehicle-side term at all.
    """
    matrix = np.asarray(costs, dtype=float)
    with Stopwatch() as timer:
        choices = matrix.argmin(axis=0)
    assignments = {int(column): int(row) for column, row in enumerate(choices)}
    total = float(matrix[choices, np.arange(matrix.shape[1])].sum())
    return assignments, total, timer.seconds


def solve_assignment(
    formulation: AssignmentFormulation,
    *,
    method: str = "hungarian",
    capacity: Optional[int] = None,
) -> ClassicalResult:
    """Solve rung 3 classically and decode through *formulation*."""
    if method not in ("hungarian", "greedy"):
        raise SolverError(f"method must be 'hungarian' or 'greedy', got {method!r}")

    costs = formulation.cost_matrix()
    if method == "hungarian":
        cap = capacity
        if cap is None and formulation.vehicle_mode == "exactly_one":
            cap = 1
        local, total, seconds = hungarian_assignment(costs, capacity=cap)
    else:
        local, total, seconds = greedy_assignment(costs)

    # Translate local row/column indices back to instance indices.
    assignments = {
        formulation.incidents[column]: formulation.vehicles[row]
        for column, row in local.items()
    }
    solution = formulation.decode(
        formulation.encode_assignments(assignments), qiskit_order=False
    )

    exact = method == "hungarian" and formulation.vehicle_mode == "exactly_one"
    details: Dict[str, Any] = {
        "method": method,
        "assignments": assignments,
        "sum_response_time": total,
        "vehicle_mode": formulation.vehicle_mode,
        # Recorded so the evaluation layer can refuse to compute a ratio between
        # two different objectives -- see the module docstring.
        "objective_matches_qubo": exact,
    }
    if not exact and method == "hungarian":
        details["note"] = (
            "Hungarian minimises total response time under a hard per-vehicle cap; "
            "the QUBO uses a soft balance term. Different objectives."
        )

    if not solution.feasible:
        _LOG.error(
            "The classical assignment decoded as INFEASIBLE (%s) -- encoding bug",
            "; ".join(solution.violations),
        )
        details["encoding_error"] = list(solution.violations)

    return ClassicalResult(
        solver=f"assignment:{method}",
        objective=solution.objective if solution.objective is not None else total,
        seconds=seconds,
        solution=solution,
        optimal=exact,
        details=details,
    )
