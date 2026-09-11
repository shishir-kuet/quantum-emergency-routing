"""QUBO layer: the mathematical core.

Everything a solver touches lives here, and nothing here imports Qiskit -- which
means the whole encoding can be developed and unit-tested without a quantum
backend installed.

Reading order, if you are following the maths:

1. :mod:`qroute.qubo.matrix` -- the ``(Q, offset)`` container, the builder, and
   the bit-order helpers. **Read the conventions in its docstring first.**
2. :mod:`qroute.qubo.ising` -- the :math:`x_i = (1 - z_i)/2` substitution that
   turns a QUBO into the Hamiltonian QAOA actually minimises.
3. :mod:`qroute.qubo.base` -- the encode/decode contract every formulation obeys.
4. The core rungs of the ladder, in increasing difficulty:
   :mod:`~qroute.qubo.shortest_path`, :mod:`~qroute.qubo.tsp`,
   :mod:`~qroute.qubo.assignment`.

Importing this package registers all formulation types, so
:func:`available_formulations` and :func:`build_formulation` work by name from
the CLI without any further imports.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from ..config import Config
from ..data.instance import RoutingInstance
from ..exceptions import FormulationError

from .assignment import (
    VEHICLE_MODES,
    AssignmentFormulation,
)

from .base import (
    Formulation,
    RouteSolution,
    available_formulations,
    get_formulation_class,
    register,
)

from .ising import (
    IsingModel,
    ising_to_qubo,
    qubo_to_ising,
)

from .matrix import (
    QUBO,
    QUBOBuilder,
    array_to_bitstring,
    bitstring_to_array,
)

from .shortest_path import (
    PathProblem,
    ShortestPathFormulation,
    build_local_path_problem,
    build_path_problem,
)

from .tsp import TSPFormulation


# ---------------------------------------------------------------------------
# Compatibility aliases
# ---------------------------------------------------------------------------
#
# Some parts of the pipeline use the older/shorter solver-facing name
# ``AssignmentQUBO``. The actual formulation class is
# ``AssignmentFormulation``. Keep both names available without duplicating
# or changing the implementation.
#
AssignmentQUBO = AssignmentFormulation


__all__ = [
    # -----------------------------------------------------------------------
    # Containers
    # -----------------------------------------------------------------------
    "QUBO",
    "QUBOBuilder",
    "IsingModel",
    "RouteSolution",

    # -----------------------------------------------------------------------
    # Conversions
    # -----------------------------------------------------------------------
    "qubo_to_ising",
    "ising_to_qubo",
    "bitstring_to_array",
    "array_to_bitstring",

    # -----------------------------------------------------------------------
    # Formulation framework
    # -----------------------------------------------------------------------
    "Formulation",
    "register",
    "available_formulations",
    "get_formulation_class",
    "build_formulation",

    # -----------------------------------------------------------------------
    # Rung 1 -- Shortest path
    # -----------------------------------------------------------------------
    "ShortestPathFormulation",
    "PathProblem",
    "build_path_problem",
    "build_local_path_problem",

    # -----------------------------------------------------------------------
    # Rung 2 -- TSP
    # -----------------------------------------------------------------------
    "TSPFormulation",

    # -----------------------------------------------------------------------
    # Rung 3 -- Assignment
    # -----------------------------------------------------------------------
    "AssignmentFormulation",
    "AssignmentQUBO",
    "VEHICLE_MODES",
]


# ---------------------------------------------------------------------------
# Expected input types for each formulation
# ---------------------------------------------------------------------------

#: Which problem container each formulation expects as its first argument.
_EXPECTED_INPUT = {
    "shortest_path": PathProblem,
    "tsp": RoutingInstance,
    "assignment": RoutingInstance,
}

ProblemLike = Union[PathProblem, RoutingInstance]


# ---------------------------------------------------------------------------
# Formulation factory
# ---------------------------------------------------------------------------

def build_formulation(
    name: str,
    problem: ProblemLike,
    config: Optional[Config] = None,
    **overrides: Any,
) -> Formulation:
    """Construct a registered formulation by name.

    This factory lives in the package ``__init__`` rather than in
    :mod:`qroute.qubo.base` for a boring but important reason: ``base`` is
    imported *by* every formulation module, so a factory there would need to
    import them back and close an import cycle.

    Parameters
    ----------
    name:
        A key from :func:`available_formulations`, e.g. ``"tsp"``.

    problem:
        A :class:`~qroute.qubo.shortest_path.PathProblem` for
        ``"shortest_path"``, or a :class:`~qroute.data.instance.RoutingInstance`
        for the other two.

    config:
        When given, penalty and normalisation settings are taken from
        ``config.qubo`` via the class's ``from_config`` hook.

    **overrides:
        Passed through to the constructor, taking precedence over *config*.

    Examples
    --------
    >>> sorted(available_formulations())
    ['assignment', 'local_routing', 'shortest_path', 'tsp']
    """

    cls = get_formulation_class(name)

    expected = _EXPECTED_INPUT.get(name)

    if expected is not None and not isinstance(problem, expected):
        raise FormulationError(
            f"Formulation '{name}' expects a {expected.__name__}, got "
            f"{type(problem).__name__}"
        )

    if config is not None:
        from_config = getattr(cls, "from_config", None)

        if from_config is not None:
            return from_config(
                problem,
                config,
                **overrides,
            )

    return cls(
        problem,
        **overrides,
    )  # type: ignore[arg-type]