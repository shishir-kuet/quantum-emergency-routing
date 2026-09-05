"""qroute -- quantum-assisted emergency vehicle routing.

A research-grade pipeline that takes a real road network (Dhaka by default),
carves out small routable instances, formulates them as QUBOs, and solves them
with both classical baselines and QAOA on a local Aer simulator.

Module map
----------
``qroute.config``       typed configuration + YAML loader
``qroute.data``         OpenStreetMap download, graph IO, instance sampling
``qroute.qubo``         QUBO container, Ising conversion, three formulations
``qroute.classical``    Dijkstra, TSP heuristics, Hungarian, annealing, brute force
``qroute.quantum``      QUBO -> Hamiltonian, QAOA ansatz, Aer execution
``qroute.evaluation``   solution-quality metrics and circuit statistics
``qroute.viz``          route maps, convergence curves, comparison charts
``qroute.pipeline``     end-to-end orchestration shared by the CLI and scripts
``qroute.cli``          command-line entry point (``qroute --help``)

The subpackages are deliberately *not* imported here: pulling in Qiskit and
OSMnx costs several seconds, and most entry points need only a slice of the
project. Import what you use, e.g.::

    from qroute.config import load_config
    from qroute.qubo.tsp import TSPFormulation
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, load_config
from .exceptions import (
    BackendError,
    ConfigError,
    DataError,
    FormulationError,
    InfeasibleSolutionError,
    MissingDependencyError,
    QRouteError,
    SolverError,
)
from .logging_utils import configure_logging, get_logger

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "configure_logging",
    "get_logger",
    "QRouteError",
    "ConfigError",
    "DataError",
    "FormulationError",
    "InfeasibleSolutionError",
    "SolverError",
    "BackendError",
    "MissingDependencyError",
]
