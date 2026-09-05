"""Exception hierarchy for :mod:`qroute`.

Every error raised deliberately by this package derives from
:class:`QRouteError`, so callers can catch the whole family with a single
``except`` clause while still being able to react to specific failures.
"""

from __future__ import annotations

__all__ = [
    "QRouteError",
    "ConfigError",
    "DataError",
    "FormulationError",
    "InfeasibleSolutionError",
    "SolverError",
    "BackendError",
    "MissingDependencyError",
]


class QRouteError(Exception):
    """Base class for all errors raised by qroute."""


class ConfigError(QRouteError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DataError(QRouteError):
    """A road network could not be fetched, loaded, or used as requested."""


class FormulationError(QRouteError):
    """A QUBO formulation was built with invalid or inconsistent inputs."""


class InfeasibleSolutionError(QRouteError):
    """A bitstring does not satisfy the constraints of its formulation."""


class SolverError(QRouteError):
    """A classical or quantum solver failed to produce a usable result."""


class BackendError(QRouteError):
    """A simulator or hardware backend could not be constructed or run."""


class MissingDependencyError(QRouteError):
    """An optional third-party package is required but not installed."""

    def __init__(self, package: str, purpose: str) -> None:
        super().__init__(
            f"The optional package '{package}' is required for {purpose}. "
            f"Install it with: python -m pip install {package}"
        )
        self.package = package
        self.purpose = purpose
