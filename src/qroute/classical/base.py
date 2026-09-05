"""Shared plumbing for the classical baselines.

Every classical solver returns a :class:`ClassicalResult` so the evaluation layer
can line them up against QAOA without special-casing each one. Two fields carry
most of the weight:

* :attr:`ClassicalResult.objective` -- the routing cost in real units, the only
  number that may be compared across solvers.
* :attr:`ClassicalResult.optimal` -- whether this result is *provably* optimal.
  Approximation ratios are meaningless unless the denominator is a true optimum,
  so heuristics set this to ``False`` and the evaluation layer refuses to compute
  a ratio against them.

Where possible a solver also decodes its answer through the formulation that
produced the QUBO, which double-checks the encoding: if a classical route and its
QUBO energy disagree about feasibility, the encoding is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import TracebackType
from typing import Any, Dict, Optional, Type

from ..qubo.base import RouteSolution

__all__ = ["ClassicalResult", "Stopwatch"]


class Stopwatch:
    """Context manager measuring wall-clock seconds.

    Wall clock rather than CPU time on purpose: the interesting comparison is
    "how long did the user wait", and for the quantum side that includes
    simulator overhead.

    >>> with Stopwatch() as timer:
    ...     total = sum(range(1000))
    >>> timer.seconds >= 0
    True
    """

    __slots__ = ("_start", "seconds")

    def __init__(self) -> None:
        self._start = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "Stopwatch":
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.seconds = perf_counter() - self._start


@dataclass(frozen=True)
class ClassicalResult:
    """The outcome of one classical solve."""

    solver: str
    objective: Optional[float]
    seconds: float
    solution: Optional[RouteSolution] = None
    optimal: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def bits(self) -> Optional[str]:
        """The solution as a Qiskit-ordered bitstring, when one was encoded."""
        return self.solution.bits if self.solution is not None else None

    @property
    def feasible(self) -> Optional[bool]:
        return self.solution.feasible if self.solution is not None else None

    def describe(self, unit: str = "") -> str:
        suffix = f" {unit}" if unit else ""
        if self.objective is None:
            return f"{self.solver}: no solution ({self.seconds * 1e3:.1f} ms)"
        tag = "optimal" if self.optimal else "heuristic"
        text = (
            f"{self.solver}: {self.objective:.2f}{suffix} [{tag}] "
            f"({self.seconds * 1e3:.1f} ms)"
        )
        if self.solution is not None and not self.solution.feasible:
            text += " !! decoded as INFEASIBLE -- check the encoding"
        return text
