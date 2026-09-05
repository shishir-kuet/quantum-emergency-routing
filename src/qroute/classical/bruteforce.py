"""Exhaustive QUBO search: the ground truth for small instances.

Enumerating all :math:`2^n` assignments is exponential and useless at scale --
which is the entire reason quantum optimisation is interesting. But at the scale
that fits on a simulator it is *invaluable*, because it answers questions no
heuristic can:

* What is the true ground state, so an approximation ratio means something?
* Is the ground state actually **feasible**? If not, the penalty weight is too
  small and every downstream result is measuring the wrong problem.
* How degenerate is the optimum? A 12-fold degenerate ground state explains why
  QAOA's success probability looks low when the sampling is in fact fine.
* What does the low-energy spectrum look like? QAOA concentrates probability near
  the bottom of the spectrum, so the gap to the first excited state predicts how
  hard the instance will be.

Enumeration is chunked so memory stays bounded, and vectorised through
:meth:`~qroute.qubo.matrix.QUBO.energies` rather than looping in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import SolverError
from ..logging_utils import get_logger
from ..qubo.base import Formulation, RouteSolution
from ..qubo.matrix import QUBO, ENUMERATION_CHUNK
from ..qubo.matrix import enumerate_assignments as _enumerate
from .base import ClassicalResult, Stopwatch

__all__ = [
    "SpectrumResult",
    "enumerate_assignments",
    "solve_qubo_bruteforce",
    "solve_formulation_bruteforce",
]

_LOG = get_logger(__name__)

#: Hard ceiling. 2**24 x 24 energy evaluations is already a minute of numpy;
#: beyond that the honest answer is "use a heuristic".
MAX_VARIABLES = 24

#: Warn above this: still fast, but no longer instant.
_WARN_VARIABLES = 20

#: Assignments per vectorised chunk, shared with the QUBO layer.
CHUNK = ENUMERATION_CHUNK


@dataclass(frozen=True, eq=False)
class SpectrumResult:
    """Everything exhaustive search learned about a QUBO."""

    ground_energy: float
    ground_state: np.ndarray
    degeneracy: int
    first_excited_energy: Optional[float]
    n_variables: int
    seconds: float
    low_energy_states: Tuple[Tuple[float, np.ndarray], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def spectral_gap(self) -> Optional[float]:
        """Energy gap between the ground state and the first excited level.

        A small gap relative to the coefficient scale means low-energy states are
        crowded together, so a finite-shot QAOA run will struggle to separate the
        optimum from near-misses no matter how well the angles are optimised.
        """
        if self.first_excited_energy is None:
            return None
        return self.first_excited_energy - self.ground_energy

    def describe(self) -> str:
        gap = self.spectral_gap
        gap_text = f"{gap:.4g}" if gap is not None else "n/a"
        return (
            f"SpectrumResult(n={self.n_variables}, ground={self.ground_energy:.6g}, "
            f"degeneracy={self.degeneracy}, gap={gap_text}, "
            f"{self.seconds * 1e3:.1f} ms)"
        )


def enumerate_assignments(n_variables: int, start: int, stop: int) -> np.ndarray:
    """Re-exported from :mod:`qroute.qubo.matrix` -- see there for the convention."""
    return _enumerate(n_variables, start, stop)


def solve_qubo_bruteforce(
    qubo: QUBO,
    *,
    max_variables: int = MAX_VARIABLES,
    top_k: int = 0,
    tolerance: float = 1e-9,
) -> SpectrumResult:
    """Find the exact ground state of *qubo* by exhaustive enumeration.

    Parameters
    ----------
    max_variables:
        Refuse to start above this many variables, rather than hanging.
    top_k:
        Also return the *k* lowest-energy states, cheapest first. Useful for
        measuring how much QAOA probability mass lands near the optimum.
    tolerance:
        Energies within this distance of the minimum count as degenerate.
    """
    n = qubo.n_variables
    if n > max_variables:
        raise SolverError(
            f"Exhaustive search over {n} variables means {2 ** n:,} assignments, "
            f"above the {max_variables}-variable limit. Use "
            f"qroute.classical.annealing.solve_qubo_annealing instead."
        )
    if n > _WARN_VARIABLES:
        _LOG.warning(
            "Enumerating %d variables (%d assignments) -- this will take a while",
            n,
            2 ** n,
        )

    total = 1 << n
    energies = np.empty(total, dtype=float)

    with Stopwatch() as timer:
        for start in range(0, total, CHUNK):
            stop = min(start + CHUNK, total)
            block = enumerate_assignments(n, start, stop)
            energies[start:stop] = qubo.energies(block)

        ground_energy = float(energies.min())
        degeneracy = int(np.count_nonzero(energies <= ground_energy + tolerance))
        ground_index = int(np.argmin(energies))
        ground_state = enumerate_assignments(n, ground_index, ground_index + 1)[0]

        excited = energies[energies > ground_energy + tolerance]
        first_excited = float(excited.min()) if excited.size else None

        low_energy: List[Tuple[float, np.ndarray]] = []
        if top_k > 0:
            k = min(int(top_k), total)
            # argpartition then sort only the k survivors: O(total) instead of
            # O(total log total) for a full sort we do not need.
            candidates = np.argpartition(energies, k - 1)[:k]
            candidates = candidates[np.argsort(energies[candidates], kind="stable")]
            low_energy = [
                (float(energies[index]), enumerate_assignments(n, index, index + 1)[0])
                for index in candidates
            ]

    result = SpectrumResult(
        ground_energy=ground_energy,
        ground_state=ground_state,
        degeneracy=degeneracy,
        first_excited_energy=first_excited,
        n_variables=n,
        seconds=timer.seconds,
        low_energy_states=tuple(low_energy),
        metadata={"assignments_evaluated": total, "label": qubo.metadata.get("formulation")},
    )
    _LOG.info("Brute force: %s", result.describe())
    return result


def solve_formulation_bruteforce(
    formulation: Formulation,
    *,
    max_variables: int = MAX_VARIABLES,
    require_feasible: bool = True,
    max_decodes: int = 200_000,
) -> ClassicalResult:
    """Exact optimum of a formulation, with a penalty-weight sanity check.

    The QUBO ground state and the best *feasible* solution are different
    questions, and comparing them is the cheapest diagnostic in this whole
    project:

    * If they coincide, the penalty weights are doing their job.
    * If the ground state is infeasible, the penalty is too weak -- some
      constraint-breaking assignment is cheaper than any real route, and every
      quantum result on this instance is optimising a broken objective. The
      returned ``details["ground_state_feasible"]`` flags exactly that, and a
      warning is logged.

    With ``require_feasible=True`` the reported objective is the best feasible
    one, found by walking the spectrum upwards from the ground state, so it is a
    valid denominator for an approximation ratio either way.
    """
    qubo = formulation.qubo()
    n = qubo.n_variables
    if n > max_variables:
        raise SolverError(
            f"Formulation '{formulation.name}' has {n} variables, above the "
            f"{max_variables}-variable exhaustive-search limit"
        )

    total = 1 << n

    with Stopwatch() as timer:
        energies = np.empty(total, dtype=float)
        for start in range(0, total, CHUNK):
            stop = min(start + CHUNK, total)
            energies[start:stop] = qubo.energies(enumerate_assignments(n, start, stop))

        order = np.argsort(energies, kind="stable")
        ground_index = int(order[0])
        ground_solution = formulation.decode(
            enumerate_assignments(n, ground_index, ground_index + 1)[0],
            qiskit_order=False,
        )

        best: Optional[RouteSolution] = None
        decodes = 0
        if ground_solution.feasible or not require_feasible:
            best = ground_solution
            decodes = 1
        else:
            for index in order[: min(total, max_decodes)]:
                decodes += 1
                candidate = formulation.decode(
                    enumerate_assignments(n, int(index), int(index) + 1)[0],
                    qiskit_order=False,
                )
                if candidate.feasible:
                    best = candidate
                    break

    if not ground_solution.feasible:
        _LOG.warning(
            "The QUBO ground state of '%s' is INFEASIBLE (%s). The penalty weight "
            "is too small -- raise qubo.penalty_scale in the config before trusting "
            "any quantum result on this instance.",
            formulation.name,
            "; ".join(ground_solution.violations) or "unspecified",
        )

    if best is None:
        raise SolverError(
            f"No feasible solution found among the {decodes:,} lowest-energy "
            f"assignments of '{formulation.name}'. The constraints may be "
            f"contradictory."
        )

    # Counts feasible *and* infeasible states, so it is an upper bound on the
    # degeneracy of the reported optimum rather than the degeneracy itself.
    at_or_below = int(np.count_nonzero(energies <= best.energy + 1e-9))
    return ClassicalResult(
        solver="bruteforce",
        objective=best.objective,
        seconds=timer.seconds,
        solution=best,
        optimal=True,
        details={
            "n_variables": n,
            "assignments_evaluated": total,
            "ground_energy": float(energies[ground_index]),
            "ground_state_feasible": ground_solution.feasible,
            "ground_state_violations": ground_solution.violations,
            "best_feasible_energy": best.energy,
            "decodes_needed": decodes,
            "states_at_or_below_best": at_or_below,
        },
    )
