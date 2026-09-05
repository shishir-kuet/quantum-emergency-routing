r"""Simulated annealing over an arbitrary QUBO.

This is QAOA's most important rival, and the fairest one: both solvers see the
*same* QUBO, both are anytime heuristics, and neither gets a hint about the
problem's structure. Any claim that a quantum method helps has to get past
simulated annealing first -- on small instances it usually will not, and saying
so plainly is more useful than a flattering benchmark.

Implemented here rather than pulled from ``dwave-neal`` on purpose: one less
dependency to break on Windows, and the incremental-energy trick below is worth
understanding rather than importing.

The delta trick
---------------
Recomputing :math:`x^\mathsf{T} Q x` after every proposed flip would be
:math:`O(n^2)` per move. Instead keep the field :math:`f = Q x` alongside the
state. The energy coefficient of variable :math:`i`, holding the others fixed, is

.. math::

    c_i = Q_{ii} + 2\sum_{j \neq i} Q_{ij} x_j = Q_{ii} + 2\,(f_i - Q_{ii} x_i)

so flipping :math:`x_i` by :math:`s = 1 - 2 x_i` changes the energy by
:math:`s\,c_i`, and the field updates as :math:`f \mathrel{+}= s\,Q_{:,i}`. Both
are :math:`O(n)`, making a full sweep :math:`O(n^2)` instead of :math:`O(n^3)`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import SolverError
from ..logging_utils import get_logger
from ..qubo.base import Formulation
from ..qubo.matrix import QUBO
from ..rng import make_rng
from .base import ClassicalResult, Stopwatch

__all__ = ["AnnealingResult", "solve_qubo_annealing", "solve_formulation_annealing"]

_LOG = get_logger(__name__)


class AnnealingResult:
    """Best state found, plus enough diagnostics to tell luck from skill."""

    __slots__ = (
        "state", "energy", "seconds", "n_restarts", "n_sweeps", "history",
        "initial_states", "hit_count",
    )

    def __init__(
        self,
        state: np.ndarray,
        energy: float,
        seconds: float,
        n_restarts: int,
        n_sweeps: int,
        history: Tuple[float, ...],
        initial_states: Tuple[np.ndarray, ...],
        hit_count: int,
    ) -> None:
        self.state = state
        self.energy = energy
        self.seconds = seconds
        self.n_restarts = n_restarts
        self.n_sweeps = n_sweeps
        #: Best energy from each restart, in order.
        self.history = history
        self.initial_states = initial_states
        #: How many restarts reached the best energy. A low hit count on an easy
        #: instance is a hint the schedule is too fast.
        self.hit_count = hit_count

    def describe(self) -> str:
        return (
            f"AnnealingResult(energy={self.energy:.6g}, "
            f"hits={self.hit_count}/{self.n_restarts}, "
            f"{self.seconds * 1e3:.1f} ms)"
        )


def _initial_temperature(qubo: QUBO) -> float:
    """A starting temperature on the scale of the largest single-flip energy change.

    Picking this from the data rather than hard-coding it matters here: QUBO
    coefficients in this project range from normalised (order 1) to raw travel
    times (order 100s), and a fixed temperature would either freeze instantly or
    random-walk forever.
    """
    diagonal = np.abs(np.diag(qubo.Q))
    row_sums = 2.0 * np.abs(qubo.Q).sum(axis=1)
    scale = float(np.max(diagonal + row_sums, initial=0.0))
    return scale if scale > 0 else 1.0


def solve_qubo_annealing(
    qubo: QUBO,
    *,
    n_restarts: int = 20,
    n_sweeps: int = 500,
    seed: int = 42,
    initial_temperature: Optional[float] = None,
    final_temperature_ratio: float = 1e-3,
) -> AnnealingResult:
    """Anneal *qubo* and return the best state found.

    Parameters
    ----------
    n_restarts:
        Independent runs from random starts. Restarts beat one very long run on
        rugged landscapes like penalised routing QUBOs, where a bad early
        commitment cannot be undone once the temperature drops.
    n_sweeps:
        Temperature steps per restart; each step proposes one flip per variable.
    initial_temperature:
        Defaults to the largest possible single-flip energy change, so almost
        every move is accepted at the start.
    final_temperature_ratio:
        Final temperature as a fraction of the initial one. Geometric cooling in
        between.
    """
    if n_restarts < 1 or n_sweeps < 1:
        raise SolverError("n_restarts and n_sweeps must both be at least 1")
    if not 0 < final_temperature_ratio < 1:
        raise SolverError("final_temperature_ratio must lie in (0, 1)")

    n = qubo.n_variables
    Q = qubo.Q
    diagonal = np.diag(Q).copy()
    rng = make_rng(seed, "annealing", qubo.metadata.get("formulation", ""))

    t_start = float(initial_temperature or _initial_temperature(qubo))
    cooling = final_temperature_ratio ** (1.0 / max(n_sweeps - 1, 1))

    best_state = np.zeros(n, dtype=np.int8)
    best_energy = float("inf")
    per_restart: List[float] = []
    initial_states: List[np.ndarray] = []

    with Stopwatch() as timer:
        for _ in range(n_restarts):
            x = rng.integers(0, 2, size=n).astype(np.int8)
            initial_states.append(x.copy())
            field = Q @ x.astype(float)
            energy = float(x @ field + qubo.offset)

            local_state = x.copy()
            local_energy = energy
            temperature = t_start

            for _sweep in range(n_sweeps):
                order = rng.permutation(n)
                thresholds = rng.random(n)
                for step, index in enumerate(order):
                    sign = 1.0 - 2.0 * x[index]
                    coefficient = diagonal[index] + 2.0 * (
                        field[index] - diagonal[index] * x[index]
                    )
                    delta = sign * coefficient
                    if delta <= 0 or thresholds[step] < np.exp(-delta / temperature):
                        x[index] = 1 - x[index]
                        field += sign * Q[:, index]
                        energy += delta
                        if energy < local_energy:
                            local_energy = energy
                            local_state = x.copy()
                temperature *= cooling

            # The incremental update is an optimization aid, not the source of
            # truth for reporting. Re-evaluate the retained state with the
            # canonical QUBO convention before comparing restarts.
            local_energy = qubo.energy(local_state)
            per_restart.append(local_energy)
            if local_energy < best_energy:
                best_energy = local_energy
                best_state = local_state

    # Recompute from scratch: the incremental energy accumulates float error over
    # tens of thousands of flips, and a drifting objective is a nasty bug to chase.
    best_energy = qubo.energy(best_state)
    hits = int(np.count_nonzero(np.isclose(np.asarray(per_restart), best_energy, atol=1e-6)))

    result = AnnealingResult(
        state=best_state,
        energy=float(best_energy),
        seconds=timer.seconds,
        n_restarts=n_restarts,
        n_sweeps=n_sweeps,
        history=tuple(per_restart),
        initial_states=tuple(initial_states),
        hit_count=hits,
    )
    _LOG.info("Annealing: %s", result.describe())
    return result


def solve_formulation_annealing(
    formulation: Formulation,
    *,
    n_restarts: int = 20,
    n_sweeps: int = 500,
    seed: int = 42,
    **kwargs: Any,
) -> ClassicalResult:
    """Anneal a formulation's QUBO and decode the winner.

    ``optimal`` is always ``False`` -- annealing gives no certificate, however
    often it happens to land on the true optimum.
    """
    annealed = solve_qubo_annealing(
        formulation.qubo(),
        n_restarts=n_restarts,
        n_sweeps=n_sweeps,
        seed=seed,
        **kwargs,
    )
    solution = formulation.decode(annealed.state, qiskit_order=False)
    details: Dict[str, Any] = {
        "n_restarts": annealed.n_restarts,
        "n_sweeps": annealed.n_sweeps,
        "restart_energies": annealed.history,
        "initial_states": [state.tolist() for state in annealed.initial_states],
        "final_energy": annealed.energy,
        "energy_convention": "qubo.energy(state), including offset",
        "energy_offset": formulation.qubo().offset,
        "energy_scale": formulation.energy_scale(),
        "normalized_energy": annealed.energy,
        "raw_energy": formulation.raw_qubo().energy(annealed.state),
        "hit_count": annealed.hit_count,
        "seed": seed,
    }
    return ClassicalResult(
        solver="annealing",
        objective=solution.objective,
        seconds=annealed.seconds,
        solution=solution,
        optimal=False,
        details=details,
    )
