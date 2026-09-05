r"""Rung 2 of the ladder: single-ambulance multi-stop tour (TSP).

One ambulance must leave its base, visit every incident exactly once, and
return. This is the textbook Travelling Salesman QUBO, and it is the rung where
the qubit cost of routing becomes vivid.

Encoding
--------
Binary variables are **position-indexed**: :math:`x_{i,p} = 1` iff location
:math:`i` is visited at position :math:`p` in the tour. The objective sums the
cost of each consecutive pair,

.. math::

    \sum_{p} \sum_{i \neq j} c_{ij}\, x_{i,p}\, x_{j,p+1}

and two families of one-hot constraints keep it a permutation:

.. math::

    \sum_p x_{i,p} = 1 \;\;\forall i, \qquad \sum_i x_{i,p} = 1 \;\;\forall p

Fixing the depot
----------------
A tour is a cycle, so every rotation of it is the same route with the same
cost. Leaving that symmetry in place wastes qubits *and* smears the QAOA
probability mass across :math:`n` degenerate optima. Pinning the depot to
position 0 removes the rotational symmetry and shrinks the register from
:math:`n^2` to :math:`(n-1)^2` variables -- for :math:`n = 6` that is 25 qubits
instead of 36, which is the difference between a comfortable simulation and an
uncomfortable one. ``fix_depot=True`` is therefore the default; set it to
``False`` to study the symmetric encoding for comparison.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..data.instance import RoutingInstance
from ..exceptions import FormulationError
from ..logging_utils import get_logger
from .base import BitsLike, Formulation, RouteSolution, register
from .matrix import QUBO, QUBOBuilder

__all__ = ["TSPFormulation"]

_LOG = get_logger(__name__)

#: Refuse to build beyond this many qubits: a statevector simulation of 26
#: qubits already needs ~1 GB, and QAOA needs several copies.
MAX_VARIABLES = 26


@register
class TSPFormulation(Formulation):
    """Position-indexed TSP QUBO over a :class:`RoutingInstance`."""

    name = "tsp"
    description = "Single-vehicle closed tour visiting every location once"

    def __init__(
        self,
        instance: RoutingInstance,
        *,
        penalty_scale: float = 2.0,
        normalise: bool = True,
        fix_depot: bool = True,
        symmetrise: bool = False,
        max_variables: int = MAX_VARIABLES,
    ) -> None:
        super().__init__(penalty_scale=penalty_scale, normalise=normalise)

        if instance.n < 3:
            raise FormulationError(
                f"A tour needs at least 3 locations, instance has {instance.n}"
            )

        self.instance = instance.symmetrised() if symmetrise else instance
        self.fix_depot = bool(fix_depot)
        self.depot = int(instance.depot)

        if self.fix_depot:
            others = [i for i in range(self.instance.n) if i != self.depot]
            self._cities: Tuple[int, ...] = tuple(others)
            self._positions: Tuple[int, ...] = tuple(range(1, self.instance.n))
        else:
            self._cities = tuple(range(self.instance.n))
            self._positions = tuple(range(self.instance.n))

        # One authoritative (city, position) -> variable index map. Every other
        # method goes through it, so there is no second place to get the
        # arithmetic wrong.
        self._index: Dict[Tuple[int, int], int] = {
            pair: k for k, pair in enumerate(product(self._cities, self._positions))
        }

        if self.num_variables > max_variables:
            raise FormulationError(
                f"This instance needs {self.num_variables} qubits "
                f"({len(self._cities)} locations x {len(self._positions)} positions), "
                f"above the {max_variables}-qubit guard. Reduce instance.n_nodes "
                f"(cost grows as (n-1)^2) or raise max_variables if you know your "
                f"machine can take it."
            )

    # -- layout -------------------------------------------------------------
    @property
    def num_variables(self) -> int:
        return len(self._index)

    @property
    def cities(self) -> Tuple[int, ...]:
        """Instance indices that the variables range over."""
        return self._cities

    @property
    def positions(self) -> Tuple[int, ...]:
        """Tour positions that the variables range over."""
        return self._positions

    def variable_index(self, city: int, position: int) -> int:
        """Variable index for "*city* is visited at *position*"."""
        try:
            return self._index[(int(city), int(position))]
        except KeyError as exc:
            raise FormulationError(
                f"No variable for city {city} at position {position}. Cities are "
                f"{self._cities}, positions are {self._positions}."
            ) from exc

    def variable_labels(self) -> Tuple[str, ...]:
        labels: List[str] = [""] * self.num_variables
        for (city, position), index in self._index.items():
            labels[index] = f"x[loc={city},pos={position}]"
        return tuple(labels)

    def encode_route(self, route: Sequence[int]) -> np.ndarray:
        """Encode a tour (as a sequence of instance indices) into a bit array.

        The inverse of :meth:`decode`, and the bridge that lets a classical tour
        be scored on the very same QUBO the quantum solver sees. If a classical
        route encodes to an energy that disagrees with its own cost, the encoding
        has a bug -- which is exactly what the round-trip tests check.
        """
        order = [int(node) for node in route]
        if self.fix_depot:
            if order and order[0] == self.depot:
                order = order[1:]
        if len(order) != len(self._positions):
            raise FormulationError(
                f"Expected a route covering {len(self._positions)} position(s), got "
                f"{len(order)} entry/entries after depot handling"
            )
        if sorted(order) != sorted(self._cities):
            raise FormulationError(
                f"Route must visit each of {self._cities} exactly once, got {tuple(order)}"
            )

        x = np.zeros(self.num_variables, dtype=np.int8)
        for position, city in zip(self._positions, order):
            x[self.variable_index(city, position)] = 1
        return x

    # -- penalty ------------------------------------------------------------
    def penalty_weight(self) -> float:
        r"""``penalty_scale * n * c_max``.

        A tour has :math:`n` legs, so no feasible tour can cost more than
        :math:`n\,c_{\max}`. Any constraint violation incurs at least one unit
        of penalty, so a weight above that bound makes cheating strictly
        unprofitable and guarantees the QUBO ground state is a valid tour. The
        configurable ``penalty_scale`` (default 2) supplies the margin.
        """
        off_diagonal = self.instance.cost[~np.eye(self.instance.n, dtype=bool)]
        c_max = float(off_diagonal.max()) if off_diagonal.size else 1.0
        return self.penalty_scale * self.instance.n * c_max

    # -- encode -------------------------------------------------------------
    def _leg_pairs(self) -> List[Tuple[int, int]]:
        """Consecutive position pairs contributing an inter-city leg."""
        positions = self._positions
        if self.fix_depot:
            return list(zip(positions[:-1], positions[1:]))
        return [
            (positions[k], positions[(k + 1) % len(positions)])
            for k in range(len(positions))
        ]

    def _build(self) -> QUBO:
        cost = self.instance.cost
        builder = QUBOBuilder(self.num_variables, labels=self.variable_labels())

        # --- objective: inter-city legs ---
        for position, next_position in self._leg_pairs():
            for city_i, city_j in product(self._cities, self._cities):
                if city_i == city_j:
                    continue
                builder.add_quadratic(
                    self.variable_index(city_i, position),
                    self.variable_index(city_j, next_position),
                    float(cost[city_i, city_j]),
                )

        # --- objective: the two depot legs (only when the depot is pinned) ---
        if self.fix_depot:
            first, last = self._positions[0], self._positions[-1]
            for city in self._cities:
                builder.add_linear(
                    self.variable_index(city, first), float(cost[self.depot, city])
                )
                builder.add_linear(
                    self.variable_index(city, last), float(cost[city, self.depot])
                )

        # --- constraints ---
        weight = self.penalty_weight()
        for city in self._cities:
            builder.add_penalty_equality(
                [self.variable_index(city, position) for position in self._positions],
                target=1,
                weight=weight,
            )
        for position in self._positions:
            builder.add_penalty_equality(
                [self.variable_index(city, position) for city in self._cities],
                target=1,
                weight=weight,
            )

        qubo = builder.build(
            metadata={
                "formulation": self.name,
                "n_locations": self.instance.n,
                "fix_depot": self.fix_depot,
                "penalty_weight": weight,
                "weight_attribute": self.instance.weight,
            }
        )
        _LOG.debug("Built TSP QUBO: %s", qubo.describe())
        return qubo

    # -- decode -------------------------------------------------------------
    def _assignment_matrix(self, x: np.ndarray) -> np.ndarray:
        """Reshape a flat solution into a ``(city, position)`` incidence matrix."""
        matrix = np.zeros((len(self._cities), len(self._positions)), dtype=int)
        city_row = {city: row for row, city in enumerate(self._cities)}
        position_column = {position: column for column, position in enumerate(self._positions)}
        for (city, position), index in self._index.items():
            matrix[city_row[city], position_column[position]] = int(x[index])
        return matrix

    def decode(self, bits: BitsLike, *, qiskit_order: bool = True) -> RouteSolution:
        x = self.as_array(bits, qiskit_order=qiskit_order)
        energy, raw_energy = self.energies(x)
        matrix = self._assignment_matrix(x)

        violations: List[str] = []
        for row, city in enumerate(self._cities):
            total = int(matrix[row].sum())
            if total != 1:
                violations.append(
                    f"location {city} visited {total} time(s), expected exactly 1"
                )
        for column, position in enumerate(self._positions):
            total = int(matrix[:, column].sum())
            if total != 1:
                violations.append(
                    f"position {position} holds {total} location(s), expected exactly 1"
                )

        route: Optional[Tuple[int, ...]] = None
        objective: Optional[float] = None
        details: Dict[str, Any] = {"n_selected": int(x.sum())}

        if not violations:
            ordered: List[int] = []
            for column in range(len(self._positions)):
                row = int(np.argmax(matrix[:, column]))
                ordered.append(self._cities[row])

            if self.fix_depot:
                route = tuple([self.depot] + ordered)
            else:
                # Rotate so the depot leads, purely for readability; the cost of
                # a cycle is rotation-invariant so this changes nothing numeric.
                if self.depot in ordered:
                    pivot = ordered.index(self.depot)
                    ordered = ordered[pivot:] + ordered[:pivot]
                route = tuple(ordered)

            objective = self.instance.tour_cost(route, closed=True)
            details["legs"] = [
                {
                    "from": route[k],
                    "to": route[(k + 1) % len(route)],
                    "cost": self.instance.cost_between(route[k], route[(k + 1) % len(route)]),
                }
                for k in range(len(route))
            ]

        return RouteSolution(
            formulation=self.name,
            bits=self.canonical_bits(x),
            energy=energy,
            raw_energy=raw_energy,
            feasible=not violations,
            violations=tuple(violations),
            objective=objective,
            route=route,
            details=details,
        )

    # -- construction from config -------------------------------------------
    @classmethod
    def from_config(
        cls, instance: RoutingInstance, config: Config, **overrides: Any
    ) -> "TSPFormulation":
        options: Dict[str, Any] = {
            "penalty_scale": config.qubo.penalty_scale,
            "normalise": config.qubo.normalise,
        }
        options.update(overrides)
        return cls(instance, **options)
