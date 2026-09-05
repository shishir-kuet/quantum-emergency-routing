r"""Rung 3 of the ladder: assigning ambulances to incidents.

Several incidents are reported at once and a fleet of ambulances sits at known
bases. Who goes where?

Encoding
--------
:math:`y_{v,c} = 1` iff vehicle :math:`v` is dispatched to incident :math:`c`,
giving :math:`V \times C` variables. The objective is the total response time

.. math::

    \sum_{v}\sum_{c} c_{\,b_v,\, i_c}\; y_{v,c}

with one hard constraint per incident -- somebody must go:

.. math::

    A \sum_c \Bigl(\sum_v y_{v,c} - 1\Bigr)^2

On the vehicle side
-------------------
Left there, the cheapest solution sends *every* incident to whichever ambulance
happens to sit closest to the cluster, which is operationally absurd. Three ways
to fix it are offered through ``vehicle_mode``:

``"exactly_one"``
    A second family of hard one-hot constraints, one per vehicle. Only
    satisfiable when :math:`V = C`, and in that case the QUBO is *exactly* the
    linear assignment problem -- which the Hungarian algorithm solves optimally
    in polynomial time. That makes this mode the ground-truth check: QAOA's
    answer must match ``scipy.optimize.linear_sum_assignment`` or something is
    wrong.

``"balance"``
    A **soft** term :math:`B \sum_v (\sum_c y_{v,c} - C/V)^2` nudging every
    vehicle toward an equal share. Works for any :math:`V, C`. Chosen over a
    hard :math:`\le` capacity constraint on purpose: inequalities in a QUBO need
    slack variables, and slack variables cost qubits we would rather spend on
    incidents. The price is that the term has an irreducible floor whenever
    :math:`C/V` is not an integer, which is one more reason to compare solvers
    on :attr:`~qroute.qubo.base.RouteSolution.objective` and never on energy.

``"none"``
    No vehicle-side term at all. Kept because watching the degenerate
    "one ambulance does everything" solution appear is the clearest way to see
    why the other two modes exist.

A caveat on the objective: minimising the *sum* of response times is what a QUBO
encodes naturally, but a dispatcher usually cares about the *worst* wait. Min-max
needs auxiliary variables, so it is out of scope here; the maximum response time
is computed and reported alongside every solution instead of being optimised.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..config import Config
from ..data.instance import RoutingInstance
from ..exceptions import FormulationError
from ..logging_utils import get_logger
from .base import BitsLike, Formulation, RouteSolution, register
from .matrix import QUBO, QUBOBuilder

__all__ = ["AssignmentFormulation", "VEHICLE_MODES"]

_LOG = get_logger(__name__)

#: Accepted values for ``vehicle_mode``.
VEHICLE_MODES = ("auto", "exactly_one", "balance", "none")

MAX_VARIABLES = 26


@register
class AssignmentFormulation(Formulation):
    """Vehicle-to-incident assignment QUBO over a dispatch instance."""

    name = "assignment"
    description = "Dispatch a fleet of ambulances to simultaneous incidents"

    def __init__(
        self,
        instance: RoutingInstance,
        *,
        penalty_scale: float = 2.0,
        normalise: bool = True,
        vehicle_mode: str = "auto",
        balance_weight: float = 0.25,
        max_variables: int = MAX_VARIABLES,
    ) -> None:
        super().__init__(penalty_scale=penalty_scale, normalise=normalise)

        vehicles = tuple(int(index) for index in instance.vehicle_indices)
        incidents = tuple(int(index) for index in instance.incident_indices)
        if not vehicles:
            raise FormulationError(
                "Instance has no vehicle_indices; build it with "
                "qroute.data.instance.build_dispatch_instance"
            )
        if not incidents:
            raise FormulationError("Instance has no incident_indices")
        overlap = set(vehicles) & set(incidents)
        if overlap:
            raise FormulationError(
                f"Indices {sorted(overlap)} are listed as both vehicle bases and "
                f"incident sites"
            )
        if balance_weight < 0:
            raise FormulationError(
                f"balance_weight must be non-negative, got {balance_weight}"
            )

        mode = str(vehicle_mode).lower()
        if mode not in VEHICLE_MODES:
            raise FormulationError(
                f"vehicle_mode must be one of {VEHICLE_MODES}, got {vehicle_mode!r}"
            )
        if mode == "auto":
            mode = "exactly_one" if len(vehicles) == len(incidents) else "balance"
            _LOG.info(
                "vehicle_mode='auto' resolved to %r for %d vehicle(s) and %d incident(s)",
                mode,
                len(vehicles),
                len(incidents),
            )
        if mode == "exactly_one" and len(vehicles) != len(incidents):
            raise FormulationError(
                f"vehicle_mode='exactly_one' requires equal counts, got "
                f"{len(vehicles)} vehicle(s) and {len(incidents)} incident(s). "
                f"Use 'balance' instead."
            )

        self.instance = instance
        self.vehicles = vehicles
        self.incidents = incidents
        self.vehicle_mode = mode
        self.balance_weight = float(balance_weight)

        if self.num_variables > max_variables:
            raise FormulationError(
                f"This instance needs {self.num_variables} qubits "
                f"({len(vehicles)} vehicle(s) x {len(incidents)} incident(s)), above "
                f"the {max_variables}-qubit guard. Reduce instance.n_ambulances or "
                f"instance.n_incidents."
            )

    # -- layout -------------------------------------------------------------
    @property
    def num_variables(self) -> int:
        return len(self.vehicles) * len(self.incidents)

    @property
    def n_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def n_incidents(self) -> int:
        return len(self.incidents)

    def variable_index(self, vehicle: int, incident: int) -> int:
        """Variable index for "*vehicle* serves *incident*", by instance index."""
        try:
            row = self.vehicles.index(int(vehicle))
        except ValueError as exc:
            raise FormulationError(
                f"{vehicle} is not a vehicle base in this instance ({self.vehicles})"
            ) from exc
        try:
            column = self.incidents.index(int(incident))
        except ValueError as exc:
            raise FormulationError(
                f"{incident} is not an incident site in this instance ({self.incidents})"
            ) from exc
        return row * self.n_incidents + column

    def variable_labels(self) -> Tuple[str, ...]:
        return tuple(
            f"y[veh={vehicle},inc={incident}]"
            for vehicle in self.vehicles
            for incident in self.incidents
        )

    def encode_assignments(self, assignments: Mapping[int, int]) -> np.ndarray:
        """Encode an ``{incident: vehicle}`` mapping into a bit array.

        The inverse of :meth:`decode`, letting the Hungarian answer be scored on
        the same QUBO the quantum solver minimises.
        """
        missing = set(self.incidents) - {int(key) for key in assignments}
        if missing:
            raise FormulationError(
                f"No vehicle assigned to incident(s) {sorted(missing)}"
            )
        x = np.zeros(self.num_variables, dtype=np.int8)
        for incident, vehicle in assignments.items():
            x[self.variable_index(vehicle, incident)] = 1
        return x

    def response_time(self, vehicle: int, incident: int) -> float:
        """Cost of sending *vehicle* to *incident*, in the instance's unit."""
        return self.instance.cost_between(vehicle, incident)

    def cost_matrix(self) -> np.ndarray:
        """The ``(V, C)`` response-time matrix, in variable row/column order."""
        return np.array(
            [
                [self.response_time(vehicle, incident) for incident in self.incidents]
                for vehicle in self.vehicles
            ],
            dtype=float,
        )

    # -- penalty ------------------------------------------------------------
    def penalty_weight(self) -> float:
        """``penalty_scale * c_max`` over the response-time matrix.

        Leaving one incident unattended saves at most the single largest response
        time, so any weight above that makes abandoning an incident strictly
        worse than serving it. Unlike the tour case there is no factor of
        :math:`n`: each incident's cost enters the objective exactly once.
        """
        costs = self.cost_matrix()
        c_max = float(costs.max()) if costs.size else 1.0
        return self.penalty_scale * (c_max if c_max > 0 else 1.0)

    def balance_penalty_weight(self) -> float:
        """Weight of the soft balance term, ``balance_weight * c_max``.

        Scaled by the largest response time rather than by
        :meth:`penalty_weight` so that it stays a *nudge*: at the default 0.25, a
        one-job load imbalance costs about a quarter of the worst single trip.
        Push it up towards :meth:`penalty_weight` and balance becomes effectively
        hard, drowning out the travel times it is supposed to trade against.
        """
        costs = self.cost_matrix()
        c_max = float(costs.max()) if costs.size else 1.0
        return self.balance_weight * (c_max if c_max > 0 else 1.0)

    # -- encode -------------------------------------------------------------
    def _build(self) -> QUBO:
        builder = QUBOBuilder(self.num_variables, labels=self.variable_labels())

        # --- objective ---
        for vehicle in self.vehicles:
            for incident in self.incidents:
                builder.add_linear(
                    self.variable_index(vehicle, incident),
                    self.response_time(vehicle, incident),
                )

        # --- every incident is served by exactly one vehicle (hard) ---
        weight = self.penalty_weight()
        for incident in self.incidents:
            builder.add_penalty_equality(
                [self.variable_index(vehicle, incident) for vehicle in self.vehicles],
                target=1,
                weight=weight,
            )

        # --- vehicle side ---
        balance_weight = 0.0
        if self.vehicle_mode == "exactly_one":
            for vehicle in self.vehicles:
                builder.add_penalty_equality(
                    [
                        self.variable_index(vehicle, incident)
                        for incident in self.incidents
                    ],
                    target=1,
                    weight=weight,
                )
        elif self.vehicle_mode == "balance" and self.balance_weight > 0:
            balance_weight = self.balance_penalty_weight()
            if balance_weight > 0:
                share = self.n_incidents / self.n_vehicles
                for vehicle in self.vehicles:
                    builder.add_penalty_equality(
                        [
                            self.variable_index(vehicle, incident)
                            for incident in self.incidents
                        ],
                        target=share,
                        weight=balance_weight,
                    )

        qubo = builder.build(
            metadata={
                "formulation": self.name,
                "n_vehicles": self.n_vehicles,
                "n_incidents": self.n_incidents,
                "vehicle_mode": self.vehicle_mode,
                "penalty_weight": weight,
                "balance_penalty_weight": balance_weight,
                "weight_attribute": self.instance.weight,
            }
        )
        _LOG.debug("Built assignment QUBO: %s", qubo.describe())
        return qubo

    # -- decode -------------------------------------------------------------
    def decode(self, bits: BitsLike, *, qiskit_order: bool = True) -> RouteSolution:
        x = self.as_array(bits, qiskit_order=qiskit_order)
        energy, raw_energy = self.energies(x)
        matrix = x.reshape(self.n_vehicles, self.n_incidents).astype(int)

        violations: List[str] = []
        for column, incident in enumerate(self.incidents):
            total = int(matrix[:, column].sum())
            if total != 1:
                violations.append(
                    f"incident {incident} assigned to {total} vehicle(s), expected 1"
                )
        if self.vehicle_mode == "exactly_one":
            for row, vehicle in enumerate(self.vehicles):
                total = int(matrix[row].sum())
                if total != 1:
                    violations.append(
                        f"vehicle {vehicle} given {total} incident(s), expected 1"
                    )

        loads = {
            vehicle: int(matrix[row].sum()) for row, vehicle in enumerate(self.vehicles)
        }
        details: Dict[str, Any] = {
            "loads": loads,
            "vehicle_mode": self.vehicle_mode,
            "idle_vehicles": [
                vehicle for vehicle, load in loads.items() if load == 0
            ],
        }

        assignments: Optional[Dict[int, int]] = None
        objective: Optional[float] = None

        if not violations:
            assignments = {}
            response_times: Dict[int, float] = {}
            for column, incident in enumerate(self.incidents):
                row = int(np.argmax(matrix[:, column]))
                vehicle = self.vehicles[row]
                assignments[incident] = vehicle
                response_times[incident] = self.response_time(vehicle, incident)

            objective = float(sum(response_times.values()))
            details["response_times"] = response_times
            # Reported, not optimised -- see the module docstring.
            details["max_response_time"] = max(response_times.values())
            details["mean_response_time"] = objective / len(response_times)
            details["load_spread"] = max(loads.values()) - min(loads.values())

        return RouteSolution(
            formulation=self.name,
            bits=self.canonical_bits(x),
            energy=energy,
            raw_energy=raw_energy,
            feasible=not violations,
            violations=tuple(violations),
            objective=objective,
            assignments=assignments,
            details=details,
        )

    # -- construction from config -------------------------------------------
    @classmethod
    def from_config(
        cls, instance: RoutingInstance, config: Config, **overrides: Any
    ) -> "AssignmentFormulation":
        options: Dict[str, Any] = {
            "penalty_scale": config.qubo.penalty_scale,
            "normalise": config.qubo.normalise,
        }
        options.update(overrides)
        return cls(instance, **options)
