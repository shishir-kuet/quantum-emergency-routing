"""The formulation interface shared by all three problem encodings.

A *formulation* is the bridge between a routing problem and a QUBO. It owns
three responsibilities:

1. **Variable layout** -- deciding what each binary variable means and giving
   it a human-readable label, so a 25-bit string can be read back as a route.
2. **Encoding** (:meth:`Formulation.qubo`) -- objective plus penalty terms.
3. **Decoding** (:meth:`Formulation.decode`) -- turning a measured bitstring
   back into a route, checking feasibility, and reporting the *true* routing
   cost in the instance's own units rather than a QUBO energy.

Keeping decode next to encode in the same class is deliberate. The classic way
to get a plausible-but-wrong QAOA result is an encode/decode mismatch (a
transposed index, or a bit-order flip); pairing them makes that mismatch a
single-file concern that a unit test can pin down.

Subclasses implement :meth:`_build`, :meth:`decode`, :meth:`penalty_weight`,
:meth:`variable_labels` and :attr:`num_variables`, and register themselves with
the :func:`register` decorator so the CLI can look them up by name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Type, Union

import numpy as np

from ..exceptions import FormulationError
from ..logging_utils import get_logger
from .ising import IsingModel, qubo_to_ising
from .matrix import QUBO, array_to_bitstring, bitstring_to_array

__all__ = [
    "RouteSolution",
    "Formulation",
    "register",
    "available_formulations",
    "get_formulation_class",
]

_LOG = get_logger(__name__)

BitsLike = Union[str, Sequence[int], np.ndarray]


# ---------------------------------------------------------------------------
# Decoded solution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteSolution:
    """A decoded candidate solution.

    The distinction between :attr:`energy` and :attr:`objective` matters:

    * :attr:`energy` is the QUBO energy, including penalty contributions and
      any normalisation scale factor. It is what the quantum optimiser sees.
    * :attr:`objective` is the real routing cost -- seconds of travel time or
      metres of distance -- and is only defined when the solution is feasible.

    Comparing quantum and classical solvers must use :attr:`objective`;
    comparing optimiser progress must use :attr:`energy`.
    """

    formulation: str
    bits: str
    energy: float
    feasible: bool
    violations: Tuple[str, ...] = ()
    objective: Optional[float] = None
    raw_energy: Optional[float] = None
    route: Optional[Tuple[int, ...]] = None
    assignments: Optional[Dict[int, int]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def describe(self, unit: str = "") -> str:
        suffix = f" {unit}" if unit else ""
        if not self.feasible:
            reasons = "; ".join(self.violations) or "unspecified"
            return f"[infeasible] {self.formulation}: {reasons} (energy {self.energy:.4g})"
        parts = [f"[feasible] {self.formulation}"]
        if self.objective is not None:
            parts.append(f"cost {self.objective:.2f}{suffix}")
        if self.route is not None:
            parts.append("route " + " -> ".join(str(index) for index in self.route))
        if self.assignments:
            parts.append(
                "assign "
                + ", ".join(
                    f"incident {incident}->vehicle {vehicle}"
                    for incident, vehicle in sorted(self.assignments.items())
                )
            )
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Type["Formulation"]] = {}


def register(cls: Type["Formulation"]) -> Type["Formulation"]:
    """Class decorator adding *cls* to the formulation registry."""
    key = getattr(cls, "name", None)
    if not key or key == "formulation":
        raise FormulationError(
            f"{cls.__name__} must define a unique class-level 'name' before it can "
            f"be registered"
        )
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise FormulationError(f"Formulation name '{key}' is already registered")
    _REGISTRY[key] = cls
    return cls


def available_formulations() -> Tuple[str, ...]:
    """Names of every registered formulation, sorted."""
    return tuple(sorted(_REGISTRY))


def get_formulation_class(name: str) -> Type["Formulation"]:
    """Look up a formulation class by its registered name."""
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise FormulationError(
            f"Unknown formulation '{name}'. Available: {', '.join(available_formulations())}"
        ) from exc


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Formulation(ABC):
    """Abstract base class for QUBO encodings of a routing problem."""

    #: Registry key, e.g. ``"tsp"``. Subclasses must override.
    name: ClassVar[str] = "formulation"
    #: One-line human description, used in CLI help and result metadata.
    description: ClassVar[str] = ""

    def __init__(self, *, penalty_scale: float = 2.0, normalise: bool = True) -> None:
        if penalty_scale <= 0:
            raise FormulationError("penalty_scale must be positive")
        self.penalty_scale = float(penalty_scale)
        self.normalise = bool(normalise)
        self._raw_qubo: Optional[QUBO] = None
        self._qubo: Optional[QUBO] = None

    # -- subclass contract --------------------------------------------------
    @property
    @abstractmethod
    def num_variables(self) -> int:
        """Number of binary variables, i.e. qubits required."""

    @abstractmethod
    def variable_labels(self) -> Tuple[str, ...]:
        """Human-readable label per variable, in index order."""

    @abstractmethod
    def penalty_weight(self) -> float:
        """The constraint penalty weight, with a documented derivation.

        Too small and the ground state is an infeasible cheat; too large and the
        objective is drowned out, flattening the energy landscape that the
        optimiser has to navigate. Each subclass derives a bound from its own
        cost scale rather than hard-coding a number.
        """

    @abstractmethod
    def _build(self) -> QUBO:
        """Construct the unnormalised QUBO. Called once, then cached."""

    @abstractmethod
    def decode(self, bits: BitsLike, *, qiskit_order: bool = True) -> RouteSolution:
        """Decode a bitstring (or 0/1 array) into a :class:`RouteSolution`."""

    # -- shared machinery ---------------------------------------------------
    def raw_qubo(self) -> QUBO:
        """The QUBO in the instance's own cost units (never normalised)."""
        if self._raw_qubo is None:
            built = self._build()
            if built.n_variables != self.num_variables:
                raise FormulationError(
                    f"{type(self).__name__}._build() produced {built.n_variables} "
                    f"variable(s) but num_variables reports {self.num_variables}"
                )
            self._raw_qubo = built
        return self._raw_qubo

    def qubo(self) -> QUBO:
        """The QUBO handed to solvers, normalised when configured."""
        if self._qubo is None:
            raw = self.raw_qubo()
            self._qubo = raw.normalised() if self.normalise else raw
        return self._qubo

    def ising(self) -> IsingModel:
        """The Ising form of :meth:`qubo`, ready for Hamiltonian construction."""
        return qubo_to_ising(self.qubo())

    def energy_scale(self) -> float:
        """Factor applied by normalisation (1.0 when normalisation is off)."""
        return float(self.qubo().metadata.get("scale_applied", 1.0))

    # -- input coercion -----------------------------------------------------
    def as_array(self, bits: BitsLike, *, qiskit_order: bool = True) -> np.ndarray:
        """Coerce a bitstring or sequence into a validated ``{0, 1}`` array.

        Accepting both shapes means classical solvers (which produce arrays) and
        quantum samplers (which produce strings) can share one decode path --
        and the bit-order flip happens in exactly one place.
        """
        if isinstance(bits, str):
            return bitstring_to_array(bits, self.num_variables, qiskit_order=qiskit_order)
        array = np.asarray(bits).reshape(-1)
        if array.size != self.num_variables:
            raise FormulationError(
                f"Expected {self.num_variables} variable(s), got {array.size}"
            )
        if not np.all(np.isin(array, (0, 1))):
            raise FormulationError("Solution array must contain only 0s and 1s")
        return array.astype(np.int8)

    def canonical_bits(self, x: Sequence[int]) -> str:
        """Render an array as a Qiskit-ordered bitstring (qubit 0 rightmost)."""
        return array_to_bitstring(x, qiskit_order=True)

    # -- convenience --------------------------------------------------------
    def energies(self, x: Sequence[int]) -> Tuple[float, float]:
        """``(energy, raw_energy)`` for an assignment."""
        array = np.asarray(x, dtype=float)
        return (
            float(self.qubo().energy(array)),
            float(self.raw_qubo().energy(array)),
        )

    def summary(self) -> str:
        qubo = self.qubo()
        return (
            f"{self.name}: {self.num_variables} qubits, "
            f"{len(qubo.quadratic())} couplings, density {qubo.density():.2f}, "
            f"penalty {self.penalty_weight():.4g}"
            + (f", normalised x{self.energy_scale():.3g}" if self.normalise else "")
        )
