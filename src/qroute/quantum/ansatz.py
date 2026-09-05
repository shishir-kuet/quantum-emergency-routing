r"""The QAOA ansatz, built explicitly.

Constructed by hand rather than imported from ``qiskit.circuit.library.QAOAAnsatz``
because the whole point of this rung of the project is to see the gates. There is
nothing hidden here:

**Initial state** -- ``H`` on every qubit, giving the uniform superposition
:math:`|+\rangle^{\otimes n}`, which is the ground state of the mixer.

**Cost layer** :math:`e^{-i\gamma \hat{H}_C}`. Since :math:`\hat{H}_C` is diagonal
this factorises exactly, with no Trotter error:

.. math::

    e^{-i\gamma h_i Z_i} = R_Z(2\gamma h_i), \qquad
    e^{-i\gamma J_{ij} Z_i Z_j} = R_{ZZ}(2\gamma J_{ij})

The factors of two come from Qiskit's convention
:math:`R_Z(\theta) = e^{-i\theta Z/2}`. Dropping them is a popular and
frustrating bug: the circuit still runs, the optimiser still converges, and the
answer is wrong by a factor of two in the angles.

**Mixer layer** :math:`e^{-i\beta \sum_i X_i} = \prod_i R_X(2\beta)`.

The identity term in the Hamiltonian is skipped: it contributes only a global
phase, which no measurement can see.

Binding parameters safely
-------------------------
``QuantumCircuit.parameters`` is sorted **by name**, so with vectors called
``beta`` and ``gamma`` it hands back every beta before every gamma. Binding a flat
list in the order you *think* you built them is therefore a silent
angle-scrambling bug. :meth:`QAOACircuit.assign` always binds through a
``{Parameter: value}`` dict, so the ordering of ``circuit.parameters`` never
matters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np

from ..exceptions import MissingDependencyError, SolverError
from ..logging_utils import get_logger
from ..qubo.ising import IsingModel
from ..rng import make_rng

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

__all__ = ["QAOACircuit", "initial_point", "INITIAL_POINT_STRATEGIES"]

_LOG = get_logger(__name__)

#: Accepted values for :func:`initial_point`.
INITIAL_POINT_STRATEGIES = ("ramp", "random", "constant")


def _import_qiskit() -> Tuple[Any, Any]:
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("qiskit", "building the QAOA circuit") from exc
    return QuantumCircuit, ParameterVector


def _schedule_pairs(
    pairs: Sequence[Tuple[int, int]], n_qubits: int
) -> List[List[Tuple[int, int]]]:
    """Group coupling pairs into rounds with no shared qubit.

    Two ``RZZ`` gates on disjoint qubit pairs commute *and* can run in the same
    time step, so grouping them this way cuts the cost layer's depth from one gate
    per coupling down to roughly the graph's maximum degree. This is greedy edge
    colouring -- not optimal, but within one colour of optimal by Vizing's
    theorem, and it costs microseconds.

    >>> _schedule_pairs([(0, 1), (2, 3), (1, 2)], 4)
    [[(0, 1), (2, 3)], [(1, 2)]]
    """
    rounds: List[List[Tuple[int, int]]] = []
    used: List[set] = []
    for i, j in pairs:
        if not (0 <= i < n_qubits and 0 <= j < n_qubits):
            raise SolverError(
                f"Coupling ({i}, {j}) is out of range for {n_qubits} qubit(s)"
            )
        for index, occupied in enumerate(used):
            if i not in occupied and j not in occupied:
                rounds[index].append((i, j))
                occupied.update((i, j))
                break
        else:
            rounds.append([(i, j)])
            used.append({i, j})
    return rounds


class QAOACircuit:
    """A parameterised QAOA circuit for a diagonal Ising Hamiltonian."""

    def __init__(
        self,
        ising: IsingModel,
        reps: int = 1,
        *,
        tolerance: float = 1e-12,
        schedule_couplings: bool = True,
    ) -> None:
        if reps < 1:
            raise SolverError(f"reps (QAOA depth p) must be at least 1, got {reps}")

        QuantumCircuit, ParameterVector = _import_qiskit()

        self.ising = ising
        self.reps = int(reps)
        self.n_qubits = ising.n_spins
        self.tolerance = float(tolerance)

        self.gamma = ParameterVector("gamma", self.reps)
        self.beta = ParameterVector("beta", self.reps)

        fields = {
            qubit: float(value)
            for qubit, value in enumerate(ising.h)
            if abs(value) > tolerance
        }
        couplings = ising.couplings(tolerance=tolerance)
        pairs = sorted(couplings)
        self._rounds = (
            _schedule_pairs(pairs, self.n_qubits) if schedule_couplings else [[pair] for pair in pairs]
        )

        circuit = QuantumCircuit(self.n_qubits, name=f"qaoa_p{self.reps}")
        circuit.h(range(self.n_qubits))

        for layer in range(self.reps):
            gamma = self.gamma[layer]
            for group in self._rounds:
                for i, j in group:
                    circuit.rzz(2.0 * gamma * couplings[(i, j)], i, j)
            for qubit, field in fields.items():
                circuit.rz(2.0 * gamma * field, qubit)
            circuit.barrier()

            beta = self.beta[layer]
            for qubit in range(self.n_qubits):
                circuit.rx(2.0 * beta, qubit)
            if layer < self.reps - 1:
                circuit.barrier()

        self.circuit = circuit
        self.n_fields = len(fields)
        self.n_couplings = len(couplings)
        _LOG.debug("Built %s", self.describe())

    # -- shape --------------------------------------------------------------
    @property
    def num_parameters(self) -> int:
        return 2 * self.reps

    @property
    def cost_layer_rounds(self) -> int:
        """Parallel rounds needed for one cost layer's two-qubit gates."""
        return len(self._rounds)

    def parameter_names(self) -> Tuple[str, ...]:
        """Flat parameter order used by :meth:`assign`: all gammas, then all betas."""
        return tuple(
            [f"gamma[{layer}]" for layer in range(self.reps)]
            + [f"beta[{layer}]" for layer in range(self.reps)]
        )

    # -- binding ------------------------------------------------------------
    def parameter_dict(self, values: Sequence[float]) -> Dict[Any, float]:
        """Map a flat ``[gamma..., beta...]`` vector to a binding dict."""
        flat = np.asarray(values, dtype=float).reshape(-1)
        if flat.size != self.num_parameters:
            raise SolverError(
                f"Expected {self.num_parameters} parameter(s) "
                f"({self.reps} gamma + {self.reps} beta), got {flat.size}"
            )
        bindings: Dict[Any, float] = {}
        for layer in range(self.reps):
            bindings[self.gamma[layer]] = float(flat[layer])
            bindings[self.beta[layer]] = float(flat[self.reps + layer])
        return bindings

    def assign(self, values: Sequence[float]) -> "QuantumCircuit":
        """Bind parameters and return a concrete circuit.

        Binds via a dict, never a positional list -- see the module docstring for
        why that distinction is load-bearing.
        """
        return self.circuit.assign_parameters(self.parameter_dict(values))

    def measured(self) -> "QuantumCircuit":
        """A copy of the parameterised circuit with all qubits measured.

        ``measure_all`` maps qubit *i* to classical bit *i*, so the returned
        counts use Qiskit's little-endian strings -- feed them through
        :func:`~qroute.qubo.matrix.bitstring_to_array` and never index them
        directly.
        """
        measured = self.circuit.copy()
        measured.measure_all()
        return measured

    # -- reporting ----------------------------------------------------------
    def describe(self) -> str:
        return (
            f"QAOACircuit(n={self.n_qubits}, p={self.reps}, "
            f"fields={self.n_fields}, couplings={self.n_couplings}, "
            f"cost-layer rounds={self.cost_layer_rounds}, "
            f"depth={self.circuit.depth()})"
        )


def initial_point(
    reps: int,
    *,
    strategy: str = "ramp",
    seed: int = 42,
    scale: float = 0.7,
) -> np.ndarray:
    """Starting angles as a flat ``[gamma..., beta...]`` vector.

    ``"ramp"`` (default) is a Trotterised annealing schedule: gamma ramps up while
    beta ramps down, mimicking a slow interpolation from the mixer's ground state
    to the problem Hamiltonian. It reliably beats random starts, and it is why
    QAOA at small ``p`` works at all without an expensive multi-start search
    (Sack & Serbyn, *Quantum annealing initialization of QAOA*, 2021).

    ``"random"`` samples uniformly and is the honest control: if the ramp's
    advantage does not survive comparison against several random starts, it is
    not real.

    The *scale* default assumes a **normalised** Hamiltonian (the config's
    ``qubo.normalise``). On raw travel-time coefficients in the hundreds these
    angles wrap many times around the circle and the ramp loses its meaning.
    """
    if reps < 1:
        raise SolverError(f"reps must be at least 1, got {reps}")
    if strategy not in INITIAL_POINT_STRATEGIES:
        raise SolverError(
            f"strategy must be one of {INITIAL_POINT_STRATEGIES}, got {strategy!r}"
        )

    if strategy == "ramp":
        fractions = (np.arange(reps) + 0.5) / reps
        gammas = fractions * scale
        betas = (1.0 - fractions) * scale
    elif strategy == "constant":
        gammas = np.full(reps, scale / 2.0)
        betas = np.full(reps, scale / 2.0)
    else:
        rng = make_rng(seed, "initial_point", str(reps))
        gammas = rng.uniform(0.0, np.pi, size=reps)
        betas = rng.uniform(0.0, np.pi / 2.0, size=reps)

    return np.concatenate([gammas, betas])
