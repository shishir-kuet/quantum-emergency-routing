r"""The QAOA driver: variational loop, expectation values, and result bookkeeping.

Deliberately built without Qiskit's ``Estimator``, ``Sampler``, or
``qiskit_algorithms.QAOA``. Three reasons, in order of importance:

1. **The Hamiltonian is diagonal.** Every expectation value is a weighted average
   of QUBO energies over measured bitstrings. Routing that through an
   ``Estimator`` adds an abstraction layer, a Pauli-grouping step, and its own
   shot budget in exchange for nothing.
2. **QAOA is a sampler, not a minimiser of means.** The quantity the optimiser
   minimises (the mean energy) is *not* the quantity you care about (the best
   sampled route). Owning the loop means both are tracked, every iteration. A
   run whose mean barely improves can still find the optimum on shot 40; a
   black-box wrapper would throw that sample away.
3. **API churn.** Primitives moved from V1 to V2 and ``qiskit-algorithms`` was
   split out of Qiskit entirely. Depending on neither keeps this code working
   across versions.

Two expectation modes
---------------------
``"shots"`` (default) samples the circuit and averages energies over the
outcomes. This is what a real device does, shot noise included -- and shot noise
matters: a gradient-free optimiser reading a noisy objective will happily chase
sampling fluctuations, which is exactly why COBYLA is the default rather than a
finite-difference method.

``"exact"`` computes :math:`\langle H \rangle = \sum_x |\psi_x|^2 E(x)` from the
statevector: a dot product, no sampling, no noise. It is the right tool for
studying the *landscape* (does :math:`p=2` beat :math:`p=1` in principle?)
because it separates "the ansatz cannot express the answer" from "we did not take
enough shots". Note that in this mode the simulator object is not used at all --
the statevector is computed directly from the circuit.

CVaR
----
With ``cvar_alpha < 1`` the objective becomes the mean of the lowest
:math:`\alpha`-fraction of sampled energies instead of the full mean (Barkoutsos
et al., *Improving Variational Quantum Optimization using CVaR*, 2020). Because
only the best sample ultimately matters, an objective that ignores the bad tail
often optimises far better on constrained problems, where most of the
distribution sits on infeasible states. It costs nothing extra: the energies are
already sorted-able once computed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..exceptions import MissingDependencyError, SolverError
from ..logging_utils import get_logger
from ..qubo.base import Formulation, RouteSolution
from ..qubo.ising import qubo_to_ising
from ..qubo.matrix import QUBO, array_to_bitstring, bitstring_to_array
from .ansatz import QAOACircuit, initial_point
from .backends import make_simulator, run_circuit, transpile_circuit
from .hamiltonian import diagonal_energies

__all__ = [
    "QAOAResult",
    "EXPECTATION_MODES",
    "SCIPY_OPTIMIZERS",
    "run_qaoa",
    "sample_energies",
    "expectation_from_counts",
    "cvar",
]

_LOG = get_logger(__name__)

#: How the objective value is obtained at each iteration.
EXPECTATION_MODES = ("shots", "exact")

#: Optimisers delegated to :func:`scipy.optimize.minimize`. ``SPSA`` is handled
#: separately by :func:`_minimize_spsa`.
SCIPY_OPTIMIZERS = ("COBYLA", "POWELL", "NELDER-MEAD")

#: ``"exact"`` mode holds a full statevector *and* an energy array of the same
#: length, so it is capped tighter than brute-force search.
MAX_EXACT_QUBITS = 20


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class QAOAResult:
    """Everything one QAOA run produced.

    ``optimal_value`` is the best *objective* (mean or CVaR energy) the optimiser
    reached; ``best_energy`` is the lowest energy in the final reported sample
    distribution. The second is the answer to the routing problem; the first
    only describes how well the variational loop went. Reporting the mean as
    though it were the result is a common and flattering mistake in the other
    direction too -- the mean is usually much worse than the best sample.
    """

    optimal_params: np.ndarray
    optimal_value: float
    best_energy: float
    best_bits: Optional[np.ndarray] = None
    best_solution: Optional[RouteSolution] = None
    counts: Dict[str, int] = field(default_factory=dict)
    history: Tuple[Tuple[int, float, float], ...] = ()
    n_evaluations: int = 0
    seconds: float = 0.0
    reps: int = 1
    shots: int = 0
    expectation_mode: str = "shots"
    cvar_alpha: float = 1.0
    n_qubits: int = 0
    optimizer: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Whether the best sample decoded to a valid route."""
        return self.best_solution is not None and self.best_solution.feasible

    @property
    def objective(self) -> Optional[float]:
        """Route cost of the best sample, in the problem's own units.

        ``None`` when nothing feasible was sampled -- which is a real outcome for
        QAOA on constrained problems at low depth, not an error.
        """
        if self.best_solution is None or not self.best_solution.feasible:
            return None
        return self.best_solution.objective

    def convergence(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(iterations, objective_values)`` arrays, ready to plot."""
        if not self.history:
            return np.empty(0, dtype=int), np.empty(0, dtype=float)
        iterations = np.array([entry[0] for entry in self.history], dtype=int)
        values = np.array([entry[1] for entry in self.history], dtype=float)
        return iterations, values

    def best_energy_trace(self) -> np.ndarray:
        """Running best sampled energy per iteration.

        Monotone by construction, and usually the more informative curve: it
        shows when the optimum was actually *found*, which the mean-energy trace
        hides completely.
        """
        if not self.history:
            return np.empty(0, dtype=float)
        return np.minimum.accumulate(
            np.array([entry[2] for entry in self.history], dtype=float)
        )

    def describe(self) -> str:
        objective = self.objective
        objective_text = "infeasible" if objective is None else f"{objective:.6g}"
        return (
            f"QAOAResult(n={self.n_qubits}, p={self.reps}, {self.optimizer}, "
            f"mode={self.expectation_mode}, evals={self.n_evaluations}, "
            f"<H>={self.optimal_value:.6g}, best E={self.best_energy:.6g}, "
            f"route={objective_text}, {self.seconds:.2f} s)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary, for writing results to disk."""
        return {
            "optimal_params": self.optimal_params.tolist(),
            "optimal_value": self.optimal_value,
            "best_energy": self.best_energy,
            "best_bits": None if self.best_bits is None else self.best_bits.tolist(),
            "objective": self.objective,
            "feasible": self.feasible,
            "violations": list(self.best_solution.violations) if self.best_solution else [],
            "n_evaluations": self.n_evaluations,
            "seconds": self.seconds,
            "reps": self.reps,
            "shots": self.shots,
            "expectation_mode": self.expectation_mode,
            "cvar_alpha": self.cvar_alpha,
            "n_qubits": self.n_qubits,
            "optimizer": self.optimizer,
            "history": [list(entry) for entry in self.history],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------
def sample_energies(
    counts: Dict[str, int], qubo: QUBO
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode *counts* into ``(assignments, energies, weights)``.

    One row per **distinct** outcome, with *weights* holding the shot counts, so
    the arrays stay small (at most the number of unique bitstrings) while the
    weighted statistics remain exact.

    Every key goes through :func:`~qroute.qubo.matrix.bitstring_to_array` with
    ``qiskit_order=True``, which is the one place the endianness of the whole
    pipeline is decided.
    """
    if not counts:
        raise SolverError("Cannot compute an expectation value from empty counts")

    n = qubo.n_variables
    keys = list(counts)
    assignments = np.empty((len(keys), n), dtype=np.int8)
    weights = np.empty(len(keys), dtype=float)
    for row, key in enumerate(keys):
        assignments[row] = bitstring_to_array(key, n, qiskit_order=True)
        weights[row] = float(counts[key])
    return assignments, qubo.energies(assignments), weights


def cvar(energies: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    r"""Conditional Value at Risk: mean of the lowest *alpha* fraction.

    With ``alpha = 1`` this is the plain weighted mean. Below that, outcomes are
    sorted by energy and accumulated until *alpha* of the total shot weight is
    covered, with the boundary outcome counted fractionally -- so the result is
    continuous in *alpha* rather than jumping as outcomes cross the cut.

    >>> import numpy as np
    >>> energies = np.array([1.0, 2.0, 3.0, 4.0])
    >>> weights = np.ones(4)
    >>> cvar(energies, weights, 1.0)
    2.5
    >>> cvar(energies, weights, 0.5)
    1.5
    """
    if not 0.0 < alpha <= 1.0:
        raise SolverError(f"cvar_alpha must be in (0, 1], got {alpha}")

    total = float(weights.sum())
    if total <= 0:
        raise SolverError("Sample weights sum to zero")
    if alpha >= 1.0:
        return float(energies @ weights / total)

    order = np.argsort(energies, kind="stable")
    budget = alpha * total
    accumulated = 0.0
    weighted_sum = 0.0
    for index in order:
        take = min(float(weights[index]), budget - accumulated)
        if take <= 0:
            break
        weighted_sum += take * float(energies[index])
        accumulated += take
        if accumulated >= budget:
            break
    if accumulated <= 0:  # pragma: no cover - alpha * total underflowed
        return float(energies[order[0]])
    return weighted_sum / accumulated


def expectation_from_counts(
    counts: Dict[str, int], qubo: QUBO, *, alpha: float = 1.0
) -> Tuple[float, float, np.ndarray]:
    """``(objective, best_energy, best_assignment)`` from measurement counts."""
    assignments, energies, weights = sample_energies(counts, qubo)
    objective = cvar(energies, weights, alpha)
    best = int(np.argmin(energies))
    return objective, float(energies[best]), assignments[best]


def _exact_expectation(
    circuit: Any, qubo: QUBO, energies: np.ndarray, alpha: float
) -> Tuple[float, float, np.ndarray]:
    """Shot-noise-free expectation from the statevector.

    Basis index ``k`` has qubit 0 as its least significant bit, matching
    :func:`~qroute.qubo.matrix.enumerate_assignments`, so ``probabilities`` and
    *energies* line up with no reindexing. That correspondence is asserted by
    :func:`qroute.quantum.hamiltonian.verify_hamiltonian`.
    """
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("qiskit", "exact statevector expectation") from exc

    probabilities = np.asarray(Statevector.from_instruction(circuit).probabilities())
    if probabilities.size != energies.size:
        raise SolverError(
            f"Statevector has {probabilities.size} amplitudes but the energy array "
            f"has {energies.size} entries -- the circuit and QUBO disagree on width"
        )

    objective = cvar(energies, probabilities, alpha)
    # The best *reachable* state, not merely the global minimum: an outcome with
    # zero amplitude was never sampled, so claiming it would overstate the run.
    reachable = np.flatnonzero(probabilities > 1e-12)
    best_index = int(reachable[np.argmin(energies[reachable])])
    bits = np.array(
        [(best_index >> k) & 1 for k in range(qubo.n_variables)], dtype=np.int8
    )
    return objective, float(energies[best_index]), bits


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def _minimize_spsa(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    seed: int = 42,
    a: float = 0.2,
    c: float = 0.1,
    alpha: float = 0.602,
    gamma: float = 0.101,
) -> Tuple[np.ndarray, float]:
    """Simultaneous Perturbation Stochastic Approximation.

    Hand-written because scipy has no SPSA and ``qiskit-algorithms`` is not a
    dependency here. It is worth having: SPSA estimates a gradient from just
    **two** objective evaluations regardless of dimension, by perturbing all
    parameters at once along a random :math:`\\pm 1` direction. On a noisy
    objective that is a real advantage -- finite differences need :math:`2d`
    evaluations and each one is corrupted by shot noise, whereas SPSA's noise
    averages out across iterations.

    The decay exponents are Spall's standard recommendations (``alpha = 0.602``,
    ``gamma = 0.101``); ``a`` and ``c`` are problem-scale dependent and these
    defaults assume a normalised Hamiltonian.
    """
    from ..rng import make_rng

    rng = make_rng(seed, "spsa")
    x = np.asarray(x0, dtype=float).copy()
    best_x = x.copy()
    best_value = float(objective(x))

    for k in range(int(maxiter)):
        ak = a / (k + 1 + 0.01 * maxiter) ** alpha
        ck = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)

        plus = float(objective(x + ck * delta))
        minus = float(objective(x - ck * delta))
        gradient = (plus - minus) / (2.0 * ck) * delta
        x = x - ak * gradient

        value = min(plus, minus)
        if value < best_value:
            best_value, best_x = value, x.copy()

    final = float(objective(x))
    if final < best_value:
        best_value, best_x = final, x.copy()
    return best_x, best_value


def _minimize_scipy(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    method: str,
    maxiter: int,
) -> Tuple[np.ndarray, float]:
    """Run a gradient-free scipy optimiser.

    All three are gradient-free by design. The energy landscape is evaluated by
    sampling, so any analytic or finite-difference gradient would be reading
    noise; COBYLA in particular is robust to that and is the config default.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("scipy", "optimising QAOA angles") from exc

    # Progress callback
    iteration_count = [0]
    def progress_callback(xk):
        iteration_count[0] += 1
        progress = (iteration_count[0] / maxiter) * 100
        _LOG.info("QAOA optimization progress: %d/%d iterations (%.1f%%)", 
                  iteration_count[0], maxiter, progress)

    if method == "COBYLA":
        scipy_method = "COBYLA"
        # rhobeg is COBYLA's initial trust-region radius. The default (1.0) is
        # large next to angles of order 0.5, so the first probes overshoot the
        # useful region entirely.
        options: Dict[str, Any] = {"maxiter": int(maxiter), "rhobeg": 0.3}
    elif method == "POWELL":
        scipy_method = "Powell"
        options = {"maxiter": int(maxiter), "maxfev": int(maxiter)}
    else:
        scipy_method = "Nelder-Mead"
        options = {"maxiter": int(maxiter), "maxfev": int(maxiter), "adaptive": True}

    outcome = minimize(objective, np.asarray(x0, dtype=float), method=scipy_method, 
                       options=options, callback=progress_callback)
    return np.asarray(outcome.x, dtype=float), float(outcome.fun)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
def run_qaoa(
    formulation: Formulation,
    config: Optional[Config] = None,
    *,
    reps: Optional[int] = None,
    optimizer: Optional[str] = None,
    maxiter: Optional[int] = None,
    shots: Optional[int] = None,
    expectation_mode: str = "shots",
    cvar_alpha: float = 1.0,
    x0: Optional[Sequence[float]] = None,
    initial_strategy: str = "ramp",
    simulator: Optional[Any] = None,
    seed: Optional[int] = None,
    schedule_couplings: bool = True,
    max_qubits: int = 24,
) -> QAOAResult:
    """Optimise QAOA angles for *formulation* and decode the best sample.

    Parameters
    ----------
    formulation:
        Any registered formulation -- shortest path, TSP, or assignment. The
        driver never inspects which: it sees a QUBO and a ``decode`` method.
    config:
        Supplies defaults for reps, optimizer, maxiter, shots and the simulator
        seed. Explicit keyword arguments override it.
    expectation_mode:
        ``"shots"`` (sampled, realistic) or ``"exact"`` (statevector, noiseless).
    cvar_alpha:
        Fraction of the best samples the objective averages over. ``1.0`` is the
        plain mean; ``0.1``-``0.25`` often works markedly better on these
        penalty-heavy QUBOs.
    x0:
        Starting angles as a flat ``[gamma..., beta...]`` vector. Defaults to
        :func:`~qroute.quantum.ansatz.initial_point` with *initial_strategy*.
    simulator:
        A pre-built ``AerSimulator``. Built from *config* if omitted, and not
        used at all in ``"exact"`` mode.

    Notes
    -----
    The circuit is transpiled **once**, while still parameterised, and angles are
    bound to the transpiled copy on every evaluation -- see
    :mod:`qroute.quantum.backends`.
    """
    from ..classical.base import Stopwatch

    qubo = formulation.qubo()
    n = qubo.n_variables
    if n > max_qubits:
        raise SolverError(
            f"Formulation '{formulation.name}' needs {n} qubits, above the "
            f"{max_qubits}-qubit limit for this run. Shrink the instance "
            f"(instance.n_cities / n_incidents) or raise max_qubits knowingly."
        )

    if expectation_mode not in EXPECTATION_MODES:
        raise SolverError(
            f"expectation_mode must be one of {EXPECTATION_MODES}, got {expectation_mode!r}"
        )
    if expectation_mode == "exact" and n > MAX_EXACT_QUBITS:
        raise SolverError(
            f"Exact statevector expectation on {n} qubits needs a 2^{n} array; the "
            f"limit is {MAX_EXACT_QUBITS}. Use expectation_mode='shots'."
        )

    settings = config.qaoa if config is not None else None
    reps = int(reps if reps is not None else (settings.reps if settings else 1))
    optimizer = str(
        optimizer if optimizer is not None else (settings.optimizer if settings else "COBYLA")
    ).upper()
    maxiter = int(maxiter if maxiter is not None else (settings.maxiter if settings else 200))
    shots = int(shots if shots is not None else (settings.shots if settings else 4096))
    if seed is None:
        seed = settings.seed_simulator if settings else 42

    if optimizer not in SCIPY_OPTIMIZERS and optimizer != "SPSA":
        raise SolverError(
            f"Unknown optimizer {optimizer!r}; expected one of "
            f"{SCIPY_OPTIMIZERS + ('SPSA',)}"
        )

    ising = qubo_to_ising(qubo)
    ansatz = QAOACircuit(ising, reps, schedule_couplings=schedule_couplings)
    if ansatz.n_qubits != n:
        raise SolverError(
            f"Ansatz has {ansatz.n_qubits} qubits but the QUBO has {n} variables"
        )

    # Priority: explicit x0 > config.qaoa.initial_point > generated strategy.
    if x0 is not None:
        angles = np.asarray(x0, dtype=float).reshape(-1)
        origin = "explicit"
    elif settings is not None and settings.initial_point is not None:
        angles = np.asarray(settings.initial_point, dtype=float)
        origin = "config"
    else:
        angles = initial_point(reps, strategy=initial_strategy, seed=int(seed))
        origin = initial_strategy
    if angles.size != 2 * reps:
        raise SolverError(
            f"Initial point must have 2 * reps = {2 * reps} entries "
            f"(gammas then betas), got {angles.size} from {origin}"
        )

    # -- build the evaluation closure ---------------------------------------
    history: List[Tuple[int, float, float]] = []
    tracker: Dict[str, Any] = {
        "evaluations": 0,
        "best_energy": float("inf"),
        "best_bits": None,
        "last_counts": {},
        "sampling_seconds": 0.0,
    }

    if expectation_mode == "exact":
        exact_energies = diagonal_energies(qubo, max_variables=MAX_EXACT_QUBITS)
        circuit_template = ansatz.circuit  # no measurements: statevector needs none
        active_simulator = None
        _LOG.info(
            "QAOA in exact mode: %d qubit(s), p=%d, %s, no shot noise",
            n,
            reps,
            optimizer,
        )
    else:
        exact_energies = None
        active_simulator = simulator if simulator is not None else make_simulator(config, seed=seed)
        circuit_template = transpile_circuit(
            ansatz.measured(),
            active_simulator,
            optimization_level=(config.backend.optimization_level if config else 3),
        )
        _LOG.info(
            "QAOA in shots mode: %d qubit(s), p=%d, %s, %d shots/eval, depth %d",
            n,
            reps,
            optimizer,
            shots,
            circuit_template.depth(),
        )

    def objective(point: np.ndarray) -> float:
        bound = circuit_template.assign_parameters(ansatz.parameter_dict(point))
        if expectation_mode == "exact":
            value, best_energy, best_bits = _exact_expectation(
                bound, qubo, exact_energies, cvar_alpha
            )
            counts: Dict[str, int] = {}
        else:
            sampling_started = perf_counter()
            counts = run_circuit(bound, active_simulator, shots=shots)
            tracker["sampling_seconds"] += perf_counter() - sampling_started
            value, best_energy, best_bits = expectation_from_counts(
                counts, qubo, alpha=cvar_alpha
            )
            tracker["last_counts"] = counts

        tracker["evaluations"] += 1
        if best_energy < tracker["best_energy"]:
            tracker["best_energy"] = best_energy
            tracker["best_bits"] = np.asarray(best_bits, dtype=np.int8).copy()
        history.append((tracker["evaluations"], float(value), float(best_energy)))
        return float(value)

    # -- optimise ------------------------------------------------------------
    with Stopwatch() as timer:
        if optimizer == "SPSA":
            best_point, best_value = _minimize_spsa(
                objective, angles, maxiter=maxiter, seed=int(seed)
            )
        else:
            best_point, best_value = _minimize_scipy(
                objective, angles, method=optimizer, maxiter=maxiter
            )

        # One final evaluation at the returned angles. scipy's ``fun`` is the best
        # value *seen*, which for a noisy objective may have been a lucky sample
        # rather than a genuinely better point; re-measuring here makes the
        # reported counts correspond to the reported parameters.
        final_value = objective(np.asarray(best_point, dtype=float))

    # The reported solution must be decoded from the final sampled distribution.
    # The optimizer may have seen a better sample during an earlier evaluation,
    # but that state is not part of the final counts and must not be reported as
    # if it were sampled in the reported distribution.
    final_counts = dict(tracker["last_counts"])
    if final_counts:
        _, final_best_energy, final_best_bits = expectation_from_counts(
            final_counts, qubo, alpha=1.0
        )
        best_bits = np.asarray(final_best_bits, dtype=np.int8)
    else:
        best_bits = tracker["best_bits"]
    best_solution: Optional[RouteSolution] = None
    if best_bits is not None:
        best_solution = formulation.decode(best_bits, qiskit_order=False)

    result = QAOAResult(
        optimal_params=np.asarray(best_point, dtype=float),
        optimal_value=float(min(best_value, final_value)),
        best_energy=float(final_best_energy if final_counts else tracker["best_energy"]),
        best_bits=best_bits,
        best_solution=best_solution,
        counts=dict(tracker["last_counts"]),
        history=tuple(history),
        n_evaluations=int(tracker["evaluations"]),
        seconds=timer.seconds,
        reps=reps,
        shots=0 if expectation_mode == "exact" else shots,
        expectation_mode=expectation_mode,
        cvar_alpha=float(cvar_alpha),
        n_qubits=n,
        optimizer=optimizer,
        metadata={
            "formulation": formulation.name,
            "initial_point": np.asarray(angles, dtype=float).tolist(),
            "initial_strategy": origin,
            "circuit_depth": int(circuit_template.depth()),
            "circuit_size": int(circuit_template.size()),
            "two_qubit_gates": int(
                sum(1 for instruction in circuit_template.data if len(instruction.qubits) >= 2)
            ),
            "ansatz_depth": int(ansatz.circuit.depth()),
            "cost_layer_rounds": ansatz.cost_layer_rounds,
            "n_couplings": ansatz.n_couplings,
            "best_bitstring": None if best_bits is None else array_to_bitstring(best_bits),
            "best_energy_seen": float(tracker["best_energy"]),
            "reported_counts_scope": "final parameter evaluation",
            "qubo": qubo.describe(),
            "optimizer_runtime_seconds": timer.seconds,
            "circuit_sampling_seconds": tracker["sampling_seconds"],
            "execution_runtime_seconds": tracker["sampling_seconds"],
            "queue_time_seconds": None,
            "qpu_execution_seconds": None,
        },
    )

    _LOG.info("QAOA done: %s", result.describe())
    if best_solution is not None and not best_solution.feasible:
        _LOG.warning(
            "The best sampled state of '%s' is infeasible (%s). At this depth that "
            "is a normal outcome, not a bug -- try cvar_alpha<1, more reps, or a "
            "larger penalty_scale.",
            formulation.name,
            "; ".join(best_solution.violations) or "unspecified",
        )
    return result
