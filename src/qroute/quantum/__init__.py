"""The quantum layer: Hamiltonian, ansatz, simulator, and the QAOA loop.

Four modules, in dependency order:

``hamiltonian``
    QUBO / Ising :math:`\\rightarrow` ``SparsePauliOp``, plus
    :func:`verify_hamiltonian`, which checks the operator's diagonal really is
    the QUBO energy function. Run it once per new formulation; a convention error
    here produces a Hamiltonian that runs perfectly and optimises the wrong
    function.

``ansatz``
    :class:`QAOACircuit`, built gate by gate from ``RZ``/``RZZ``/``RX`` so the
    factors of two in Qiskit's rotation convention are visible rather than
    assumed, and :func:`initial_point` for the annealing-inspired ramp.

``backends``
    ``AerSimulator`` construction and transpile-once-bind-many.

``qaoa``
    :func:`run_qaoa`, the variational loop -- shot-based or exact expectation,
    optional CVaR objective, and per-iteration tracking of both the mean energy
    and the best single sample.

The intended order of operations on any new formulation
------------------------------------------------------
1. ``verify_hamiltonian(formulation.qubo())`` -- conventions correct?
2. ``solve_formulation_bruteforce(formulation)`` -- is the QUBO ground state
   *feasible*? If not, fix the penalty weight before running anything quantum.
3. ``run_qaoa(formulation, config, expectation_mode="exact")`` -- can the ansatz
   express a good answer at all, with no shot noise in the way?
4. ``run_qaoa(formulation, config)`` -- what happens with a finite shot budget?

Skipping straight to step 4 is the usual way to spend a week debugging a
penalty weight.
"""

from __future__ import annotations

from .ansatz import INITIAL_POINT_STRATEGIES, QAOACircuit, initial_point
from .backends import describe_backend, make_simulator, run_circuit, transpile_circuit
from .hamiltonian import (
    diagonal_energies,
    hamiltonian_terms,
    ising_to_sparse_pauli_op,
    qubo_to_sparse_pauli_op,
    verify_hamiltonian,
)
from .qaoa import (
    EXPECTATION_MODES,
    SCIPY_OPTIMIZERS,
    QAOAResult,
    cvar,
    expectation_from_counts,
    run_qaoa,
    sample_energies,
)

__all__ = [
    # hamiltonian
    "ising_to_sparse_pauli_op",
    "qubo_to_sparse_pauli_op",
    "hamiltonian_terms",
    "verify_hamiltonian",
    "diagonal_energies",
    # ansatz
    "QAOACircuit",
    "initial_point",
    "INITIAL_POINT_STRATEGIES",
    # backends
    "make_simulator",
    "transpile_circuit",
    "run_circuit",
    "describe_backend",
    # qaoa
    "run_qaoa",
    "QAOAResult",
    "EXPECTATION_MODES",
    "SCIPY_OPTIMIZERS",
    "sample_energies",
    "expectation_from_counts",
    "cvar",
]
