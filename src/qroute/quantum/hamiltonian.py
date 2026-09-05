r"""Ising model :math:`\rightarrow` Qiskit ``SparsePauliOp``.

The problem Hamiltonian is **diagonal** in the computational basis, which is the
single most useful fact in this whole package. Because
:math:`z_i \leftrightarrow Z_i` (with :math:`Z|0\rangle = +|0\rangle`, matching
the :math:`x_i = (1 - z_i)/2` convention from :mod:`qroute.qubo.ising`), the
Ising energy becomes

.. math::

    \hat{H}_C = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + c\,\mathbb{1}

and every computational basis state is an eigenstate with eigenvalue equal to the
QUBO energy of the corresponding bitstring. Three consequences worth internalising:

1. The cost unitary :math:`e^{-i\gamma \hat{H}_C}` is exactly a product of ``RZ``
   and ``RZZ`` rotations -- no Trotter error, no approximation.
2. The exact expectation value is
   :math:`\langle \hat{H}_C \rangle = \sum_x |\psi_x|^2 E(x)`: a dot product of
   the statevector's probabilities with the energy array. No ``Estimator``
   needed, no shot noise.
3. Measuring in the computational basis loses nothing. QAOA here is a *sampler*,
   and the answer you keep is the best sample, not the mean.

Label ordering
--------------
Qiskit Pauli labels read **right-to-left**: in ``"IIZ"`` the ``Z`` acts on qubit
0. :func:`ising_to_sparse_pauli_op` builds labels explicitly rather than relying
on a helper, and :func:`verify_hamiltonian` checks the result against the QUBO
energies term by term -- so if the convention ever shifts, a test fails instead
of a result being quietly wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from ..exceptions import MissingDependencyError, SolverError
from ..logging_utils import get_logger
from ..qubo.ising import IsingModel, qubo_to_ising
from ..qubo.matrix import QUBO, all_energies

if TYPE_CHECKING:  # pragma: no cover
    from qiskit.quantum_info import SparsePauliOp

__all__ = [
    "ising_to_sparse_pauli_op",
    "qubo_to_sparse_pauli_op",
    "hamiltonian_terms",
    "verify_hamiltonian",
    "diagonal_energies",
]

_LOG = get_logger(__name__)


def _import_sparse_pauli_op() -> Any:
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError("qiskit", "building the problem Hamiltonian") from exc
    return SparsePauliOp


def _pauli_label(n_qubits: int, positions: Dict[int, str]) -> str:
    """Build a Pauli label with *positions* mapping qubit index to letter.

    Qubit 0 is the **rightmost** character.

    >>> _pauli_label(3, {0: "Z"})
    'IIZ'
    >>> _pauli_label(3, {0: "Z", 2: "Z"})
    'ZIZ'
    """
    characters = ["I"] * n_qubits
    for qubit, letter in positions.items():
        if not 0 <= qubit < n_qubits:
            raise SolverError(f"Qubit index {qubit} out of range for {n_qubits} qubits")
        characters[qubit] = letter
    return "".join(reversed(characters))


def hamiltonian_terms(
    ising: IsingModel,
    *,
    include_constant: bool = True,
    tolerance: float = 1e-12,
) -> List[Tuple[str, float]]:
    """The Hamiltonian as a list of ``(pauli_label, coefficient)`` pairs.

    Terms below *tolerance* are dropped: they contribute nothing to the energy but
    would each add a gate to the cost layer, so pruning them is a free reduction
    in circuit depth.
    """
    n = ising.n_spins
    terms: List[Tuple[str, float]] = []

    for qubit, field in enumerate(ising.h):
        if abs(field) > tolerance:
            terms.append((_pauli_label(n, {qubit: "Z"}), float(field)))

    for (i, j), coupling in sorted(ising.couplings(tolerance=tolerance).items()):
        terms.append((_pauli_label(n, {i: "Z", j: "Z"}), float(coupling)))

    if include_constant and abs(ising.constant) > tolerance:
        terms.append(("I" * n, float(ising.constant)))

    if not terms:
        # SparsePauliOp.from_list rejects an empty list; a zero identity is the
        # honest representation of a Hamiltonian with no surviving terms.
        terms.append(("I" * n, 0.0))
    return terms


def ising_to_sparse_pauli_op(
    ising: IsingModel,
    *,
    include_constant: bool = True,
    tolerance: float = 1e-12,
) -> "SparsePauliOp":
    """Convert an :class:`~qroute.qubo.ising.IsingModel` to a ``SparsePauliOp``.

    The constant term is a multiple of the identity. Including it changes nothing
    about which state minimises the energy -- and in the circuit it is only a
    global phase -- but it *does* make expectation values directly comparable to
    QUBO energies, which is worth the one extra term.
    """
    SparsePauliOp = _import_sparse_pauli_op()
    terms = hamiltonian_terms(
        ising, include_constant=include_constant, tolerance=tolerance
    )
    operator = SparsePauliOp.from_list(terms)
    _LOG.debug(
        "Hamiltonian: %d qubit(s), %d Pauli term(s)", ising.n_spins, len(terms)
    )
    return operator


def qubo_to_sparse_pauli_op(qubo: QUBO, **kwargs: Any) -> "SparsePauliOp":
    """Shortcut: QUBO to ``SparsePauliOp`` via the Ising form."""
    return ising_to_sparse_pauli_op(qubo_to_ising(qubo), **kwargs)


def diagonal_energies(qubo: QUBO, *, max_variables: int = 22) -> np.ndarray:
    """The Hamiltonian diagonal, indexed by statevector basis index.

    Basis index :math:`k` has qubit 0 as its least significant bit, which is
    exactly the convention of
    :func:`~qroute.qubo.matrix.enumerate_assignments` -- so
    ``probabilities @ diagonal_energies(qubo)`` is the exact expectation value
    with no reindexing and no shot noise.

    The default limit is lower than the brute-force solver's because a
    statevector of the same size has to live in memory beside this array.
    """
    return all_energies(qubo, max_variables=max_variables)


def verify_hamiltonian(
    qubo: QUBO,
    operator: Optional["SparsePauliOp"] = None,
    *,
    max_variables: int = 12,
    tolerance: float = 1e-8,
) -> Tuple[bool, float]:
    """Check that the operator's diagonal really is the QUBO energy function.

    Returns ``(matches, max_absolute_difference)``. This is the guard rail for the
    two conventions that are easy to get backwards -- the
    :math:`x = (1 - z)/2` orientation and Qiskit's right-to-left Pauli labels.
    Getting either wrong produces a Hamiltonian that is perfectly valid, runs
    without error, and optimises the wrong function.

    Densifies the operator, so keep *max_variables* small; 12 qubits is a
    4096x4096 matrix, which is plenty to catch a convention error.
    """
    n = qubo.n_variables
    if n > max_variables:
        raise SolverError(
            f"Verification densifies a 2^{n} x 2^{n} matrix; the limit is "
            f"{max_variables} qubits. Verify on a small instance instead -- the "
            f"conventions do not depend on size."
        )
    if operator is None:
        operator = qubo_to_sparse_pauli_op(qubo)

    matrix = np.asarray(operator.to_matrix())
    off_diagonal = float(np.abs(matrix - np.diag(np.diag(matrix))).max(initial=0.0))
    if off_diagonal > tolerance:
        raise SolverError(
            f"Problem Hamiltonian is not diagonal (max off-diagonal {off_diagonal:.3g}); "
            f"only Z-type terms should be present"
        )

    expected = all_energies(qubo, max_variables=max_variables)
    actual = np.real(np.diag(matrix))
    difference = float(np.abs(actual - expected).max(initial=0.0))
    matches = difference <= tolerance
    if not matches:
        _LOG.error(
            "Hamiltonian diagonal disagrees with the QUBO by up to %.3g -- check the "
            "Ising conversion and Pauli label ordering",
            difference,
        )
    return matches, difference
