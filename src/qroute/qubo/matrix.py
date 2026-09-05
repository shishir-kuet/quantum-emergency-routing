"""The QUBO container and its builder.

Conventions (read this before touching any formulation)
-------------------------------------------------------
A QUBO here is the pair ``(Q, offset)`` defining

.. math::

    E(x) = x^\\mathsf{T} Q x + \\text{offset}, \\qquad x \\in \\{0, 1\\}^n

with **Q stored symmetrically**: the diagonal holds linear coefficients (valid
because :math:`x_i^2 = x_i`) and each quadratic coefficient :math:`q_{ij}` is
split evenly across ``Q[i, j]`` and ``Q[j, i]``. Expanding,

.. math::

    E(x) = \\sum_i Q_{ii} x_i + \\sum_{i<j} 2 Q_{ij} x_i x_j + \\text{offset}

The symmetric convention is chosen over upper-triangular because it makes
``x @ Q @ x`` correct as written, keeps the Ising transform symmetric, and
matches what ``SparsePauliOp`` construction expects.

Bit ordering
------------
Qiskit returns measurement outcomes as strings whose **rightmost character is
qubit 0**. Formulation code indexes variables left-to-right from 0. Every
crossing of that boundary must go through :func:`bitstring_to_array` /
:func:`array_to_bitstring` -- never index a raw Qiskit bitstring directly. This
is the single most common source of silently-wrong QAOA results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..exceptions import FormulationError

__all__ = [
    "QUBO",
    "QUBOBuilder",
    "bitstring_to_array",
    "array_to_bitstring",
    "enumerate_assignments",
    "all_energies",
]

TermSpec = Union[Mapping[int, float], Iterable[int]]

#: Rows per chunk when enumerating the whole space. 2**16 keeps peak memory in
#: the low megabytes whatever ``n`` is.
ENUMERATION_CHUNK = 1 << 16


# ---------------------------------------------------------------------------
# Bit-order helpers
# ---------------------------------------------------------------------------
def bitstring_to_array(bits: str, n_variables: int, *, qiskit_order: bool = True) -> np.ndarray:
    """Convert a measurement bitstring to a ``{0, 1}`` array indexed by variable.

    Parameters
    ----------
    bits:
        The bitstring. Spaces (which Qiskit inserts between classical
        registers) are stripped.
    n_variables:
        Expected length after stripping.
    qiskit_order:
        ``True`` (default) treats the input as little-endian -- rightmost
        character is variable 0, as produced by Qiskit samplers. ``False``
        treats it as already left-to-right.

    Examples
    --------
    >>> bitstring_to_array("100", 3).tolist()   # qubit 0 is the '0' on the right
    [0, 0, 1]
    >>> bitstring_to_array("100", 3, qiskit_order=False).tolist()
    [1, 0, 0]
    """
    clean = str(bits).replace(" ", "").strip()
    if len(clean) != n_variables:
        raise FormulationError(
            f"Bitstring has {len(clean)} bit(s) but the formulation has "
            f"{n_variables} variable(s): {bits!r}"
        )
    if any(character not in "01" for character in clean):
        raise FormulationError(f"Bitstring contains non-binary characters: {bits!r}")

    ordered = clean[::-1] if qiskit_order else clean
    return np.fromiter((int(character) for character in ordered), dtype=np.int8, count=n_variables)


def array_to_bitstring(x: Sequence[int], *, qiskit_order: bool = True) -> str:
    """Inverse of :func:`bitstring_to_array`.

    >>> array_to_bitstring([0, 0, 1])
    '100'
    >>> array_to_bitstring([0, 0, 1], qiskit_order=False)
    '001'
    """
    text = "".join("1" if int(value) else "0" for value in x)
    return text[::-1] if qiskit_order else text


# ---------------------------------------------------------------------------
# Exhaustive enumeration
# ---------------------------------------------------------------------------
def enumerate_assignments(n_variables: int, start: int, stop: int) -> np.ndarray:
    """All assignments for integers ``[start, stop)``, one per row.

    Variable ``k`` is bit ``k`` of the integer, so variable 0 is the least
    significant bit. That is deliberately the *same* convention Qiskit uses for
    statevector indices, which is what lets a statevector's probability array be
    dotted straight into an energy array without any reindexing.

    >>> enumerate_assignments(2, 0, 4).tolist()
    [[0, 0], [1, 0], [0, 1], [1, 1]]
    """
    integers = np.arange(start, stop, dtype=np.int64)
    shifts = np.arange(n_variables, dtype=np.int64)
    return ((integers[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def all_energies(qubo: "QUBO", *, max_variables: int = 24) -> np.ndarray:
    """Energies of every assignment, indexed as in :func:`enumerate_assignments`.

    Computed in chunks so memory stays bounded by the output array itself. The
    result is the diagonal of the problem Hamiltonian, which is why the quantum
    side can reuse it to get an exact, shot-noise-free expectation value.
    """
    n = qubo.n_variables
    if n > max_variables:
        raise FormulationError(
            f"Enumerating {n} variables means {2 ** n:,} energies "
            f"({2 ** n * 8 / 2 ** 30:.1f} GiB), above the {max_variables}-variable "
            f"limit"
        )
    total = 1 << n
    energies = np.empty(total, dtype=float)
    for start in range(0, total, ENUMERATION_CHUNK):
        stop = min(start + ENUMERATION_CHUNK, total)
        energies[start:stop] = qubo.energies(enumerate_assignments(n, start, stop))
    return energies


# ---------------------------------------------------------------------------
# The QUBO
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class QUBO:
    """An immutable QUBO instance."""

    Q: np.ndarray
    offset: float = 0.0
    labels: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.Q, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise FormulationError(f"Q must be a square matrix, got shape {matrix.shape}")
        if not np.allclose(matrix, matrix.T, atol=1e-9, rtol=0.0):
            raise FormulationError(
                "Q must be symmetric. Use QUBOBuilder, which maintains symmetry "
                "automatically."
            )
        # Bypass frozen-ness once, to normalise the stored dtype.
        object.__setattr__(self, "Q", matrix)
        if self.labels and len(self.labels) != matrix.shape[0]:
            raise FormulationError(
                f"Got {len(self.labels)} label(s) for {matrix.shape[0]} variable(s)"
            )

    # -- shape --------------------------------------------------------------
    @property
    def n_variables(self) -> int:
        return int(self.Q.shape[0])

    def label(self, index: int) -> str:
        return self.labels[index] if self.labels else f"x[{index}]"

    # -- coefficient views --------------------------------------------------
    def linear(self) -> np.ndarray:
        """The linear coefficients (the diagonal of ``Q``)."""
        return np.diag(self.Q).copy()

    def quadratic(self, *, tolerance: float = 1e-12) -> Dict[Tuple[int, int], float]:
        """Non-zero quadratic coefficients ``{(i, j): q_ij}`` for ``i < j``.

        Note these are the *physical* coefficients :math:`q_{ij} = 2 Q_{ij}`,
        i.e. what multiplies :math:`x_i x_j` in the expanded energy.
        """
        terms: Dict[Tuple[int, int], float] = {}
        n = self.n_variables
        for i in range(n):
            for j in range(i + 1, n):
                coefficient = 2.0 * self.Q[i, j]
                if abs(coefficient) > tolerance:
                    terms[(i, j)] = float(coefficient)
        return terms

    def max_abs_coefficient(self) -> float:
        """Largest absolute coefficient, used for normalisation and for
        estimating how much dynamic range a device would need."""
        peak = float(np.abs(self.linear()).max(initial=0.0))
        quadratic = self.quadratic()
        if quadratic:
            peak = max(peak, max(abs(value) for value in quadratic.values()))
        return peak

    def density(self) -> float:
        """Fraction of the :math:`n(n-1)/2` possible couplings that are non-zero.

        Dense QUBOs need all-to-all connectivity, which on real hardware means
        many SWAP gates -- so this number is a rough proxy for how badly the
        circuit will suffer during transpilation.
        """
        n = self.n_variables
        possible = n * (n - 1) // 2
        return len(self.quadratic()) / possible if possible else 0.0

    # -- evaluation ---------------------------------------------------------
    def energy(self, x: Sequence[int]) -> float:
        """Energy of a single assignment."""
        vector = np.asarray(x, dtype=float).reshape(-1)
        if vector.size != self.n_variables:
            raise FormulationError(
                f"Expected {self.n_variables} variable(s), got {vector.size}"
            )
        return float(vector @ self.Q @ vector + self.offset)

    def energies(self, X: np.ndarray) -> np.ndarray:
        """Energies of a batch of assignments, one per row of *X*."""
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_variables:
            raise FormulationError(
                f"Expected an (m, {self.n_variables}) array, got {matrix.shape}"
            )
        return np.einsum("ij,jk,ik->i", matrix, self.Q, matrix) + self.offset

    def energy_of_bitstring(self, bits: str, *, qiskit_order: bool = True) -> float:
        """Energy of a measurement bitstring, handling bit-order conversion."""
        return self.energy(bitstring_to_array(bits, self.n_variables, qiskit_order=qiskit_order))

    # -- transforms ---------------------------------------------------------
    def scaled(self, factor: float) -> "QUBO":
        """Return ``factor * self``. Scaling cannot change the argmin."""
        if factor == 0:
            raise FormulationError("Refusing to scale a QUBO by zero")
        metadata = dict(self.metadata)
        metadata["scale_applied"] = metadata.get("scale_applied", 1.0) * float(factor)
        return QUBO(
            Q=self.Q * float(factor),
            offset=self.offset * float(factor),
            labels=self.labels,
            metadata=metadata,
        )

    def normalised(self) -> "QUBO":
        """Rescale so the largest absolute coefficient is 1.

        Purely cosmetic for exact solvers, but it matters for QAOA: the cost
        unitary applies rotations of angle proportional to the coefficients, so
        raw travel-time coefficients in the hundreds send every angle spinning
        many times around the circle and make the optimiser's landscape
        needlessly hostile.
        """
        peak = self.max_abs_coefficient()
        if peak <= 0:
            return self
        return self.scaled(1.0 / peak)

    # -- reporting ----------------------------------------------------------
    def describe(self) -> str:
        return (
            f"QUBO(n={self.n_variables}, couplings={len(self.quadratic())}, "
            f"density={self.density():.2f}, "
            f"max|coeff|={self.max_abs_coefficient():.4g}, offset={self.offset:.4g})"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Q": self.Q.tolist(),
            "offset": self.offset,
            "labels": list(self.labels),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------
class QUBOBuilder:
    """Accumulate QUBO terms, then :meth:`build` an immutable :class:`QUBO`.

    All ``add_*`` methods are additive, so overlapping contributions from the
    objective and several penalties simply sum -- which is exactly how
    constrained problems are assembled.

    Examples
    --------
    A one-hot constraint over two variables, with a preference for the first:

    >>> builder = QUBOBuilder(2, labels=("a", "b"))
    >>> builder.add_linear(0, -1.0)                      # doctest: +ELLIPSIS
    <...QUBOBuilder object at ...>
    >>> builder.add_penalty_equality([0, 1], target=1, weight=10.0)  # doctest: +ELLIPSIS
    <...QUBOBuilder object at ...>
    >>> qubo = builder.build()
    >>> qubo.energy([1, 0]), qubo.energy([0, 1]), qubo.energy([1, 1])
    (-1.0, 0.0, 9.0)
    """

    def __init__(self, n_variables: int, labels: Optional[Sequence[str]] = None) -> None:
        if n_variables < 1:
            raise FormulationError(f"A QUBO needs at least one variable, got {n_variables}")
        self._n = int(n_variables)
        self._Q = np.zeros((self._n, self._n), dtype=float)
        self._offset = 0.0
        self._labels: Tuple[str, ...] = tuple(labels) if labels is not None else ()
        if self._labels and len(self._labels) != self._n:
            raise FormulationError(
                f"Got {len(self._labels)} label(s) for {self._n} variable(s)"
            )

    # -- guards -------------------------------------------------------------
    def _check(self, index: int) -> int:
        value = int(index)
        if not 0 <= value < self._n:
            raise FormulationError(
                f"Variable index {index} out of range for {self._n} variable(s)"
            )
        return value

    # -- primitives ---------------------------------------------------------
    def add_constant(self, value: float) -> "QUBOBuilder":
        self._offset += float(value)
        return self

    def add_linear(self, index: int, coefficient: float) -> "QUBOBuilder":
        """Add ``coefficient * x[index]``."""
        i = self._check(index)
        self._Q[i, i] += float(coefficient)
        return self

    def add_quadratic(self, i: int, j: int, coefficient: float) -> "QUBOBuilder":
        """Add ``coefficient * x[i] * x[j]``.

        A term with ``i == j`` is folded into the linear part, since
        :math:`x_i^2 = x_i` for binary variables.
        """
        a, b = self._check(i), self._check(j)
        value = float(coefficient)
        if a == b:
            self._Q[a, a] += value
        else:
            self._Q[a, b] += value / 2.0
            self._Q[b, a] += value / 2.0
        return self

    # -- composite ----------------------------------------------------------
    @staticmethod
    def _as_terms(terms: TermSpec) -> Dict[int, float]:
        """Normalise a term spec to ``{index: coefficient}``.

        A bare iterable of indices is read as all-ones coefficients, which is
        what one-hot constraints need.
        """
        if isinstance(terms, Mapping):
            return {int(key): float(value) for key, value in terms.items()}
        collected: Dict[int, float] = {}
        for index in terms:
            key = int(index)
            if key in collected:
                raise FormulationError(
                    f"Variable {key} appears twice in a constraint; pass an explicit "
                    f"coefficient mapping instead"
                )
            collected[key] = 1.0
        return collected

    def add_penalty_equality(
        self, terms: TermSpec, target: float, weight: float
    ) -> "QUBOBuilder":
        r"""Add :math:`w \left(\sum_k a_k x_k - t\right)^2`.

        This is the workhorse for hard constraints. Expanding with
        :math:`x_k^2 = x_k` gives

        .. math::

            w \sum_k (a_k^2 - 2 t a_k) x_k
            + 2 w \sum_{k<l} a_k a_l x_k x_l
            + w t^2

        Passing a plain list of indices with ``target=1`` yields the standard
        one-hot penalty.
        """
        if weight <= 0:
            raise FormulationError(f"Penalty weight must be positive, got {weight}")

        coefficients = self._as_terms(terms)
        for index in coefficients:
            self._check(index)

        w = float(weight)
        t = float(target)
        self.add_constant(w * t * t)

        items = list(coefficients.items())
        for index, a in items:
            self.add_linear(index, w * (a * a - 2.0 * t * a))
        for position, (index_i, a) in enumerate(items):
            for index_j, b in items[position + 1 :]:
                self.add_quadratic(index_i, index_j, 2.0 * w * a * b)
        return self

    def add_linear_terms(self, terms: Mapping[int, float]) -> "QUBOBuilder":
        """Add several linear coefficients at once."""
        for index, coefficient in terms.items():
            self.add_linear(index, coefficient)
        return self

    # -- output -------------------------------------------------------------
    def build(self, metadata: Optional[Mapping[str, Any]] = None) -> QUBO:
        """Freeze the accumulated terms into a :class:`QUBO`."""
        return QUBO(
            Q=self._Q.copy(),
            offset=self._offset,
            labels=self._labels,
            metadata=dict(metadata or {}),
        )
