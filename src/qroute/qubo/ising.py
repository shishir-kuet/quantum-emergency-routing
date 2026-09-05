r"""QUBO :math:`\leftrightarrow` Ising conversion.

QAOA acts on spins, not bits. The bridge is the substitution

.. math::

    x_i = \frac{1 - z_i}{2}, \qquad z_i \in \{-1, +1\}

which pairs :math:`x_i = 0` with :math:`z_i = +1` (the :math:`|0\rangle`
eigenstate of :math:`Z`) and :math:`x_i = 1` with :math:`z_i = -1`. That
orientation is not arbitrary -- it is what makes :math:`Z` in the Hamiltonian
agree with Qiskit's computational basis, so a measured ``1`` really does mean
"variable selected".

Substituting into
:math:`E = \sum_i Q_{ii} x_i + \sum_{i<j} q_{ij} x_i x_j + \text{offset}`
(where :math:`q_{ij} = 2 Q_{ij}`) and collecting terms gives

.. math::

    J_{ij} &= \tfrac{1}{4} q_{ij} \\
    h_i &= -\tfrac{1}{2} Q_{ii} - \tfrac{1}{4} \sum_{j \neq i} q_{ij} \\
    c &= \text{offset} + \tfrac{1}{2} \sum_i Q_{ii}
         + \tfrac{1}{4} \sum_{i<j} q_{ij}

so that :math:`E = \sum_i h_i z_i + \sum_{i<j} J_{ij} z_i z_j + c`.

The transform is exact and invertible; :func:`ising_to_qubo` implements the
inverse and the test suite round-trips random instances through both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ..exceptions import FormulationError
from .matrix import QUBO

__all__ = ["IsingModel", "qubo_to_ising", "ising_to_qubo"]


@dataclass(frozen=True, eq=False)
class IsingModel:
    r"""Ising Hamiltonian :math:`\sum_i h_i z_i + \sum_{i<j} J_{ij} z_i z_j + c`.

    Attributes
    ----------
    h:
        Local fields, shape ``(n,)``.
    J:
        Couplings, shape ``(n, n)``, stored **strictly upper-triangular** --
        only ``J[i, j]`` with ``i < j`` is populated, so there is exactly one
        entry per pair and no factor-of-two ambiguity.
    constant:
        Energy offset. Irrelevant to the argmin, essential for comparing
        expectation values against QUBO energies.
    """

    h: np.ndarray
    J: np.ndarray
    constant: float = 0.0
    labels: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h = np.asarray(self.h, dtype=float).reshape(-1)
        J = np.asarray(self.J, dtype=float)
        n = h.size
        if J.shape != (n, n):
            raise FormulationError(f"J shape {J.shape} incompatible with h size {n}")
        if not np.allclose(np.tril(J), 0.0, atol=1e-12):
            raise FormulationError(
                "J must be strictly upper-triangular (one entry per pair, i < j)"
            )
        object.__setattr__(self, "h", h)
        object.__setattr__(self, "J", J)

    @property
    def n_spins(self) -> int:
        return int(self.h.size)

    def couplings(self, *, tolerance: float = 1e-12) -> Dict[Tuple[int, int], float]:
        """Non-zero couplings as ``{(i, j): J_ij}`` with ``i < j``."""
        indices = np.argwhere(np.abs(self.J) > tolerance)
        return {(int(i), int(j)): float(self.J[i, j]) for i, j in indices}

    def energy(self, z: Sequence[int]) -> float:
        """Energy of a spin configuration with entries in ``{-1, +1}``."""
        spins = np.asarray(z, dtype=float).reshape(-1)
        if spins.size != self.n_spins:
            raise FormulationError(
                f"Expected {self.n_spins} spin(s), got {spins.size}"
            )
        if not np.all(np.isin(spins, (-1.0, 1.0))):
            raise FormulationError("Spins must all be -1 or +1")
        return float(self.h @ spins + spins @ self.J @ spins + self.constant)

    def describe(self) -> str:
        return (
            f"IsingModel(n={self.n_spins}, couplings={len(self.couplings())}, "
            f"max|h|={np.abs(self.h).max(initial=0.0):.4g}, "
            f"max|J|={np.abs(self.J).max(initial=0.0):.4g}, "
            f"constant={self.constant:.4g})"
        )


def qubo_to_ising(qubo: QUBO) -> IsingModel:
    """Convert a :class:`~qroute.qubo.matrix.QUBO` to an :class:`IsingModel`.

    Examples
    --------
    A single variable with energy ``E(x) = x``:

    >>> import numpy as np
    >>> from qroute.qubo.matrix import QUBO
    >>> ising = qubo_to_ising(QUBO(Q=np.array([[1.0]])))
    >>> float(ising.h[0]), ising.constant
    (-0.5, 0.5)
    >>> ising.energy([+1]), ising.energy([-1])     # x = 0, then x = 1
    (0.0, 1.0)
    """
    n = qubo.n_variables
    diagonal = np.diag(qubo.Q).astype(float)

    # q_ij = 2 * Q_ij for i != j; take only the upper triangle.
    q_upper = 2.0 * np.triu(qubo.Q, k=1)

    J = q_upper / 4.0

    # Each pair (i, j) contributes -q_ij / 4 to both h_i and h_j.
    pair_row_sums = q_upper.sum(axis=1) + q_upper.sum(axis=0)
    h = -diagonal / 2.0 - pair_row_sums / 4.0

    constant = float(qubo.offset + diagonal.sum() / 2.0 + q_upper.sum() / 4.0)

    metadata = dict(qubo.metadata)
    metadata["from_qubo"] = True
    return IsingModel(h=h, J=J, constant=constant, labels=qubo.labels, metadata=metadata)


def ising_to_qubo(ising: IsingModel) -> QUBO:
    r"""Inverse of :func:`qubo_to_ising`, substituting :math:`z_i = 1 - 2 x_i`.

    >>> import numpy as np
    >>> from qroute.qubo.matrix import QUBO
    >>> original = QUBO(Q=np.array([[1.0, 0.5], [0.5, -2.0]]), offset=0.25)
    >>> recovered = ising_to_qubo(qubo_to_ising(original))
    >>> bool(np.allclose(recovered.Q, original.Q))
    True
    >>> bool(np.isclose(recovered.offset, original.offset))
    True
    """
    n = ising.n_spins
    J_upper = np.triu(ising.J, k=1)

    # q_ij = 4 J_ij  ->  symmetric storage Q_ij = Q_ji = 2 J_ij
    Q = 2.0 * (J_upper + J_upper.T)

    # Q_ii = -2 h_i - 2 * sum_{j != i} J_ij
    pair_row_sums = J_upper.sum(axis=1) + J_upper.sum(axis=0)
    diagonal = -2.0 * ising.h - 2.0 * pair_row_sums
    Q[np.diag_indices(n)] = diagonal

    offset = float(ising.constant + ising.h.sum() + J_upper.sum())

    metadata = dict(ising.metadata)
    metadata["from_ising"] = True
    return QUBO(Q=Q, offset=offset, labels=ising.labels, metadata=metadata)
