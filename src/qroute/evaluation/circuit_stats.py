r"""Circuit resource accounting: depth, gate counts, and what they imply.

Solution quality is only half of a NISQ result. The other half is what the
circuit costs, because on real hardware the two are directly opposed: more
layers (:math:`p`) improves the ansatz's expressivity *and* multiplies the
accumulated error until the output is noise.

Why two-qubit gates get their own field
--------------------------------------
On superconducting devices the two-qubit gate (CX, ECR, CZ) is roughly an order
of magnitude noisier than any single-qubit rotation, so the two-qubit count -- not
the total gate count, and not even the depth -- is the number that predicts
whether a circuit will produce a signal. A QAOA cost layer needs one ``RZZ`` per
QUBO coupling, and each ``RZZ`` compiles to two CX gates, so:

.. math::

    n_{\text{2q}} \approx 2 \, p \, |\text{couplings}|

which for a dense :math:`n`-variable QUBO is :math:`\approx p\,n(n-1)`. That
quadratic growth in the *number of variables* is the real constraint on problem
size, and it is why the position-indexed TSP encoding (:math:`(n-1)^2` variables)
hits a wall around six or seven cities.

Fidelity estimate
-----------------
:func:`estimate_fidelity` multiplies per-gate success probabilities:
:math:`F \approx (1 - \epsilon_1)^{n_1} (1 - \epsilon_2)^{n_2}`. This is a crude
upper bound -- it ignores crosstalk, idling decoherence, measurement error, and
any correlation between errors -- but crude is enough to make the point. At
realistic error rates a few hundred two-qubit gates leaves essentially nothing,
and knowing that *before* queueing a hardware job saves a lot of time.

The default error rates are order-of-magnitude figures for a good
superconducting device circa 2025 and are **not** calibration data for any
specific machine. Anything published needs real backend properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ..logging_utils import get_logger
from ..qubo.matrix import QUBO

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

__all__ = [
    "CircuitStats",
    "circuit_stats",
    "estimate_fidelity",
    "predict_two_qubit_count",
    "compare_transpilation",
]

_LOG = get_logger(__name__)

#: Indicative single- and two-qubit error rates. Order of magnitude only.
DEFAULT_ONE_QUBIT_ERROR = 3e-4
DEFAULT_TWO_QUBIT_ERROR = 8e-3
DEFAULT_READOUT_ERROR = 1e-2

#: Gate names counted as two-qubit entangling operations across Qiskit versions
#: and target basis sets.
TWO_QUBIT_GATES = frozenset(
    {"cx", "cz", "cy", "ch", "ecr", "rzz", "rxx", "ryy", "rzx", "swap", "iswap", "cp", "crz"}
)


@dataclass(frozen=True, eq=False)
class CircuitStats:
    """Resource footprint of one circuit."""

    n_qubits: int
    depth: int
    size: int
    two_qubit_gates: int
    one_qubit_gates: int
    two_qubit_depth: int
    gate_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def entangling_fraction(self) -> float:
        """Share of operations that are two-qubit gates."""
        return self.two_qubit_gates / self.size if self.size else 0.0

    def describe(self) -> str:
        return (
            f"CircuitStats(qubits={self.n_qubits}, depth={self.depth}, "
            f"ops={self.size}, 2q={self.two_qubit_gates} "
            f"({self.entangling_fraction:.0%}), 2q-depth={self.two_qubit_depth})"
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "size": self.size,
            "two_qubit_gates": self.two_qubit_gates,
            "one_qubit_gates": self.one_qubit_gates,
            "two_qubit_depth": self.two_qubit_depth,
            "entangling_fraction": self.entangling_fraction,
            "gate_counts": dict(self.gate_counts),
            "metadata": dict(self.metadata),
        }


def circuit_stats(circuit: "QuantumCircuit", **metadata: Any) -> CircuitStats:
    """Measure a circuit's resource footprint.

    ``depth(filter_function=...)`` gives the *critical path through two-qubit
    gates only*, which is the better proxy for accumulated error than total depth:
    single-qubit rotations are nearly free, and a transpiler that trades one CX
    for six single-qubit gates has made the circuit better, not worse, even though
    total depth went up.
    """
    counts = {str(name): int(value) for name, value in circuit.count_ops().items()}

    two_qubit = 0
    one_qubit = 0
    for instruction in circuit.data:
        operation = instruction.operation
        name = operation.name
        if name in {"barrier", "measure", "delay", "reset", "snapshot"}:
            continue
        qubit_count = len(instruction.qubits)
        if qubit_count >= 2 or name in TWO_QUBIT_GATES:
            two_qubit += 1
        else:
            one_qubit += 1

    try:
        two_qubit_depth = int(
            circuit.depth(lambda instruction: len(instruction.qubits) >= 2)
        )
    except Exception:  # pragma: no cover - older Qiskit passed a 3-tuple instead
        _LOG.debug("depth(filter_function=...) unsupported; reporting 2q-depth as 0")
        two_qubit_depth = 0

    stats = CircuitStats(
        n_qubits=int(circuit.num_qubits),
        depth=int(circuit.depth()),
        size=int(circuit.size()),
        two_qubit_gates=two_qubit,
        one_qubit_gates=one_qubit,
        two_qubit_depth=two_qubit_depth,
        gate_counts=counts,
        metadata=dict(metadata),
    )
    _LOG.debug("%s", stats.describe())
    return stats


def predict_two_qubit_count(qubo: QUBO, reps: int, *, cx_per_rzz: int = 2) -> int:
    r"""Predicted CX count for a QAOA circuit on *qubo*, before transpiling.

    One ``RZZ`` per non-zero coupling per layer, each decomposing to
    *cx_per_rzz* CX gates. Useful as a sanity check on transpiler output: if the
    transpiled count is far *above* this, routing SWAPs are dominating and the
    coupling map -- not the encoding -- is the bottleneck.

    >>> import numpy as np
    >>> from qroute.qubo.matrix import QUBO
    >>> qubo = QUBO(Q=np.array([[1.0, 0.5], [0.5, 1.0]]))
    >>> predict_two_qubit_count(qubo, reps=3)
    6
    """
    return int(len(qubo.quadratic()) * int(reps) * int(cx_per_rzz))


def estimate_fidelity(
    stats: CircuitStats,
    *,
    one_qubit_error: float = DEFAULT_ONE_QUBIT_ERROR,
    two_qubit_error: float = DEFAULT_TWO_QUBIT_ERROR,
    readout_error: float = DEFAULT_READOUT_ERROR,
) -> Tuple[float, Dict[str, float]]:
    """Crude fidelity upper bound, plus the per-source breakdown.

    Returns ``(fidelity, contributions)`` where *contributions* holds the
    surviving fidelity from each source separately, so it is obvious which one is
    doing the damage -- almost always the two-qubit gates.

    Read the module docstring before quoting this number anywhere.
    """
    one_qubit_term = (1.0 - one_qubit_error) ** stats.one_qubit_gates
    two_qubit_term = (1.0 - two_qubit_error) ** stats.two_qubit_gates
    readout_term = (1.0 - readout_error) ** stats.n_qubits
    fidelity = one_qubit_term * two_qubit_term * readout_term
    return (
        float(fidelity),
        {
            "one_qubit": float(one_qubit_term),
            "two_qubit": float(two_qubit_term),
            "readout": float(readout_term),
        },
    )


def compare_transpilation(
    logical: "QuantumCircuit",
    transpiled: "QuantumCircuit",
    *,
    qubo: Optional[QUBO] = None,
    reps: Optional[int] = None,
) -> str:
    """Human-readable before/after report for one transpilation.

    The overhead ratio is the honest measure of what hardware connectivity costs.
    A dense QUBO needs all-to-all coupling; a device offers a sparse lattice; the
    transpiler bridges the gap with SWAP networks, and this is the bill.
    """
    before = circuit_stats(logical, stage="logical")
    after = circuit_stats(transpiled, stage="transpiled")
    fidelity, breakdown = estimate_fidelity(after)

    lines = [
        f"{'metric':<24}{'logical':>12}{'transpiled':>14}{'factor':>10}",
        "-" * 60,
    ]

    def row(label: str, first: int, second: int) -> str:
        factor = second / first if first else float("nan")
        return f"{label:<24}{first:>12,}{second:>14,}{factor:>9.2f}x"

    lines.append(row("qubits", before.n_qubits, after.n_qubits))
    lines.append(row("depth", before.depth, after.depth))
    lines.append(row("operations", before.size, after.size))
    lines.append(row("two-qubit gates", before.two_qubit_gates, after.two_qubit_gates))
    lines.append(row("two-qubit depth", before.two_qubit_depth, after.two_qubit_depth))

    if qubo is not None and reps is not None:
        predicted = predict_two_qubit_count(qubo, reps)
        lines.append("")
        if predicted:
            overhead = after.two_qubit_gates / predicted
            lines.append(
                f"Predicted CX from couplings: {predicted:,} "
                f"(actual {after.two_qubit_gates:,}, {overhead:.2f}x -- "
                f"anything well above 1.0 is SWAP routing, not encoding)"
            )
        else:
            lines.append("Predicted CX from couplings: 0 (no quadratic terms)")

    lines.append("")
    lines.append(
        f"Rough fidelity bound: {fidelity:.4f} "
        f"(1q {breakdown['one_qubit']:.4f} x 2q {breakdown['two_qubit']:.4f} "
        f"x readout {breakdown['readout']:.4f})"
    )
    lines.append("Indicative error rates, not device calibration data.")
    return "\n".join(lines)
