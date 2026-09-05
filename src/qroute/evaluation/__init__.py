"""Turning raw solver output into defensible numbers.

Two modules, answering the two halves of a NISQ result:

``metrics``
    How good is the answer? Approximation ratio (only against a *provable*
    optimum), optimality gap, feasibility rate, success probability, and
    time-to-solution -- plus :func:`compare_results`, which assembles a
    side-by-side table and structurally refuses to divide by a heuristic.

``circuit_stats``
    What did it cost? Depth, gate counts, two-qubit depth, a transpilation
    before/after report, and a deliberately crude fidelity bound.

Read both columns together. A route 3% off optimal is a good result; a route 3%
off optimal that needed 900 two-qubit gates on a device with 0.8% two-qubit error
is a good *simulation* result and nothing more, and the honest write-up says so.
"""

from __future__ import annotations

from .circuit_stats import (
    DEFAULT_ONE_QUBIT_ERROR,
    DEFAULT_READOUT_ERROR,
    DEFAULT_TWO_QUBIT_ERROR,
    CircuitStats,
    circuit_stats,
    compare_transpilation,
    estimate_fidelity,
    predict_two_qubit_count,
)
from .metrics import (
    DEFAULT_TOLERANCE,
    ComparisonRow,
    SampleStatistics,
    analyse_counts,
    approximation_ratio,
    comparison_table,
    compare_results,
    feasibility_rate,
    optimality_gap,
    success_probability,
    time_to_solution,
)

__all__ = [
    # metrics
    "SampleStatistics",
    "ComparisonRow",
    "approximation_ratio",
    "optimality_gap",
    "analyse_counts",
    "success_probability",
    "feasibility_rate",
    "time_to_solution",
    "compare_results",
    "comparison_table",
    "DEFAULT_TOLERANCE",
    # circuit_stats
    "CircuitStats",
    "circuit_stats",
    "estimate_fidelity",
    "predict_two_qubit_count",
    "compare_transpilation",
    "DEFAULT_ONE_QUBIT_ERROR",
    "DEFAULT_TWO_QUBIT_ERROR",
    "DEFAULT_READOUT_ERROR",
]
