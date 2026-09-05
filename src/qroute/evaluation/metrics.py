r"""Metrics for comparing a quantum result against a classical one honestly.

Most of the effort in this module goes into refusing to compute numbers that
would be misleading. Three traps in particular:

**An approximation ratio needs a provable optimum.** Dividing by a heuristic's
answer produces a number that looks like a ratio and means nothing -- and it can
exceed 1, which is how you discover the denominator was wrong. Every function
here demands ``optimal=True`` on its reference, or refuses.

**QAOA's mean energy is not its answer.** The variational objective is the mean
(or CVaR) energy; the *result* is the best feasible sample. Reporting the mean as
the solution quality understates QAOA badly; reporting the best sample without
also reporting how rare it was overstates it just as badly. Hence
:func:`success_probability` alongside :func:`approximation_ratio`.

**Feasibility is a first-class outcome.** On penalty-encoded constrained
problems most of the Hilbert space is invalid routes. A run where 2% of shots are
feasible and one of them is optimal is a very different result from one where 90%
are feasible and none is optimal, and a single "cost" column hides that
completely.

Timing
------
:func:`time_to_solution` is the fairest single-number comparison available on a
simulator, but it is still not a claim about hardware. Simulating :math:`n`
qubits costs :math:`O(2^n)` classically, so a simulator's wall-clock time says
nothing about what a device would take. Treat these numbers as a check that the
classical baseline was run competently, not as evidence of quantum advantage --
and say so in any write-up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..classical.base import ClassicalResult
from ..exceptions import QRouteError
from ..logging_utils import get_logger
from ..qubo.base import Formulation, RouteSolution
from ..quantum.qaoa import QAOAResult, sample_energies

__all__ = [
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
]

_LOG = get_logger(__name__)

#: Costs within this relative distance of the optimum count as "optimal found".
DEFAULT_TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Ratios and gaps
# ---------------------------------------------------------------------------
def approximation_ratio(
    achieved: Optional[float],
    reference: float,
    *,
    reference_is_optimal: bool = True,
) -> Optional[float]:
    """``reference / achieved`` for a minimisation problem.

    Returns a number in :math:`(0, 1]` where **1.0 means optimal** and smaller is
    worse, which is the orientation used throughout the QAOA literature. Returns
    ``None`` when *achieved* is ``None`` (nothing feasible was found) -- that is a
    real result and should be reported as "no feasible solution", not as a ratio
    of zero.

    >>> approximation_ratio(120.0, 100.0)
    0.8333333333333334
    >>> approximation_ratio(None, 100.0) is None
    True
    """
    if not reference_is_optimal:
        raise QRouteError(
            "Refusing to compute an approximation ratio against a non-optimal "
            "reference. Use optimality_gap() with an explicit note, or solve the "
            "instance exactly first (classical.bruteforce / classical.tsp with "
            "method='bruteforce')."
        )
    if achieved is None:
        return None
    if reference <= 0:
        raise QRouteError(
            f"Reference cost must be positive to form a ratio, got {reference}. "
            f"A zero-cost optimum means the instance is degenerate."
        )
    if achieved <= 0:
        raise QRouteError(
            f"Achieved cost {achieved} is non-positive, which is impossible for a "
            f"travel-time objective -- check the decode path"
        )
    return float(reference) / float(achieved)


def optimality_gap(achieved: Optional[float], reference: float) -> Optional[float]:
    """Excess cost over *reference*, as a percentage.

    ``0.0`` means the reference cost was matched, ``25.0`` means a quarter more
    expensive. Unlike :func:`approximation_ratio` this makes no claim that the
    reference is optimal, so it is the right function for "how far off the
    heuristic is the quantum answer".
    """
    if achieved is None:
        return None
    if reference <= 0:
        raise QRouteError(f"Reference cost must be positive, got {reference}")
    return 100.0 * (float(achieved) - float(reference)) / float(reference)


# ---------------------------------------------------------------------------
# Shot-level analysis
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SampleStatistics:
    """What a measured distribution actually contains.

    Decoding every distinct outcome is affordable because counts collapse
    duplicates: 4096 shots of a 20-qubit circuit typically yield a few thousand
    unique strings at most, and far fewer once the optimiser has concentrated the
    distribution.
    """

    n_shots: int
    n_unique: int
    feasible_shots: int
    optimal_shots: int
    mean_energy: float
    best_energy: float
    best_solution: Optional[RouteSolution]
    best_feasible_objective: Optional[float]
    feasible_objectives: Tuple[float, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feasibility_rate(self) -> float:
        """Fraction of shots that decoded to a valid route."""
        return self.feasible_shots / self.n_shots if self.n_shots else 0.0

    @property
    def success_probability(self) -> Optional[float]:
        """Fraction of shots that hit the optimum, if an optimum was supplied.

        This is the number that decides whether a QAOA run is *useful*. A
        best-sample cost equal to the optimum with a success probability of
        1/4096 means the circuit is barely better than random guessing at this
        size; the same cost at 0.3 means the distribution genuinely concentrated.
        """
        if self.metadata.get("reference_objective") is None:
            return None
        return self.optimal_shots / self.n_shots if self.n_shots else 0.0

    @property
    def mean_feasible_objective(self) -> Optional[float]:
        if not self.feasible_objectives:
            return None
        return float(np.mean(self.feasible_objectives))

    def describe(self) -> str:
        success = self.success_probability
        success_text = "n/a" if success is None else f"{success:.4f}"
        best = self.best_feasible_objective
        best_text = "none" if best is None else f"{best:.2f}"
        return (
            f"SampleStatistics(shots={self.n_shots}, unique={self.n_unique}, "
            f"feasible={self.feasibility_rate:.3f}, p(optimal)={success_text}, "
            f"best feasible cost={best_text}, <E>={self.mean_energy:.4g})"
        )


def analyse_counts(
    counts: Mapping[str, int],
    formulation: Formulation,
    *,
    reference_objective: Optional[float] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_unique: int = 200_000,
) -> SampleStatistics:
    """Decode every distinct outcome in *counts* and summarise the distribution.

    Parameters
    ----------
    reference_objective:
        The optimal route cost, if known. Supplying it enables
        :attr:`SampleStatistics.success_probability`; omitting it leaves that
        ``None`` rather than silently comparing against the best sample (which
        would always give 100% and mean nothing).
    tolerance:
        Relative tolerance for "equals the optimum".
    """
    if not counts:
        raise QRouteError("Cannot analyse an empty counts dictionary")
    if len(counts) > max_unique:
        raise QRouteError(
            f"{len(counts):,} distinct outcomes exceeds the {max_unique:,} decode "
            f"limit; reduce shots or raise max_unique deliberately"
        )

    qubo = formulation.qubo()
    assignments, energies, weights = sample_energies(dict(counts), qubo)

    n_shots = int(weights.sum())
    feasible_shots = 0
    optimal_shots = 0
    best_energy = float(energies.min())
    best_solution: Optional[RouteSolution] = None
    best_objective: Optional[float] = None
    feasible_objectives: List[float] = []

    threshold = (
        None
        if reference_objective is None
        else float(reference_objective) * (1.0 + tolerance) + tolerance
    )

    for row in range(assignments.shape[0]):
        shots = int(weights[row])
        solution = formulation.decode(assignments[row], qiskit_order=False)
        if not solution.feasible:
            continue
        feasible_shots += shots
        objective = solution.objective
        if objective is None:  # pragma: no cover - feasible implies an objective
            continue
        feasible_objectives.extend([float(objective)] * shots)
        if best_objective is None or objective < best_objective:
            best_objective = float(objective)
            best_solution = solution
        if threshold is not None and objective <= threshold:
            optimal_shots += shots

    statistics = SampleStatistics(
        n_shots=n_shots,
        n_unique=len(counts),
        feasible_shots=feasible_shots,
        optimal_shots=optimal_shots,
        mean_energy=float(energies @ weights / weights.sum()),
        best_energy=best_energy,
        best_solution=best_solution,
        best_feasible_objective=best_objective,
        feasible_objectives=tuple(feasible_objectives),
        metadata={
            "formulation": formulation.name,
            "reference_objective": reference_objective,
            "tolerance": tolerance,
        },
    )
    _LOG.info("Distribution: %s", statistics.describe())
    return statistics


def feasibility_rate(counts: Mapping[str, int], formulation: Formulation) -> float:
    """Shortcut for :attr:`SampleStatistics.feasibility_rate`."""
    return analyse_counts(counts, formulation).feasibility_rate


def success_probability(
    counts: Mapping[str, int],
    formulation: Formulation,
    reference_objective: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Probability that a single shot lands on an optimal route."""
    statistics = analyse_counts(
        counts, formulation, reference_objective=reference_objective, tolerance=tolerance
    )
    return statistics.success_probability or 0.0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
def time_to_solution(
    seconds: float, success_prob: Optional[float], *, target: float = 0.99
) -> Optional[float]:
    r"""Expected time to see an optimal sample with probability *target*.

    .. math::

        \mathrm{TTS} = t_{\text{run}} \cdot
        \frac{\log(1 - \text{target})}{\log(1 - p_s)}

    The standard figure of merit for stochastic optimisers, because it folds a
    low success probability and a fast run into one comparable number: a sampler
    that succeeds 1% of the time but runs 1000x faster genuinely is competitive.

    Returns ``None`` when :math:`p_s = 0` (never succeeded, so the time is
    unbounded) and *seconds* when :math:`p_s = 1`.
    """
    if success_prob is None or success_prob <= 0.0:
        return None
    if success_prob >= 1.0:
        return float(seconds)
    repeats = np.log1p(-target) / np.log1p(-success_prob)
    return float(seconds) * float(repeats)


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ComparisonRow:
    """One solver's line in a comparison table."""

    solver: str
    objective: Optional[float]
    feasible: bool
    seconds: float
    optimal: bool = False
    approximation: Optional[float] = None
    gap_percent: Optional[float] = None
    success_prob: Optional[float] = None
    feasibility: Optional[float] = None
    tts: Optional[float] = None
    n_qubits: Optional[int] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "solver": self.solver,
            "objective": self.objective,
            "feasible": self.feasible,
            "seconds": self.seconds,
            "optimal": self.optimal,
            "approximation": self.approximation,
            "gap_percent": self.gap_percent,
            "success_prob": self.success_prob,
            "feasibility": self.feasibility,
            "tts": self.tts,
            "n_qubits": self.n_qubits,
            "notes": self.notes,
        }


def _reference_from(results: Sequence[ClassicalResult]) -> Tuple[Optional[float], bool]:
    """Best available reference cost, and whether it is provably optimal.

    Prefers an exact solver's answer. Falls back to the best heuristic cost, but
    reports ``False`` so callers know an approximation ratio is off the table.
    """
    exact = [
        result
        for result in results
        if result.optimal and result.feasible and result.objective is not None
    ]
    if exact:
        return float(min(result.objective for result in exact)), True
    feasible = [
        result for result in results if result.feasible and result.objective is not None
    ]
    if feasible:
        return float(min(result.objective for result in feasible)), False
    return None, False


def compare_results(
    formulation: Formulation,
    classical: Sequence[ClassicalResult],
    quantum: Optional[QAOAResult] = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    analyse_distribution: bool = True,
) -> List[ComparisonRow]:
    """Build a comparison table from classical results and one QAOA run.

    The reference cost is taken from the best *provably optimal* classical result
    if there is one; otherwise ratios are left blank and only percentage gaps are
    filled in. That distinction is the whole point of this function -- it makes it
    structurally impossible to publish an approximation ratio computed against a
    two-opt heuristic.
    """
    reference, is_optimal = _reference_from(classical)
    rows: List[ComparisonRow] = []

    for result in classical:
        objective = result.objective if result.feasible else None
        rows.append(
            ComparisonRow(
                solver=result.solver,
                objective=objective,
                feasible=result.feasible,
                seconds=result.seconds,
                optimal=result.optimal,
                approximation=(
                    approximation_ratio(objective, reference)
                    if reference is not None and is_optimal
                    else None
                ),
                gap_percent=(
                    optimality_gap(objective, reference) if reference is not None else None
                ),
                notes=str(result.details.get("note", "")),
            )
        )

    if quantum is not None:
        statistics: Optional[SampleStatistics] = None
        if analyse_distribution and quantum.counts:
            try:
                statistics = analyse_counts(
                    quantum.counts,
                    formulation,
                    reference_objective=reference if is_optimal else None,
                    tolerance=tolerance,
                )
            except QRouteError as exc:  # too many unique outcomes, most likely
                _LOG.warning("Skipping distribution analysis: %s", exc)

        objective = quantum.objective
        if statistics is not None and statistics.best_feasible_objective is not None:
            # The distribution may contain a feasible route better than the
            # lowest-*energy* sample: penalties can make a slightly costlier
            # valid route score below a cheap invalid one. Take the better.
            if objective is None or statistics.best_feasible_objective < objective:
                objective = statistics.best_feasible_objective

        success = statistics.success_probability if statistics is not None else None
        rows.append(
            ComparisonRow(
                solver=f"qaoa(p={quantum.reps},{quantum.expectation_mode})",
                objective=objective,
                feasible=objective is not None,
                seconds=quantum.seconds,
                optimal=False,
                approximation=(
                    approximation_ratio(objective, reference)
                    if reference is not None and is_optimal
                    else None
                ),
                gap_percent=(
                    optimality_gap(objective, reference) if reference is not None else None
                ),
                success_prob=success,
                feasibility=statistics.feasibility_rate if statistics is not None else None,
                tts=time_to_solution(quantum.seconds, success),
                n_qubits=quantum.n_qubits,
                notes=(
                    "simulator wall-clock; not a hardware timing claim"
                    if quantum.expectation_mode == "shots"
                    else "exact expectation, no shot noise"
                ),
            )
        )

    if reference is not None and not is_optimal:
        _LOG.warning(
            "No provably optimal classical result available, so approximation "
            "ratios are omitted and gaps are measured against the best heuristic "
            "(%.4g). Solve exactly if the instance is small enough.",
            reference,
        )
    return rows


def comparison_table(rows: Sequence[ComparisonRow], *, unit: str = "s") -> str:
    """Render comparison rows as a fixed-width table for the console or a log."""
    header = (
        f"{'solver':<28} {'cost (' + unit + ')':>14} {'ratio':>7} {'gap %':>8} "
        f"{'p(opt)':>9} {'feas':>7} {'time (s)':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        cost = "infeasible" if row.objective is None else f"{row.objective:,.2f}"
        ratio = "-" if row.approximation is None else f"{row.approximation:.4f}"
        gap = "-" if row.gap_percent is None else f"{row.gap_percent:+.2f}"
        success = "-" if row.success_prob is None else f"{row.success_prob:.5f}"
        feasible = "-" if row.feasibility is None else f"{row.feasibility:.3f}"
        marker = "*" if row.optimal else " "
        lines.append(
            f"{row.solver + marker:<28} {cost:>14} {ratio:>7} {gap:>8} "
            f"{success:>9} {feasible:>7} {row.seconds:>9.3f}"
        )
    lines.append("")
    lines.append("* = provably optimal, so a valid denominator for the ratio column.")
    lines.append("ratio = optimum / achieved; 1.0000 is optimal, lower is worse.")
    return "\n".join(lines)
