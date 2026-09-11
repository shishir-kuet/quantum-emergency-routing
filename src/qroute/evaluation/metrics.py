"""
Metrics for comparing quantum results against classical references honestly.

The module separates three concepts:

1. Original-problem reference
   - Example: Dijkstra for shortest path.
   - Used as the scientifically valid optimum/reference.

2. QUBO-level classical baselines
   - Brute force over the QUBO.
   - Simulated annealing over the QUBO.

3. QAOA measurement distribution
   - Feasibility.
   - Best feasible sampled objective.
   - Exact/near-optimal probabilities.
   - Time-to-solution.

Approximation ratios are computed only when the supplied reference is
explicitly marked as provably optimal.
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

DEFAULT_TOLERANCE = 1e-6


# ============================================================================
# RATIOS AND GAPS
# ============================================================================

def approximation_ratio(
    achieved: Optional[float],
    reference: float,
    *,
    reference_is_optimal: bool = True,
) -> Optional[float]:
    """
    Return the minimisation approximation ratio.

    ratio = reference / achieved

    A value of 1.0 means the achieved solution matches the optimum.

    The ratio is only valid when ``reference_is_optimal=True``.
    """

    if not reference_is_optimal:
        raise QRouteError(
            "Refusing to compute an approximation ratio against a "
            "non-optimal reference."
        )

    if achieved is None:
        return None

    if reference <= 0:
        raise QRouteError(
            f"Reference cost must be positive, got {reference}."
        )

    if achieved <= 0:
        raise QRouteError(
            f"Achieved cost must be positive, got {achieved}."
        )

    return float(reference) / float(achieved)


def optimality_gap(
    achieved: Optional[float],
    reference: float,
) -> Optional[float]:
    """
    Return excess cost over the reference as a percentage.

    gap = (achieved - reference) / reference * 100

    0% means the achieved objective matches the reference.
    """

    if achieved is None:
        return None

    if reference <= 0:
        raise QRouteError(
            f"Reference cost must be positive, got {reference}."
        )

    return (
        100.0
        * (float(achieved) - float(reference))
        / float(reference)
    )


# ============================================================================
# SHOT-LEVEL ANALYSIS
# ============================================================================

@dataclass(frozen=True, eq=False)
class SampleStatistics:
    """
    Summary of the complete QAOA measurement distribution.
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

    near_optimal_shots_1pct: int = 0
    near_optimal_shots_2pct: int = 0
    near_optimal_shots_5pct: int = 0
    near_optimal_shots_10pct: int = 0

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def feasibility_rate(self) -> float:
        """Fraction of all shots that produced feasible solutions."""

        if self.n_shots == 0:
            return 0.0

        return (
            self.feasible_shots
            / self.n_shots
        )

    @property
    def success_probability(self) -> Optional[float]:
        """
        Fraction of shots that hit the supplied optimum.

        Returns None when no reference optimum was supplied.
        """

        if self.metadata.get(
            "reference_objective"
        ) is None:
            return None

        if self.n_shots == 0:
            return 0.0

        return (
            self.optimal_shots
            / self.n_shots
        )

    @property
    def mean_feasible_objective(self) -> Optional[float]:
        """Mean objective among feasible shots."""

        if not self.feasible_objectives:
            return None

        return float(
            np.mean(
                self.feasible_objectives
            )
        )

    @property
    def near_optimal_probability_1pct(self) -> float:
        return (
            self.near_optimal_shots_1pct
            / self.n_shots
            if self.n_shots
            else 0.0
        )

    @property
    def near_optimal_probability_2pct(self) -> float:
        return (
            self.near_optimal_shots_2pct
            / self.n_shots
            if self.n_shots
            else 0.0
        )

    @property
    def near_optimal_probability_5pct(self) -> float:
        return (
            self.near_optimal_shots_5pct
            / self.n_shots
            if self.n_shots
            else 0.0
        )

    @property
    def near_optimal_probability_10pct(self) -> float:
        return (
            self.near_optimal_shots_10pct
            / self.n_shots
            if self.n_shots
            else 0.0
        )

    def describe(self) -> str:
        """Human-readable summary."""

        success = self.success_probability

        success_text = (
            "n/a"
            if success is None
            else f"{success:.4f}"
        )

        best = self.best_feasible_objective

        best_text = (
            "none"
            if best is None
            else f"{best:.2f}"
        )

        return (
            f"SampleStatistics("
            f"shots={self.n_shots}, "
            f"unique={self.n_unique}, "
            f"feasible={self.feasibility_rate:.3f}, "
            f"p(optimal)={success_text}, "
            f"best feasible cost={best_text}, "
            f"<E>={self.mean_energy:.4g})"
        )


def analyse_counts(
    counts: Mapping[str, int],
    formulation: Formulation,
    *,
    reference_objective: Optional[float] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    max_unique: int = 200_000,
) -> SampleStatistics:
    """
    Decode every distinct QAOA outcome and analyse the full distribution.

    Parameters
    ----------
    counts:
        QAOA measurement counts.

    formulation:
        QUBO formulation used to decode the states.

    reference_objective:
        Known optimal objective. If supplied, exact and near-optimal
        probabilities are computed.

    tolerance:
        Relative/absolute tolerance for exact-optimal classification.

    max_unique:
        Safety limit on the number of distinct bitstrings decoded.
    """

    if not counts:
        raise QRouteError(
            "Cannot analyse an empty counts dictionary."
        )

    if len(counts) > max_unique:
        raise QRouteError(
            f"{len(counts):,} distinct outcomes exceeds "
            f"the {max_unique:,} decode limit."
        )

    qubo = formulation.qubo()

    assignments, energies, weights = sample_energies(
        dict(counts),
        qubo,
    )

    n_shots = int(
        weights.sum()
    )

    if n_shots <= 0:
        raise QRouteError(
            "Counts contain no positive shots."
        )

    feasible_shots = 0
    optimal_shots = 0

    near_optimal_shots_1pct = 0
    near_optimal_shots_2pct = 0
    near_optimal_shots_5pct = 0
    near_optimal_shots_10pct = 0

    best_energy = float(
        energies.min()
    )

    best_solution: Optional[
        RouteSolution
    ] = None

    best_objective: Optional[
        float
    ] = None

    feasible_objectives: List[
        float
    ] = []

    # ==================================================================
    # OPTIMALITY THRESHOLDS
    # ==================================================================

    if reference_objective is None:

        exact_threshold = None
        threshold_1pct = None
        threshold_2pct = None
        threshold_5pct = None
        threshold_10pct = None

    else:

        reference = float(
            reference_objective
        )

        exact_threshold = (
            reference * (1.0 + tolerance)
            + tolerance
        )

        threshold_1pct = (
            reference * 1.01
        )

        threshold_2pct = (
            reference * 1.02
        )

        threshold_5pct = (
            reference * 1.05
        )

        threshold_10pct = (
            reference * 1.10
        )

    # ==================================================================
    # DECODE ALL UNIQUE STATES
    # ==================================================================

    # ==================================================================
# DECODE ALL UNIQUE STATES
# ==================================================================

    for bitstring, shot_count in counts.items():

        shots = int(shot_count)

        solution = formulation.decode(
        bitstring,
        qiskit_order=True,
        )

        if not solution.feasible:
            continue

        feasible_shots += shots

        objective = solution.objective

        if objective is None:
            continue

        objective = float(objective)

        feasible_objectives.extend(
            [objective] * shots
        )

    # --------------------------------------------------------------
    # Best feasible objective
    # --------------------------------------------------------------

        if (best_objective is None or objective < best_objective):
            best_objective = objective
            best_solution = solution

    # --------------------------------------------------------------
    # Exact optimum
    # --------------------------------------------------------------

        if (exact_threshold is not None and objective <= exact_threshold):
            optimal_shots += shots

    # --------------------------------------------------------------
    # Near-optimal states
    # --------------------------------------------------------------

        if (threshold_1pct is not None and objective <= threshold_1pct):
            near_optimal_shots_1pct += shots

        if (threshold_2pct is not None and objective <= threshold_2pct):
            near_optimal_shots_2pct += shots

        if (threshold_5pct is not None and objective <= threshold_5pct):
            near_optimal_shots_5pct += shots

        if (threshold_10pct is not None and objective <= threshold_10pct):
            near_optimal_shots_10pct += shots

    # ==================================================================
    # MEAN QUBO ENERGY
    # ==================================================================

    mean_energy = float(
        energies @ weights
        / weights.sum()
    )

    statistics = SampleStatistics(
        n_shots=n_shots,
        n_unique=len(counts),
        feasible_shots=feasible_shots,
        optimal_shots=optimal_shots,
        mean_energy=mean_energy,
        best_energy=best_energy,
        best_solution=best_solution,
        best_feasible_objective=best_objective,
        feasible_objectives=tuple(
            feasible_objectives
        ),
        near_optimal_shots_1pct=(
            near_optimal_shots_1pct
        ),
        near_optimal_shots_2pct=(
            near_optimal_shots_2pct
        ),
        near_optimal_shots_5pct=(
            near_optimal_shots_5pct
        ),
        near_optimal_shots_10pct=(
            near_optimal_shots_10pct
        ),
        metadata={
            "formulation": formulation.name,
            "reference_objective": reference_objective,
            "tolerance": tolerance,
        },
    )

    _LOG.info(
        "Distribution: %s",
        statistics.describe(),
    )

    return statistics


def feasibility_rate(
    counts: Mapping[str, int],
    formulation: Formulation,
) -> float:
    """Return the feasibility rate of the sampled distribution."""

    return analyse_counts(
        counts,
        formulation,
    ).feasibility_rate


def success_probability(
    counts: Mapping[str, int],
    formulation: Formulation,
    reference_objective: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> float:
    """Return probability of sampling an optimal solution."""

    statistics = analyse_counts(
        counts,
        formulation,
        reference_objective=reference_objective,
        tolerance=tolerance,
    )

    return (
        statistics.success_probability
        or 0.0
    )


# ============================================================================
# TIME TO SOLUTION
# ============================================================================

def time_to_solution(
    seconds: float,
    success_prob: Optional[float],
    *,
    target: float = 0.99,
) -> Optional[float]:
    """
    Expected time to obtain an optimal sample with probability ``target``.

    TTS = runtime * log(1-target) / log(1-success_probability)
    """

    if success_prob is None:
        return None

    if success_prob <= 0.0:
        return None

    if success_prob >= 1.0:
        return float(seconds)

    if not 0.0 < target < 1.0:
        raise QRouteError(
            f"target must be between 0 and 1, got {target}"
        )

    repeats = (
        np.log1p(-target)
        / np.log1p(-success_prob)
    )

    return (
        float(seconds)
        * float(repeats)
    )


# ============================================================================
# COMPARISON ROW
# ============================================================================

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
        """Return a serialisable dictionary."""

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


# ============================================================================
# RESULT DETAILS HELPER
# ============================================================================

def _result_note(
    result: ClassicalResult,
) -> str:
    """Safely extract a result note."""

    details = getattr(
        result,
        "details",
        None,
    )

    if not isinstance(
        details,
        Mapping,
    ):
        return ""

    return str(
        details.get(
            "note",
            "",
        )
    )


# ============================================================================
# EXPLICIT REFERENCE
# ============================================================================

def _reference_values(
    reference: Optional[ClassicalResult],
) -> Tuple[
    Optional[float],
    bool,
]:
    """
    Extract the explicit reference objective.

    Returns
    -------
    (objective, is_provably_optimal)
    """

    if reference is None:
        return None, False

    feasible = bool(
        getattr(
            reference,
            "feasible",
            False,
        )
    )

    optimal = bool(
        getattr(
            reference,
            "optimal",
            False,
        )
    )

    objective = getattr(
        reference,
        "objective",
        None,
    )

    if (
        not feasible
        or objective is None
    ):
        return None, False

    return (
        float(objective),
        optimal,
    )


# ============================================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================================

def compare_results(
    formulation: Formulation,
    classical: Sequence[ClassicalResult],
    quantum: Optional[QAOAResult] = None,
    *,
    reference: Optional[ClassicalResult] = None,
    tolerance: float = DEFAULT_TOLERANCE,
    analyse_distribution: bool = True,
) -> List[ComparisonRow]:
    """
    Compare QUBO classical baselines and QAOA against an explicit reference.

    Parameters
    ----------
    formulation:
        QUBO formulation.

    classical:
        QUBO-level classical baselines only.

        Examples:
        - brute force QUBO
        - simulated annealing

        The original-problem reference should NOT be placed here.

    quantum:
        Optional QAOA result.

    reference:
        Explicit original-problem reference.

        For shortest path this should normally be Dijkstra.

    tolerance:
        Tolerance used for exact-optimal sample classification.

    analyse_distribution:
        Analyse QAOA counts when available.

    Returns
    -------
    list[ComparisonRow]
        Comparison rows.
    """

    # ==================================================================
    # EXPLICIT REFERENCE
    # ==================================================================

    reference_objective, reference_is_optimal = (
        _reference_values(
            reference
        )
    )

    rows: List[
        ComparisonRow
    ] = []

    # ==================================================================
    # REFERENCE ROW
    # ==================================================================

    if reference_objective is not None:

        reference_solver = getattr(
            reference,
            "solver",
            "reference",
        )

        rows.append(
            ComparisonRow(
                solver=str(
                    reference_solver
                ),
                objective=reference_objective,
                feasible=True,
                seconds=float(
                    getattr(
                        reference,
                        "seconds",
                        0.0,
                    )
                ),
                optimal=reference_is_optimal,
                approximation=(
                    1.0
                    if reference_is_optimal
                    else None
                ),
                gap_percent=0.0,
                success_prob=(
                    1.0
                    if reference_is_optimal
                    else None
                ),
                feasibility=1.0,
                tts=None,
                n_qubits=None,
                notes=(
                    "original-problem reference"
                    if reference_is_optimal
                    else
                    "reference is feasible but not "
                    "provably optimal"
                ),
            )
        )

    # ==================================================================
    # QUBO CLASSICAL BASELINES
    # ==================================================================

    for result in classical:

        feasible = bool(
            getattr(
                result,
                "feasible",
                False,
            )
        )

        objective = (
            getattr(
                result,
                "objective",
                None,
            )
            if feasible
            else None
        )

        objective = (
            float(objective)
            if objective is not None
            else None
        )

        result_optimal = bool(
            getattr(
                result,
                "optimal",
                False,
            )
        )

        # --------------------------------------------------------------
        # Approximation ratio
        # --------------------------------------------------------------

        if (
            objective is not None
            and reference_objective is not None
            and reference_is_optimal
        ):

            approximation = approximation_ratio(
                objective,
                reference_objective,
                reference_is_optimal=True,
            )

        else:

            approximation = None

        # --------------------------------------------------------------
        # Gap
        # --------------------------------------------------------------

        if (
            objective is not None
            and reference_objective is not None
        ):

            gap = optimality_gap(
                objective,
                reference_objective,
            )

        else:

            gap = None

        rows.append(
            ComparisonRow(
                solver=str(
                    getattr(
                        result,
                        "solver",
                        "classical",
                    )
                ),
                objective=objective,
                feasible=feasible,
                seconds=float(
                    getattr(
                        result,
                        "seconds",
                        0.0,
                    )
                ),
                optimal=result_optimal,
                approximation=approximation,
                gap_percent=gap,
                success_prob=None,
                feasibility=(
                    1.0
                    if feasible
                    else 0.0
                ),
                tts=None,
                n_qubits=None,
                notes=_result_note(
                    result
                ),
            )
        )

    # ==================================================================
    # QAOA
    # ==================================================================

    if quantum is not None:

        statistics: Optional[
            SampleStatistics
        ] = None

        # --------------------------------------------------------------
        # Distribution analysis
        # --------------------------------------------------------------

        if (
            analyse_distribution
            and getattr(
                quantum,
                "counts",
                None,
            )
        ):

            try:

                statistics = analyse_counts(
                    quantum.counts,
                    formulation,
                    reference_objective=(
                        reference_objective
                        if reference_is_optimal
                        else None
                    ),
                    tolerance=tolerance,
                )

            except QRouteError as exc:

                _LOG.warning(
                    "Skipping QAOA distribution analysis: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # QAOA objective
        # --------------------------------------------------------------

        objective = getattr(
            quantum,
            "objective",
            None,
        )

        if objective is not None:
            objective = float(
                objective
            )

        # The actual routing result should be the best feasible sampled
        # route, not merely the variational mean energy.

        if (
            statistics is not None
            and statistics.best_feasible_objective
            is not None
        ):

            sampled_best = (
                statistics.best_feasible_objective
            )

            if (
                objective is None
                or sampled_best < objective
            ):

                objective = sampled_best

        # --------------------------------------------------------------
        # QAOA feasibility
        # --------------------------------------------------------------

        qaoa_feasible = (
            objective is not None
        )

        # --------------------------------------------------------------
        # Approximation ratio
        # --------------------------------------------------------------

        if (
            objective is not None
            and reference_objective is not None
            and reference_is_optimal
        ):

            approximation = approximation_ratio(
                objective,
                reference_objective,
                reference_is_optimal=True,
            )

        else:

            approximation = None

        # --------------------------------------------------------------
        # Gap
        # --------------------------------------------------------------

        if (
            objective is not None
            and reference_objective is not None
        ):

            gap = optimality_gap(
                objective,
                reference_objective,
            )

        else:

            gap = None

        # --------------------------------------------------------------
        # Success probability
        # --------------------------------------------------------------

        success = (
            statistics.success_probability
            if statistics is not None
            else None
        )

        # --------------------------------------------------------------
        # Feasibility probability
        # --------------------------------------------------------------

        feasibility = (
            statistics.feasibility_rate
            if statistics is not None
            else None
        )

        # --------------------------------------------------------------
        # TTS
        # --------------------------------------------------------------

        tts = time_to_solution(
            float(
                getattr(
                    quantum,
                    "seconds",
                    0.0,
                )
            ),
            success,
        )

        # --------------------------------------------------------------
        # Notes
        # --------------------------------------------------------------

        expectation_mode = getattr(
            quantum,
            "expectation_mode",
            "shots",
        )

        if expectation_mode == "shots":

            notes = (
                "simulator wall-clock; "
                "not a hardware timing claim"
            )

        else:

            notes = (
                "exact expectation; "
                "no shot noise"
            )

        # --------------------------------------------------------------
        # Row
        # --------------------------------------------------------------

        rows.append(
            ComparisonRow(
                solver=(
                    f"qaoa("
                    f"p={getattr(quantum, 'reps', '?')},"
                    f"{expectation_mode})"
                ),
                objective=objective,
                feasible=qaoa_feasible,
                seconds=float(
                    getattr(
                        quantum,
                        "seconds",
                        0.0,
                    )
                ),
                optimal=(
                    reference_is_optimal
                    and objective is not None
                    and (
                        abs(
                            objective
                            - reference_objective
                        )
                        <= (
                            reference_objective
                            * tolerance
                            + tolerance
                        )
                    )
                )
                if reference_objective is not None
                else False,
                approximation=approximation,
                gap_percent=gap,
                success_prob=success,
                feasibility=feasibility,
                tts=tts,
                n_qubits=getattr(
                    quantum,
                    "n_qubits",
                    None,
                ),
                notes=notes,
            )
        )

    # ==================================================================
    # WARN WHEN NO PROVABLE REFERENCE EXISTS
    # ==================================================================

    if (
        reference_objective is not None
        and not reference_is_optimal
    ):

        _LOG.warning(
            "Reference objective %.6g is not marked as provably optimal. "
            "Approximation ratios are omitted.",
            reference_objective,
        )

    elif reference_objective is None:

        _LOG.warning(
            "No valid reference objective supplied. "
            "Approximation ratios and reference gaps are unavailable."
        )

    return rows


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def comparison_table(
    rows: Sequence[ComparisonRow],
    *,
    unit: str = "s",
) -> str:
    """
    Render comparison rows as a fixed-width console table.
    """

    header = (
        f"{'solver':<28} "
        f"{'cost':>14} "
        f"{'ratio':>8} "
        f"{'gap %':>9} "
        f"{'p(opt)':>10} "
        f"{'feas':>8} "
        f"{'time (s)':>10}"
    )

    lines = [
        header,
        "-" * len(header),
    ]

    for row in rows:

        cost = (
            "infeasible"
            if row.objective is None
            else f"{row.objective:,.2f}"
        )

        ratio = (
            "-"
            if row.approximation is None
            else f"{row.approximation:.4f}"
        )

        gap = (
            "-"
            if row.gap_percent is None
            else f"{row.gap_percent:+.2f}"
        )

        success = (
            "-"
            if row.success_prob is None
            else f"{row.success_prob:.5f}"
        )

        feasible = (
            "-"
            if row.feasibility is None
            else f"{row.feasibility:.3f}"
        )

        marker = (
            "*"
            if row.optimal
            else ""
        )

        lines.append(
            f"{row.solver + marker:<28} "
            f"{cost:>14} "
            f"{ratio:>8} "
            f"{gap:>9} "
            f"{success:>10} "
            f"{feasible:>8} "
            f"{row.seconds:>10.3f}"
        )

    lines.append("")

    lines.append(
        "* = provably optimal/reference-matched result."
    )

    lines.append(
        "ratio = reference optimum / achieved objective."
    )

    lines.append(
        "ratio = 1.0000 means the achieved objective matches "
        "the optimal reference."
    )

    lines.append(
        "p(opt) = probability of sampling an optimal solution."
    )

    lines.append(
        "feas = fraction of all QAOA shots producing feasible solutions."
    )

    return "\n".join(lines)