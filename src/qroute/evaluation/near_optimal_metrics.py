"""
Near-optimal probability metrics for QAOA analysis.

This module calculates P(within t%) for various tolerance levels,
allowing detailed analysis of QAOA's solution distribution quality.

Key distinction:
    - Exact optimal:
        Did QAOA sample the exact reference-optimal solution?

    - Near-optimal:
        Did QAOA sample feasible solutions within a given tolerance
        of the classical reference optimum?

These are fundamentally different metrics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..qubo.base import Formulation

__all__ = [
    "classify_near_optimal_samples",
    "compute_near_optimal_probabilities",
    "extended_sample_statistics",
]


DEFAULT_TOLERANCES = [
    0.01,
    0.02,
    0.05,
    0.10,
]


def classify_near_optimal_samples(
    counts: Dict[str, int],
    formulation: Formulation,
    reference_objective: Optional[float],
    *,
    tolerances: Optional[List[float]] = None,
) -> Tuple[List[float], Dict[float, int]]:
    """
    Classify measured samples according to feasibility and
    proximity to the reference optimum.

    Parameters
    ----------
    counts:
        Measurement outcomes in the form:

            {
                "010101": 50,
                "111000": 20,
                ...
            }

    formulation:
        QUBO formulation used to decode and evaluate bitstrings.

    reference_objective:
        Classical reference objective.

    tolerances:
        Relative tolerance values.

        Example:

            [0.01, 0.02, 0.05, 0.10]

        corresponds to:

            1%, 2%, 5%, 10%.

    Returns
    -------
    feasible_objectives:
        Objective value repeated once per feasible shot.

    near_optimal_counts:
        Number of feasible shots within each tolerance.

    Notes
    -----
    This is a minimization problem.

    A solution is within tolerance t when:

        objective <= reference_objective * (1 + t)

    For example, with reference = 100:

        1%  -> objective <= 101
        5%  -> objective <= 105
        10% -> objective <= 110
    """

    if tolerances is None:
        tolerances = DEFAULT_TOLERANCES.copy()

    # Validate tolerances.
    for tolerance in tolerances:
        if tolerance < 0:
            raise ValueError(
                f"Tolerance must be non-negative: {tolerance}"
            )

    feasible_objectives: List[float] = []

    near_optimal_counts: Dict[float, int] = {
        tolerance: 0
        for tolerance in tolerances
    }

    # No reference means no meaningful optimality classification.
    if (
        reference_objective is None
        or reference_objective <= 0
    ):
        return (
            feasible_objectives,
            near_optimal_counts,
        )

    for bitstring, raw_count in counts.items():

        count = int(raw_count)

        if count <= 0:
            continue

        # --------------------------------------------------
        # Decode measured bitstring.
        # --------------------------------------------------

        try:
            solution = formulation.decode(
                bitstring,
                qiskit_order=True,
            )
        except Exception:
            # Invalid/un-decodable state.
            continue

        # --------------------------------------------------
        # Only feasible samples count.
        # --------------------------------------------------

        if not solution.feasible:
            continue

        if solution.objective is None:
            continue

        objective = float(
            solution.objective
        )

        # One entry per shot.
        feasible_objectives.extend(
            [objective] * count
        )

        # --------------------------------------------------
        # Near-optimal classification.
        #
        # IMPORTANT:
        #
        # threshold = reference * (1 + tolerance)
        #
        # Do NOT multiply a precomputed maximum threshold
        # by another tolerance.
        # --------------------------------------------------

        for tolerance in tolerances:

            threshold = (
                reference_objective
                * (1.0 + tolerance)
            )

            if objective <= threshold:

                near_optimal_counts[
                    tolerance
                ] += count

    return (
        feasible_objectives,
        near_optimal_counts,
    )


def compute_near_optimal_probabilities(
    near_optimal_counts: Dict[float, int],
    total_shots: int,
) -> Dict[float, float]:
    """
    Convert near-optimal shot counts into probabilities.

    Probability is calculated over all measurement shots,
    including infeasible samples.

    Parameters
    ----------
    near_optimal_counts:
        Mapping:

            tolerance -> number of shots

    total_shots:
        Total number of measurement shots.

    Returns
    -------
    Dict[float, float]
        Mapping:

            tolerance -> probability
    """

    if total_shots <= 0:
        return {
            tolerance: 0.0
            for tolerance in near_optimal_counts
        }

    return {
        tolerance: (
            float(count)
            / float(total_shots)
        )
        for tolerance, count
        in near_optimal_counts.items()
    }


def compute_exact_optimal_probability(
    counts: Dict[str, int],
    formulation: Formulation,
    reference_objective: Optional[float],
    *,
    atol: float = 1e-6,
) -> float:
    """
    Calculate probability of sampling an exact
    reference-optimal feasible solution.

    The probability is measured over all shots.
    """

    total_shots = sum(
        int(count)
        for count in counts.values()
    )

    if (
        total_shots <= 0
        or reference_objective is None
        or reference_objective <= 0
    ):
        return 0.0

    optimal_shots = 0

    for bitstring, raw_count in counts.items():

        count = int(raw_count)

        if count <= 0:
            continue

        try:
            solution = formulation.decode(
                bitstring,
                qiskit_order=True,
            )
        except Exception:
            continue

        if not solution.feasible:
            continue

        if solution.objective is None:
            continue

        if (
            abs(
                float(solution.objective)
                - float(reference_objective)
            )
            <= atol
        ):
            optimal_shots += count

    return (
        float(optimal_shots)
        / float(total_shots)
    )


def extended_sample_statistics(
    counts: Dict[str, int],
    formulation: Formulation,
    reference_objective: Optional[float],
) -> Dict[str, float | int | None]:
    """
    Calculate complete distribution-aware QAOA statistics.

    Returned metrics include:

        n_shots

        feasibility_rate

        best_feasible_objective

        mean_feasible_objective

        exact_optimal_probability

        near_optimal_probability_1pct

        near_optimal_probability_2pct

        near_optimal_probability_5pct

        near_optimal_probability_10pct

        absolute_gap

        optimality_gap_percent

        approximation_ratio
    """

    total_shots = sum(
        int(count)
        for count in counts.values()
    )

    # ------------------------------------------------------
    # Classify all samples.
    # ------------------------------------------------------

    (
        feasible_objectives,
        near_optimal_counts,
    ) = classify_near_optimal_samples(
        counts=counts,
        formulation=formulation,
        reference_objective=reference_objective,
    )

    # ------------------------------------------------------
    # Convert near-optimal counts to probabilities.
    # ------------------------------------------------------

    probabilities = (
        compute_near_optimal_probabilities(
            near_optimal_counts,
            total_shots,
        )
    )

    # ------------------------------------------------------
    # Feasibility.
    # ------------------------------------------------------

    feasible_shots = len(
        feasible_objectives
    )

    feasibility_rate = (
        feasible_shots / total_shots
        if total_shots > 0
        else 0.0
    )

    # ------------------------------------------------------
    # Best feasible objective.
    # ------------------------------------------------------

    best_feasible_objective = None

    if feasible_objectives:
        best_feasible_objective = float(
            min(feasible_objectives)
        )

    # ------------------------------------------------------
    # Mean feasible objective.
    # ------------------------------------------------------

    mean_feasible_objective = None

    if feasible_objectives:
        mean_feasible_objective = float(
            np.mean(feasible_objectives)
        )

    # ------------------------------------------------------
    # Exact optimal probability.
    # ------------------------------------------------------

    exact_optimal_probability = (
        compute_exact_optimal_probability(
            counts=counts,
            formulation=formulation,
            reference_objective=reference_objective,
        )
    )

    # ------------------------------------------------------
    # Gap metrics.
    # ------------------------------------------------------

    absolute_gap = None
    optimality_gap_percent = None
    approximation_ratio = None

    if (
        best_feasible_objective is not None
        and reference_objective is not None
        and reference_objective > 0
    ):

        absolute_gap = (
            best_feasible_objective
            - reference_objective
        )

        optimality_gap_percent = (
            absolute_gap
            / reference_objective
            * 100.0
        )

        approximation_ratio = (reference_objective/ best_feasible_objective
        )

    # ------------------------------------------------------
    # Final statistics.
    # ------------------------------------------------------

    stats = {

        "n_shots":
            total_shots,

        "feasible_shots":
            feasible_shots,

        "feasibility_rate":
            feasibility_rate,

        "best_feasible_objective":
            best_feasible_objective,

        "mean_feasible_objective":
            mean_feasible_objective,

        "exact_optimal_probability":
            exact_optimal_probability,

        "near_optimal_probability_1pct":
            probabilities.get(
                0.01,
                0.0,
            ),

        "near_optimal_probability_2pct":
            probabilities.get(
                0.02,
                0.0,
            ),

        "near_optimal_probability_5pct":
            probabilities.get(
                0.05,
                0.0,
            ),

        "near_optimal_probability_10pct":
            probabilities.get(
                0.10,
                0.0,
            ),

        "absolute_gap":
            absolute_gap,

        "optimality_gap_percent":
            optimality_gap_percent,

        "approximation_ratio":
            approximation_ratio,
    }

    return stats