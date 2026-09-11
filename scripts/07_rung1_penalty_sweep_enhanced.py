"""
Enhanced Rung 1 shortest-path QAOA penalty sweep with multiple seeds.

This experiment studies the effect of the QUBO penalty scale on QAOA
performance for the reduced shortest-path formulation.

Research design
---------------
For every penalty value:

    penalty
       |
       +-- same graph
       +-- same source / target
       +-- same QAOA depth
       +-- same shot count
       +-- same optimizer budget
       +-- same set of random seeds
              |
              +-- QAOA

Reference semantics
-------------------
The reference objective comes from the original shortest-path problem:

    Dijkstra -> reference optimum

The QAOA solution is evaluated from its measured distribution:

    counts
       |
       +-- feasibility rate
       +-- best feasible objective
       +-- exact optimal probability
       +-- near-optimal probability (1%, 2%, 5%, 10%)
       +-- mean feasible objective
       +-- optimality gap
       +-- approximation ratio

No metric is manually recomputed here. The pipeline/evaluation layer is the
single source of truth for these metrics.

Outputs
-------
results/shortest_path_penalty_sweep_raw.json
    All seed-level results.

results/shortest_path_penalty_sweep_aggregate.json
    Per-penalty mean/std aggregates.

results/shortest_path_penalty_sweep.csv
    Human-readable summary table.

results/paper_figures/
    Publication-quality figures.

Usage
-----
python scripts/07_rung1_penalty_sweep_enhanced.py \
    --seeds 10 \
    --penalties 0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0,10.0,15.0,20.0 \
    --p 3 \
    --shots 4096 \
    --maxiter 200
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

_SRC = Path(__file__).resolve().parents[1] / "src"

if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qroute.config import load_config
from qroute.logging_utils import get_logger
from qroute.pipeline import (
    build_rung,
    load_network,
    run_experiment,
)

_LOG = get_logger(__name__)


# ============================================================================
# PLOTTING
# ============================================================================

def setup_matplotlib():
    """Configure matplotlib for publication-quality output."""

    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        return plt

    except ImportError:
        _LOG.warning(
            "matplotlib not available; figures will be skipped"
        )
        return None


plt = setup_matplotlib()


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_PENALTIES = [
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
]

DEFAULT_SEEDS = 10
DEFAULT_P = 3

# 4096 gives substantially better resolution than 1024 for low-probability
# feasible / optimal events.
DEFAULT_SHOTS = 4096

DEFAULT_MAXITER = 200


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(
    value: Any,
) -> Optional[float]:
    """Convert a value to float, preserving None."""

    if value is None:
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def _safe_mean_std(
    values: List[Optional[float]],
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute mean and standard deviation while ignoring None values.
    """

    valid = [
        float(value)
        for value in values
        if value is not None
    ]

    if not valid:
        return None, None

    return (
        float(np.mean(valid)),
        float(np.std(valid)),
    )


# ============================================================================
# SEED-LEVEL METRIC EXTRACTION
# ============================================================================

def extract_seed_metrics(
    result: Any,
    penalty_scale: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Extract all relevant metrics from ExperimentResult.

    IMPORTANT:
    No metric is manually recomputed here.

    The pipeline's evaluation layer is the single source of truth.
    """

    metadata = getattr(
        result,
        "metadata",
        {},
    )

    quantum = getattr(
        result,
        "quantum",
        None,
    )

    reference_objective = _safe_float(
        getattr(
            result,
            "reference_objective",
            None,
        )
    )

    # ------------------------------------------------------------------
    # Primary QAOA result
    # ------------------------------------------------------------------

    qaoa_objective = _safe_float(
        metadata.get(
            "best_feasible_objective"
        )
    )

    # Fallback only if extended metrics are unavailable.
    if qaoa_objective is None:
        qaoa_objective = _safe_float(
            metadata.get(
                "best_sampled_objective"
            )
        )

    # ------------------------------------------------------------------
    # Distribution metrics
    # ------------------------------------------------------------------

    feasibility_rate = _safe_float(
        metadata.get(
            "feasibility_rate"
        )
    )

    success_probability = _safe_float(
        metadata.get(
            "success_probability"
        )
    )

    exact_optimal_probability = _safe_float(
        metadata.get(
            "exact_optimal_probability"
        )
    )

    # ``p_opt`` is retained as a convenient experiment-level name.
    p_opt = (
        exact_optimal_probability
        if exact_optimal_probability is not None
        else success_probability
    )

    # ------------------------------------------------------------------
    # Near-optimal probabilities
    # ------------------------------------------------------------------

    p_1pct = _safe_float(
        metadata.get(
            "near_optimal_probability_1pct"
        )
    )

    p_2pct = _safe_float(
        metadata.get(
            "near_optimal_probability_2pct"
        )
    )

    p_5pct = _safe_float(
        metadata.get(
            "near_optimal_probability_5pct"
        )
    )

    p_10pct = _safe_float(
        metadata.get(
            "near_optimal_probability_10pct"
        )
    )

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    absolute_gap = _safe_float(
        metadata.get(
            "absolute_gap"
        )
    )

    optimality_gap_percent = _safe_float(
        metadata.get(
            "optimality_gap_percent"
        )
    )

    approximation_ratio = _safe_float(
        metadata.get(
            "approximation_ratio"
        )
    )

    mean_feasible_objective = _safe_float(
        metadata.get(
            "mean_feasible_objective"
        )
    )

    # ------------------------------------------------------------------
    # Quantum runtime / circuit information
    # ------------------------------------------------------------------

    runtime = 0.0
    qubits = 0
    reps = None
    shots = None

    if quantum is not None:

        runtime = _safe_float(
            getattr(
                quantum,
                "seconds",
                0.0,
            )
        ) or 0.0

        qubits = int(
            getattr(
                quantum,
                "n_qubits",
                0,
            )
            or 0
        )

        reps = getattr(
            quantum,
            "reps",
            None,
        )

        shots = getattr(
            quantum,
            "shots",
            None,
        )

    # ------------------------------------------------------------------
    # Circuit statistics
    # ------------------------------------------------------------------

    circuit_stats = metadata.get(
        "circuit_stats"
    )

    circuit_depth = None
    two_qubit_gates = None

    if isinstance(
        circuit_stats,
        dict,
    ):
        circuit_depth = circuit_stats.get(
            "depth"
        )

        two_qubit_gates = (
            circuit_stats.get(
                "two_qubit_gates"
            )
        )

    # ------------------------------------------------------------------
    # Return complete seed result
    # ------------------------------------------------------------------

    return {
        "penalty_scale": penalty_scale,
        "seed": seed,

        # Reference
        "reference_objective": reference_objective,
        "reference_solver": metadata.get(
            "reference_solver"
        ),
        "reference_is_optimal": metadata.get(
            "reference_is_optimal"
        ),

        # QAOA
        "qaoa_objective": qaoa_objective,

        # Distribution
        "feasibility_rate": feasibility_rate,
        "p_opt": p_opt,
        "exact_optimal_probability": exact_optimal_probability,

        "near_optimal_probability_1pct": p_1pct,
        "near_optimal_probability_2pct": p_2pct,
        "near_optimal_probability_5pct": p_5pct,
        "near_optimal_probability_10pct": p_10pct,

        # Quality
        "mean_feasible_objective": mean_feasible_objective,
        "absolute_gap": absolute_gap,
        "optimality_gap_percent": optimality_gap_percent,
        "approximation_ratio": approximation_ratio,

        # Runtime / circuit
        "qaoa_runtime_seconds": runtime,
        "qubits": qubits,
        "shots": shots,
        "reps": reps,
        "circuit_depth": circuit_depth,
        "two_qubit_gates": two_qubit_gates,

        # Full metadata retained for reproducibility
        "reference_metadata": {
            "reference_objective": reference_objective,
            "reference_solver": metadata.get(
                "reference_solver"
            ),
            "reference_is_optimal": metadata.get(
                "reference_is_optimal"
            ),
        },
    }


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_seed_results(
    seed_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate multiple seed-level results for one penalty.

    Mean and population standard deviation are reported across successful
    seeds.
    """

    if not seed_results:
        return {}

    penalty_scale = seed_results[0][
        "penalty_scale"
    ]

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def aggregate(
        key: str,
    ) -> Tuple[Optional[float], Optional[float]]:
        return _safe_mean_std(
            [
                _safe_float(
                    result.get(key)
                )
                for result in seed_results
            ]
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    reference_values = [
        _safe_float(
            result.get(
                "reference_objective"
            )
        )
        for result in seed_results
    ]

    qaoa_mean, qaoa_std = aggregate(
        "qaoa_objective"
    )

    qaoa_values = [
        value
        for value in (
            _safe_float(
                result.get(
                    "qaoa_objective"
                )
            )
            for result in seed_results
        )
        if value is not None
    ]

    absolute_gap_mean, absolute_gap_std = aggregate(
        "absolute_gap"
    )

    gap_mean, gap_std = aggregate(
        "optimality_gap_percent"
    )

    ratio_mean, ratio_std = aggregate(
        "approximation_ratio"
    )

    feasibility_mean, feasibility_std = aggregate(
        "feasibility_rate"
    )

    p_opt_mean, p_opt_std = aggregate(
        "p_opt"
    )

    exact_opt_mean, exact_opt_std = aggregate(
        "exact_optimal_probability"
    )

    p_1_mean, p_1_std = aggregate(
        "near_optimal_probability_1pct"
    )

    p_2_mean, p_2_std = aggregate(
        "near_optimal_probability_2pct"
    )

    p_5_mean, p_5_std = aggregate(
        "near_optimal_probability_5pct"
    )

    p_10_mean, p_10_std = aggregate(
        "near_optimal_probability_10pct"
    )

    mean_feasible_mean, mean_feasible_std = aggregate(
        "mean_feasible_objective"
    )

    runtime_mean, runtime_std = aggregate(
        "qaoa_runtime_seconds"
    )

    depth_mean, depth_std = aggregate(
        "circuit_depth"
    )

    two_qubit_mean, two_qubit_std = aggregate(
        "two_qubit_gates"
    )

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    reference_valid = [
        value
        for value in reference_values
        if value is not None
    ]

    reference_mean = (
        float(np.mean(reference_valid))
        if reference_valid
        else None
    )

    reference_std = (
        float(np.std(reference_valid))
        if reference_valid
        else None
    )

    # ------------------------------------------------------------------
    # Aggregate result
    # ------------------------------------------------------------------

    return {
        "penalty_scale": penalty_scale,

        "n_seeds": len(seed_results),

        # Reference
        "reference_objective_mean": reference_mean,
        "reference_objective_std": reference_std,

        # Best feasible QAOA result
        "qaoa_objective_mean": qaoa_mean,
        "qaoa_objective_std": qaoa_std,
        "qaoa_objective_min": (
            min(qaoa_values)
            if qaoa_values
            else None
        ),
        "qaoa_objective_max": (
            max(qaoa_values)
            if qaoa_values
            else None
        ),

        # Quality
        "absolute_gap_mean": absolute_gap_mean,
        "absolute_gap_std": absolute_gap_std,

        "optimality_gap_percent_mean": gap_mean,
        "optimality_gap_percent_std": gap_std,

        "approximation_ratio_mean": ratio_mean,
        "approximation_ratio_std": ratio_std,

        # Feasibility
        "feasibility_rate_mean": feasibility_mean,
        "feasibility_rate_std": feasibility_std,

        # Exact optimum
        "p_opt_mean": p_opt_mean,
        "p_opt_std": p_opt_std,

        "exact_optimal_probability_mean": exact_opt_mean,
        "exact_optimal_probability_std": exact_opt_std,

        # Near optimal
        "near_optimal_1pct_mean": p_1_mean,
        "near_optimal_1pct_std": p_1_std,

        "near_optimal_2pct_mean": p_2_mean,
        "near_optimal_2pct_std": p_2_std,

        "near_optimal_5pct_mean": p_5_mean,
        "near_optimal_5pct_std": p_5_std,

        "near_optimal_10pct_mean": p_10_mean,
        "near_optimal_10pct_std": p_10_std,

        # Feasible objective distribution
        "mean_feasible_objective_mean": mean_feasible_mean,
        "mean_feasible_objective_std": mean_feasible_std,

        # Runtime
        "runtime_mean": runtime_mean,
        "runtime_std": runtime_std,

        # Circuit
        "circuit_depth_mean": depth_mean,
        "circuit_depth_std": depth_std,

        "two_qubit_gates_mean": two_qubit_mean,
        "two_qubit_gates_std": two_qubit_std,
    }


# ============================================================================
# FIGURE 1
# ============================================================================

def plot_quality_vs_penalty(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Best feasible QAOA objective vs penalty scale."""

    if plt is None:
        return

    valid = [
        result
        for result in results
        if result.get(
            "qaoa_objective_mean"
        ) is not None
    ]

    if not valid:
        _LOG.info(
            "No valid results for quality figure"
        )
        return

    penalties = [
        result["penalty_scale"]
        for result in valid
    ]

    objectives = [
        result["qaoa_objective_mean"]
        for result in valid
    ]

    errors = [
        result.get(
            "qaoa_objective_std"
        ) or 0.0
        for result in valid
    ]

    reference = valid[0].get(
        "reference_objective_mean"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        penalties,
        objectives,
        yerr=errors,
        marker="o",
        linewidth=2,
        capsize=5,
        label="QAOA best feasible (mean ± std)",
    )

    if reference is not None:
        ax.axhline(
            reference,
            linestyle="--",
            linewidth=2,
            label="Reference optimum (Dijkstra)",
        )

    ax.set_xlabel(
        "QUBO Penalty Scale",
        fontsize=12,
    )

    ax.set_ylabel(
        "Route Objective (Cost)",
        fontsize=12,
    )

    ax.set_title(
        "Rung 1: Best Feasible Route Objective vs Penalty Scale",
        fontsize=13,
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend(
        fontsize=11
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    _LOG.info(
        "Saved: %s",
        output_path,
    )


# ============================================================================
# FIGURE 2
# ============================================================================

def plot_feasibility_vs_penalty(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Feasibility rate vs penalty scale."""

    if plt is None:
        return

    valid = [
        result
        for result in results
        if result.get(
            "feasibility_rate_mean"
        ) is not None
    ]

    if not valid:
        _LOG.info(
            "No valid results for feasibility figure"
        )
        return

    penalties = [
        result["penalty_scale"]
        for result in valid
    ]

    means = [
        result["feasibility_rate_mean"] * 100.0
        for result in valid
    ]

    errors = [
        (
            result.get(
                "feasibility_rate_std"
            )
            or 0.0
        ) * 100.0
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        penalties,
        means,
        yerr=errors,
        marker="s",
        linewidth=2,
        capsize=5,
        label="Feasibility rate (mean ± std)",
    )

    ax.set_xlabel(
        "QUBO Penalty Scale",
        fontsize=12,
    )

    ax.set_ylabel(
        "Feasible-Solution Rate (%)",
        fontsize=12,
    )

    ax.set_title(
        "Rung 1: Feasibility Rate vs Penalty Scale",
        fontsize=13,
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend(
        fontsize=11
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    _LOG.info(
        "Saved: %s",
        output_path,
    )


# ============================================================================
# FIGURE 3
# ============================================================================

def plot_gap_vs_penalty(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Optimality gap vs penalty scale."""

    if plt is None:
        return

    valid = [
        result
        for result in results
        if result.get(
            "optimality_gap_percent_mean"
        ) is not None
    ]

    if not valid:
        _LOG.info(
            "No valid results for gap figure"
        )
        return

    penalties = [
        result["penalty_scale"]
        for result in valid
    ]

    means = [
        result[
            "optimality_gap_percent_mean"
        ]
        for result in valid
    ]

    errors = [
        result.get(
            "optimality_gap_percent_std"
        ) or 0.0
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.errorbar(
        penalties,
        means,
        yerr=errors,
        marker="^",
        linewidth=2,
        capsize=5,
        label="Optimality gap (mean ± std)",
    )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
        label="Optimal",
    )

    ax.set_xlabel(
        "QUBO Penalty Scale",
        fontsize=12,
    )

    ax.set_ylabel(
        "Optimality Gap (%)",
        fontsize=12,
    )

    ax.set_title(
        "Rung 1: Optimality Gap vs Penalty Scale",
        fontsize=13,
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend(
        fontsize=11
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    _LOG.info(
        "Saved: %s",
        output_path,
    )


# ============================================================================
# FIGURE 4
# ============================================================================

def plot_near_optimal_vs_penalty(
    results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Near-optimal sampling probability vs penalty scale."""

    if plt is None:
        return

    valid = [
        result
        for result in results
        if result.get(
            "near_optimal_1pct_mean"
        ) is not None
    ]

    if not valid:
        _LOG.info(
            "No valid results for near-optimal figure"
        )
        return

    penalties = [
        result["penalty_scale"]
        for result in valid
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    tolerance_levels = [
        (
            "1%",
            "near_optimal_1pct_mean",
            "near_optimal_1pct_std",
        ),
        (
            "2%",
            "near_optimal_2pct_mean",
            "near_optimal_2pct_std",
        ),
        (
            "5%",
            "near_optimal_5pct_mean",
            "near_optimal_5pct_std",
        ),
        (
            "10%",
            "near_optimal_10pct_mean",
            "near_optimal_10pct_std",
        ),
    ]

    for (
        label,
        mean_key,
        std_key,
    ) in tolerance_levels:

        means = [
            (
                result.get(
                    mean_key
                ) or 0.0
            ) * 100.0
            for result in valid
        ]

        stds = [
            (
                result.get(
                    std_key
                ) or 0.0
            ) * 100.0
            for result in valid
        ]

        ax.errorbar(
            penalties,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=5,
            label=f"Within {label} of optimum",
        )

    ax.set_xlabel(
        "QUBO Penalty Scale",
        fontsize=12,
    )

    ax.set_ylabel(
        "Sampling Probability (%)",
        fontsize=12,
    )

    ax.set_title(
        "Rung 1: Near-Optimal Sampling Probability vs Penalty Scale",
        fontsize=13,
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y",
    )

    ax.legend(
        fontsize=10
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    _LOG.info(
        "Saved: %s",
        output_path,
    )


# ============================================================================
# STORAGE
# ============================================================================

def save_raw_results(
    results: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Save all seed-level results."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    _LOG.info(
        "Saved raw results: %s",
        path,
    )


def save_aggregate_results(
    aggregates: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Save aggregated results."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            aggregates,
            file,
            indent=2,
        )

    _LOG.info(
        "Saved aggregate results: %s",
        path,
    )


def save_csv_summary(
    aggregates: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Save a human-readable CSV summary."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "Penalty",
        "N Seeds",
        "Reference",
        "QAOA Obj (mean ± std)",
        "Gap % (mean ± std)",
        "Approx. Ratio (mean ± std)",
        "Feasibility % (mean ± std)",
        "P(opt) %",
        "P(1%) %",
        "P(2%) %",
        "P(5%) %",
        "P(10%) %",
        "Mean Feasible Obj",
        "Runtime (s)",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for agg in aggregates:

            qaoa_mean = agg.get(
                "qaoa_objective_mean"
            )

            qaoa_std = (
                agg.get(
                    "qaoa_objective_std"
                )
                or 0.0
            )

            gap_mean = agg.get(
                "optimality_gap_percent_mean"
            )

            gap_std = (
                agg.get(
                    "optimality_gap_percent_std"
                )
                or 0.0
            )

            ratio_mean = agg.get(
                "approximation_ratio_mean"
            )

            ratio_std = (
                agg.get(
                    "approximation_ratio_std"
                )
                or 0.0
            )

            ref = agg.get(
                "reference_objective_mean"
            )

            feas_mean = (
                agg.get(
                    "feasibility_rate_mean"
                )
                or 0.0
            ) * 100.0

            feas_std = (
                agg.get(
                    "feasibility_rate_std"
                )
                or 0.0
            ) * 100.0

            p_opt = (
                agg.get(
                    "p_opt_mean"
                )
                or 0.0
            ) * 100.0

            p_1 = (
                agg.get(
                    "near_optimal_1pct_mean"
                )
                or 0.0
            ) * 100.0

            p_2 = (
                agg.get(
                    "near_optimal_2pct_mean"
                )
                or 0.0
            ) * 100.0

            p_5 = (
                agg.get(
                    "near_optimal_5pct_mean"
                )
                or 0.0
            ) * 100.0

            p_10 = (
                agg.get(
                    "near_optimal_10pct_mean"
                )
                or 0.0
            ) * 100.0

            mean_feasible = agg.get(
                "mean_feasible_objective_mean"
            )

            runtime = agg.get(
                "runtime_mean"
            )

            writer.writerow(
                {
                    "Penalty": (
                        f"{agg['penalty_scale']:.2f}"
                    ),

                    "N Seeds": agg.get(
                        "n_seeds",
                        0,
                    ),

                    "Reference": (
                        f"{ref:.4f}"
                        if ref is not None
                        else "N/A"
                    ),

                    "QAOA Obj (mean ± std)": (
                        f"{qaoa_mean:.4f} ± "
                        f"{qaoa_std:.4f}"
                        if qaoa_mean is not None
                        else "N/A"
                    ),

                    "Gap % (mean ± std)": (
                        f"{gap_mean:.4f} ± "
                        f"{gap_std:.4f}"
                        if gap_mean is not None
                        else "N/A"
                    ),

                    "Approx. Ratio (mean ± std)": (
                        f"{ratio_mean:.6f} ± "
                        f"{ratio_std:.6f}"
                        if ratio_mean is not None
                        else "N/A"
                    ),

                    "Feasibility % (mean ± std)": (
                        f"{feas_mean:.4f} ± "
                        f"{feas_std:.4f}"
                    ),

                    "P(opt) %": f"{p_opt:.6f}",

                    "P(1%) %": f"{p_1:.6f}",

                    "P(2%) %": f"{p_2:.6f}",

                    "P(5%) %": f"{p_5:.6f}",

                    "P(10%) %": f"{p_10:.6f}",

                    "Mean Feasible Obj": (
                        f"{mean_feasible:.4f}"
                        if mean_feasible is not None
                        else "N/A"
                    ),

                    "Runtime (s)": (
                        f"{runtime:.4f}"
                        if runtime is not None
                        else "N/A"
                    ),
                }
            )

    _LOG.info(
        "Saved CSV summary: %s",
        path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Run the enhanced Rung 1 penalty sweep."""

    parser = argparse.ArgumentParser(
        description=(
            "Enhanced Rung 1 shortest-path QAOA "
            "penalty sweep with multiple seeds"
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=(
            "Number of independent seeds per penalty "
            f"(default: {DEFAULT_SEEDS})"
        ),
    )

    parser.add_argument(
        "--penalties",
        type=str,
        default=",".join(
            str(p)
            for p in DEFAULT_PENALTIES
        ),
        help=(
            "Comma-separated penalty values"
        ),
    )

    parser.add_argument(
        "--p",
        type=int,
        default=DEFAULT_P,
        help=(
            f"QAOA depth (default: {DEFAULT_P})"
        ),
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=(
            f"Measurement shots per experiment "
            f"(default: {DEFAULT_SHOTS})"
        ),
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_MAXITER,
        help=(
            f"Optimizer iterations "
            f"(default: {DEFAULT_MAXITER})"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory",
    )

    args = parser.parse_args()

    # ==================================================================
    # Parse penalties
    # ==================================================================

    try:

        penalties = [
            float(value.strip())
            for value in args.penalties.split(",")
        ]

    except ValueError:

        print(
            "Error: Could not parse penalty values"
        )

        sys.exit(1)

    if not penalties:

        print(
            "Error: No penalty values supplied"
        )

        sys.exit(1)

    if args.seeds <= 0:

        print(
            "Error: --seeds must be positive"
        )

        sys.exit(1)

    if args.shots <= 0:

        print(
            "Error: --shots must be positive"
        )

        sys.exit(1)

    if args.p <= 0:

        print(
            "Error: --p must be positive"
        )

        sys.exit(1)

    # ==================================================================
    # Experiment configuration
    # ==================================================================

    _LOG.info(
        "=" * 70
    )

    _LOG.info(
        "Enhanced Rung 1 Penalty Sweep"
    )

    _LOG.info(
        "=" * 70
    )

    _LOG.info(
        "Seeds per penalty: %d",
        args.seeds,
    )

    _LOG.info(
        "Penalties: %s",
        penalties,
    )

    _LOG.info(
        "QAOA depth (p): %d",
        args.p,
    )

    _LOG.info(
        "Shots: %d",
        args.shots,
    )

    _LOG.info(
        "Optimizer iterations: %d",
        args.maxiter,
    )

    _LOG.info("")

    # ==================================================================
    # Load project configuration
    # ==================================================================

    config = load_config()

    graph = load_network(
        config,
        offline=True,
    )

    # ==================================================================
    # Build one base formulation to obtain the same source / target.
    # ==================================================================

    try:

        base_config = replace_config(
            config,
            penalty_scale=penalties[0],
        )

        base_formulation = build_rung(
            "shortest_path",
            base_config,
            graph,
        )

    except Exception as exc:

        _LOG.error(
            "Could not build base shortest-path formulation: %s",
            exc,
        )

        raise

    source = getattr(
        base_formulation,
        "source",
        None,
    )

    target = getattr(
        base_formulation,
        "target",
        None,
    )

    _LOG.info(
        "Shortest-path source: %s",
        source,
    )

    _LOG.info(
        "Shortest-path target: %s",
        target,
    )

    # ==================================================================
    # Main sweep
    # ==================================================================

    all_results: List[
        Dict[str, Any]
    ] = []
    total_runs = len(penalties) * args.seeds
    completed_runs = 0

    print(
        f"\nTotal QAOA runs: {total_runs}",
        flush=True,
    )
    for penalty_index, penalty_scale in enumerate(
        penalties,
        start=1,
    ):

        _LOG.info(
            "-" * 70
        )

        _LOG.info(
            "Penalty %d/%d: %.4g",
            penalty_index,
            len(penalties),
            penalty_scale,
        )

        penalty_results: List[
            Dict[str, Any]
        ] = []

        # --------------------------------------------------------------
        # Same seed set for every penalty.
        # This is essential for fair paired comparison.
        # --------------------------------------------------------------

        for seed_offset in range(
            args.seeds
        ):

            seed = 42 + seed_offset

            _LOG.info(
                "  Seed %d/%d: %d",
                seed_offset + 1,
                args.seeds,
                seed,
            )

            try:

                # ------------------------------------------------------
                # Build penalty-specific configuration
                # ------------------------------------------------------

                sweep_config = replace_config(
                    config,
                    penalty_scale=penalty_scale,
                    reps=args.p,
                    shots=args.shots,
                    seed=seed,
                )

                # ------------------------------------------------------
                # Build penalty-specific formulation
                # ------------------------------------------------------

                formulation = build_rung(
                    "shortest_path",
                    sweep_config,
                    graph,
                    source=source,
                    target=target,
                )

                # ------------------------------------------------------
                # Run complete experiment
                # ------------------------------------------------------

                result = run_experiment(
                    "shortest_path",
                    sweep_config,
                    graph,
                    formulation=formulation,
                    source=source,
                    target=target,
                    reps=args.p,
                    shots=args.shots,
                    maxiter=args.maxiter,
                    bruteforce=False,
                    annealing=True,
                    quantum=True,
                    verify=False,
                    max_qubits=40,
                    seed=seed,
                    annealing_seed=seed,
                )

                # ------------------------------------------------------
                # Extract metrics from pipeline metadata
                # ------------------------------------------------------

                seed_result = extract_seed_metrics(
                    result=result,
                    penalty_scale=penalty_scale,
                    seed=seed,
                )

                penalty_results.append(
                    seed_result
                )

                all_results.append(
                    seed_result
                )

                _LOG.info(
                    (
                        "    objective=%s | "
                        "feasibility=%s | "
                        "P(opt)=%s | "
                        "P(5%%)=%s"
                    ),
                    seed_result.get(
                        "qaoa_objective"
                    ),
                    seed_result.get(
                        "feasibility_rate"
                    ),
                    seed_result.get(
                        "p_opt"
                    ),
                    seed_result.get(
                        "near_optimal_probability_5pct"
                    ),
                )

            except Exception as exc:

                _LOG.warning(
                    "  Seed %d failed: %s",
                    seed,
                    exc,
                )

                # ------------------------------------------------------
                # Preserve failed-run information.
                # ------------------------------------------------------

                failed_result = {
                    "penalty_scale": penalty_scale,
                    "seed": seed,
                    "status": "failed",
                    "error": str(exc),
                }

                all_results.append(
                    failed_result
                )

            finally:

                completed_runs += 1

                progress = (
                    completed_runs
                    / total_runs
                    * 100.0
                )

                print(
                    f"\rProgress: "
                    f"{completed_runs}/{total_runs} "
                    f"({progress:6.2f}%)",
                    end="",
                    flush=True,
                )
            

        _LOG.info(
            "Penalty %.4g completed: %d successful seeds",
            penalty_scale,
            len(penalty_results),
        )

    # ==================================================================
    # Aggregate
    # ==================================================================

    _LOG.info(
        "=" * 70
    )

    _LOG.info(
        "Aggregating results..."
    )

    successful_results = [
        result
        for result in all_results
        if result.get(
            "status",
            "success",
        ) != "failed"
    ]

    aggregates: List[
        Dict[str, Any]
    ] = []

    for penalty in penalties:

        penalty_subset = [
            result
            for result in successful_results
            if result.get(
                "penalty_scale"
            ) == penalty
        ]

        if not penalty_subset:
            _LOG.warning(
                "No successful results for penalty %.4g",
                penalty,
            )
            continue

        aggregate = aggregate_seed_results(
            penalty_subset
        )

        aggregates.append(
            aggregate
        )

    # ==================================================================
    # Save
    # ==================================================================

    results_dir = Path(
        args.output_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _LOG.info(
        "Saving results..."
    )

    save_raw_results(
        all_results,
        results_dir
        / "shortest_path_penalty_sweep_raw.json",
    )

    save_aggregate_results(
        aggregates,
        results_dir
        / "shortest_path_penalty_sweep_aggregate.json",
    )

    save_csv_summary(
        aggregates,
        results_dir
        / "shortest_path_penalty_sweep.csv",
    )

    # ==================================================================
    # Figures
    # ==================================================================

    _LOG.info(
        "Generating figures..."
    )

    figures_dir = (
        results_dir
        / "paper_figures"
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_quality_vs_penalty(
        aggregates,
        figures_dir
        / "figure_04_rung1_quality_vs_penalty.png",
    )

    plot_feasibility_vs_penalty(
        aggregates,
        figures_dir
        / "figure_05_rung1_feasibility_vs_penalty.png",
    )

    plot_gap_vs_penalty(
        aggregates,
        figures_dir
        / "figure_06_rung1_optimality_gap_vs_penalty.png",
    )

    plot_near_optimal_vs_penalty(
        aggregates,
        figures_dir
        / "figure_07_rung1_near_optimal_probabilities.png",
    )

    # ==================================================================
    # Final summary
    # ==================================================================

    _LOG.info(
        "=" * 70
    )

    _LOG.info(
        "Experiment complete!"
    )

    _LOG.info(
        "Successful seed runs: %d",
        len(successful_results),
    )

    _LOG.info(
        "Failed seed runs: %d",
        len(all_results) - len(successful_results),
    )

    _LOG.info(
        "Raw results: %s",
        results_dir
        / "shortest_path_penalty_sweep_raw.json",
    )

    _LOG.info(
        "Aggregate: %s",
        results_dir
        / "shortest_path_penalty_sweep_aggregate.json",
    )

    _LOG.info(
        "Summary CSV: %s",
        results_dir
        / "shortest_path_penalty_sweep.csv",
    )

    _LOG.info(
        "Figures: %s",
        figures_dir,
    )

    _LOG.info(
        "=" * 70
    )


# ============================================================================
# CONFIGURATION HELPER
# ============================================================================

def replace_config(
    config: Any,
    *,
    penalty_scale: Optional[float] = None,
    reps: Optional[int] = None,
    shots: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Create a configuration copy with QUBO/QAOA settings overridden.

    The original configuration object is never mutated.
    """

    from dataclasses import replace

    # -------------------------
    # QUBO configuration
    # -------------------------
    qubo_updates: Dict[str, Any] = {}

    if penalty_scale is not None:
        qubo_updates["penalty_scale"] = penalty_scale

    new_qubo_config = replace(
        config.qubo,
        **qubo_updates,
    )

    # -------------------------
    # QAOA configuration
    # -------------------------
    qaoa_updates: Dict[str, Any] = {}

    if reps is not None:
        qaoa_updates["reps"] = reps

    if shots is not None:
        qaoa_updates["shots"] = shots

    if seed is not None:
        qaoa_updates["seed_simulator"] = seed

    new_qaoa_config = replace(
        config.qaoa,
        **qaoa_updates,
    )

    # -------------------------
    # Return new Config
    # -------------------------
    return replace(
        config,
        qubo=new_qubo_config,
        qaoa=new_qaoa_config,
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

