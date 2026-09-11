"""
Rung 1 shortest-path QAOA penalty sweep.

Evaluates how the QUBO penalty scale affects QAOA routing quality.

Primary metrics:
    - Best feasible objective
    - Absolute optimality gap
    - Percentage optimality gap
    - Approximation ratio
    - Feasibility rate
    - Exact-optimal sampling probability
    - Near-optimality indicators

The exact-optimal probability may be zero when QAOA does not sample
the exact reference-optimal route. This is reported honestly rather
than replaced with a surrogate.

Outputs:
    results/shortest_path_penalty_sweep.json
    results/shortest_path_penalty_sweep.csv

Figures:
    results/paper_figures/figure_04_rung1_quality_vs_penalty.png
    results/paper_figures/figure_05_rung1_feasibility_vs_penalty.png
    results/paper_figures/figure_06_rung1_optimality_gap_vs_penalty.png
"""


from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path


# ============================================================
# Project source path
# ============================================================

_SRC = Path(__file__).resolve().parents[1] / "src"

if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ============================================================
# Project imports
# ============================================================

from qroute.config import load_config
from qroute.pipeline import (
    build_rung,
    load_network,
    run_experiment,
)


# ============================================================
# Plotting
# ============================================================

import matplotlib.pyplot as plt


# ============================================================
# Constants
# ============================================================

NEAR_OPTIMAL_TOLERANCES = [
    0.01,   # 1%
    0.02,   # 2%
    0.05,   # 5%
    0.10,   # 10%
]


# ============================================================
# Helpers
# ============================================================

def safe_float(value):
    """
    Convert a value to float when possible.
    Return None for missing/invalid values.
    """

    if value is None:
        return None

    try:
        value = float(value)

    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def calculate_metrics(
    reference_objective,
    qaoa_objective,
):
    """
    Calculate routing-quality metrics.

    This assumes shortest-path minimization:

        smaller objective = better

    Returns:
        absolute_gap
        percentage_gap
        approximation_ratio
        near_optimal flags
    """

    reference = safe_float(
        reference_objective
    )

    qaoa = safe_float(
        qaoa_objective
    )

    metrics = {
        "absolute_gap": None,
        "percentage_gap": None,
        "approximation_ratio": None,

        "within_1_percent": False,
        "within_2_percent": False,
        "within_5_percent": False,
        "within_10_percent": False,
    }

    if reference is None or qaoa is None:
        return metrics

    if reference <= 0:
        return metrics

    # --------------------------------------------------------
    # Absolute gap
    # --------------------------------------------------------

    metrics["absolute_gap"] = (
        qaoa - reference
    )

    # --------------------------------------------------------
    # Percentage gap
    # --------------------------------------------------------

    metrics["percentage_gap"] = (
        (qaoa - reference)
        / reference
        * 100.0
    )

    # --------------------------------------------------------
    # Approximation ratio
    #
    # For minimization:
    #
    #     QAOA / reference
    #
    # 1.0 = optimal
    # >1.0 = worse than optimum
    # --------------------------------------------------------

    metrics["approximation_ratio"] = (
        qaoa / reference
    )

    # --------------------------------------------------------
    # Near-optimality
    # --------------------------------------------------------

    relative_gap = (
        qaoa - reference
    ) / reference

    metrics["within_1_percent"] = (
        relative_gap <= 0.01
    )

    metrics["within_2_percent"] = (
        relative_gap <= 0.02
    )

    metrics["within_5_percent"] = (
        relative_gap <= 0.05
    )

    metrics["within_10_percent"] = (
        relative_gap <= 0.10
    )

    return metrics


# ============================================================
# Generate figures
# ============================================================

def create_quality_figure(
    rows,
    output_path,
):
    """
    Figure:
        Best feasible objective vs penalty scale.

    Reference optimum is shown as a horizontal line.
    """

    valid_rows = [
        row
        for row in rows
        if row["qaoa_objective"] is not None
    ]

    if not valid_rows:
        print(
            "Skipping quality figure: "
            "no feasible QAOA results."
        )
        return

    penalties = [
        row["penalty_scale"]
        for row in valid_rows
    ]

    objectives = [
        row["qaoa_objective"]
        for row in valid_rows
    ]

    reference = valid_rows[0][
        "reference_objective"
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        penalties,
        objectives,
        marker="o",
        linewidth=2,
        label="QAOA best feasible objective",
    )

    plt.axhline(
        reference,
        linestyle="--",
        linewidth=2,
        label="Reference optimum",
    )

    plt.xlabel(
        "QUBO penalty scale"
    )

    plt.ylabel(
        "Route objective"
    )

    plt.title(
        "Rung 1 QAOA Quality versus Penalty Scale"
    )

    plt.xticks(
        penalties
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Feasibility figure
# ============================================================

def create_feasibility_figure(
    rows,
    output_path,
):
    """
    Figure:
        Feasibility rate vs penalty scale.
    """

    penalties = [
        row["penalty_scale"]
        for row in rows
    ]

    feasibility = [
        (
            None
            if row["feasibility_rate"] is None
            else row["feasibility_rate"] * 100.0
        )
        for row in rows
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        penalties,
        feasibility,
        marker="o",
        linewidth=2,
    )

    plt.xlabel(
        "QUBO penalty scale"
    )

    plt.ylabel(
        "Feasibility rate (%)"
    )

    plt.title(
        "Rung 1 QAOA Feasibility versus Penalty Scale"
    )

    plt.xticks(
        penalties
    )

    plt.ylim(
        0,
        max(
            1.0,
            max(
                value
                for value in feasibility
                if value is not None
            ) * 1.2,
        ),
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Optimality-gap figure
# ============================================================

def create_gap_figure(
    rows,
    output_path,
):
    """
    Figure:
        Percentage optimality gap vs penalty scale.

    Lower is better.
    """

    penalties = [
        row["penalty_scale"]
        for row in rows
    ]

    gaps = [
        row["percentage_gap"]
        for row in rows
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        penalties,
        gaps,
        marker="o",
        linewidth=2,
    )

    plt.axhline(
        0.0,
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel(
        "QUBO penalty scale"
    )

    plt.ylabel(
        "Optimality gap (%)"
    )

    plt.title(
        "Rung 1 QAOA Optimality Gap versus Penalty Scale"
    )

    plt.xticks(
        penalties
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main(argv=None) -> int:

    # ========================================================
    # CLI
    # ========================================================

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--penalties",
        nargs="+",
        type=float,
        default=[
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
        ],
        help="QUBO penalty scales to evaluate.",
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44, 45, 46],
        help="Random seeds for independent QAOA runs.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots.",
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum QAOA optimizer iterations.",
    )

    parser.add_argument(
        "--p",
        type=int,
        default=3,
        help="QAOA circuit depth.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file.",
    )

    args = parser.parse_args(argv)

    # ========================================================
    # Load configuration
    # ========================================================

    config = load_config()

    # ========================================================
    # Load road network
    # ========================================================

    graph = load_network(
        config,
        offline=True,
    )

    # ========================================================
    # Build base shortest-path formulation
    # ========================================================

    base = build_rung(
        "shortest_path",
        config,
        graph,
    )

    source = base.problem.source
    target = base.problem.target

    # ========================================================
    # Output locations
    # ========================================================

    paths = config.build_paths(
        create=True
    )

    output = (
        args.output
        or paths.results_dir
        / "shortest_path_penalty_sweep.json"
    )

    figure_dir = (
        paths.results_dir
        / "paper_figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Run sweep
    # ========================================================

    rows = []

    total = len(
        args.penalties
    )

    print()
    print("=" * 80)
    print("RUNG 1 QAOA PENALTY SWEEP")
    print("=" * 80)

    print(
        f"Source       : {source}"
    )

    print(
        f"Target       : {target}"
    )

    print(
        f"QAOA depth   : p={args.p}"
    )

    print(
        f"Shots        : {args.shots}"
    )

    print(
        f"Max iterations: {args.maxiter}"
    )

    print(
        f"Seeds        : {args.seeds}"
    )

    print()

    # ========================================================
    # Each penalty
    # ========================================================

    for index, penalty in enumerate(
        args.penalties,
        start=1,
    ):

        print(
            "-" * 80
        )

        print(
            f"[{index}/{total}] "
            f"Penalty scale = {penalty}"
        )

        # ----------------------------------------------------
        # Override QUBO penalty
        # ----------------------------------------------------

        sweep_config = config.with_overrides(
            qubo=replace(
                config.qubo,
                penalty_scale=float(
                    penalty
                ),
            )
        )

        # ----------------------------------------------------
        # Build formulation
        # ----------------------------------------------------

        formulation = build_rung(
            "shortest_path",
            sweep_config,
            graph,
            source=source,
            target=target,
        )

        # ----------------------------------------------------
        # Run QAOA for each seed
        # ----------------------------------------------------

        for seed_index, seed in enumerate(args.seeds, start=1):

            print(
                f"  [{seed_index}/{len(args.seeds)}] "
                f"Seed = {seed}"
            )

            result = run_experiment(
                "shortest_path",
                sweep_config,
                graph,
                formulation=formulation,
                reps=args.p,
                shots=args.shots,
                maxiter=args.maxiter,
                bruteforce=False,
                annealing=True,
                quantum=True,
                max_qubits=40,
                seed=seed,
            )

            qaoa = result.quantum

            # ----------------------------------------------------
            # Basic values
            # ----------------------------------------------------

            reference_objective = safe_float(
                result.reference_objective
            )

            qaoa_objective = (
                None
                if qaoa is None
                else safe_float(
                    qaoa.objective
                )
            )

            qaoa_feasible = (
                None
                if qaoa is None
                else bool(
                    qaoa.feasible
                )
            )

            feasibility_rate = safe_float(
                result.metadata.get(
                    "feasibility_rate"
                )
            )

            p_opt = safe_float(
                result.metadata.get(
                    "success_probability"
                )
            )

            p_near_1pct = safe_float(
                result.metadata.get(
                    "near_optimal_probability_1pct"
                )
            )

            p_near_2pct = safe_float(
                result.metadata.get(
                    "near_optimal_probability_2pct"
                )
            )

            p_near_5pct = safe_float(
                result.metadata.get(
                    "near_optimal_probability_5pct"
                )
            )

            p_near_10pct = safe_float(
                result.metadata.get(
                    "near_optimal_probability_10pct"
                )
            )

            # ----------------------------------------------------
            # Calculate quality metrics
            # ----------------------------------------------------

            metrics = calculate_metrics(
                reference_objective,
                qaoa_objective,
            )

            # ----------------------------------------------------
            # Build row
            # ----------------------------------------------------

            row = {
                # Experiment
                "penalty_scale": penalty,
                "seed": seed,
                "source": source,
                "target": target,

                "qaoa_layers": args.p,
                "shots": args.shots,
                "maxiter": args.maxiter,

                # Reference
                "reference_objective":
                    reference_objective,

                # QAOA
                "qaoa_objective":
                    qaoa_objective,

                "qaoa_feasible":
                    qaoa_feasible,

                # Sampling
                "feasibility_rate":
                    feasibility_rate,

                "p_opt":
                    p_opt,

                "p_near_1pct":
                    p_near_1pct,

                "p_near_2pct":
                    p_near_2pct,

                "p_near_5pct":
                    p_near_5pct,

                "p_near_10pct":
                    p_near_10pct,

                # Quality metrics
                "absolute_gap":
                    metrics["absolute_gap"],

                "percentage_gap":
                    metrics["percentage_gap"],

                "approximation_ratio":
                    metrics["approximation_ratio"],

                # Near-optimality
                "within_1_percent":
                    metrics["within_1_percent"],

                "within_2_percent":
                    metrics["within_2_percent"],

                "within_5_percent":
                    metrics["within_5_percent"],

                "within_10_percent":
                    metrics["within_10_percent"],

                # Runtime
                "qaoa_runtime_seconds": (
                    None
                    if qaoa is None
                    else safe_float(
                        qaoa.seconds
                    )
                ),

                # Circuit resources
                "qubits": (
                    None
                    if qaoa is None
                    else qaoa.n_qubits
                ),

                "circuit_depth": (
                    None
                    if qaoa is None
                    else qaoa.metadata.get(
                        "circuit_depth"
                    )
                ),

                "two_qubit_gates": (
                    None
                    if qaoa is None
                    else qaoa.metadata.get(
                        "two_qubit_gates"
                    )
                ),
            }

            rows.append(row)

            # ----------------------------------------------------
            # Console report
            # ----------------------------------------------------

            print(
                f"    Reference objective : "
                f"{reference_objective}"
            )

            print(
                f"    QAOA objective      : "
                f"{qaoa_objective}"
            )

            print(
                f"    Feasible            : "
                f"{qaoa_feasible}"
            )

            print(
                f"    Feasibility rate    : "
                f"{None if feasibility_rate is None else feasibility_rate * 100:.4f}%"
                if feasibility_rate is not None
                else "    Feasibility rate    : None"
            )

            print(
                f"    Exact P(opt)        : "
                f"{None if p_opt is None else p_opt * 100:.4f}%"
                if p_opt is not None
                else "    Exact P(opt)        : None"
            )

            print(
                f"    Near-optimal P(1%)  : "
                f"{None if p_near_1pct is None else p_near_1pct * 100:.4f}%"
                if p_near_1pct is not None
                else "    Near-optimal P(1%)  : None"
            )

            print(
                f"    Absolute gap        : "
                f"{metrics['absolute_gap']}"
            )

            print(
                f"    Optimality gap      : "
                f"{None if metrics['percentage_gap'] is None else metrics['percentage_gap']:.4f}%"
                if metrics["percentage_gap"] is not None
                else "    Optimality gap      : None"
            )

            print(
                f"    Approximation ratio : "
                f"{metrics['approximation_ratio']}"
            )

    # ========================================================
    # Save JSON
    # ========================================================

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            rows,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Save CSV
    # ========================================================

    csv_output = output.with_suffix(
        ".csv"
    )

    with csv_output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # Generate figures
    # ========================================================

    quality_figure = (
        figure_dir
        / "figure_04_rung1_quality_vs_penalty.png"
    )

    feasibility_figure = (
        figure_dir
        / "figure_05_rung1_feasibility_vs_penalty.png"
    )

    gap_figure = (
        figure_dir
        / "figure_06_rung1_optimality_gap_vs_penalty.png"
    )

    create_quality_figure(
        rows,
        quality_figure,
    )

    create_feasibility_figure(
        rows,
        feasibility_figure,
    )

    create_gap_figure(
        rows,
        gap_figure,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 80)
    print("RUNG 1 PENALTY SWEEP COMPLETE")
    print("=" * 80)

    print(
        f"JSON : {output}"
    )

    print(
        f"CSV  : {csv_output}"
    )

    print()
    print("Figures:")

    print(
        f"  {quality_figure}"
    )

    print(
        f"  {feasibility_figure}"
    )

    print(
        f"  {gap_figure}"
    )

    print()
    print("Summary (mean ± std across seeds):")
    print()

    print(
        f"{'Penalty':>10} "
        f"{'QAOA Obj':>14} "
        f"{'Gap %':>10} "
        f"{'Feasible %':>12} "
        f"{'P(opt) %':>10} "
        f"{'P(1%) %':>10}"
    )

    print(
        "-" * 75
    )

    # Group by penalty and compute statistics
    from collections import defaultdict
    import statistics

    penalty_groups = defaultdict(list)
    for row in rows:
        penalty_groups[row["penalty_scale"]].append(row)

    for penalty in sorted(penalty_groups.keys()):
        group = penalty_groups[penalty]

        objectives = [r["qaoa_objective"] for r in group if r["qaoa_objective"] is not None]
        gaps = [r["percentage_gap"] for r in group if r["percentage_gap"] is not None]
        feasibilities = [r["feasibility_rate"] for r in group if r["feasibility_rate"] is not None]
        p_opts = [r["p_opt"] for r in group if r["p_opt"] is not None]
        p_near_1pcts = [r["p_near_1pct"] for r in group if r["p_near_1pct"] is not None]

        mean_obj = statistics.mean(objectives) if objectives else None
        mean_gap = statistics.mean(gaps) if gaps else None
        mean_feas = statistics.mean(feasibilities) if feasibilities else None
        mean_popt = statistics.mean(p_opts) if p_opts else None
        mean_pnear1 = statistics.mean(p_near_1pcts) if p_near_1pcts else None

        std_obj = statistics.stdev(objectives) if len(objectives) > 1 else 0
        std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0
        std_feas = statistics.stdev(feasibilities) if len(feasibilities) > 1 else 0
        std_popt = statistics.stdev(p_opts) if len(p_opts) > 1 else 0
        std_pnear1 = statistics.stdev(p_near_1pcts) if len(p_near_1pcts) > 1 else 0

        obj_str = f"{mean_obj:.2f}±{std_obj:.2f}" if mean_obj is not None else "N/A"
        gap_str = f"{mean_gap:.2f}±{std_gap:.2f}" if mean_gap is not None else "N/A"
        feas_str = f"{mean_feas*100:.2f}±{std_feas*100:.2f}" if mean_feas is not None else "N/A"
        popt_str = f"{mean_popt*100:.2f}±{std_popt*100:.2f}" if mean_popt is not None else "N/A"
        pnear1_str = f"{mean_pnear1*100:.2f}±{std_pnear1*100:.2f}" if mean_pnear1 is not None else "N/A"

        print(
            f"{penalty:>10.2f} "
            f"{obj_str:>14} "
            f"{gap_str:>10} "
            f"{feas_str:>12} "
            f"{popt_str:>10} "
            f"{pnear1_str:>10}"
        )

    print()

    return 0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )