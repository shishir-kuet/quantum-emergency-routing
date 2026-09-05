"""
Analyze QAOA noise-study results.

This script evaluates noise impact using metrics that are sensitive
to changes in the QAOA measurement distribution:

    1. Optimal-solution probability
    2. Feasibility rate
    3. Best decoded objective
    4. Percentage degradation relative to noiseless execution

Results are aggregated across independent seeds using mean ± standard
deviation.

Input:
    results/assignment_noise_study.json

Outputs:
    results/paper_figures/
        figure_12_noise_probability.png
        figure_13_noise_feasibility.png
        figure_14_noise_metrics_summary.png

    results/noise_metrics_summary.json
"""


from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


# ============================================================
# Optional plotting import
# ============================================================

import matplotlib.pyplot as plt


# ============================================================
# Argument parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Analyze QAOA noise-study metrics."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/assignment_noise_study.json"
        ),
        help="Path to noise-study JSON file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/paper_figures"
        ),
        help="Directory for generated figures.",
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/noise_metrics_summary.json"
        ),
        help="Path for aggregated summary JSON.",
    )

    return parser


# ============================================================
# Helpers
# ============================================================

def mean(values: list[float]) -> float | None:

    if not values:
        return None

    return statistics.mean(values)


def std(values: list[float]) -> float:

    if len(values) <= 1:
        return 0.0

    return statistics.stdev(values)


def percentage_change(
    baseline: float | None,
    value: float | None,
) -> float | None:

    if baseline is None or value is None:
        return None

    if baseline == 0:
        return None

    return (
        (value - baseline)
        / abs(baseline)
        * 100.0
    )


def degradation(
    baseline: float | None,
    value: float | None,
) -> float | None:

    change = percentage_change(
        baseline,
        value,
    )

    if change is None:
        return None

    return -change


# ============================================================
# Load results
# ============================================================

def load_results(
    path: Path,
) -> list[dict]:

    if not path.exists():

        raise FileNotFoundError(
            f"Noise-study result file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            "Expected the JSON file to contain a list of results."
        )

    return data


# ============================================================
# Group results
# ============================================================

def group_by_execution(
    rows: list[dict],
) -> dict[str, list[dict]]:

    groups: dict[str, list[dict]] = {}

    for row in rows:

        label = row.get(
            "execution",
            "unknown",
        )

        groups.setdefault(
            label,
            [],
        ).append(row)

    return groups


# ============================================================
# Aggregate metrics
# ============================================================

def aggregate_metrics(
    rows: list[dict],
) -> list[dict]:

    groups = group_by_execution(rows)

    # Preserve the experimental order rather than sorting
    # alphabetically.
    order = [
        "noiseless",
        "depolarizing",
        "depolarizing_0.1%",
        "depolarizing_0.5%",
        "depolarizing_1%",
        "depolarizing_2%",
        "depolarizing_5%",
    ]

    labels = []

    for label in order:

        if label in groups:
            labels.append(label)

    # Include any unexpected labels at the end.
    for label in groups:

        if label not in labels:
            labels.append(label)

    # --------------------------------------------------------
    # Find noiseless baseline
    # --------------------------------------------------------

    baseline_rows = groups.get(
        "noiseless",
        [],
    )

    baseline_objectives = [
        float(row["objective"])
        for row in baseline_rows
        if row.get("objective") is not None
    ]

    baseline_probabilities = [
        float(row["success_probability"])
        for row in baseline_rows
        if row.get("success_probability") is not None
    ]

    baseline_feasibilities = [
        float(row["feasibility_rate"])
        for row in baseline_rows
        if row.get("feasibility_rate") is not None
    ]

    baseline_objective = mean(
        baseline_objectives
    )

    baseline_probability = mean(
        baseline_probabilities
    )

    baseline_feasibility = mean(
        baseline_feasibilities
    )

    # --------------------------------------------------------
    # Aggregate every noise level
    # --------------------------------------------------------

    summary = []

    for label in labels:

        group = groups[label]

        objectives = [
            float(row["objective"])
            for row in group
            if row.get("objective") is not None
        ]

        probabilities = [
            float(row["success_probability"])
            for row in group
            if row.get("success_probability") is not None
        ]

        feasibilities = [
            float(row["feasibility_rate"])
            for row in group
            if row.get("feasibility_rate") is not None
        ]

        runtimes = [
            float(row["runtime_seconds"])
            for row in group
            if row.get("runtime_seconds") is not None
        ]

        depths = [
            float(row["circuit_depth"])
            for row in group
            if row.get("circuit_depth") is not None
        ]

        two_qubit_gates = [
            float(row["two_qubit_gates"])
            for row in group
            if row.get("two_qubit_gates") is not None
        ]

        # ----------------------------------------------------
        # Mean and standard deviation
        # ----------------------------------------------------

        mean_objective = mean(objectives)
        std_objective = std(objectives)

        mean_probability = mean(probabilities)
        std_probability = std(probabilities)

        mean_feasibility = mean(feasibilities)
        std_feasibility = std(feasibilities)

        mean_runtime = mean(runtimes)
        std_runtime = std(runtimes)

        mean_depth = mean(depths)
        mean_two_qubit = mean(two_qubit_gates)

        # ----------------------------------------------------
        # Convert probabilities to percentages
        # ----------------------------------------------------

        probability_percent = (
            None
            if mean_probability is None
            else mean_probability * 100.0
        )

        probability_std_percent = (
            std_probability * 100.0
        )

        feasibility_percent = (
            None
            if mean_feasibility is None
            else mean_feasibility * 100.0
        )

        feasibility_std_percent = (
            std_feasibility * 100.0
        )

        # ----------------------------------------------------
        # Degradation relative to noiseless
        # ----------------------------------------------------

        probability_degradation = degradation(
            baseline_probability,
            mean_probability,
        )

        feasibility_degradation = degradation(
            baseline_feasibility,
            mean_feasibility,
        )

        # Objective degradation is included only as a
        # descriptive metric. In your current experiment the
        # best decoded objective may remain unchanged.
        objective_change = percentage_change(
            baseline_objective,
            mean_objective,
        )

        summary.append(
            {
                "execution": label,

                "runs": len(group),

                "mean_objective": mean_objective,
                "std_objective": std_objective,

                "mean_success_probability":
                    mean_probability,

                "std_success_probability":
                    std_probability,

                "success_probability_percent":
                    probability_percent,

                "success_probability_std_percent":
                    probability_std_percent,

                "mean_feasibility_rate":
                    mean_feasibility,

                "std_feasibility_rate":
                    std_feasibility,

                "feasibility_percent":
                    feasibility_percent,

                "feasibility_std_percent":
                    feasibility_std_percent,

                "success_probability_degradation_percent":
                    probability_degradation,

                "feasibility_degradation_percent":
                    feasibility_degradation,

                "objective_change_percent":
                    objective_change,

                "mean_runtime_seconds":
                    mean_runtime,

                "std_runtime_seconds":
                    std_runtime,

                "mean_circuit_depth":
                    mean_depth,

                "mean_two_qubit_gates":
                    mean_two_qubit,
            }
        )

    return summary


# ============================================================
# Print summary
# ============================================================

def print_summary(
    summary: list[dict],
) -> None:

    print()
    print("=" * 90)
    print("QAOA NOISE PERFORMANCE SUMMARY")
    print("=" * 90)

    print()

    print(
        f"{'Execution':25s}"
        f"{'P(opt)':>12s}"
        f"{'Feasible':>14s}"
        f"{'Objective':>16s}"
    )

    print("-" * 90)

    for row in summary:

        probability = row[
            "success_probability_percent"
        ]

        feasibility = row[
            "feasibility_percent"
        ]

        objective = row[
            "mean_objective"
        ]

        probability_text = (
            "N/A"
            if probability is None
            else f"{probability:.3f}%"
        )

        feasibility_text = (
            "N/A"
            if feasibility is None
            else f"{feasibility:.3f}%"
        )

        objective_text = (
            "N/A"
            if objective is None
            else f"{objective:.6f}"
        )

        print(
            f"{row['execution']:25s}"
            f"{probability_text:>12s}"
            f"{feasibility_text:>14s}"
            f"{objective_text:>16s}"
        )

    print()


# ============================================================
# Figure 1
# Optimal-solution probability
# ============================================================

def plot_success_probability(
    summary: list[dict],
    output_path: Path,
) -> None:

    labels = [
        row["execution"]
        for row in summary
    ]

    values = [
        row["success_probability_percent"]
        for row in summary
    ]

    errors = [
        row["success_probability_std_percent"]
        for row in summary
    ]

    x = list(range(len(labels)))

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.errorbar(
        x,
        values,
        yerr=errors,
        marker="o",
        capsize=4,
        linewidth=2,
    )

    plt.xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        "Optimal-solution probability (%)"
    )

    plt.xlabel(
        "Noise condition"
    )

    plt.title(
        "Impact of Noise on QAOA Optimal-Solution Probability"
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
# Figure 2
# Feasibility rate
# ============================================================

def plot_feasibility(
    summary: list[dict],
    output_path: Path,
) -> None:

    labels = [
        row["execution"]
        for row in summary
    ]

    values = [
        row["feasibility_percent"]
        for row in summary
    ]

    errors = [
        row["feasibility_std_percent"]
        for row in summary
    ]

    x = list(range(len(labels)))

    plt.figure(
        figsize=(9, 5.5)
    )

    plt.errorbar(
        x,
        values,
        yerr=errors,
        marker="o",
        capsize=4,
        linewidth=2,
    )

    plt.xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        "Feasible-solution rate (%)"
    )

    plt.xlabel(
        "Noise condition"
    )

    plt.title(
        "Impact of Noise on QAOA Feasibility"
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
# Figure 3
# Combined performance
# ============================================================

def plot_combined_metrics(
    summary: list[dict],
    output_path: Path,
) -> None:

    labels = [
        row["execution"]
        for row in summary
    ]

    probability = [
        row["success_probability_percent"]
        for row in summary
    ]

    feasibility = [
        row["feasibility_percent"]
        for row in summary
    ]

    x = list(range(len(labels)))

    # Width for grouped bars
    width = 0.36

    plt.figure(
        figsize=(10, 6)
    )

    x_probability = [
        value - width / 2
        for value in x
    ]

    x_feasibility = [
        value + width / 2
        for value in x
    ]

    plt.bar(
        x_probability,
        probability,
        width=width,
        label="Optimal-solution probability",
    )

    plt.bar(
        x_feasibility,
        feasibility,
        width=width,
        label="Feasibility rate",
    )

    plt.xticks(
        x,
        labels,
        rotation=25,
        ha="right",
    )

    plt.ylabel(
        "Rate (%)"
    )

    plt.xlabel(
        "Noise condition"
    )

    plt.title(
        "QAOA Performance Under Simulated Noise"
    )

    plt.legend()

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
# Save summary
# ============================================================

def save_summary(
    summary: list[dict],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# Main
# ============================================================

def main(argv=None) -> int:

    args = build_parser().parse_args(argv)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        f"Loading noise-study results from:\n"
        f"  {args.input}"
    )

    rows = load_results(
        args.input
    )

    print(
        f"Loaded {len(rows)} experiment runs."
    )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary = aggregate_metrics(
        rows
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_summary(
        summary
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    save_summary(
        summary,
        args.summary,
    )

    # --------------------------------------------------------
    # Generate figures
    # --------------------------------------------------------

    probability_path = (
        args.output_dir
        / "figure_12_noise_probability.png"
    )

    feasibility_path = (
        args.output_dir
        / "figure_13_noise_feasibility.png"
    )

    combined_path = (
        args.output_dir
        / "figure_14_noise_metrics_summary.png"
    )

    plot_success_probability(
        summary,
        probability_path,
    )

    plot_feasibility(
        summary,
        feasibility_path,
    )

    plot_combined_metrics(
        summary,
        combined_path,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("=" * 90)
    print("NOISE METRIC ANALYSIS COMPLETE")
    print("=" * 90)

    print()
    print("Generated:")
    print(
        f"  {probability_path}"
    )
    print(
        f"  {feasibility_path}"
    )
    print(
        f"  {combined_path}"
    )
    print(
        f"  {args.summary}"
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