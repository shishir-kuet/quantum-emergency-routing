"""Generate paper figures and statistical CSV summaries from saved experiments.

This command only plots measurements that exist in saved JSON/CSV files. Missing
hardware, noise, or local-Rung-2 inputs are reported in ``figure_manifest.json``
instead of being invented.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "paper_figures"
BENCHMARK = RESULTS / "assignment_benchmark.json"
NOISE = RESULTS / "assignment_noise_study.json"
PENALTY = RESULTS / "shortest_path_penalty_sweep.json"
RUNG1_DEPTH_FILES = sorted(
    path
    for path in RESULTS.glob("shortest_path_p*.json")
    if re.fullmatch(r"shortest_path_p\d+", path.stem)
)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with (OUTPUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict], group_key: str, value_key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is not None:
            groups[row[group_key]].append(float(value))
    summary = []
    for group, values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        summary.append({
            group_key: group,
            "n": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
            "median": float(np.median(array)),
            "best": float(array.min()),
            "worst": float(array.max()),
        })
    return summary


def _plot_group(rows, x_key, value_key, title, ylabel, filename):
    summary = _summary(rows, x_key, value_key)
    if not summary:
        return False
    x = [row[x_key] for row in summary]
    means = [row["mean"] for row in summary]
    errors = [row["std"] for row in summary]
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.errorbar(x, means, yerr=errors, marker="o", capsize=4, linewidth=2)
    axes.set_title(title)
    axes.set_xlabel(x_key)
    axes.set_ylabel(ylabel)
    axes.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=300)
    plt.close(figure)
    _save_csv(filename.replace(".png", ".csv"), summary)
    return True


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    benchmark = _load_json(BENCHMARK) or []
    noise = _load_json(NOISE) or []
    penalty_rows = _load_json(PENALTY) or []

    # Rung 1 depth results are saved one JSON file per p value by
    # scripts/03_qaoa_experiment.py --depths. Use the reported route objective,
    # feasibility, and p(opt); do not substitute QUBO energy for travel cost.
    rung1_rows = []
    for path in RUNG1_DEPTH_FILES:
        result = _load_json(path)
        # Depth runs are ExperimentResult objects. Ignore any accidental list
        # or other JSON artifact matching the broad filename prefix.
        if not isinstance(result, dict) or not result.get("quantum"):
            continue
        quantum = result["quantum"]
        metadata = result.get("metadata", {})
        rung1_rows.append(
            {
                "p": quantum.get("reps"),
                "objective": quantum.get("objective"),
                "reference_objective": result.get("reference_objective"),
                "feasibility_rate": metadata.get("feasibility_rate"),
                "p_opt": metadata.get("success_probability"),
                "runtime_seconds": quantum.get("seconds"),
                "circuit_depth": quantum.get("metadata", {}).get("circuit_depth"),
            }
        )

    if rung1_rows:
        rung1_rows.sort(key=lambda row: row["p"])
        _save_csv("rung1_depth_summary.csv", rung1_rows)
        feasible_rows = [row for row in rung1_rows if row["objective"] is not None]
        if feasible_rows:
            figure, axes = plt.subplots(figsize=(7, 4.5))
            axes.plot(
                [row["p"] for row in feasible_rows],
                [row["objective"] for row in feasible_rows],
                marker="o",
                label="QAOA best feasible objective",
            )
            reference = feasible_rows[0]["reference_objective"]
            if reference is not None:
                axes.axhline(reference, linestyle="--", color="#f59e0b", label="candidate-space reference")
            axes.set_title("Rung 1 QAOA quality versus p")
            axes.set_xlabel("QAOA layers (p)")
            axes.set_ylabel("route objective (seconds)")
            axes.grid(alpha=0.25)
            axes.legend()
            figure.tight_layout()
            figure.savefig(OUTPUT / "figure_03_rung1_quality_vs_p.png", dpi=300)
            plt.close(figure)
            manifest["figure_03"] = "generated from results/shortest_path_p*.json"
        else:
            manifest["figure_03"] = "missing_data: no feasible QAOA samples in depth results"

        figure, axes = plt.subplots(figsize=(7, 4.5))
        axes.plot(
            [row["p"] for row in rung1_rows],
            [row["p_opt"] or 0.0 for row in rung1_rows],
            marker="o",
            color="#10b981",
        )
        axes.set_title("Rung 1 QAOA optimal-sample probability versus p")
        axes.set_xlabel("QAOA layers (p)")
        axes.set_ylabel("p(opt)")
        axes.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(OUTPUT / "figure_05_rung1_popt_vs_p.png", dpi=300)
        plt.close(figure)
        _save_csv("rung1_popt_summary.csv", rung1_rows)
        manifest["figure_05"] = "generated from results/shortest_path_p*.json"
    else:
        manifest["figure_03"] = "missing_data: run a Rung 1 depth matrix"
        manifest["figure_05"] = "missing_data: run a Rung 1 depth matrix with p(opt)"

    # Existing road/candidate figures are reused without changing their data.
    existing_route = next(RESULTS.glob("figures/**/shortest_path_route.png"), None)
    existing_comparison = next(RESULTS.glob("figures/**/shortest_path_comparison.png"), None)
    if existing_route:
        shutil.copy2(existing_route, OUTPUT / "figure_02_rung1_candidate_space.png")
        manifest["figure_02"] = "generated from saved Rung 1 route figure"
    else:
        manifest["figure_02"] = "missing_data: run a Rung 1 experiment with plots"
    if existing_comparison:
        shutil.copy2(existing_comparison, OUTPUT / "figure_01_study_area_reference.png")
        manifest["figure_01"] = "generated from saved road-network experiment figure"
    else:
        manifest["figure_01"] = "missing_data: run data fetch and road-network plotting"

    # Assignment quality and sampling figures from the saved benchmark matrix.
    assignment_rows = [row for row in benchmark if row.get("qaoa_objective") is not None]
    if assignment_rows:
        quality_rows = []
        for row in assignment_rows:
            for solver, key in (("Exact", "exact_optimum"), ("SA", "sa_objective"), ("QAOA", "qaoa_objective")):
                if row.get(key) is not None:
                    quality_rows.append({"solver": solver, "objective": row[key]})
        if quality_rows:
            groups = defaultdict(list)
            for row in quality_rows:
                groups[row["solver"]].append(float(row["objective"]))
            figure, axes = plt.subplots(figsize=(7, 4.5))
            labels = list(groups)
            means = [np.mean(groups[label]) for label in labels]
            errors = [np.std(groups[label], ddof=1) if len(groups[label]) > 1 else 0.0 for label in labels]
            axes.bar(labels, means, yerr=errors, capsize=4, color=["#f59e0b", "#10b981", "#6366f1"])
            axes.set_title("Rung 3 assignment solution quality")
            axes.set_ylabel("objective")
            figure.tight_layout()
            figure.savefig(OUTPUT / "figure_09_assignment_quality.png", dpi=300)
            plt.close(figure)
            _save_csv("assignment_solver_summary.csv", [
                {"solver": label, "n": len(groups[label]), "mean": float(np.mean(groups[label])),
                 "std": float(np.std(groups[label], ddof=1)) if len(groups[label]) > 1 else 0.0,
                 "median": float(np.median(groups[label])), "best": float(np.min(groups[label])),
                 "worst": float(np.max(groups[label]))}
                for label in labels
            ])
            manifest["figure_09"] = "generated from exact, SA, and QAOA benchmark columns"
        _plot_group(assignment_rows, "p", "p_opt", "Rung 3 QAOA sampling reliability", "p(opt)", "figure_10_assignment_popt.png")
        manifest["figure_10"] = "generated from results/assignment_benchmark.json"
        _save_csv("assignment_quality_summary.csv", _summary(assignment_rows, "p", "qaoa_objective"))
        _save_csv("assignment_popt_summary.csv", _summary(assignment_rows, "p", "p_opt"))
        _save_csv("assignment_feasibility_summary.csv", _summary(assignment_rows, "p", "feasibility_rate"))
    else:
        for number in (9, 10):
            manifest[f"figure_{number:02d}"] = "missing_data: run scripts/04_assignment_benchmark.py"

    # Resource scaling is only plotted when explicit resource fields exist.
    resource_rows = [row for row in benchmark if row.get("qaoa_depth") is not None]
    if resource_rows:
        _plot_group(resource_rows, "p", "qaoa_depth", "QAOA circuit depth by p", "depth", "figure_06_qubit_or_depth_scaling.png")
        _plot_group(resource_rows, "p", "qaoa_two_qubit_gates", "QAOA two-qubit gates by p", "two-qubit gates", "figure_07_circuit_resources.png")
        manifest["figure_06"] = "resource data available in assignment benchmark"
        manifest["figure_07"] = "resource data available in assignment benchmark"
    else:
        manifest["figure_06"] = "missing_data: run resource benchmark"
        manifest["figure_07"] = "missing_data: run resource benchmark"

    if noise:
        figure, axes = plt.subplots(figsize=(7, 4.5))
        labels = [row["execution"] for row in noise]
        values = [row["objective"] or np.nan for row in noise]
        axes.bar(labels, values, color=["#10b981", "#ef4444"])
        axes.set_title("Noise impact on QAOA assignment objective")
        axes.set_ylabel("objective")
        figure.tight_layout()
        figure.savefig(OUTPUT / "figure_11_noise_impact.png", dpi=300)
        plt.close(figure)
        manifest["figure_11"] = "generated from results/assignment_noise_study.json"
    else:
        manifest["figure_11"] = "missing_data: run scripts/05_noise_study.py"

    if penalty_rows:
        penalty_rows = sorted(penalty_rows, key=lambda row: row["penalty_scale"])
        figure, axes = plt.subplots(figsize=(7, 4.5))
        x = [row["penalty_scale"] for row in penalty_rows]
        feasibility = [row.get("feasibility_rate") or 0.0 for row in penalty_rows]
        axes.plot(x, feasibility, marker="o", color="#10b981", label="QAOA feasibility rate")
        axes.set_xlabel("QUBO penalty scale")
        axes.set_ylabel("feasibility rate")
        axes.set_ylim(0.0, 1.0)
        axes.set_title("Rung 1 feasibility versus penalty scale")
        axes.grid(alpha=0.25)
        axes.legend()
        figure.tight_layout()
        figure.savefig(OUTPUT / "figure_04_rung1_feasibility_vs_penalty.png", dpi=300)
        plt.close(figure)
        _save_csv("rung1_penalty_summary.csv", penalty_rows)
        manifest["figure_04"] = "generated from results/shortest_path_penalty_sweep.json"
    else:
        manifest["figure_04"] = "missing_data: run scripts/07_rung1_penalty_sweep.py"

    assignment_geometry = next(
        RESULTS.glob("figures/**/assignment_assignment.png"), None
    )
    if assignment_geometry:
        shutil.copy2(assignment_geometry, OUTPUT / "figure_08_rung3_assignment_example.png")
        manifest["figure_08"] = "generated from geometry-aware Rung 3 assignment figure"
    else:
        manifest["figure_08"] = "missing_data: run scripts/08_assignment_figure.py"

    manifest["figure_12"] = "missing_data: hardware experiment not performed"
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Paper output: {OUTPUT}")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
