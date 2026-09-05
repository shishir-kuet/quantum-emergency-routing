"""Run reproducible Rung 3 assignment benchmarks.

Every row uses one identical assignment formulation. Dijkstra, when used while
building a real dispatch instance, contributes travel-time coefficients only;
Hungarian/exhaustive search, annealing, and QAOA optimize the assignment QUBO.

Example:
    python scripts/04_assignment_benchmark.py --p 1 2 3 4 --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qroute.config import load_config
from qroute.evaluation import approximation_ratio, optimality_gap
from qroute.pipeline import build_rung, load_network, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--p", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[41, 42, 43])
    parser.add_argument(
        "--optimizers", nargs="+", default=["COBYLA", "SPSA"], choices=["COBYLA", "SPSA"]
    )
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--offline", action="store_true", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.shots is not None or args.maxiter is not None:
        changes = {}
        if args.shots is not None:
            changes["shots"] = args.shots
        if args.maxiter is not None:
            changes["maxiter"] = args.maxiter
        config = config.with_overrides(qaoa=replace(config.qaoa, **changes))

    graph = load_network(config, offline=args.offline)
    formulation = build_rung("assignment", config, graph)
    instance = formulation.instance
    paths = config.build_paths(create=True)
    output = args.output or (paths.results_dir / "assignment_benchmark.json")

    rows = []
    for optimizer in args.optimizers:
        for depth in sorted(set(args.p)):
            if depth < 1:
                continue
            for seed in sorted(set(args.seeds)):
                result = run_experiment(
                    "assignment",
                    config,
                    graph,
                    formulation=formulation,
                    reps=depth,
                    optimizer=optimizer,
                    seed=seed,
                    annealing_seed=seed,
                    bruteforce=True,
                    annealing=True,
                    quantum=True,
                    maxiter=args.maxiter,
                    shots=args.shots,
                )
                qaoa = result.quantum
                sa = next((item for item in result.classical if item.solver == "annealing"), None)
                objective = None if qaoa is None else qaoa.objective
                reference = result.reference_objective
                row = {
                    "instance_id": instance.metadata.get("instance_id", "assignment-instance"),
                    "seed": seed,
                    "vehicles": len(formulation.vehicles),
                    "incidents": len(formulation.incidents),
                    "travel_time_matrix": instance.cost.tolist(),
                    "problem_size": formulation.num_variables,
                    "optimizer": optimizer,
                    "p": depth,
                    "shots": None if qaoa is None else qaoa.shots,
                    "iterations": None if qaoa is None else qaoa.n_evaluations,
                    "qaoa_runtime_seconds": None if qaoa is None else qaoa.seconds,
                    "qaoa_objective": objective,
                    "qaoa_feasible": None if qaoa is None else qaoa.feasible,
                    "best_sampled_bitstring": (
                        None if qaoa is None or qaoa.best_bits is None
                        else formulation.canonical_bits(qaoa.best_bits)
                    ),
                    "exact_optimum": reference,
                    "optimality_gap_percent": (
                        None if objective is None or reference is None
                        else optimality_gap(objective, reference)
                    ),
                    "approximation_ratio": (
                        None if objective is None or reference is None
                        else approximation_ratio(objective, reference)
                    ),
                    "p_opt": result.metadata.get("success_probability"),
                    "feasibility_rate": result.metadata.get("feasibility_rate"),
                    "sa_objective": None if sa is None else sa.objective,
                    "sa_feasible": None if sa is None else sa.feasible,
                    "sa_runtime_seconds": None if sa is None else sa.seconds,
                    "sa_seed": None if sa is None else sa.details.get("seed"),
                    "sa_restart_count": None if sa is None else sa.details.get("n_restarts"),
                    "sa_sweeps": None if sa is None else sa.details.get("n_sweeps"),
                    "sa_initial_states": None if sa is None else sa.details.get("initial_states"),
                    "sa_restart_energies": None if sa is None else sa.details.get("restart_energies"),
                    "sa_final_bitstring": None if sa is None else sa.bits,
                    "sa_raw_energy": None if sa is None else sa.details.get("raw_energy"),
                    "sa_normalized_energy": None if sa is None else sa.details.get("normalized_energy"),
                    "sa_energy_offset": None if sa is None else sa.details.get("energy_offset"),
                    "sa_energy_scale": None if sa is None else sa.details.get("energy_scale"),
                    "qaoa_depth": None if qaoa is None else qaoa.metadata.get("circuit_depth"),
                    "qaoa_two_qubit_gates": None if qaoa is None else qaoa.metadata.get("two_qubit_gates"),
                    "qaoa_top_states": (
                        [] if qaoa is None else sorted(qaoa.counts.items(), key=lambda item: item[1], reverse=True)[:10]
                    ),
                }
                rows.append(row)
                print(
                    f"optimizer={optimizer} p={depth} seed={seed} "
                    f"objective={objective} p(opt)={row['p_opt']} "
                    f"feasible_rate={row['feasibility_rate']}"
                )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    fieldnames = [key for key, value in rows[0].items() if not isinstance(value, (list, dict))] if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)
    print(f"JSON: {output}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
