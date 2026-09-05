"""Run a reproducible Rung 1 penalty sweep for paper Figure 4."""

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
from qroute.pipeline import build_rung, load_network, run_experiment


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--penalties", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0])
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config()
    graph = load_network(config, offline=True)
    base = build_rung("shortest_path", config, graph)
    source = base.problem.source
    target = base.problem.target
    paths = config.build_paths(create=True)
    output = args.output or paths.results_dir / "shortest_path_penalty_sweep.json"

    rows = []
    for penalty in args.penalties:
        sweep_config = config.with_overrides(
            qubo=replace(config.qubo, penalty_scale=float(penalty))
        )
        formulation = build_rung(
            "shortest_path", sweep_config, graph, source=source, target=target
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
        )
        qaoa = result.quantum
        rows.append(
            {
                "penalty_scale": penalty,
                "source": source,
                "target": target,
                "reference_objective": result.reference_objective,
                "qaoa_objective": None if qaoa is None else qaoa.objective,
                "qaoa_feasible": None if qaoa is None else qaoa.feasible,
                "feasibility_rate": result.metadata.get("feasibility_rate"),
                "p_opt": result.metadata.get("success_probability"),
                "qaoa_runtime_seconds": None if qaoa is None else qaoa.seconds,
                "qubits": None if qaoa is None else qaoa.n_qubits,
                "circuit_depth": None if qaoa is None else qaoa.metadata.get("circuit_depth"),
            }
        )
        print(rows[-1])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"JSON: {output}")
    print(f"CSV: {output.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
