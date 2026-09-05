"""Generate a geometry-aware Rung 3 assignment figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qroute.config import load_config
from qroute.pipeline import build_rung, load_network, run_experiment, write_figures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=5)
    args = parser.parse_args(argv)

    config = load_config()
    graph = load_network(config, offline=True)
    formulation = build_rung("assignment", config, graph)
    result = run_experiment(
        "assignment",
        config,
        graph,
        formulation=formulation,
        reps=args.p,
        shots=args.shots,
        maxiter=args.maxiter,
        bruteforce=True,
        annealing=True,
        quantum=True,
        max_qubits=24,
    )
    figures = write_figures(config, result, graph)
    assignment_figures = [path for path in figures if path.name.endswith("assignment.png")]
    print("Assignment figure(s):")
    for path in assignment_figures:
        print(path)
    if not assignment_figures:
        raise SystemExit("No geometry-aware assignment figure was generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
