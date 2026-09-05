"""Run a small noiseless-versus-noisy QAOA study on the assignment QUBO.

Noise and hardware runs are reporting experiments, not substitutes for exact
QUBO validation. This script uses the installed qiskit-aer noise APIs and keeps
hardware execution opt-in and outside the automated workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qroute.config import load_config
from qroute.pipeline import build_rung, run_experiment
from qroute.quantum.backends import make_simulator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    graph = __import__("qroute.pipeline", fromlist=["load_network"]).load_network(
        config, offline=True
    )
    formulation = build_rung("assignment", config, graph)
    paths = config.build_paths(create=True)
    output = args.output or paths.results_dir / "assignment_noise_study.json"

    try:
        from qiskit_aer.noise import NoiseModel, depolarizing_error
    except ImportError as exc:
        raise SystemExit(f"qiskit-aer noise APIs are unavailable: {exc}") from exc

    noisy_model = NoiseModel()
    one_qubit_error = depolarizing_error(0.01, 1)
    two_qubit_error = depolarizing_error(0.02, 2)
    noisy_model.add_all_qubit_quantum_error(one_qubit_error, ["h", "rx", "rz"])
    noisy_model.add_all_qubit_quantum_error(two_qubit_error, ["rzz"])

    rows = []
    for label, noise_model in (("noiseless", None), ("depolarizing", noisy_model)):
        simulator = make_simulator(config, noise_model=noise_model, seed=config.qaoa.seed_simulator)
        result = run_experiment(
            "assignment",
            config,
            graph,
            formulation=formulation,
            reps=args.p,
            maxiter=args.maxiter,
            shots=args.shots,
            simulator=simulator,
            bruteforce=True,
            annealing=True,
            quantum=True,
        )
        qaoa = result.quantum
        rows.append(
            {
                "execution": label,
                "backend": "AerSimulator",
                "noise_model": None if noise_model is None else "depolarizing(1q=.01,2q=.02)",
                "p": args.p,
                "shots": args.shots,
                "objective": None if qaoa is None else qaoa.objective,
                "feasible": None if qaoa is None else qaoa.feasible,
                "p_opt": result.metadata.get("success_probability"),
                "feasibility_rate": result.metadata.get("feasibility_rate"),
                "runtime_seconds": None if qaoa is None else qaoa.seconds,
                "qubits": None if qaoa is None else qaoa.n_qubits,
                "depth": None if qaoa is None else qaoa.metadata.get("circuit_depth"),
                "two_qubit_gates": None if qaoa is None else qaoa.metadata.get("two_qubit_gates"),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Noise study results: {output}")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
