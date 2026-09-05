"""
QAOA noise-sensitivity study using Qiskit Aer.

This experiment compares noiseless and noisy QAOA execution
using simulated depolarizing and thermal-relaxation noise.

No IBM Quantum hardware backend is used.

Outputs:
    results/assignment_noise_study.json
"""

from __future__ import annotations

import argparse
import json
import sys
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
from qroute.pipeline import build_rung, run_experiment
from qroute.quantum.backends import make_simulator


# ============================================================
# Argument parser
# ============================================================

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Run QAOA noise study using Qiskit Aer."
    )

    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file.",
    )

    parser.add_argument(
        "--p",
        type=int,
        default=3,
        help="QAOA depth/repetitions.",
    )

    parser.add_argument(
        "--shots",
        type=int,
        default=512,
        help="Number of measurement shots.",
    )

    parser.add_argument(
        "--maxiter",
        type=int,
        default=30,
        help="Maximum optimizer iterations.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=5,
        help="Number of independent seeds for each noise level.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path.",
    )

    return parser


# ============================================================
# Build Aer noise model
# ============================================================

def build_noise_model(
    p1: float,
    p2: float,
    include_thermal: bool = True,
):
    """
    Create a Qiskit Aer NoiseModel.

    p1:
        1-qubit depolarizing probability.

    p2:
        2-qubit depolarizing probability.

    Thermal parameters are simulator assumptions.
    They are NOT measured hardware parameters.
    """

    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        thermal_relaxation_error,
    )

    noise_model = NoiseModel()

    # --------------------------------------------------------
    # 1-qubit noise
    # --------------------------------------------------------

    if p1 > 0.0:

        depolarizing_1q = depolarizing_error(
            p1,
            1,
        )

    else:

        depolarizing_1q = None

    # --------------------------------------------------------
    # 2-qubit noise
    # --------------------------------------------------------

    if p2 > 0.0:

        depolarizing_2q = depolarizing_error(
            p2,
            2,
        )

    else:

        depolarizing_2q = None

    # --------------------------------------------------------
    # Thermal relaxation parameters
    # --------------------------------------------------------

    thermal_1q = None
    thermal_2q = None

    if include_thermal:

        # Simulator assumptions
        T1 = 100e-6
        T2 = 50e-6

        # Gate durations
        gate_time_1q = 20e-9
        gate_time_2q = 40e-9

        # 1-qubit thermal error
        thermal_1q = thermal_relaxation_error(
            T1,
            T2,
            gate_time_1q,
        )

        # 2-qubit thermal error
        #
        # Tensor two independent 1-qubit thermal errors.
        thermal_2q_single = thermal_relaxation_error(
            T1,
            T2,
            gate_time_2q,
        )

        thermal_2q = thermal_2q_single.tensor(
            thermal_2q_single
        )

    # ========================================================
    # Combine 1-qubit errors
    # ========================================================

    if depolarizing_1q is not None and thermal_1q is not None:

        combined_1q = depolarizing_1q.compose(
            thermal_1q
        )

    elif depolarizing_1q is not None:

        combined_1q = depolarizing_1q

    elif thermal_1q is not None:

        combined_1q = thermal_1q

    else:

        combined_1q = None

    # ========================================================
    # Combine 2-qubit errors
    # ========================================================

    if depolarizing_2q is not None and thermal_2q is not None:

        combined_2q = depolarizing_2q.compose(
            thermal_2q
        )

    elif depolarizing_2q is not None:

        combined_2q = depolarizing_2q

    elif thermal_2q is not None:

        combined_2q = thermal_2q

    else:

        combined_2q = None

    # ========================================================
    # Attach errors to gates
    # ========================================================

    if combined_1q is not None:

        noise_model.add_all_qubit_quantum_error(
            combined_1q,
            ["h", "rx", "rz"],
        )

    if combined_2q is not None:

        noise_model.add_all_qubit_quantum_error(
            combined_2q,
            ["rzz"],
        )

    return noise_model


# ============================================================
# Main
# ============================================================

def main(argv=None) -> int:

    args = build_parser().parse_args(argv)

    # ========================================================
    # Load configuration
    # ========================================================

    config = load_config(args.config)

    # ========================================================
    # Load road network
    # ========================================================

    pipeline_module = __import__(
        "qroute.pipeline",
        fromlist=["load_network"],
    )

    graph = pipeline_module.load_network(
        config,
        offline=True,
    )

    # ========================================================
    # Build assignment formulation
    # ========================================================

    formulation = build_rung(
        "assignment",
        config,
        graph,
    )

    # ========================================================
    # Output path
    # ========================================================

    paths = config.build_paths(
        create=True
    )

    output = (
        args.output
        or paths.results_dir
        / "assignment_noise_study.json"
    )

    # ========================================================
    # Noise sweep
    # ========================================================

    noise_levels = [
        {
            "label": "noiseless",
            "p1": 0.0,
            "p2": 0.0,
        },
        {
            "label": "depolarizing",
            "p1": 0.050,
            "p2": 0.100,
        },
    ]

    rows = []

    total_runs = (
        len(noise_levels) * args.seeds
    )

    run_number = 0

    # ========================================================
    # Run experiment
    # ========================================================

    for noise_config in noise_levels:

        label = noise_config["label"]
        p1 = noise_config["p1"]
        p2 = noise_config["p2"]

        print()
        print("=" * 70)
        print(f"Noise level : {label}")
        print(f"1Q error    : {p1}")
        print(f"2Q error    : {p2}")
        print("=" * 70)

        # ----------------------------------------------------
        # Noiseless
        # ----------------------------------------------------

        if p1 == 0.0 and p2 == 0.0:

            noise_model = None

        # ----------------------------------------------------
        # Noisy Aer
        # ----------------------------------------------------

        else:

            noise_model = build_noise_model(
                p1=p1,
                p2=p2,
                include_thermal=True,
            )

        # ----------------------------------------------------
        # Independent seeds
        # ----------------------------------------------------

        for seed_index in range(args.seeds):

            run_number += 1

            seed = (
                config.qaoa.seed_simulator
                + seed_index
            )

            print(
                f"[{run_number}/{total_runs}] "
                f"{label} | seed={seed}"
            )

            # ------------------------------------------------
            # Aer simulator
            # ------------------------------------------------

            simulator = make_simulator(
                config,
                noise_model=noise_model,
                seed=seed,
            )

            # ------------------------------------------------
            # QAOA experiment
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Store results
            # ------------------------------------------------

            row = {
                "execution": label,

                "backend": "Qiskit AerSimulator",

                "seed": seed,
                "seed_index": seed_index,

                "depolarizing_1q": p1,
                "depolarizing_2q": p2,

                "thermal_relaxation": (
                    noise_model is not None
                ),

                "T1_us": (
                    100.0
                    if noise_model is not None
                    else None
                ),

                "T2_us": (
                    50.0
                    if noise_model is not None
                    else None
                ),

                "p": args.p,
                "shots": args.shots,
                "maxiter": args.maxiter,

                "objective": (
                    None
                    if qaoa is None
                    else qaoa.objective
                ),

                "feasible": (
                    None
                    if qaoa is None
                    else qaoa.feasible
                ),

                "success_probability": (
                    result.metadata.get(
                        "success_probability"
                    )
                ),

                "feasibility_rate": (
                    result.metadata.get(
                        "feasibility_rate"
                    )
                ),

                "runtime_seconds": (
                    None
                    if qaoa is None
                    else qaoa.seconds
                ),

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

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"    objective          : "
                f"{row['objective']}"
            )

            print(
                f"    success probability: "
                f"{row['success_probability']}"
            )

            print(
                f"    feasibility rate   : "
                f"{row['feasibility_rate']}"
            )

            print(
                f"    feasible            : "
                f"{row['feasible']}"
            )

    # ========================================================
    # Save raw results
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
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("NOISE STUDY COMPLETE")
    print("=" * 70)

    print(f"Total runs : {len(rows)}")
    print(f"Output     : {output}")

    print()
    print("Mean results")
    print("-" * 70)

    for noise_config in noise_levels:

        label = noise_config["label"]

        matching = [
            row
            for row in rows
            if row["execution"] == label
        ]

        objective_values = [
            row["objective"]
            for row in matching
            if row["objective"] is not None
        ]

        probability_values = [
            row["success_probability"]
            for row in matching
            if row["success_probability"] is not None
        ]

        feasibility_values = [
            row["feasibility_rate"]
            for row in matching
            if row["feasibility_rate"] is not None
        ]

        mean_objective = (
            sum(objective_values)
            / len(objective_values)
            if objective_values
            else None
        )

        mean_probability = (
            sum(probability_values)
            / len(probability_values)
            if probability_values
            else None
        )

        mean_feasibility = (
            sum(feasibility_values)
            / len(feasibility_values)
            if feasibility_values
            else None
        )

        print(
            f"{label:25s} "
            f"objective={mean_objective!s:12s} "
            f"p_opt={mean_probability!s:12s} "
            f"feasibility={mean_feasibility}"
        )

    print()

    return 0


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    raise SystemExit(main())