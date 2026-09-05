"""Step 2: classical baselines on all three rungs -- no Qiskit involved.

Run this before any quantum experiment. It answers two questions that decide
whether a QAOA run will mean anything:

1. **What is the optimum?** Dijkstra, exhaustive tour enumeration and the
   Hungarian algorithm are all exact at these sizes, so the reference cost is a
   true optimum and approximation ratios are well defined.
2. **Is the QUBO encoding correct?** Exhaustive search over the QUBO should find
   a ground state that (a) is feasible and (b) has the same cost as the exact
   classical solver. If either check fails, the penalty weight or the encoding is
   wrong, and no amount of quantum tuning will rescue it.

    python scripts/02_classical_baseline.py
    python scripts/02_classical_baseline.py --nodes 6 --plot
    python scripts/02_classical_baseline.py --rungs tsp

Exit code 1 if any encoding check fails -- so this is safe to put in CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dataclasses import replace  # noqa: E402

from qroute.config import load_config  # noqa: E402
from qroute.exceptions import QRouteError  # noqa: E402
from qroute.logging_utils import configure_logging, get_logger  # noqa: E402
from qroute.pipeline import RUNGS, build_rung, load_network, run_classical  # noqa: E402
from qroute.quantum.hamiltonian import verify_hamiltonian  # noqa: E402

_LOG = get_logger("scripts.classical")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--rungs", nargs="+", default=list(RUNGS), choices=list(RUNGS))
    parser.add_argument("--nodes", type=int, default=None, help="override n_nodes")
    parser.add_argument("--penalty", type=float, default=None, help="override penalty_scale")
    parser.add_argument("--plot", action="store_true", help="write a map per rung")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    config = load_config(args.config)
    if args.nodes is not None:
        config = config.with_overrides(instance=replace(config.instance, n_nodes=args.nodes))
    if args.penalty is not None:
        config = config.with_overrides(qubo=replace(config.qubo, penalty_scale=args.penalty))

    graph = load_network(config, offline=True)
    unit = "s" if config.instance.weight == "travel_time" else "m"
    paths = config.build_paths(create=True)
    failures = []

    for rung in args.rungs:
        print()
        print("=" * 72)
        print(f"RUNG: {rung}")
        print("=" * 72)

        formulation = build_rung(rung, config, graph)
        print(formulation.summary())
        print()

        # Convention check first: cheap, and a failure here invalidates everything.
        if formulation.num_variables <= 12:
            matches, difference = verify_hamiltonian(formulation.qubo())
            print(
                f"Hamiltonian <-> QUBO: {'PASS' if matches else 'FAIL'} "
                f"(max |diff| {difference:.2e})"
            )
            if not matches:
                failures.append(f"{rung}: Hamiltonian does not match the QUBO")
        else:
            print(
                f"Hamiltonian check skipped ({formulation.num_variables} qubits is too "
                f"large for a dense matrix; conventions do not depend on size)"
            )

        results = run_classical(formulation, graph=graph, seed=config.project.seed)
        print()
        for result in results:
            print("  " + result.describe(unit))

        # Encoding check: does the QUBO ground state agree with the exact solver?
        exhaustive = next((r for r in results if r.solver == "bruteforce"), None)
        exact = next(
            (r for r in results if r.optimal and r.solver != "bruteforce"), None
        )
        if exhaustive is not None:
            feasible = exhaustive.details.get("ground_state_feasible")
            print()
            print(
                "Ground state feasible: "
                + ("yes" if feasible else "NO -- raise qubo.penalty_scale")
            )
            if not feasible:
                failures.append(f"{rung}: QUBO ground state violates the constraints")
            if (
                exact is not None
                and exhaustive.objective is not None
                and exact.objective is not None
            ):
                gap = abs(exhaustive.objective - exact.objective)
                agree = gap <= 1e-6 * max(1.0, abs(exact.objective))
                print(
                    f"QUBO optimum vs {exact.solver}: "
                    + (
                        "agree"
                        if agree
                        else f"DISAGREE by {gap:.6g} {unit} -- the encoding loses solutions"
                    )
                )
                if not agree:
                    failures.append(
                        f"{rung}: QUBO optimum {exhaustive.objective:.4g} != "
                        f"{exact.solver} optimum {exact.objective:.4g}"
                    )

        if args.plot:
            _plot(config, formulation, graph, results, unit, paths.figures_dir)

    print()
    print("=" * 72)
    if failures:
        print("ENCODING PROBLEMS FOUND:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nFix these before running any quantum experiment.")
        return 1

    print("All encoding checks passed.")
    print("Next: python scripts/03_qaoa_experiment.py")
    return 0


def _plot(config, formulation, graph, results, unit, figures_dir) -> None:
    """Write a map of the best classical solution for one rung."""
    from qroute.viz import plot_assignment, plot_node_path, plot_route, use_headless_backend

    use_headless_backend()
    best = min(
        (
            result
            for result in results
            if result.objective is not None
            and result.solution is not None
            and (result.solution.route or result.solution.assignments)
        ),
        key=lambda result: result.objective,
        default=None,
    )
    if best is None:
        return

    target = figures_dir / f"{formulation.name}_classical.png"
    if formulation.name == "shortest_path":
        plot_node_path(
            graph,
            best.solution.route,
            path=target,
            candidate_edges=formulation.problem.edges,
            label=f"{best.solver} ({best.objective:,.0f} {unit})",
            unit=unit,
            title=f"Rung 1: {best.solver} optimum over "
            f"{formulation.num_variables} candidate segments",
        )
    elif best.solution.assignments:
        plot_assignment(formulation.instance, best.solution.assignments, graph, path=target)
    else:
        plot_route(
            formulation.instance,
            best.solution.route,
            graph,
            path=target,
            closed=formulation.name == "tsp",
            title=f"{formulation.name}: {best.solver} optimum",
        )
    print(f"  map: {target}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QRouteError as exc:
        _LOG.error("%s", exc)
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
