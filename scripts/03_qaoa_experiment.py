"""Step 3: the QAOA experiment -- QUBO to Hamiltonian to circuit to comparison.

Runs the full pipeline on one rung and writes results and figures. The default
settings are chosen to *work*, not to impress: rung 1, p=2, 4096 shots, COBYLA.

    python scripts/03_qaoa_experiment.py
    python scripts/03_qaoa_experiment.py --rung tsp --nodes 4 -p 3
    python scripts/03_qaoa_experiment.py --exact              # no shot noise
    python scripts/03_qaoa_experiment.py --cvar 0.2           # CVaR objective
    python scripts/03_qaoa_experiment.py --depths 1 2 3 4     # depth sweep

Reading the output
------------------
The comparison table is the point. Three numbers matter, in this order:

* **feasible** -- did the best sample decode to a valid route at all? On
  constrained problems this is where low-depth QAOA usually fails first, and a
  cost of "n/a" is a real result, not a crash.
* **ratio** -- optimum / achieved, so 1.000 means optimal. It is only printed
  when the reference is provably optimal.
* **p(opt)** -- the fraction of shots that landed exactly on an optimal route.
  This is the honest measure of how useful the sampler is, and it is usually much
  smaller than the ratio column suggests, because one lucky shot out of 4096 is
  enough to make the ratio 1.000.

A run where ratio is 1.000 and p(opt) is 0.0005 means: the circuit can produce
the answer, but you needed thousands of shots to see it once. Say so in the
write-up.
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
from qroute.pipeline import (  # noqa: E402
    RUNGS,
    build_rung,
    circuit_report,
    load_network,
    run_experiment,
    save_experiment,
    write_figures,
)

_LOG = get_logger("scripts.qaoa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--rung", default="shortest_path", choices=list(RUNGS))
    parser.add_argument("--nodes", type=int, default=None, help="override n_nodes")
    parser.add_argument("--penalty", type=float, default=None, help="override penalty_scale")
    parser.add_argument("-p", "--reps", type=int, default=None, help="QAOA layers")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument(
        "--optimizer", default=None, choices=["COBYLA", "SPSA", "POWELL", "NELDER-MEAD"]
    )
    parser.add_argument(
        "--exact",
        dest="expectation_mode",
        action="store_const",
        const="exact",
        default="shots",
        help="statevector expectation instead of sampling",
    )
    parser.add_argument("--cvar", type=float, default=1.0, metavar="ALPHA")
    parser.add_argument(
        "--initial", default="ramp", choices=["ramp", "random", "constant"]
    )
    parser.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=None,
        help="sweep these values of p instead of running once",
    )
    parser.add_argument("--max-qubits", type=int, default=30)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    parser.set_defaults(plots=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)

    config = load_config(args.config)
    if args.nodes is not None:
        config = config.with_overrides(instance=replace(config.instance, n_nodes=args.nodes))
    if args.penalty is not None:
        config = config.with_overrides(qubo=replace(config.qubo, penalty_scale=args.penalty))
    qaoa_changes = {
        key: value
        for key, value in (
            ("shots", args.shots),
            ("maxiter", args.maxiter),
            ("optimizer", args.optimizer),
        )
        if value is not None
    }
    if qaoa_changes:
        config = config.with_overrides(qaoa=replace(config.qaoa, **qaoa_changes))

    print(config.summary())
    graph = load_network(config, offline=True)
    paths = config.build_paths(create=True)

    if args.depths:
        return _sweep(args, config, graph, paths)

    result = run_experiment(
        args.rung,
        config,
        graph,
        reps=args.reps,
        expectation_mode=args.expectation_mode,
        cvar_alpha=args.cvar,
        initial_strategy=args.initial,
        max_qubits=args.max_qubits,
    )
    print()
    print(result.report())

    report = circuit_report(result, config)
    if report:
        print()
        print(report)

    destination = save_experiment(result, paths.results_dir / f"{args.rung}.json")
    print(f"\nResults: {destination}")

    if args.plots and config.output.save_figures:
        for figure_path in write_figures(config, result, graph):
            print(f"figure: {figure_path}")
    return 0


def _sweep(args, config, graph, paths) -> int:
    """Run the same instance at several depths and plot quality against p.

    The formulation is built once and reused, so every depth attacks the
    *identical* QUBO -- otherwise a change in the sampled instance would be
    indistinguishable from a change in depth, and the plot would be worthless.
    """
    formulation = build_rung(args.rung, config, graph)
    print()
    print(formulation.summary())

    depths = sorted({int(depth) for depth in args.depths if int(depth) >= 1})
    quantum, success = {}, {}
    reference = None

    for index, depth in enumerate(depths):
        print()
        print(f"--- p = {depth} " + "-" * 40)
        result = run_experiment(
            args.rung,
            config,
            graph,
            formulation=formulation,
            reps=depth,
            expectation_mode=args.expectation_mode,
            cvar_alpha=args.cvar,
            initial_strategy=args.initial,
            max_qubits=args.max_qubits,
            # The classical side is depth-independent: pay for it once.
            bruteforce=index == 0,
            annealing=index == 0,
            verify=index == 0,
        )
        if index == 0:
            reference = result.ground_energy
            print(result.table())
        if result.quantum is None:
            continue
        quantum[depth] = result.quantum
        success[depth] = result.metadata.get("success_probability")
        save_experiment(result, paths.results_dir / f"{args.rung}_p{depth}.json")
        print(
            f"  best energy {result.quantum.best_energy:.6g} | "
            f"cost {result.quantum.objective} | "
            f"p(opt) {success[depth]} | {result.quantum.seconds:.1f} s"
        )

    if not quantum:
        print("\nNo depth produced a quantum result.")
        return 1

    if args.plots and config.output.save_figures:
        from qroute.viz import plot_depth_scaling, use_headless_backend

        use_headless_backend()
        target = paths.figures_dir / f"{args.rung}_depth_scan.png"
        plot_depth_scaling(
            quantum,
            path=target,
            reference_energy=reference,
            success_probabilities=success,
            title=f"{args.rung}: quality against QAOA depth",
        )
        print(f"\nFigure: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QRouteError as exc:
        _LOG.error("%s", exc)
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
