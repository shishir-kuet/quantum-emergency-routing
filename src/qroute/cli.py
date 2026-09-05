"""Command-line interface.

``qroute <command>`` after ``pip install -e .``, or ``python -m qroute`` without
installing. Every command is a thin wrapper over :mod:`qroute.pipeline` -- the CLI
parses arguments and prints, and contains no experiment logic of its own, so a
result obtained from the command line and one obtained from a notebook are the
same computation.

Commands
--------
``info``
    Print the resolved configuration, the registered formulations, and the paths
    that will be used. Run this first when something behaves unexpectedly.
``fetch``
    Download the study-area road network from OpenStreetMap and cache it. Needs
    an internet connection exactly once.
``classical``
    Solve one rung with the classical baselines only. Fast, and the right place
    to confirm an encoding before spending simulator time on it.
``run``
    The full experiment: classical baselines, then QAOA, then the comparison
    table and figures.
``ladder``
    ``run`` on all three rungs in order.
``scan``
    Sweep QAOA depth *p* and plot quality against circuit depth.

Exit codes: ``0`` success, ``1`` a handled :class:`~qroute.exceptions.QRouteError`
(bad config, missing data, infeasible encoding), ``130`` interrupted.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .config import Config, load_config
from .exceptions import QRouteError
from .logging_utils import configure_logging, get_logger
from .pipeline import (
    RUNGS,
    ExperimentResult,
    build_rung,
    circuit_report,
    load_network,
    run_classical,
    run_experiment,
    save_experiment,
    save_summary,
    write_figures,
)
from .qubo import available_formulations

__all__ = ["main", "build_parser"]

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level parser.

    Exposed so that tests can check argument wiring without running anything,
    and so ``--help`` text can be rendered into documentation.
    """
    parser = argparse.ArgumentParser(
        prog="qroute",
        description=(
            "Quantum-assisted emergency vehicle routing: QUBO formulations, QAOA "
            "on the Aer simulator, and honest classical baselines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"qroute {__version__}")
    parser.add_argument(
        "-c", "--config", type=Path, default=None, help="path to a YAML config file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console logging verbosity",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the project seed")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # -- info ---------------------------------------------------------------
    info = subparsers.add_parser("info", help="show configuration, paths and formulations")
    info.add_argument(
        "--backend",
        action="store_true",
        help="also probe the Aer simulator (imports Qiskit, so it is slower)",
    )
    info.set_defaults(func=_cmd_info)

    # -- fetch --------------------------------------------------------------
    fetch = subparsers.add_parser("fetch", help="download and cache the road network")
    fetch.add_argument(
        "--force", action="store_true", help="re-download even if a cache exists"
    )
    fetch.add_argument("--plot", action="store_true", help="also write a map of the network")
    fetch.set_defaults(func=_cmd_fetch)

    # -- shared experiment options -----------------------------------------
    def add_problem_options(sub: argparse.ArgumentParser, *, multi: bool = False) -> None:
        if multi:
            sub.add_argument(
                "--rungs",
                nargs="+",
                default=list(RUNGS),
                choices=list(RUNGS),
                help="which rungs to run, in order",
            )
        else:
            sub.add_argument(
                "rung",
                nargs="?",
                default="shortest_path",
                choices=list(RUNGS),
                help="which problem to solve",
            )
        sub.add_argument("--nodes", type=int, default=None, help="instance size (n_nodes)")
        sub.add_argument(
            "--ambulances", type=int, default=None, help="number of ambulances (assignment)"
        )
        sub.add_argument(
            "--incidents", type=int, default=None, help="number of incidents (assignment)"
        )
        sub.add_argument(
            "--weight",
            default=None,
            choices=["travel_time", "length"],
            help="edge cost: seconds or metres",
        )
        sub.add_argument(
            "--penalty",
            type=float,
            default=None,
            help="constraint penalty scale (multiple of the largest cost)",
        )
        sub.add_argument(
            "--offline",
            action="store_true",
            help="use only the cached network; never touch the network",
        )
        sub.add_argument(
            "--no-bruteforce",
            dest="bruteforce",
            action="store_false",
            help="skip exhaustive QUBO search (loses the optimality guarantee)",
        )
        sub.set_defaults(bruteforce=True)

    def add_quantum_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-p", "--reps", type=int, default=None, help="QAOA layers")
        sub.add_argument("--shots", type=int, default=None, help="shots per evaluation")
        sub.add_argument(
            "--optimizer",
            default=None,
            choices=["COBYLA", "SPSA", "POWELL", "NELDER-MEAD"],
            help="classical optimiser for the variational loop",
        )
        sub.add_argument("--maxiter", type=int, default=None, help="optimiser iteration budget")
        sub.add_argument(
            "--exact",
            dest="expectation_mode",
            action="store_const",
            const="exact",
            default="shots",
            help="use the statevector expectation instead of sampling (no shot noise)",
        )
        sub.add_argument(
            "--cvar",
            type=float,
            default=1.0,
            metavar="ALPHA",
            help="CVaR fraction; 1.0 is the plain mean, 0.1-0.25 usually helps",
        )
        sub.add_argument(
            "--initial",
            default="ramp",
            choices=["ramp", "random", "constant"],
            help="initial angles",
        )
        sub.add_argument(
            "--max-qubits",
            type=int,
            default=24,
            help="refuse to simulate more qubits than this",
        )
        sub.add_argument(
            "--no-plots", dest="plots", action="store_false", help="skip figure generation"
        )
        sub.set_defaults(plots=True)

    # -- classical ----------------------------------------------------------
    classical = subparsers.add_parser(
        "classical", help="classical baselines only (fast, no Qiskit needed)"
    )
    add_problem_options(classical)
    classical.add_argument(
        "--plot", action="store_true", help="write a map of the classical route"
    )
    classical.set_defaults(func=_cmd_classical)

    # -- run ----------------------------------------------------------------
    run = subparsers.add_parser("run", help="full experiment on one rung")
    add_problem_options(run)
    add_quantum_options(run)
    run.add_argument(
        "--circuit-report",
        action="store_true",
        help="also print transpiled gate counts and a crude fidelity estimate",
    )
    run.set_defaults(func=_cmd_run)

    # -- ladder -------------------------------------------------------------
    ladder = subparsers.add_parser("ladder", help="run every rung in order")
    add_problem_options(ladder, multi=True)
    add_quantum_options(ladder)
    ladder.set_defaults(func=_cmd_ladder)

    # -- scan ---------------------------------------------------------------
    scan = subparsers.add_parser("scan", help="sweep QAOA depth p")
    add_problem_options(scan)
    add_quantum_options(scan)
    scan.add_argument(
        "--depths",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="the values of p to try",
    )
    scan.set_defaults(func=_cmd_scan)

    return parser


def _apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """Fold command-line overrides into *config*.

    Only sections that actually changed are replaced, so ``config.source_file``
    and everything else survives untouched.
    """
    sections: Dict[str, Any] = {}

    if getattr(args, "seed", None) is not None:
        sections["project"] = replace(config.project, seed=int(args.seed))

    instance_kwargs: Dict[str, Any] = {}
    if getattr(args, "nodes", None) is not None:
        instance_kwargs["n_nodes"] = int(args.nodes)
    if getattr(args, "ambulances", None) is not None:
        instance_kwargs["n_ambulances"] = int(args.ambulances)
    if getattr(args, "incidents", None) is not None:
        instance_kwargs["n_incidents"] = int(args.incidents)
    if getattr(args, "weight", None) is not None:
        instance_kwargs["weight"] = str(args.weight)
    if instance_kwargs:
        sections["instance"] = replace(config.instance, **instance_kwargs)

    if getattr(args, "penalty", None) is not None:
        sections["qubo"] = replace(config.qubo, penalty_scale=float(args.penalty))

    qaoa_kwargs: Dict[str, Any] = {}
    if getattr(args, "reps", None) is not None:
        qaoa_kwargs["reps"] = int(args.reps)
    if getattr(args, "shots", None) is not None:
        qaoa_kwargs["shots"] = int(args.shots)
    if getattr(args, "optimizer", None) is not None:
        qaoa_kwargs["optimizer"] = str(args.optimizer)
    if getattr(args, "maxiter", None) is not None:
        qaoa_kwargs["maxiter"] = int(args.maxiter)
    if qaoa_kwargs:
        sections["qaoa"] = replace(config.qaoa, **qaoa_kwargs)

    return config.with_overrides(**sections) if sections else config


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def _cmd_info(config: Config, args: argparse.Namespace) -> int:
    paths = config.build_paths()
    print(config.summary())
    print()
    print("paths")
    print(f"  root        : {paths.root}")
    print(f"  raw data    : {paths.raw_dir}")
    print(f"  graph cache : {paths.graph_file}"
          f"{'  (present)' if paths.graph_file.is_file() else '  (not downloaded yet)'}")
    print(f"  results     : {paths.results_dir}")
    print(f"  figures     : {paths.figures_dir}")
    print()
    print("formulations : " + ", ".join(sorted(available_formulations())))
    print("rungs        : " + " -> ".join(RUNGS))

    if args.backend:
        print()
        try:
            from .quantum.backends import describe_backend, make_simulator

            print(describe_backend(make_simulator(config)))
        except QRouteError as exc:
            print(f"backend      : unavailable ({exc})")
    return 0


def _cmd_fetch(config: Config, args: argparse.Namespace) -> int:
    graph = load_network(config, force=args.force)
    from .data.graph_io import format_graph_summary, graph_summary

    print(format_graph_summary(graph_summary(graph)))

    if args.plot:
        from .viz import plot_graph, use_headless_backend

        use_headless_backend()
        paths = config.build_paths(create=True)
        target = paths.figures_dir / "network.png"
        plot_graph(graph, path=target, title=f"{config.project.name}: road network")
        print(f"\nMap written to {target}")
    return 0


def _cmd_classical(config: Config, args: argparse.Namespace) -> int:
    graph = load_network(config, offline=args.offline)
    formulation = build_rung(args.rung, config, graph)
    print(formulation.summary())
    print()

    results = run_classical(
        formulation,
        graph=graph,
        bruteforce=args.bruteforce,
        seed=config.project.seed,
    )
    unit = "s" if config.instance.weight == "travel_time" else "m"
    for result in results:
        print("  " + result.describe(unit))

    best = min(
        (result for result in results if result.objective is not None),
        key=lambda result: result.objective,
        default=None,
    )
    if best is not None and best.solution is not None:
        print()
        print("best: " + best.solution.describe(unit))

    if getattr(args, "plot", False):
        _plot_classical(config, formulation, graph, results)
    return 0


def _plot_classical(
    config: Config, formulation: Any, graph: Any, results: Sequence[Any]
) -> None:
    """Draw the best classical route."""
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
        _LOG.info("Nothing to draw: no classical result decoded to a route")
        return

    from .viz import plot_assignment, plot_node_path, plot_route, use_headless_backend

    use_headless_backend()
    paths = config.build_paths(create=True)
    target = paths.figures_dir / f"{formulation.name}_classical.png"
    unit = "s" if config.instance.weight == "travel_time" else "m"
    title = f"{formulation.name}: {best.solver} route"

    if formulation.name == "shortest_path":
        problem = getattr(formulation, "problem", None)
        plot_node_path(
            graph,
            best.solution.route,
            path=target,
            candidate_edges=None if problem is None else problem.edges,
            candidate_paths=(
                None
                if problem is None
                else tuple(
                    (path, problem.path_cost(path))
                    for path in problem.candidate_paths
                )
            ),
            label=f"{best.solver} ({best.objective:,.0f} {unit})",
            unit=unit,
            title=title,
        )
    else:
        instance = getattr(formulation, "instance", None)
        if instance is None:
            _LOG.info("Nothing to draw: formulation carries no instance geometry")
            return
        if best.solution.assignments:
            plot_assignment(instance, best.solution.assignments, graph, path=target)
        else:
            plot_route(
                instance,
                best.solution.route,
                graph,
                path=target,
                title=title,
                closed=formulation.name == "tsp",
            )
    print(f"\nMap written to {target}")


def _cmd_run(config: Config, args: argparse.Namespace) -> int:
    graph = load_network(config, offline=args.offline)
    result = run_experiment(
        args.rung,
        config,
        graph,
        reps=args.reps,
        expectation_mode=args.expectation_mode,
        cvar_alpha=args.cvar,
        bruteforce=args.bruteforce,
        max_qubits=args.max_qubits,
        initial_strategy=args.initial,
    )
    print()
    print(result.report())

    if args.circuit_report:
        report = circuit_report(result, config)
        if report:
            print()
            print(report)

    paths = config.build_paths(create=True)
    save_experiment(result, paths.results_dir / f"{args.rung}.json")

    if args.plots and config.output.save_figures:
        for figure_path in write_figures(config, result, graph):
            print(f"figure: {figure_path}")
    return 0


def _cmd_ladder(config: Config, args: argparse.Namespace) -> int:
    graph = load_network(config, offline=args.offline)
    paths = config.build_paths(create=True)
    results: List[ExperimentResult] = []

    for rung in args.rungs:
        print()
        print("=" * 72)
        try:
            result = run_experiment(
                rung,
                config,
                graph,
                reps=args.reps,
                expectation_mode=args.expectation_mode,
                cvar_alpha=args.cvar,
                bruteforce=args.bruteforce,
                max_qubits=args.max_qubits,
                initial_strategy=args.initial,
            )
        except QRouteError as exc:
            # One rung failing should not lose the rungs that already worked.
            _LOG.error("Rung '%s' failed: %s", rung, exc)
            continue
        print(result.report())
        save_experiment(result, paths.results_dir / f"{rung}.json")
        results.append(result)
        if args.plots and config.output.save_figures:
            write_figures(config, result, graph)

    if not results:
        print("\nNo rung completed successfully.")
        return 1

    summary_path = save_summary(results, paths.results_dir / "ladder_summary.txt")
    print(f"\nSummary written to {summary_path}")
    return 0


def _cmd_scan(config: Config, args: argparse.Namespace) -> int:
    graph = load_network(config, offline=args.offline)
    formulation = build_rung(args.rung, config, graph)
    print(formulation.summary())

    depths = sorted({int(depth) for depth in args.depths if int(depth) >= 1})
    if not depths:
        raise QRouteError("--depths needs at least one positive integer")

    reference: Optional[float] = None
    success: Dict[int, Optional[float]] = {}
    quantum_results: Dict[int, Any] = {}
    rows: List[Any] = []

    for index, depth in enumerate(depths):
        print()
        print(f"--- p = {depth} ---")
        result = run_experiment(
            args.rung,
            config,
            graph,
            formulation=formulation,
            reps=depth,
            expectation_mode=args.expectation_mode,
            cvar_alpha=args.cvar,
            # The classical side does not depend on p: solve it once.
            bruteforce=args.bruteforce and index == 0,
            annealing=index == 0,
            verify=index == 0,
            max_qubits=args.max_qubits,
            initial_strategy=args.initial,
        )
        if result.quantum is None:
            _LOG.warning("No quantum result at p=%d; skipping", depth)
            continue
        quantum_results[depth] = result.quantum
        success[depth] = result.metadata.get("success_probability")
        if index == 0:
            reference = result.ground_energy
            rows = list(result.rows)
        print(f"  best energy {result.quantum.best_energy:.6g} | "
              f"objective {result.quantum.objective} | "
              f"{result.quantum.n_evaluations} evaluations | "
              f"{result.quantum.seconds:.1f} s")

    if not quantum_results:
        print("\nNo depth produced a quantum result.")
        return 1

    paths = config.build_paths(create=True)
    if args.plots and config.output.save_figures:
        from .viz import plot_depth_scaling, use_headless_backend

        use_headless_backend()
        target = paths.figures_dir / f"{args.rung}_depth_scan.png"
        plot_depth_scaling(
            quantum_results,
            path=target,
            reference_energy=reference,
            success_probabilities=success,
            title=f"{args.rung}: quality against QAOA depth",
        )
        print(f"\nFigure written to {target}")

    if rows:
        from .evaluation import comparison_table

        print()
        print(comparison_table(rows, unit="s" if config.instance.weight == "travel_time" else "m"))
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = _apply_overrides(load_config(args.config), args)
        return int(args.func(config, args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except QRouteError as exc:
        # Expected, actionable failures: print the message, not a traceback.
        _LOG.error("%s", exc)
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
