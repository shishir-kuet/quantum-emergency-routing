"""End-to-end experiment orchestration.

Everything above this module is a library: formulations, solvers, metrics, plots.
This is where they get wired into a runnable experiment, so that the CLI and the
numbered ``scripts/`` are both thin wrappers over the same code path rather than
two divergent copies of the same logic.

One deliberate ordering choice runs through the whole module: **the classical
baseline is always solved first**, and its result is retained as a reference for
validation and reporting. It is not passed into the quantum or annealing solver.
Not for convenience -- for honesty. An approximation ratio needs a denominator
that was fixed before the numerator was known, and brute force also answers the
question that decides whether the quantum run means anything at all:
*is the QUBO ground state feasible?* If the penalty weight is too small, the
quantum optimiser is faithfully minimising the wrong function, and no amount of
tuning ``reps`` will fix it. :func:`run_experiment` checks that first and says so
loudly.

The rungs
---------
``"shortest_path"``
    Pick an origin and destination on the real road graph, take the union of the
    *k* cheapest simple paths, and select edges under flow conservation. Compare
    against Dijkstra, which is exact.

``"tsp"``
    One ambulance visiting several stops and returning. Position-indexed encoding,
    :math:`(n-1)^2` variables. Compare against exhaustive tour enumeration.

``"assignment"``
    Several ambulances, several incidents, minimise response time.
    :math:`V \\times C` variables. Compare against the Hungarian algorithm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .classical import (
    ClassicalResult,
    solve_assignment,
    solve_formulation_annealing,
    solve_formulation_bruteforce,
    solve_shortest_path,
    solve_tsp,
)
from .config import Config
from .data import build_dispatch_instance, build_tour_instance, fetch_road_network, load_graph
from .evaluation import (
    ComparisonRow,
    analyse_counts,
    circuit_stats,
    compare_results,
    comparison_table,
)
from .exceptions import DataError, MissingDependencyError, QRouteError, SolverError
from .logging_utils import get_logger
from .qubo import build_formulation
from .qubo.base import Formulation
from .qubo.shortest_path import (
    ShortestPathFormulation,
    build_local_path_problem,
)
from .quantum import QAOAResult, run_qaoa, verify_hamiltonian
from .rng import make_rng

__all__ = [
    "RUNGS",
    "ExperimentResult",
    "load_network",
    "build_rung",
    "pick_endpoints",
    "run_classical",
    "run_experiment",
    "circuit_report",
    "save_experiment",
    "save_summary",
    "write_figures",
]

_LOG = get_logger(__name__)

#: The three problems, in the order they should be attempted. Each is a working
#: milestone; each is strictly harder than the one before.
RUNGS = ("shortest_path", "local_routing", "assignment")

#: Above this many variables, exhaustive search is skipped automatically.
BRUTEFORCE_LIMIT = 22


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ExperimentResult:
    """One rung, solved classically and (optionally) quantum-mechanically."""

    rung: str
    formulation: Formulation
    classical: Tuple[ClassicalResult, ...]
    quantum: Optional[QAOAResult] = None
    rows: Tuple[ComparisonRow, ...] = ()
    reference_objective: Optional[float] = None
    reference_is_optimal: bool = False
    ground_energy: Optional[float] = None
    penalty_ok: Optional[bool] = None
    hamiltonian_verified: Optional[bool] = None
    unit: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def table(self) -> str:
        return comparison_table(self.rows, unit=self.unit or "cost")

    def report(self) -> str:
        """Multi-line summary suitable for a console or a log file."""
        lines = [
            f"=== {self.rung} ===",
            self.formulation.summary(),
        ]
        if self.hamiltonian_verified is not None:
            lines.append(
                "Hamiltonian check: "
                + ("PASS" if self.hamiltonian_verified else "FAIL -- conventions are wrong")
            )
        if self.penalty_ok is not None:
            lines.append(
                "Penalty check    : "
                + (
                    "PASS (ground state is feasible)"
                    if self.penalty_ok
                    else "FAIL -- the QUBO ground state breaks the constraints; "
                    "raise qubo.penalty_scale"
                )
            )
        lines.append("")
        lines.append(self.table())
        if self.quantum is not None and self.quantum.best_solution is not None:
            lines.append("")
            lines.append("Best quantum sample: " + self.quantum.best_solution.describe(self.unit))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rung": self.rung,
            "formulation": self.formulation.name,
            "summary": self.formulation.summary(),
            "n_variables": self.formulation.num_variables,
            "unit": self.unit,
            "reference_objective": self.reference_objective,
            "reference_is_optimal": self.reference_is_optimal,
            "ground_energy": self.ground_energy,
            "penalty_ok": self.penalty_ok,
            "hamiltonian_verified": self.hamiltonian_verified,
            "classical": [
                {
                    "solver": result.solver,
                    "objective": result.objective,
                    "feasible": result.feasible,
                    "optimal": result.optimal,
                    "seconds": result.seconds,
                    "details": _jsonable(result.details),
                }
                for result in self.classical
            ],
            "quantum": None if self.quantum is None else self.quantum.to_dict(),
            "rows": [row.as_dict() for row in self.rows],
            "metadata": _jsonable(self.metadata),
        }


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of numpy and tuple-heavy structures to JSON types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_network(
    config: Config, *, force: bool = False, offline: bool = False
) -> nx.MultiDiGraph:
    """Return the study-area road graph, from cache or from OpenStreetMap.

    With ``offline=True`` only the cache is consulted, and a missing cache is an
    error rather than a download. Useful in tests and on a train.
    """
    paths = config.build_paths(create=True)
    if offline:
        if not paths.graph_file.is_file():
            raise DataError(
                f"No cached road network at {paths.graph_file}. Run "
                f"'qroute fetch' (or scripts/01_fetch_data.py) once with an "
                f"internet connection."
            )
        _LOG.info("Loading cached network (offline): %s", paths.graph_file)
        return load_graph(
            paths.graph_file, default_speed_kph=config.data.default_speed_kph
        )

    outcome = fetch_road_network(config, force=force)
    _LOG.info("%s", outcome.describe())
    return outcome.graph


def pick_endpoints(
    graph: nx.Graph, config: Config, *, min_separation_m: float = 600.0
) -> Tuple[Any, Any]:
    """Choose a source and target that are far enough apart to be interesting.

    Two adjacent intersections make a shortest-path problem with one edge and no
    decision to make. This samples candidate pairs and keeps the first that clears
    *min_separation_m* of straight-line distance, falling back to the farthest
    pair it saw. Deterministic given ``project.seed``.
    """
    from .data.instance import haversine_m

    nodes = [
        node
        for node, data in graph.nodes(data=True)
        if "x" in data and "y" in data
    ]
    if len(nodes) < 2:
        raise DataError("Need at least two coordinate-bearing nodes to pick endpoints")

    rng = make_rng(config.project.seed, "endpoints")
    best: Tuple[float, Any, Any] = (-1.0, nodes[0], nodes[1])
    for _ in range(400):
        first, second = rng.choice(len(nodes), size=2, replace=False)
        source, target = nodes[int(first)], nodes[int(second)]
        distance = haversine_m(
            float(graph.nodes[source]["y"]),
            float(graph.nodes[source]["x"]),
            float(graph.nodes[target]["y"]),
            float(graph.nodes[target]["x"]),
        )
        if distance > best[0]:
            best = (distance, source, target)
        if distance >= min_separation_m and nx.has_path(graph, source, target):
            _LOG.info(
                "Endpoints %s -> %s, %.0f m apart", source, target, distance
            )
            return source, target

    _LOG.warning(
        "No pair cleared %.0f m; using the farthest sampled pair (%.0f m)",
        min_separation_m,
        best[0],
    )
    return best[1], best[2]


# ---------------------------------------------------------------------------
# Formulations
# ---------------------------------------------------------------------------
def build_rung(
    rung: str,
    config: Config,
    graph: Optional[nx.Graph] = None,
    *,
    source: Optional[Any] = None,
    target: Optional[Any] = None,
    k_paths: int = 3,
    **overrides: Any,
) -> Formulation:
    """Construct the formulation for one rung of the ladder.

    *graph* is required for ``"shortest_path"`` and ``"local_routing"``. The
    local rung expects a small bounded subgraph; the two assignment/instance
    formulations sample nodes from the supplied road graph.
    """
    if rung not in RUNGS and rung != "tsp":
        raise QRouteError(f"Unknown rung {rung!r}; expected one of {RUNGS}")
    if graph is None:
        raise QRouteError(f"Rung '{rung}' needs a road graph")

    if rung in ("shortest_path", "local_routing"):
        if source is None or target is None:
            source, target = pick_endpoints(graph, config)
        if rung == "shortest_path":
            formulation = ShortestPathFormulation.from_graph(
                graph, source, target, config, k_paths=k_paths, **overrides
            )
        else:
            problem = build_local_path_problem(
                graph,
                source,
                target,
                weight=config.instance.weight,
                max_edges=overrides.pop("max_edges", 24),
            )
            formulation = ShortestPathFormulation.from_config(
                problem, config, **overrides
            )
    elif rung == "tsp":
        instance = build_tour_instance(graph, config)
        formulation = build_formulation("tsp", instance, config, **overrides)
    else:
        instance = build_dispatch_instance(graph, config)
        formulation = build_formulation("assignment", instance, config, **overrides)

    _LOG.info("Built %s", formulation.summary())
    return formulation


def _unit_for(formulation: Formulation, config: Config) -> str:
    weight = config.instance.weight
    return "s" if weight == "travel_time" else "m"


# ---------------------------------------------------------------------------
# Classical stage
# ---------------------------------------------------------------------------
def run_classical(
    formulation: Formulation,
    *,
    graph: Optional[nx.Graph] = None,
    bruteforce: bool = True,
    annealing: bool = True,
    bruteforce_limit: int = BRUTEFORCE_LIMIT,
    seed: int = 42,
) -> List[ClassicalResult]:
    """Run every applicable classical solver on *formulation*.

    Three roles, and they are not interchangeable:

    * The **problem-specific exact solver** (Dijkstra / tour enumeration /
      Hungarian) gives the true optimum in the problem's own terms. This is the
      reference.
    * **Exhaustive QUBO search** answers a different question -- whether the
      *encoding* is right. Its ground state should match the exact solver's
      answer; if it does not, the penalty weight or the encoding is wrong.
    * **Simulated annealing** is the fair competitor. It attacks the identical
      QUBO, so it is the only baseline that answers "does the quantum approach
      beat a good classical heuristic *on the same formulation*?"
    """
    results: List[ClassicalResult] = []
    name = formulation.name

    if name == "shortest_path":
        results.append(solve_shortest_path(formulation, full_graph=graph))
    elif name == "tsp":
        results.append(solve_tsp(formulation, method="auto"))
        try:
            results.append(solve_tsp(formulation, method="nearest_neighbour"))
        except SolverError as exc:  # pragma: no cover - defensive
            _LOG.warning("Nearest-neighbour tour failed: %s", exc)
    elif name == "assignment":
        results.append(solve_assignment(formulation, method="hungarian"))
        results.append(solve_assignment(formulation, method="greedy"))

    if bruteforce:
        if formulation.num_variables <= bruteforce_limit:
            results.append(solve_formulation_bruteforce(formulation))
        else:
            _LOG.info(
                "Skipping exhaustive search: %d variables is above the limit of %d",
                formulation.num_variables,
                bruteforce_limit,
            )

    if annealing:
        results.append(solve_formulation_annealing(formulation, seed=seed))

    for result in results:
        _LOG.info("%s", result.describe())
    return results


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------
def run_experiment(
    rung: str,
    config: Config,
    graph: Optional[nx.Graph] = None,
    *,
    formulation: Optional[Formulation] = None,
    source: Optional[Any] = None,
    target: Optional[Any] = None,
    quantum: bool = True,
    reps: Optional[int] = None,
    expectation_mode: str = "shots",
    cvar_alpha: float = 1.0,
    verify: bool = True,
    bruteforce: bool = True,
    annealing: bool = True,
    seed: Optional[int] = None,
    annealing_seed: Optional[int] = None,
    max_qubits: int = 24,
    **qaoa_kwargs: Any,
) -> ExperimentResult:
    """Solve one rung classically, then with QAOA, and compare the two.

    The order of the checks matters and is not negotiable:

    1. **Verify the Hamiltonian** (small instances only). Catches the two
       conventions that are easy to invert: :math:`x = (1-z)/2` and Qiskit's
       right-to-left Pauli labels. Either mistake yields a Hamiltonian that runs
       cleanly and optimises something else.
    2. **Solve classically**, including exhaustive QUBO search where affordable,
       to establish the reference cost *and* to confirm the QUBO ground state is
       feasible.
    3. **Run QAOA** and compare -- with ratios only where the reference is
       provably optimal.
    """
    if formulation is None:
        formulation = build_rung(
            rung, config, graph, source=source, target=target
        )

    unit = _unit_for(formulation, config)
    n = formulation.num_variables

    verified: Optional[bool] = None
    if verify and n <= 12:
        verified, difference = verify_hamiltonian(formulation.qubo())
        _LOG.info(
            "Hamiltonian verification: %s (max |difference| %.3g)",
            "pass" if verified else "FAIL",
            difference,
        )
        if not verified:
            raise SolverError(
                "The problem Hamiltonian does not reproduce the QUBO energies. "
                "Stopping: every result after this point would be meaningless."
            )
    elif verify:
        _LOG.info(
            "Skipping Hamiltonian verification: %d qubits would need a %dx%d dense "
            "matrix. Conventions do not depend on size -- verify on a smaller instance.",
            n,
            2 ** n,
            2 ** n,
        )

    classical = run_classical(
        formulation,
        graph=graph,
        bruteforce=bruteforce,
        annealing=annealing,
        seed=config.project.seed if annealing_seed is None else annealing_seed,
    )

    penalty_ok: Optional[bool] = None
    ground_energy: Optional[float] = None
    exact_objective: Optional[float] = None
    for result in classical:
        if result.solver == "bruteforce":
            penalty_ok = bool(result.details.get("ground_state_feasible"))
            ground_energy = result.details.get("ground_energy")
            exact_objective = result.objective
            break

    if penalty_ok is False:
        raise SolverError(
            "The QUBO ground state is infeasible. Increase qubo.penalty_scale "
            "and validate the formulation before running QAOA."
        )

    if exact_objective is not None and formulation.name == "assignment":
        reference_assignment = next(
            (item for item in classical if item.solver == "assignment:hungarian"),
            None,
        )
        if (
            reference_assignment is not None
            and reference_assignment.objective is not None
            and not np.isclose(exact_objective, reference_assignment.objective)
        ):
            raise SolverError(
                "Assignment QUBO ground-state objective does not match the exact "
                "classical assignment objective."
            )

    quantum_result: Optional[QAOAResult] = None
    if quantum:
        if n > max_qubits:
            _LOG.warning(
                "Skipping QAOA: %d qubits is above the %d-qubit limit for this run",
                n,
                max_qubits,
            )
        else:
            quantum_result = run_qaoa(
                formulation,
                config,
                reps=reps,
                expectation_mode=expectation_mode,
                cvar_alpha=cvar_alpha,
                max_qubits=max_qubits,
                seed=config.project.seed if seed is None else seed,
                **qaoa_kwargs,
            )

    rows = compare_results(formulation, classical, quantum_result)
    reference = next(
        (
            row.objective
            for row in rows
            if row.optimal and row.objective is not None
        ),
        None,
    )
    reference_is_optimal = reference is not None
    if reference is None:
        reference = next((row.objective for row in rows if row.objective is not None), None)

    metadata: Dict[str, Any] = {
        "config": config.summary(),
        "n_variables": n,
        "weight": config.instance.weight,
        "seed": config.project.seed,
        "exact_qubo_objective": exact_objective,
    }
    problem = getattr(formulation, "problem", None)
    if formulation.name == "shortest_path" and formulation.problem.metadata.get(
        "candidate_generation"
    ) != "none; direct local edge variables":
        candidate_path_costs = [
            {
                "rank": rank,
                "path": list(path),
                "cost": problem.path_cost(path),
            }
            for rank, path in enumerate(problem.candidate_paths, start=1)
        ] if problem else []
        metadata.update(
            {
                "optimization_scope": "Reduced candidate-space optimization",
                "candidate_generation": "Dijkstra/Yen k-shortest-path preprocessing",
                "k_paths_requested": (
                    problem.metadata.get("k_paths_requested") if problem else None
                ),
                "candidate_path_count": len(problem.candidate_paths) if problem else None,
                "candidate_edge_count": problem.n_edges if problem else None,
                "candidate_path_costs": candidate_path_costs,
                "standalone_reference_solver": "Dijkstra",
                "source": problem.source if problem else None,
                "target": problem.target if problem else None,
                "instance_id": (
                    f"shortest_path:{problem.source}:{problem.target}"
                    if problem else None
                ),
            }
        )
    elif formulation.name == "shortest_path":
        metadata.update(
            {
                "optimization_scope": "Direct local graph optimization",
                "candidate_generation": "None; local graph edges are QUBO variables",
                "candidate_path_count": len(problem.candidate_paths) if problem else None,
                "candidate_edge_count": problem.n_edges if problem else None,
                "standalone_reference_solver": "Dijkstra on the local graph",
            }
        )
    elif formulation.name == "assignment":
        metadata.update(
            {
                "optimization_scope": "Assignment QUBO",
                "candidate_generation": "None; Dijkstra may generate travel-time coefficients",
                "standalone_reference_solver": "Hungarian/exact enumeration",
                "dijkstra_role": "travel-time coefficient generation only",
                "instance_id": "assignment:" + ",".join(map(str, formulation.vehicles))
                + "->" + ",".join(map(str, formulation.incidents)),
                "vehicles": list(formulation.vehicles),
                "incidents": list(formulation.incidents),
                "travel_time_matrix": formulation.cost_matrix().tolist(),
            }
        )
    elif formulation.name == "tsp":
        metadata.update(
            {
                "optimization_scope": "Complete distilled instance QUBO",
                "candidate_generation": "Shortest-path travel-time preprocessing",
            }
        )
    if quantum_result is not None and quantum_result.counts:
        try:
            statistics = analyse_counts(
                quantum_result.counts,
                formulation,
                reference_objective=reference if reference_is_optimal else None,
            )
            metadata["distribution"] = statistics.describe()
            metadata["feasibility_rate"] = statistics.feasibility_rate
            metadata["success_probability"] = statistics.success_probability
            total_shots = sum(quantum_result.counts.values())
            sampled_states = []
            for bits, count in sorted(
                quantum_result.counts.items(), key=lambda item: item[1], reverse=True
            )[:10]:
                decoded = formulation.decode(bits, qiskit_order=True)
                sampled_states.append(
                    {
                        "bitstring": bits,
                        "count": int(count),
                        "probability": float(count / total_shots) if total_shots else 0.0,
                        "feasible": bool(decoded.feasible),
                        "route": list(decoded.route) if decoded.route else None,
                        "assignments": decoded.assignments,
                        "objective": decoded.objective,
                    }
                )
            metadata["top_sampled_states"] = sampled_states
            metadata["sampled_shots"] = total_shots
            metadata["best_bits_in_counts"] = bool(
                quantum_result.best_bits is not None
                and formulation.canonical_bits(quantum_result.best_bits)
                in quantum_result.counts
            )
        except QRouteError as exc:
            _LOG.warning("Distribution analysis skipped: %s", exc)

    outcome = ExperimentResult(
        rung=rung,
        formulation=formulation,
        classical=tuple(classical),
        quantum=quantum_result,
        rows=tuple(rows),
        reference_objective=reference,
        reference_is_optimal=reference_is_optimal,
        ground_energy=ground_energy,
        penalty_ok=penalty_ok,
        hamiltonian_verified=verified,
        unit=unit,
        metadata=metadata,
    )
    return outcome


def circuit_report(result: ExperimentResult, config: Config) -> Optional[str]:
    """Transpilation and resource report for the run's circuit, if there was one.

    Rebuilds the ansatz rather than storing the circuit on the result: circuits
    are not JSON-serialisable and keeping one alive would pin the whole Qiskit
    object graph in memory for the lifetime of the result.
    """
    if result.quantum is None:
        return None
    from .evaluation.circuit_stats import compare_transpilation
    from .quantum.ansatz import QAOACircuit
    from .quantum.backends import make_simulator, transpile_circuit

    qubo = result.formulation.qubo()
    ansatz = QAOACircuit(result.formulation.ising(), result.quantum.reps)
    logical = ansatz.measured()
    try:
        simulator = make_simulator(config)
        transpiled = transpile_circuit(
            logical, simulator, optimization_level=config.backend.optimization_level
        )
    except QRouteError as exc:
        _LOG.warning("Could not transpile for the report: %s", exc)
        return circuit_stats(logical, stage="logical").describe()

    return compare_transpilation(logical, transpiled, qubo=qubo, reps=result.quantum.reps)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_experiment(result: ExperimentResult, path: Path | str) -> Path:
    """Write an experiment result to JSON.

    Results are written even when a run went badly -- an infeasible outcome with
    the penalty diagnostics attached is data, and re-running experiments to
    recover a number you already had is a waste of an afternoon.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=False)
    _LOG.info("Wrote %s", target)
    return target


def save_summary(results: Sequence[ExperimentResult], path: Path | str) -> Path:
    """Write a plain-text report covering several rungs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(result.report() for result in results)
    target.write_text(text + "\n", encoding="utf-8")
    _LOG.info("Wrote %s", target)
    return target


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def write_figures(
    config: Config, result: ExperimentResult, graph: Optional[nx.Graph] = None
) -> List[Path]:
    """Write the standard figure set for one experiment; return the paths written.

    Individual plot failures are logged and swallowed. A missing PNG is an
    inconvenience; losing a finished experiment because matplotlib disliked a tick
    label would be a real loss. A missing matplotlib is likewise not fatal -- the
    numbers are already saved.
    """
    written: List[Path] = []
    try:
        from .viz import (
            plot_assignment,
            plot_comparison,
            plot_convergence,
            plot_energy_distribution,
            plot_node_path,
            plot_route,
            use_headless_backend,
        )

        use_headless_backend()
    except MissingDependencyError as exc:
        _LOG.warning("Skipping figures: %s", exc)
        return written

    figures = config.build_paths(create=True).figures_dir
    stem = result.rung

    def attempt(name: str, target: Path, draw: Any) -> None:
        try:
            draw()
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            _LOG.warning("%s plot failed: %s", name, exc)
        else:
            written.append(target)

    comparison_path = figures / f"{stem}_comparison.png"
    attempt(
        "comparison",
        comparison_path,
        lambda: plot_comparison(
            result.rows,
            path=comparison_path,
            unit=result.unit,
            title=f"{stem}: route cost by solver",
        ),
    )

    if result.quantum is not None:
        convergence_path = figures / f"{stem}_convergence.png"
        attempt(
            "convergence",
            convergence_path,
            lambda: plot_convergence(
                result.quantum,
                path=convergence_path,
                reference_energy=result.ground_energy,
            ),
        )

        spectrum = None
        if result.formulation.num_variables <= 18:
            try:
                from .qubo.matrix import all_energies

                spectrum = all_energies(result.formulation.qubo())
            except QRouteError as exc:  # pragma: no cover - defensive
                _LOG.debug("No spectrum background: %s", exc)

        energies_path = figures / f"{stem}_energies.png"
        attempt(
            "energy distribution",
            energies_path,
            lambda: plot_energy_distribution(
                result.quantum,
                result.formulation.qubo(),
                path=energies_path,
                spectrum=spectrum,
                ground_energy=result.ground_energy,
            ),
        )

    # Always show both classical and quantum routes for comparison
    solution = None
    comparison_solution = None
    comparison_label = None
    
    # Get quantum solution if available
    if result.quantum is not None and result.quantum.feasible:
        solution = result.quantum.best_solution
    
    # Get classical solution for comparison
    classical_result = next(
        (item for item in result.classical if item.solver == "dijkstra"),
        None,
    )
    if classical_result and classical_result.solution:
        comparison_solution = classical_result.solution
        comparison_label = f"best candidate / classical (dijkstra): {classical_result.objective:.2f} s"
    
    # Fallback to any classical solution if dijkstra not available
    if comparison_solution is None:
        for item in result.classical:
            if item.solution is not None and item.solution.feasible:
                comparison_solution = item.solution
                comparison_label = f"classical ({item.solver}): {item.objective:.2f} s"
                break
    if solution is None:
        _LOG.info("No feasible solution to draw")
        return written

    classical_route = None
    if comparison_solution and comparison_solution.route:
        classical_route = comparison_solution.route
    
    # Add travel time labels
    quantum_cost = "" if solution.objective is None else f" ({solution.objective:.2f} s)"
    quantum_label = f"quantum (QAOA){quantum_cost}"
    
    if result.rung == "shortest_path":
        if solution.route and graph is not None:
            problem = getattr(result.formulation, "problem", None)
            route_path = figures / f"{stem}_route.png"
            attempt(
                "route",
                route_path,
                lambda: plot_node_path(
                    graph,
                    solution.route,
                    path=route_path,
                    candidate_edges=None if problem is None else problem.edges,
                    candidate_paths=(
                        None
                        if problem is None
                        else tuple(
                            (path, problem.path_cost(path))
                            for path in problem.candidate_paths
                        )
                    ),
                    comparison=classical_route,
                    comparison_label=comparison_label,
                    label=quantum_label,
                    unit=result.unit,
                    title=(
                        f"Rung 1: {result.formulation.num_variables} candidate "
                        f"segments, one binary variable each"
                    ),
                ),
            )
        return written

    instance = getattr(result.formulation, "instance", None)
    if instance is None:
        return written

    if solution.assignments:
        assignment_path = figures / f"{stem}_assignment.png"
        alternatives = []
        for item, color, label in (
            (next((item for item in result.classical if item.solver == "assignment:hungarian"), None), "#f59e0b", "Exact / Hungarian"),
            (next((item for item in result.classical if item.solver == "annealing"), None), "#10b981", "Simulated annealing"),
        ):
            if item is not None and item.solution is not None and item.solution.assignments:
                alternatives.append((label, item.solution.assignments, color))
        attempt(
            "assignment",
            assignment_path,
            lambda: plot_assignment(
                instance,
                solution.assignments,
                graph,
                path=assignment_path,
                alternatives=alternatives,
                title="Rung 3 assignment decisions: QAOA vs classical solvers",
            ),
        )
    elif solution.route:
        route_path = figures / f"{stem}_route.png"
        attempt(
            "route",
            route_path,
            lambda: plot_route(
                instance,
                solution.route,
                graph,
                path=route_path,
                comparison=classical_route,
                closed=result.rung == "tsp",
                title=f"{stem}: best route{cost}",
            ),
        )
    return written
