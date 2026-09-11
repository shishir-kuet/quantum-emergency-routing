"""
Main experiment pipeline for Quantum Emergency Routing.

Research workflow
-----------------
1. Load the road network.
2. Build the selected routing/QUBO formulation.
3. Verify the Hamiltonian when feasible.
4. Run the original-problem reference solver.
5. Run QUBO-level classical baselines.
6. Verify the QUBO ground state when feasible.
7. Run QAOA.
8. Analyse the complete QAOA shot distribution.
9. Compute extended distribution-aware metrics.
10. Collect circuit statistics when available.
11. Return a reproducible ExperimentResult.

Reference semantics
-------------------
For shortest_path:

    reference
        -> Dijkstra on the original routing problem

    classical
        -> exact/brute-force search of the QUBO

    annealing
        -> simulated annealing on the same QUBO

    quantum
        -> QAOA on the same QUBO

The original-problem reference and QUBO baselines are kept as separate
fields in ExperimentResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from .classical import (
    ClassicalResult,
    solve_formulation_annealing,
    solve_formulation_bruteforce,
    solve_shortest_path,
    solve_assignment,
    solve_tsp,
)

from .config import Config

from .data import (
    build_dispatch_instance,
    build_tour_instance,
    fetch_road_network,
    load_graph,
)

from .evaluation import (
    ComparisonRow,
    analyse_counts,
    circuit_stats,
    compare_results,
    comparison_table,
    extended_sample_statistics,
)

from .exceptions import (
    DataError,
    QRouteError,
)

from .qubo import (
    AssignmentFormulation,
    Formulation,
    ShortestPathFormulation,
    TSPFormulation,
    build_formulation,
)

from .qubo.shortest_path import (
    build_local_path_problem,
)

from .quantum import (
    QAOAResult,
    run_qaoa,
)

from .rng import (
    make_rng,
)


# ============================================================================
# CONSTANTS
# ============================================================================

RUNGS = (
    "shortest_path",
    "local_routing",
    "assignment",
    "tsp",
)

BRUTEFORCE_LIMIT = 22
DEFAULT_LOCAL_MAX_EDGES = 24
DEFAULT_ENDPOINT_TRIALS = 400
DEFAULT_MIN_ENDPOINT_SEPARATION_M = 600.0


__all__ = [
    "RUNGS",
    "ExperimentResult",
    "load_network",
    "pick_endpoints",
    "build_rung",
    "run_classical",
    "verify_qubo_ground_state",
    "run_quantum",
    "run_experiment",
    "make_comparison_table",
    "circuit_report",
]


_LOG_PREFIX = "[pipeline]"


# ============================================================================
# NETWORK LOADING
# ============================================================================

def load_network(
    config: Config,
    *,
    force: bool = False,
    offline: bool = False,
) -> nx.MultiDiGraph:
    """
    Load the cached road network or fetch it when necessary.

    Parameters
    ----------
    config:
        Project configuration.

    force:
        Force a fresh network fetch.

    offline:
        Only use the cached GraphML file.

    Returns
    -------
    nx.MultiDiGraph
        Loaded road network.
    """

    paths = config.build_paths(create=True)

    # ------------------------------------------------------------------
    # Offline mode
    # ------------------------------------------------------------------

    if offline:

        if not paths.graph_file.is_file():
            raise DataError(
                f"No cached road network found at "
                f"{paths.graph_file}. "
                f"Run scripts/01_fetch_data.py first."
            )

        return load_graph(
            paths.graph_file,
            default_speed_kph=config.data.default_speed_kph,
        )

    # ------------------------------------------------------------------
    # Normal fetch/load workflow
    # ------------------------------------------------------------------

    outcome = fetch_road_network(
        config,
        force=force,
    )

    graph = getattr(
        outcome,
        "graph",
        outcome,
    )

    if graph is None:
        raise DataError(
            "fetch_road_network() did not return a graph."
        )

    return graph


# ============================================================================
# ENDPOINT SELECTION
# ============================================================================

def pick_endpoints(
    graph: nx.Graph,
    config: Config,
    *,
    min_separation_m: float = DEFAULT_MIN_ENDPOINT_SEPARATION_M,
    trials: int = DEFAULT_ENDPOINT_TRIALS,
) -> Tuple[Any, Any]:
    """
    Select reproducible source/target nodes from the graph.

    The selection prefers coordinate-bearing nodes with a reasonably
    large geographic separation.

    Parameters
    ----------
    graph:
        Road network.

    config:
        Project configuration.

    min_separation_m:
        Preferred minimum geographic separation.

    trials:
        Number of random candidate pairs to inspect.

    Returns
    -------
    tuple
        ``(source, target)``
    """

    try:
        from .data.instance import haversine_m
    except ImportError as exc:
        raise DataError(
            "Could not import haversine_m from qroute.data.instance."
        ) from exc

    nodes = [
        node
        for node, data in graph.nodes(data=True)
        if "x" in data and "y" in data
    ]

    if len(nodes) < 2:
        raise DataError(
            "Need at least two coordinate-bearing graph nodes "
            "to select source and target."
        )

    rng = make_rng(
        config.project.seed,
        "endpoints",
    )

    best_distance = -1.0
    best_source = nodes[0]
    best_target = nodes[1]

    n_trials = max(
        1,
        min(
            trials,
            max(1, len(nodes) * 4),
        ),
    )

    for _ in range(n_trials):

        indices = rng.choice(
            len(nodes),
            size=2,
            replace=False,
        )

        source = nodes[int(indices[0])]
        target = nodes[int(indices[1])]

        source_data = graph.nodes[source]
        target_data = graph.nodes[target]

        distance = haversine_m(
            float(source_data["y"]),
            float(source_data["x"]),
            float(target_data["y"]),
            float(target_data["x"]),
        )

        if distance > best_distance:

            best_distance = distance
            best_source = source
            best_target = target

        if distance >= min_separation_m:
            return source, target

    return best_source, best_target


# ============================================================================
# FORMULATION BUILDING
# ============================================================================

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
    """
    Build a QUBO formulation for the selected research rung.

    Parameters
    ----------
    rung:
        One of:

        - ``shortest_path``
        - ``local_routing``
        - ``assignment``
        - ``tsp``

    config:
        Project configuration.

    graph:
        Already-loaded road network.

    source, target:
        Explicit endpoints for path-based rungs.

    k_paths:
        Number of candidate paths for the reduced shortest-path
        formulation.

    overrides:
        Additional formulation-specific options.

    Returns
    -------
    Formulation
        Constructed QUBO formulation.
    """

    if rung not in RUNGS:

        raise QRouteError(
            f"Unknown rung {rung!r}. "
            f"Available rungs: {RUNGS}"
        )

    if graph is None:

        raise QRouteError(
            "build_rung() requires a graph. "
            "Load it first using load_network(config)."
        )

    # ==================================================================
    # PATH-BASED RUNGS
    # ==================================================================

    if rung in (
        "shortest_path",
        "local_routing",
    ):

        # --------------------------------------------------------------
        # Select endpoints automatically when not supplied
        # --------------------------------------------------------------

        if source is None or target is None:

            source, target = pick_endpoints(
                graph,
                config,
            )

        # --------------------------------------------------------------
        # Reduced candidate-path shortest path
        # --------------------------------------------------------------

        if rung == "shortest_path":

            formulation = ShortestPathFormulation.from_graph(
                graph,
                source,
                target,
                config,
                k_paths=k_paths,
                **overrides,
            )

            return formulation

        # --------------------------------------------------------------
        # Local-routing formulation
        # --------------------------------------------------------------

        max_edges = overrides.pop(
            "max_edges",
            DEFAULT_LOCAL_MAX_EDGES,
        )

        problem = build_local_path_problem(
            graph,
            source,
            target,
            weight=config.instance.weight,
            max_edges=max_edges,
        )

        formulation = ShortestPathFormulation.from_config(
            problem,
            config,
            **overrides,
        )

        return formulation

    # ==================================================================
    # ASSIGNMENT
    # ==================================================================

    if rung == "assignment":

        instance = build_dispatch_instance(
            graph,
            config,
        )

        return build_formulation(
            "assignment",
            instance,
            config,
            **overrides,
        )

    # ==================================================================
    # TSP
    # ==================================================================

    if rung == "tsp":

        instance = build_tour_instance(
            graph,
            config,
        )

        return build_formulation(
            "tsp",
            instance,
            config,
            **overrides,
        )

    raise QRouteError(
        f"Unsupported rung: {rung}"
    )


# ============================================================================
# EXPERIMENT RESULT
# ============================================================================

@dataclass
class ExperimentResult:
    """
    Complete result of one experiment.

    reference:
        Original-problem reference solver.

    classical:
        Exact QUBO baseline when brute-force is enabled and feasible.

    annealing:
        Simulated annealing on the same QUBO.

    quantum:
        QAOA result.
    """

    rung: str

    reference: Optional[ClassicalResult] = None

    classical: Optional[ClassicalResult] = None

    annealing: Optional[ClassicalResult] = None

    quantum: Optional[QAOAResult] = None

    reference_objective: Optional[float] = None

    comparison: List[ComparisonRow] = field(
        default_factory=list
    )

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "rung": self.rung,
            "reference": _jsonable(self.reference),
            "classical": _jsonable(self.classical),
            "annealing": _jsonable(self.annealing),
            "quantum": _jsonable(self.quantum),
            "reference_objective": _jsonable(
                self.reference_objective
            ),
            "comparison": _jsonable(
                self.comparison
            ),
            "metadata": _jsonable(
                self.metadata
            ),
        }


# ============================================================================
# SERIALISATION
# ============================================================================

def _jsonable(value: Any) -> Any:
    """
    Convert common Python / NumPy / dataclass-like objects into
    JSON-compatible structures.
    """

    if value is None:
        return None

    if isinstance(value, dict):

        return {
            str(key): _jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            _jsonable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return value.item()

    if hasattr(value, "to_dict"):

        try:
            return _jsonable(
                value.to_dict()
            )
        except Exception:
            pass

    if hasattr(value, "__dict__"):

        try:
            return _jsonable(
                vars(value)
            )
        except Exception:
            pass

    return value


# ============================================================================
# CLASSICAL STAGE
# ============================================================================

def run_classical(
    rung: str,
    graph: nx.Graph,
    formulation: Formulation,
    *,
    bruteforce: bool = True,
    annealing: bool = True,
    seed: Optional[int] = None,
    annealing_seed: Optional[int] = None,
    bruteforce_limit: int = BRUTEFORCE_LIMIT,
) -> Tuple[
    Optional[ClassicalResult],
    Optional[ClassicalResult],
    Optional[ClassicalResult],
]:
    """
    Run the original-problem reference and QUBO classical baselines.

    Returns
    -------
    tuple
        ``(reference, classical, annealing)``

    The reference is deliberately kept separate from the QUBO
    baselines.
    """

    reference_result = None
    classical_result = None
    annealing_result = None

    # ==================================================================
    # ORIGINAL-PROBLEM REFERENCE
    # ==================================================================

    if rung == "shortest_path":

        reference_result = solve_shortest_path(
            formulation,
            full_graph=graph,
        )

    # ==================================================================
    # ASSIGNMENT REFERENCE
    # ==================================================================

    elif rung == "assignment":

        try:

            # The assignment formulation should expose the original
            # instance through its problem/instance attributes.
            instance = getattr(
                formulation,
                "instance",
                None,
            )

            if instance is not None:

                reference_result = solve_assignment(
                    instance,
                )

        except Exception:
            reference_result = None

    # ==================================================================
    # TSP REFERENCE
    # ==================================================================

    elif rung == "tsp":

        try:

            instance = getattr(
                formulation,
                "instance",
                None,
            )

            if instance is not None:

                reference_result = solve_tsp(
                    instance,
                )

        except Exception:
            reference_result = None

    # ==================================================================
    # EXACT QUBO BASELINE
    # ==================================================================

    if bruteforce:

        n_variables = getattr(
            formulation,
            "num_variables",
            formulation.num_variables,
        )

        if n_variables <= bruteforce_limit:

            classical_result = solve_formulation_bruteforce(
                formulation,
            )

    # ==================================================================
    # SIMULATED ANNEALING
    # ==================================================================

    if annealing:

        effective_seed = (
            annealing_seed
            if annealing_seed is not None
            else seed
        )

        annealing_result = solve_formulation_annealing(
            formulation,
            seed=effective_seed,
        )

    return (
        reference_result,
        classical_result,
        annealing_result,
    )


# ============================================================================
# QUBO GROUND-STATE VERIFICATION
# ============================================================================

def verify_qubo_ground_state(
    formulation: Formulation,
    *,
    max_qubits: int = 24,
) -> Dict[str, Any]:
    """
    Verify the QUBO ground state through exhaustive enumeration.

    This is only performed for small instances because exhaustive
    enumeration scales exponentially.
    """

    n_qubits = formulation.num_variables

    if n_qubits > max_qubits:

        return {
            "performed": False,
            "reason": (
                f"QUBO has {n_qubits} qubits; "
                f"maximum allowed is {max_qubits}."
            ),
        }

    result = solve_formulation_bruteforce(
        formulation,
    )

    feasible = getattr(
        result,
        "feasible",
        None,
    )

    objective = getattr(
        result,
        "objective",
        None,
    )

    bits = getattr(
        result,
        "bits",
        None,
    )

    return {
        "performed": True,
        "feasible": feasible,
        "objective": objective,
        "bits": bits,
        "solver": getattr(
            result,
            "solver",
            "bruteforce",
        ),
        "optimal": getattr(
            result,
            "optimal",
            None,
        ),
    }


# ============================================================================
# QUANTUM STAGE
# ============================================================================

def run_quantum(
    formulation: Formulation,
    *,
    config: Optional[Config] = None,
    reps: Optional[int] = None,
    optimizer: Optional[str] = None,
    maxiter: Optional[int] = None,
    shots: Optional[int] = None,
    expectation_mode: str = "shots",
    cvar_alpha: float = 1.0,
    seed: Optional[int] = None,
    max_qubits: int = 24,
    **qaoa_kwargs: Any,
) -> QAOAResult:
    """
    Run QAOA using the project's native ``run_qaoa()`` API.
    """

    return run_qaoa(
        formulation,
        config=config,
        reps=reps,
        optimizer=optimizer,
        maxiter=maxiter,
        shots=shots,
        expectation_mode=expectation_mode,
        cvar_alpha=cvar_alpha,
        seed=seed,
        max_qubits=max_qubits,
        **qaoa_kwargs,
    )


# ============================================================================
# REFERENCE HELPERS
# ============================================================================

def _get_reference_objective(
    reference: Optional[ClassicalResult],
) -> Optional[float]:
    """
    Return the reference objective only when the reference is both
    feasible and explicitly marked optimal.
    """

    if reference is None:
        return None

    if not getattr(
        reference,
        "feasible",
        False,
    ):
        return None

    if not getattr(
        reference,
        "optimal",
        False,
    ):
        return None

    objective = getattr(
        reference,
        "objective",
        None,
    )

    if objective is None:
        return None

    return float(objective)


# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def _analyse_quantum_distribution(
    quantum_result: QAOAResult,
    formulation: Formulation,
    reference_objective: Optional[float],
    metadata: Dict[str, Any],
) -> None:
    """
    Analyse all QAOA samples and store distribution-aware metrics.
    """

    counts = getattr(
        quantum_result,
        "counts",
        None,
    )

    if not counts:
        return

    # ==================================================================
    # BASIC DISTRIBUTION METRICS
    # ==================================================================

    distribution_statistics = analyse_counts(
        counts,
        formulation,
        reference_objective=reference_objective,
    )

    metadata["distribution"] = (
        distribution_statistics.describe()
    )

    metadata["feasibility_rate"] = (
        distribution_statistics.feasibility_rate
    )

    metadata["success_probability"] = (
        distribution_statistics.success_probability
    )

    metadata["near_optimal_probability_1pct"] = (
        distribution_statistics.near_optimal_probability_1pct
    )

    metadata["near_optimal_probability_2pct"] = (
        distribution_statistics.near_optimal_probability_2pct
    )

    metadata["near_optimal_probability_5pct"] = (
        distribution_statistics.near_optimal_probability_5pct
    )

    metadata["near_optimal_probability_10pct"] = (
        distribution_statistics.near_optimal_probability_10pct
    )

    # ==================================================================
    # EXTENDED METRICS
    # ==================================================================

    extended_statistics = extended_sample_statistics(
        counts=counts,
        formulation=formulation,
        reference_objective=reference_objective,
    )

    metadata.update(
        _jsonable(
            extended_statistics
        )
    )

    # ==================================================================
    # SHOTS
    # ==================================================================

    total_shots = int(
        sum(counts.values())
    )

    metadata["sampled_shots"] = total_shots

    metadata["unique_states"] = int(
        len(counts)
    )

    # ==================================================================
    # TOP STATES
    # ==================================================================

    try:

        sorted_counts = sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_states = []

        for bitstring, shot_count in sorted_counts[:10]:

            decoded = formulation.decode(
                bitstring,
                qiskit_order=True,
            )

            top_states.append(
                {
                    "bitstring": bitstring,
                    "shots": int(shot_count),
                    "probability": (
                        float(shot_count)
                        / float(total_shots)
                        if total_shots
                        else 0.0
                    ),
                    "feasible": getattr(
                        decoded,
                        "feasible",
                        None,
                    ),
                    "objective": getattr(
                        decoded,
                        "objective",
                        None,
                    ),
                }
            )

        metadata["top_sampled_states"] = top_states

    except Exception as exc:

        metadata[
            "top_sampled_states_error"
        ] = str(exc)

    # ==================================================================
    # BEST FEASIBLE SAMPLED STATE
    # ==================================================================

    best_bits = None
    best_objective = None

    for bitstring in counts:

        try:

            decoded = formulation.decode(
                bitstring,
                qiskit_order=True,
            )

            if not getattr(
                decoded,
                "feasible",
                False,
            ):
                continue

            objective = getattr(
                decoded,
                "objective",
                None,
            )

            if objective is None:
                continue

            objective = float(
                objective
            )

            if (
                best_objective is None
                or objective < best_objective
            ):

                best_objective = objective
                best_bits = bitstring

        except Exception:
            continue

    metadata[
        "best_bits_in_counts"
    ] = best_bits

    metadata[
        "best_sampled_objective"
    ] = best_objective

    # ==================================================================
    # GAP AGAINST REFERENCE
    # ==================================================================

    if (
        reference_objective is not None
        and best_objective is not None
    ):

        metadata[
            "best_sampled_absolute_gap"
        ] = (
            best_objective
            - reference_objective
        )

        metadata[
            "best_sampled_gap_percent"
        ] = (
            (
                best_objective
                - reference_objective
            )
            / reference_objective
            * 100.0
            if reference_objective != 0
            else None
        )

        metadata[
            "best_sampled_approximation_ratio"
        ] = (
            reference_objective
            / best_objective
            if best_objective != 0
            else None
        )


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

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
    optimizer: Optional[str] = None,
    maxiter: Optional[int] = None,
    shots: Optional[int] = None,
    expectation_mode: str = "shots",
    cvar_alpha: float = 1.0,
    verify: bool = True,
    bruteforce: bool = True,
    annealing: bool = True,
    seed: Optional[int] = None,
    annealing_seed: Optional[int] = None,
    k_paths: int = 3,
    max_qubits: int = 24,
    bruteforce_limit: int = BRUTEFORCE_LIMIT,
    **qaoa_kwargs: Any,
) -> ExperimentResult:
    """
    Execute the complete research workflow.

    Parameters
    ----------
    rung:
        Research rung.

    config:
        Project configuration.

    graph:
        Optional pre-loaded road network.

    formulation:
        Optional pre-built formulation.

    source, target:
        Explicit endpoints for routing experiments.

    quantum:
        Whether to execute QAOA.

    reps:
        QAOA depth.

    optimizer:
        QAOA optimizer.

    maxiter:
        Maximum QAOA optimizer iterations.

    shots:
        Number of QAOA measurement shots.

    expectation_mode:
        ``"shots"`` or ``"exact"``.

    cvar_alpha:
        CVaR parameter.

    verify:
        Enable Hamiltonian and QUBO verification.

    bruteforce:
        Run exact QUBO enumeration when within the limit.

    annealing:
        Run simulated annealing.

    seed:
        Experiment seed.

    annealing_seed:
        Separate annealing seed.

    k_paths:
        Candidate path count for shortest-path formulation.

    max_qubits:
        Maximum QAOA/exhaustive verification size.

    bruteforce_limit:
        Maximum number of binary variables for brute force.
    """

    if rung not in RUNGS:

        raise QRouteError(
            f"Unknown rung {rung!r}; expected one of {RUNGS}"
        )

    effective_seed = (
        config.project.seed
        if seed is None
        else seed
    )

    # ==================================================================
    # 1. BUILD FORMULATION
    # ==================================================================

    if formulation is None:

        if graph is None:

            raise QRouteError(
                "A graph is required when formulation is not supplied."
            )

        formulation = build_rung(
            rung,
            config,
            graph,
            source=source,
            target=target,
            k_paths=k_paths,
            **qaoa_kwargs.pop(
                "formulation_overrides",
                {},
            ),
        )

    # ==================================================================
    # 2. BASIC METADATA
    # ==================================================================

    metadata: Dict[str, Any] = {

        "rung": rung,

        "num_qubits": int(
            formulation.num_variables
        ),

        "num_variables": int(
            formulation.num_variables
        ),

        "seed": effective_seed,

        "config_seed": int(
            config.project.seed
        ),

        "qubo_penalty_scale": float(
            config.qubo.penalty_scale
        ),

        "qubo_normalise": bool(
            config.qubo.normalise
        ),

        "qaoa_reps": (
            reps
            if reps is not None
            else config.qaoa.reps
        ),

        "qaoa_optimizer": (
            optimizer
            if optimizer is not None
            else config.qaoa.optimizer
        ),

        "qaoa_maxiter": (
            maxiter
            if maxiter is not None
            else config.qaoa.maxiter
        ),

        "qaoa_shots": (
            shots
            if shots is not None
            else config.qaoa.shots
        ),

        "expectation_mode": expectation_mode,

        "cvar_alpha": float(
            cvar_alpha
        ),
    }

    # ==================================================================
    # 3. HAMILTONIAN VERIFICATION
    # ==================================================================

    if verify:

        if formulation.num_variables <= 12:

            try:

                verification = (
                    formulation.verify_hamiltonian()
                )

                metadata[
                    "hamiltonian_verification"
                ] = _jsonable(
                    verification
                )

            except Exception as exc:

                metadata[
                    "hamiltonian_verification"
                ] = {
                    "performed": False,
                    "error": str(exc),
                }

        else:

            metadata[
                "hamiltonian_verification"
            ] = {
                "performed": False,
                "reason": (
                    f"Skipped for "
                    f"{formulation.num_variables} qubits."
                ),
            }

    # ==================================================================
    # 4. CLASSICAL STAGE
    # ==================================================================

    if graph is None:

        # For experiments using a pre-built formulation, some classical
        # reference solvers may not need the graph. Keep this safe.
        graph_for_classical = None

    else:

        graph_for_classical = graph

    if (
        rung == "shortest_path"
        and graph_for_classical is None
    ):

        # Try to recover the full graph only when possible.
        try:

            graph_for_classical = load_network(
                config,
                offline=True,
            )

        except Exception:

            graph_for_classical = None

    if graph_for_classical is not None:

        (
            reference_result,
            classical_result,
            annealing_result,
        ) = run_classical(
            rung,
            graph_for_classical,
            formulation,
            bruteforce=bruteforce,
            annealing=annealing,
            seed=effective_seed,
            annealing_seed=annealing_seed,
            bruteforce_limit=bruteforce_limit,
        )

    else:

        reference_result = None

        classical_result = (
            solve_formulation_bruteforce(
                formulation
            )
            if (
                bruteforce
                and formulation.num_variables <= bruteforce_limit
            )
            else None
        )

        annealing_result = (
            solve_formulation_annealing(
                formulation,
                seed=(
                    annealing_seed
                    if annealing_seed is not None
                    else effective_seed
                ),
            )
            if annealing
            else None
        )

    # ==================================================================
    # 5. REFERENCE METADATA
    # ==================================================================

    if reference_result is not None:

        metadata.update(
            {
                "reference_solver": getattr(
                    reference_result,
                    "solver",
                    None,
                ),

                "reference_is_optimal": bool(
                    getattr(
                        reference_result,
                        "optimal",
                        False,
                    )
                ),

                "reference_feasible": bool(
                    getattr(
                        reference_result,
                        "feasible",
                        False,
                    )
                ),

                "reference_objective": getattr(
                    reference_result,
                    "objective",
                    None,
                ),
            }
        )

    else:

        metadata.update(
            {
                "reference_solver": None,
                "reference_is_optimal": False,
                "reference_feasible": False,
                "reference_objective": None,
            }
        )

    # ==================================================================
    # 6. REFERENCE OBJECTIVE
    # ==================================================================

    reference_objective = (
        _get_reference_objective(
            reference_result
        )
    )

    # ==================================================================
    # 7. QUBO GROUND-STATE VERIFICATION
    # ==================================================================

    if verify:

        metadata[
            "qubo_ground_state_verification"
        ] = verify_qubo_ground_state(
            formulation,
            max_qubits=max_qubits,
        )

    # ==================================================================
    # 8. QAOA
    # ==================================================================

    quantum_result: Optional[QAOAResult] = None

    if quantum:

        if formulation.num_variables > max_qubits:

            metadata[
                "quantum_skipped"
            ] = True

            metadata[
                "quantum_skip_reason"
            ] = (
                f"{formulation.num_variables} qubits exceeds "
                f"max_qubits={max_qubits}"
            )

        else:

            quantum_result = run_quantum(
                formulation,
                config=config,
                reps=reps,
                optimizer=optimizer,
                maxiter=maxiter,
                shots=shots,
                expectation_mode=expectation_mode,
                cvar_alpha=cvar_alpha,
                seed=effective_seed,
                max_qubits=max_qubits,
                **qaoa_kwargs,
            )

    # ==================================================================
    # 9. DISTRIBUTION ANALYSIS
    # ==================================================================

    if quantum_result is not None:

        _analyse_quantum_distribution(
            quantum_result,
            formulation,
            reference_objective,
            metadata,
        )

    # ==================================================================
    # 10. CIRCUIT STATISTICS
    # ==================================================================
    #
    # QAOAResult does not necessarily expose a circuit object.
    # Therefore, circuit statistics are collected only if the result
    # actually contains one.
    # ==================================================================

    if quantum_result is not None:

        circuit = getattr(
            quantum_result,
            "circuit",
            None,
        )

        if circuit is not None:

            try:

                stats = circuit_stats(
                    circuit
                )

                metadata[
                    "circuit_stats"
                ] = _jsonable(
                    stats
                )

            except Exception as exc:

                metadata[
                    "circuit_stats_error"
                ] = str(exc)

    # ==================================================================
    # 11. COMPARISON
    # ==================================================================
    #
    # NOTE:
    # The existing compare_results() implementation determines its
    # reference from the classical-results sequence. We therefore keep
    # the explicit reference separately in ExperimentResult and metadata.
    #
    # The QUBO baseline list is built separately below.
    # ==================================================================

    comparison_inputs: List[ClassicalResult] = []

    if classical_result is not None:

        comparison_inputs.append(
            classical_result
        )

    if annealing_result is not None:

        comparison_inputs.append(
            annealing_result
        )

    if (
        comparison_inputs
        or quantum_result is not None
    ):

        comparison = compare_results(
            formulation,
            comparison_inputs,
            quantum_result,
            reference=reference_result,
        )

    else:

        comparison = []

    # ==================================================================
    # 12. STORE COMPARISON METADATA
    # ==================================================================

    metadata[
        "comparison_reference_separate"
    ] = True

    if reference_result is not None:

        metadata[
            "comparison_reference_solver"
        ] = getattr(
            reference_result,
            "solver",
            None,
        )

    # ==================================================================
    # 13. FINAL RESULT
    # ==================================================================

    return ExperimentResult(
        rung=rung,

        reference=reference_result,

        classical=classical_result,

        annealing=annealing_result,

        quantum=quantum_result,

        reference_objective=reference_objective,

        comparison=comparison,

        metadata=metadata,
    )


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def make_comparison_table(
    result: ExperimentResult,
) -> str:
    """Render the comparison table."""

    return comparison_table(
        result.comparison
    )


# ============================================================================
# CIRCUIT REPORT
# ============================================================================

def circuit_report(
    quantum_result: Optional[QAOAResult],
    *,
    backend: Any = None,
) -> Dict[str, Any]:
    """
    Return circuit statistics and optional transpilation information.

    If QAOAResult does not contain a circuit object, the function
    returns an empty circuit section rather than failing.
    """

    report: Dict[str, Any] = {}

    if quantum_result is None:
        return report

    circuit = getattr(
        quantum_result,
        "circuit",
        None,
    )

    if circuit is not None:

        try:

            report[
                "circuit_stats"
            ] = _jsonable(
                circuit_stats(
                    circuit
                )
            )

        except Exception as exc:

            report[
                "circuit_stats_error"
            ] = str(exc)

    # ==================================================================
    # TRANSPILATION
    # ==================================================================

    if (
        backend is not None
        and circuit is not None
    ):

        try:

            from .evaluation import (
                compare_transpilation,
            )

            report[
                "transpilation"
            ] = _jsonable(
                compare_transpilation(
                    circuit,
                    backend,
                )
            )

        except Exception as exc:

            report[
                "transpilation_error"
            ] = str(exc)

    return report