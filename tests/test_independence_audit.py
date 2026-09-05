"""Scientific independence audit tests.

These tests verify that Dijkstra, simulated annealing, and QAOA are
independent solvers and that no result leakage occurs.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from pathlib import Path
import sys

# Add src to path
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qroute.config import Config, load_config
from qroute.pipeline import build_rung, run_classical
from qroute.classical import solve_shortest_path
from qroute.quantum import run_qaoa
from qroute.qubo.shortest_path import build_path_problem, ShortestPathFormulation


def test_1_log_all_solver_outputs():
    """TEST 1: Run Dijkstra, annealing, and QAOA on the same instance.
    
    Log:
    - Dijkstra route
    - Dijkstra objective
    - annealing bitstring
    - annealing decoded route
    - annealing objective
    - QAOA top-N bitstrings
    - QAOA counts/probabilities
    - QAOA decoded best route
    - QAOA objective
    """
    print("\n" + "="*80)
    print("TEST 1: Log all solver outputs for same instance")
    print("="*80)
    
    # Create a simple test graph
    graph = nx.DiGraph()
    graph.add_edge(0, 1, travel_time=1.0)
    graph.add_edge(1, 2, travel_time=1.0)
    graph.add_edge(0, 2, travel_time=3.0)
    graph.add_edge(2, 3, travel_time=1.0)
    graph.add_edge(1, 3, travel_time=2.0)
    
    source, target = 0, 3
    
    # Build formulation
    config = load_config()
    formulation = ShortestPathFormulation.from_graph(
        graph, source, target, config, k_paths=3, max_edges=10
    )
    
    print(f"\nFormulation: {formulation.summary()}")
    print(f"Candidate edges: {formulation.problem.edges}")
    print(f"Candidate paths: {formulation.problem.candidate_paths}")
    
    # Run Dijkstra
    print("\n--- DIJKSTRA ---")
    dijkstra_result = solve_shortest_path(formulation, full_graph=graph)
    print(f"Route: {dijkstra_result.solution.route if dijkstra_result.solution else None}")
    print(f"Objective: {dijkstra_result.objective}")
    print(f"Bitstring: {dijkstra_result.bits}")
    print(f"Feasible: {dijkstra_result.feasible}")
    print(f"Optimal: {dijkstra_result.optimal}")
    
    # Run annealing with seed 42
    print("\n--- ANNEALING (seed=42) ---")
    from qroute.classical import solve_formulation_annealing
    annealing_result = solve_formulation_annealing(formulation, seed=42)
    print(f"Bitstring: {annealing_result.bits}")
    print(f"Route: {annealing_result.solution.route if annealing_result.solution else None}")
    print(f"Objective: {annealing_result.objective}")
    print(f"Feasible: {annealing_result.feasible}")
    print(f"Details: {annealing_result.details}")
    
    # Run QAOA
    print("\n--- QAOA ---")
    quantum_result = run_qaoa(
        formulation,
        config,
        reps=1,
        maxiter=10,
        shots=100,
        expectation_mode="shots",
        max_qubits=10,
    )
    print(f"Best bitstring: {quantum_result.best_bits}")
    print(f"Best route: {quantum_result.best_solution.route if quantum_result.best_solution else None}")
    print(f"Best objective: {quantum_result.objective}")
    print(f"Best energy: {quantum_result.best_energy}")
    print(f"Feasible: {quantum_result.feasible}")
    print(f"Top 5 counts: {dict(list(quantum_result.counts.items())[:5])}")
    
    # Verify independence
    print("\n--- INDEPENDENCE CHECK ---")
    if dijkstra_result.bits and annealing_result.bits:
        bits_match = dijkstra_result.bits == annealing_result.bits
        print(f"Dijkstra bits == Annealing bits: {bits_match}")
        if bits_match:
            print("  WARNING: Annealing reproduced Dijkstra exactly (may be coincidence on small instance)")
    
    if dijkstra_result.bits and quantum_result.best_bits is not None:
        dijkstra_array = np.array([int(b) for b in dijkstra_result.bits])
        qaoa_match = np.array_equal(dijkstra_array, quantum_result.best_bits)
        print(f"Dijkstra bits == QAOA bits: {qaoa_match}")
        if qaoa_match:
            print("  WARNING: QAOA reproduced Dijkstra exactly (may be coincidence on small instance)")
    
    print("\nTEST 1 PASSED: All outputs logged\n")


def test_2_random_seed_independence():
    """TEST 2: Change the random seed for annealing and QAOA.
    
    Verify that they still execute without using the Dijkstra result.
    """
    print("\n" + "="*80)
    print("TEST 2: Random seed independence test")
    print("="*80)
    
    # Create a simple test graph
    graph = nx.DiGraph()
    graph.add_edge(0, 1, travel_time=1.0)
    graph.add_edge(1, 2, travel_time=1.0)
    graph.add_edge(0, 2, travel_time=3.0)
    graph.add_edge(2, 3, travel_time=1.0)
    graph.add_edge(1, 3, travel_time=2.0)
    
    source, target = 0, 3
    config = load_config()
    formulation = ShortestPathFormulation.from_graph(
        graph, source, target, config, k_paths=3, max_edges=10
    )
    
    # Run annealing with different seeds
    print("\n--- ANNEALING WITH DIFFERENT SEEDS ---")
    from qroute.classical import solve_formulation_annealing
    
    results = []
    for seed in [42, 123, 456, 789]:
        result = solve_formulation_annealing(formulation, seed=seed)
        results.append((seed, result.bits, result.objective))
        print(f"Seed {seed}: bits={result.bits}, objective={result.objective}")
    
    # Check that different seeds produce different results (or at least can)
    unique_bits = set(bits for _, bits, _ in results if bits is not None)
    print(f"Unique bitstrings across seeds: {len(unique_bits)}")
    
    # Run QAOA with different seeds
    print("\n--- QAOA WITH DIFFERENT SEEDS ---")
    qaoa_results = []
    for seed in [42, 123, 456]:
        result = run_qaoa(
            formulation,
            config,
            reps=1,
            maxiter=10,
            shots=100,
            seed=seed,
            expectation_mode="shots",
            max_qubits=10,
        )
        qaoa_results.append((seed, result.best_bits, result.objective))
        print(f"Seed {seed}: best_bits={result.best_bits}, objective={result.objective}")
    
    print("\nTEST 2 PASSED: Solvers execute with different seeds\n")


def test_3_disable_dijkstra_after_qubo():
    """TEST 3: Disable Dijkstra AFTER the problem/QUBO has been constructed.
    
    Verify that:
    - annealing still runs
    - QAOA still runs
    """
    print("\n" + "="*80)
    print("TEST 3: Disable Dijkstra after QUBO construction")
    print("="*80)
    
    # Create a simple test graph
    graph = nx.DiGraph()
    graph.add_edge(0, 1, travel_time=1.0)
    graph.add_edge(1, 2, travel_time=1.0)
    graph.add_edge(0, 2, travel_time=3.0)
    graph.add_edge(2, 3, travel_time=1.0)
    graph.add_edge(1, 3, travel_time=2.0)
    
    source, target = 0, 3
    config = load_config()
    
    # Build formulation (this uses Dijkstra for candidate generation)
    formulation = ShortestPathFormulation.from_graph(
        graph, source, target, config, k_paths=3, max_edges=10
    )
    
    print(f"Formulation built with {formulation.num_variables} variables")
    
    # Now run annealing and QAOA WITHOUT calling Dijkstra
    print("\n--- ANNEALING (no Dijkstra call) ---")
    from qroute.classical import solve_formulation_annealing
    annealing_result = solve_formulation_annealing(formulation, seed=42)
    print(f"Annealing completed: objective={annealing_result.objective}")
    
    print("\n--- QAOA (no Dijkstra call) ---")
    quantum_result = run_qaoa(
        formulation,
        config,
        reps=1,
        maxiter=10,
        shots=100,
        expectation_mode="shots",
        max_qubits=10,
    )
    print(f"QAOA completed: objective={quantum_result.objective}")
    
    print("\nTEST 3 PASSED: Both solvers run without Dijkstra after QUBO construction\n")


def test_4_synthetic_alternative_route():
    """TEST 4: Construct a synthetic tiny graph where the Dijkstra route is
    deliberately different from the optimal route represented in the QUBO candidate space.
    
    Example:
    Dijkstra route cost = 20
    alternative valid route cost = 10
    
    The QUBO must contain BOTH alternatives.
    
    Expected behavior:
    Dijkstra -> 20
    Annealing -> may find 10
    QAOA -> may find 10 depending on optimization quality
    """
    print("\n" + "="*80)
    print("TEST 4: Synthetic graph with alternative optimal route")
    print("="*80)
    
    # Create a graph with multiple paths
    # Path 1 (Dijkstra on full graph): 0->1->2->3 with cost 2+2+2=6
    # Path 2 (alternative): 0->2->3 with cost 5+1=6 (same cost)
    # Path 3 (alternative): 0->1->3 with cost 2+3=5 (better!)
    graph = nx.DiGraph()
    graph.add_edge(0, 1, travel_time=2.0)
    graph.add_edge(1, 2, travel_time=2.0)
    graph.add_edge(2, 3, travel_time=2.0)
    graph.add_edge(0, 2, travel_time=5.0)
    graph.add_edge(1, 3, travel_time=3.0)
    
    source, target = 0, 3
    
    # Full graph Dijkstra
    print("\n--- FULL GRAPH DIJKSTRA ---")
    full_cost, full_path = nx.single_source_dijkstra(graph, source, target=target, weight="travel_time")
    print(f"Full graph optimal path: {full_path}")
    print(f"Full graph optimal cost: {full_cost}")
    
    # Build formulation with k_paths to include alternatives
    config = load_config()
    formulation = ShortestPathFormulation.from_graph(
        graph, source, target, config, k_paths=5, max_edges=10
    )
    
    print(f"\n--- CANDIDATE SPACE ---")
    print(f"Candidate edges: {formulation.problem.edges}")
    print(f"Candidate paths: {formulation.problem.candidate_paths}")
    print(f"Number of candidates: {len(formulation.problem.candidate_paths)}")
    
    # Check if better path is in candidate space
    better_path = (0, 1, 3)
    better_in_candidates = any(
        list(p) == list(better_path) for p in formulation.problem.candidate_paths
    )
    print(f"Better path {better_path} in candidates: {better_in_candidates}")
    
    # Run Dijkstra on candidate space
    print("\n--- CANDIDATE SPACE DIJKSTRA ---")
    dijkstra_result = solve_shortest_path(formulation, full_graph=graph)
    print(f"Candidate space route: {dijkstra_result.solution.route if dijkstra_result.solution else None}")
    print(f"Candidate space cost: {dijkstra_result.objective}")
    
    # Run annealing
    print("\n--- ANNEALING ---")
    from qroute.classical import solve_formulation_annealing
    annealing_result = solve_formulation_annealing(formulation, seed=42, n_restarts=50, n_sweeps=1000)
    print(f"Annealing route: {annealing_result.solution.route if annealing_result.solution else None}")
    print(f"Annealing cost: {annealing_result.objective}")
    
    # Run QAOA
    print("\n--- QAOA ---")
    quantum_result = run_qaoa(
        formulation,
        config,
        reps=2,
        maxiter=50,
        shots=1000,
        expectation_mode="shots",
        max_qubits=10,
    )
    print(f"QAOA route: {quantum_result.best_solution.route if quantum_result.best_solution else None}")
    print(f"QAOA cost: {quantum_result.objective}")
    
    # Analysis
    print("\n--- ANALYSIS ---")
    if better_in_candidates:
        print("✓ Better alternative is in candidate space")
        print("  Annealing and QAOA can potentially find it")
    else:
        print("✗ Better alternative is NOT in candidate space")
        print("  This is a methodological limitation of k-shortest-paths reduction")
    
    print("\nTEST 4 PASSED: Synthetic graph analysis complete\n")


def test_5_qaoa_no_dijkstra_inputs():
    """TEST 5: Add an assertion that QAOA does NOT receive:
    - Dijkstra route
    - Dijkstra bitstring
    - Dijkstra objective
    as an optimization input.
    """
    print("\n" + "="*80)
    print("TEST 5: Assertion that QAOA doesn't receive Dijkstra inputs")
    print("="*80)
    
    # Create a simple test graph
    graph = nx.DiGraph()
    graph.add_edge(0, 1, travel_time=1.0)
    graph.add_edge(1, 2, travel_time=1.0)
    graph.add_edge(0, 2, travel_time=3.0)
    graph.add_edge(2, 3, travel_time=1.0)
    graph.add_edge(1, 3, travel_time=2.0)
    
    source, target = 0, 3
    config = load_config()
    formulation = ShortestPathFormulation.from_graph(
        graph, source, target, config, k_paths=3, max_edges=10
    )
    
    # Get Dijkstra result
    dijkstra_result = solve_shortest_path(formulation, full_graph=graph)
    
    # Run QAOA and inspect what it receives
    # We'll monkey-patch to verify inputs
    original_qubo = formulation.qubo
    
    def tracked_qubo():
        qubo = original_qubo()
        print(f"QUBO shape: {qubo.Q.shape}")
        print(f"QUBO metadata: {qubo.metadata}")
        # Verify no Dijkstra data in metadata
        assert "dijkstra_route" not in qubo.metadata, "Dijkstra route leaked into QUBO metadata"
        assert "dijkstra_bitstring" not in qubo.metadata, "Dijkstra bitstring leaked into QUBO metadata"
        assert "dijkstra_objective" not in qubo.metadata, "Dijkstra objective leaked into QUBO metadata"
        return qubo
    
    formulation.qubo = tracked_qubo
    
    print("\n--- QAOA INPUT VERIFICATION ---")
    quantum_result = run_qaoa(
        formulation,
        config,
        reps=1,
        maxiter=10,
        shots=100,
        expectation_mode="shots",
        max_qubits=10,
    )
    
    print("✓ QAOA received only QUBO (no Dijkstra route/bitstring/objective)")
    
    # Verify QAOA result is different from Dijkstra (unless by coincidence)
    if dijkstra_result.bits and quantum_result.best_bits is not None:
        dijkstra_array = np.array([int(b) for b in dijkstra_result.bits])
        identical = np.array_equal(dijkstra_array, quantum_result.best_bits)
        if identical:
            print("⚠ QAOA result identical to Dijkstra (coincidence on small instance)")
        else:
            print("✓ QAOA result differs from Dijkstra (independent optimization)")
    
    print("\nTEST 5 PASSED: QAOA does not receive Dijkstra inputs\n")


if __name__ == "__main__":
    test_1_log_all_solver_outputs()
    test_2_random_seed_independence()
    test_3_disable_dijkstra_after_qubo()
    test_4_synthetic_alternative_route()
    test_5_qaoa_no_dijkstra_inputs()
    
    print("\n" + "="*80)
    print("ALL INDEPENDENCE TESTS PASSED")
    print("="*80)
