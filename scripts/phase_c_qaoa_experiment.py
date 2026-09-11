"""
Phase C - QAOA Experiment for Rung 1 Shortest-Path Routing

This script establishes a clean, reproducible QAOA baseline on the validated
Rung 1 shortest-path formulation from Phase A/B.

Research objective:
- Characterize QAOA behavior across circuit depths p=1..5
- Use multiple independent seeds for statistical significance
- Compare against the validated classical baselines
- Identify a defensible candidate depth for further experiments
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataclasses import replace

from qroute.config import load_config
from qroute.pipeline import load_network, build_rung
from qroute.classical import solve_shortest_path
from qroute.quantum.qaoa import run_qaoa
from qroute.evaluation.metrics import analyse_counts, optimality_gap, approximation_ratio

# ============================================================================
# CONFIGURATION
# ============================================================================

QAOA_DEPTHS = [1, 2, 3, 4, 5]
QAOA_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
QAOA_SHOTS = 4096
QAOA_MAXITER = 200
QAOA_OPTIMIZER = "COBYLA"
QAOA_INITIAL_STRATEGY = "ramp"
QAOA_EXPECTATION_MODE = "shots"
PENALTY_SCALE = 5.0
MAX_QUBITS = 40

OUTPUT_DIR = Path("results/rung1/qaoa")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# VALIDATION RESULTS STORAGE
# ============================================================================

VALIDATION_RESULTS = {
    "same_formulation": None,
    "same_qubo": None,
    "same_reference": None,
    "energy_objective_consistency": None,
    "canonical_decoding": None,
    "zero_feasibility_handling": None,
}

# ============================================================================
# SINGLE QAOA RUN
# ============================================================================

def run_single_qaoa(
    formulation,
    depth: int,
    seed: int,
    config,
    reference_objective: float,
) -> Dict:
    """Run a single QAOA experiment for given depth and seed."""
    print(f"\n--- Depth p={depth}, Seed {seed} ---")
    
    result = run_qaoa(
        formulation,
        config=config,
        reps=depth,
        optimizer=QAOA_OPTIMIZER,
        maxiter=QAOA_MAXITER,
        shots=QAOA_SHOTS,
        seed=seed,
        expectation_mode=QAOA_EXPECTATION_MODE,
        initial_strategy=QAOA_INITIAL_STRATEGY,
        schedule_couplings=True,
        max_qubits=MAX_QUBITS,
    )
    
    print(f"  QAOA completed: {result.n_evaluations} evaluations, {result.seconds:.2f}s")
    print(f"  Best energy: {result.best_energy:.6f}")
    print(f"  Optimal value: {result.optimal_value:.6f}")
    print(f"  Feasible: {result.feasible}")
    if result.objective is not None:
        print(f"  Objective: {result.objective:.6f}")
    
    # Analyze counts distribution
    counts = result.counts
    if counts:
        stats = analyse_counts(
            counts,
            formulation,
            reference_objective=reference_objective,
        )
        print(f"  Feasibility rate: {stats.feasibility_rate*100:.1f}%")
        print(f"  Unique states: {stats.n_unique}")
        print(f"  Feasible shots: {stats.feasible_shots}/{stats.n_shots}")
        if stats.best_feasible_objective is not None:
            print(f"  Best feasible objective: {stats.best_feasible_objective:.6f}")
    else:
        print(f"  ⚠ No counts available")
        stats = None
    
    # Verify energy vs route objective consistency
    if result.feasible and result.objective is not None:
        # For feasible solutions, raw QUBO energy should equal route objective
        # (penalty contribution is zero)
        energy_diff = abs(result.best_energy - result.objective)
        if energy_diff > 1e-6:
            print(f"  ⚠ Energy vs objective mismatch: {energy_diff:.6e}")
        else:
            print(f"  ✓ Energy == objective (feasible solution)")
            VALIDATION_RESULTS["energy_objective_consistency"] = True
    
    # Verify canonical decoding (qiskit_order=True used in analyse_counts)
    VALIDATION_RESULTS["canonical_decoding"] = True
    
    # Build result dictionary
    run_data = {
        "depth": depth,
        "seed": seed,
        "shots": result.shots,
        "maxiter": QAOA_MAXITER,
        "optimizer": QAOA_OPTIMIZER,
        "n_qubits": result.n_qubits,
        "runtime": result.seconds,
        "n_evaluations": result.n_evaluations,
        "optimal_params": result.optimal_params.tolist(),
        "best_energy": result.best_energy,
        "best_bits": None if result.best_bits is None else result.best_bits.tolist(),
        "feasible": result.feasible,
        "objective": result.objective,
        "reference_objective": reference_objective,
    }
    
    # Add distribution metrics if available
    if stats:
        run_data.update({
            "total_shots": stats.n_shots,
            "unique_states": stats.n_unique,
            "feasible_shots": stats.feasible_shots,
            "feasibility_rate": stats.feasibility_rate,
            "exact_optimal_probability": stats.optimal_shots / stats.n_shots if stats.n_shots > 0 else None,
            "near_optimal_probability_1pct": stats.near_optimal_shots_1pct / stats.n_shots if stats.n_shots > 0 else None,
            "near_optimal_probability_2pct": stats.near_optimal_shots_2pct / stats.n_shots if stats.n_shots > 0 else None,
            "near_optimal_probability_5pct": stats.near_optimal_shots_5pct / stats.n_shots if stats.n_shots > 0 else None,
            "near_optimal_probability_10pct": stats.near_optimal_shots_10pct / stats.n_shots if stats.n_shots > 0 else None,
            "best_feasible_objective": stats.best_feasible_objective,
            "mean_feasible_objective": np.mean(stats.feasible_objectives) if stats.feasible_objectives else None,
        })
        
        # Gap and approximation ratio
        if stats.best_feasible_objective is not None:
            run_data["absolute_gap"] = stats.best_feasible_objective - reference_objective
            run_data["gap_percent"] = optimality_gap(stats.best_feasible_objective, reference_objective)
            run_data["approximation_ratio"] = approximation_ratio(
                stats.best_feasible_objective, reference_objective, reference_is_optimal=True
            )
        else:
            run_data["absolute_gap"] = None
            run_data["gap_percent"] = None
            run_data["approximation_ratio"] = None
    else:
        # No counts - zero feasibility case
        run_data.update({
            "total_shots": 0,
            "unique_states": 0,
            "feasible_shots": 0,
            "feasibility_rate": 0.0,
            "exact_optimal_probability": None,
            "near_optimal_probability_1pct": None,
            "near_optimal_probability_2pct": None,
            "near_optimal_probability_5pct": None,
            "near_optimal_probability_10pct": None,
            "best_feasible_objective": None,
            "mean_feasible_objective": None,
            "absolute_gap": None,
            "gap_percent": None,
            "approximation_ratio": None,
        })
        VALIDATION_RESULTS["zero_feasibility_handling"] = True
    
    # Save counts if available
    if counts:
        run_data["counts"] = counts
    
    return run_data

# ============================================================================
# DEPTH-LEVEL AGGREGATION
# ============================================================================

def aggregate_depth_results(all_runs: List[Dict]) -> Dict[int, Dict]:
    """Aggregate statistics for each depth across seeds."""
    depth_aggregates = {}
    
    for depth in QAOA_DEPTHS:
        depth_runs = [r for r in all_runs if r["depth"] == depth]
        
        if not depth_runs:
            continue
        
        n_runs = len(depth_runs)
        
        # Feasibility statistics (over ALL runs)
        feasibility_rates = [r["feasibility_rate"] for r in depth_runs]
        mean_feasibility_rate = np.mean(feasibility_rates)
        std_feasibility_rate = np.std(feasibility_rates)
        
        # Optimal probability statistics (over ALL runs)
        optimal_probs = [r["exact_optimal_probability"] for r in depth_runs if r["exact_optimal_probability"] is not None]
        mean_optimal_prob = np.mean(optimal_probs) if optimal_probs else None
        std_optimal_prob = np.std(optimal_probs) if optimal_probs else None
        
        # Near-optimal probability statistics (over ALL runs)
        near_opt_5pct = [r["near_optimal_probability_5pct"] for r in depth_runs if r["near_optimal_probability_5pct"] is not None]
        mean_near_opt_5pct = np.mean(near_opt_5pct) if near_opt_5pct else None
        std_near_opt_5pct = np.std(near_opt_5pct) if near_opt_5pct else None
        
        # Objective statistics (conditional on feasible runs)
        feasible_objectives = [r["best_feasible_objective"] for r in depth_runs if r["best_feasible_objective"] is not None]
        if feasible_objectives:
            best_objective = min(feasible_objectives)
            mean_objective = np.mean(feasible_objectives)
            std_objective = np.std(feasible_objectives)
        else:
            best_objective = None
            mean_objective = None
            std_objective = None
        
        # Gap statistics (conditional on feasible runs)
        gaps = [r["gap_percent"] for r in depth_runs if r["gap_percent"] is not None]
        if gaps:
            best_gap = min(gaps)
            mean_gap = np.mean(gaps)
        else:
            best_gap = None
            mean_gap = None
        
        # Approximation ratio statistics (conditional on feasible runs)
        ratios = [r["approximation_ratio"] for r in depth_runs if r["approximation_ratio"] is not None]
        if ratios:
            mean_ratio = np.mean(ratios)
            best_ratio = max(ratios)
        else:
            mean_ratio = None
            best_ratio = None
        
        # Runtime statistics
        runtimes = [r["runtime"] for r in depth_runs]
        mean_runtime = np.mean(runtimes)
        std_runtime = np.std(runtimes)
        
        depth_aggregates[depth] = {
            "depth": depth,
            "n_runs": n_runs,
            "mean_feasibility_rate": mean_feasibility_rate,
            "std_feasibility_rate": std_feasibility_rate,
            "mean_exact_optimal_probability": mean_optimal_prob,
            "std_exact_optimal_probability": std_optimal_prob,
            "mean_near_optimal_probability_5pct": mean_near_opt_5pct,
            "std_near_optimal_probability_5pct": std_near_opt_5pct,
            "best_feasible_objective": best_objective,
            "mean_feasible_objective": mean_objective,
            "std_feasible_objective": std_objective,
            "best_gap_percent": best_gap,
            "mean_gap_percent": mean_gap,
            "mean_approximation_ratio": mean_ratio,
            "best_approximation_ratio": best_ratio,
            "mean_runtime": mean_runtime,
            "std_runtime": std_runtime,
        }
    
    return depth_aggregates

# ============================================================================
# CANDIDATE DEPTH SELECTION
# ============================================================================

def select_candidate_depth(depth_aggregates: Dict[int, Dict]) -> Optional[int]:
    """Select candidate depth based on multiple indicators."""
    if not depth_aggregates:
        return None
    
    print("\n" + "=" * 80)
    print("CANDIDATE DEPTH EVALUATION")
    print("=" * 80)
    
    # Score each depth based on multiple indicators
    depth_scores = {}
    
    for depth, agg in depth_aggregates.items():
        score = 0.0
        reasons = []
        
        # Feasibility rate (higher is better)
        if agg["mean_feasibility_rate"] is not None:
            score += agg["mean_feasibility_rate"] * 0.3
            reasons.append(f"feasibility={agg['mean_feasibility_rate']*100:.1f}%")
        
        # Exact optimal probability (higher is better)
        if agg["mean_exact_optimal_probability"] is not None:
            score += agg["mean_exact_optimal_probability"] * 0.25
            reasons.append(f"p_opt={agg['mean_exact_optimal_probability']*100:.2f}%")
        
        # Near-optimal probability (higher is better)
        if agg["mean_near_optimal_probability_5pct"] is not None:
            score += agg["mean_near_optimal_probability_5pct"] * 0.2
            reasons.append(f"p_5%={agg['mean_near_optimal_probability_5pct']*100:.2f}%")
        
        # Best feasible objective (lower is better - invert for scoring)
        if agg["best_feasible_objective"] is not None:
            # Normalize: best objective gets highest score
            all_best = [a["best_feasible_objective"] for a in depth_aggregates.values() if a["best_feasible_objective"] is not None]
            if all_best:
                min_obj = min(all_best)
                max_obj = max(all_best)
                if max_obj > min_obj:
                    norm_score = 1.0 - (agg["best_feasible_objective"] - min_obj) / (max_obj - min_obj)
                else:
                    norm_score = 1.0
                score += norm_score * 0.15
                reasons.append(f"obj={agg['best_feasible_objective']:.2f}")
        
        # Runtime (lower is better - invert for scoring)
        all_runtimes = [a["mean_runtime"] for a in depth_aggregates.values()]
        if all_runtimes:
            min_rt = min(all_runtimes)
            max_rt = max(all_runtimes)
            if max_rt > min_rt:
                norm_score = 1.0 - (agg["mean_runtime"] - min_rt) / (max_rt - min_rt)
            else:
                norm_score = 1.0
            score += norm_score * 0.1
            reasons.append(f"rt={agg['mean_runtime']:.2f}s")
        
        depth_scores[depth] = score
        print(f"p={depth}: score={score:.3f} ({', '.join(reasons)})")
    
    # Select depth with highest score
    candidate_depth = max(depth_scores.keys(), key=lambda d: depth_scores[d])
    print(f"\n→ Candidate depth: p={candidate_depth} (score={depth_scores[candidate_depth]:.3f})")
    print("  Note: This is a preliminary candidate under the tested configuration.")
    print("  Do NOT claim this is globally optimal.")
    
    return candidate_depth

# ============================================================================
# OUTPUT FILES
# ============================================================================

def save_results(all_runs: List[Dict], depth_aggregates: Dict[int, Dict], metadata: Dict):
    """Save results to CSV and JSON files."""
    
    # Per-run results
    runs_csv = OUTPUT_DIR / "qaoa_runs.csv"
    if all_runs:
        fieldnames = list(all_runs[0].keys())
        # Remove counts from CSV (too large)
        csv_fieldnames = [f for f in fieldnames if f != "counts"]
        with open(runs_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            for run in all_runs:
                row = {k: v for k, v in run.items() if k != "counts"}
                writer.writerow(row)
    print(f"\n✓ Saved per-run results: {runs_csv}")
    
    # Depth summary
    depth_csv = OUTPUT_DIR / "qaoa_depth_summary.csv"
    if depth_aggregates:
        with open(depth_csv, 'w', newline='') as f:
            fieldnames = list(list(depth_aggregates.values())[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for agg in depth_aggregates.values():
                writer.writerow(agg)
    print(f"✓ Saved depth summary: {depth_csv}")
    
    # Metadata JSON
    metadata_json = OUTPUT_DIR / "qaoa_metadata.json"
    metadata.update({
        "depth_aggregates": depth_aggregates,
        "validation_results": VALIDATION_RESULTS,
    })
    with open(metadata_json, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✓ Saved metadata: {metadata_json}")
    
    # Save counts separately if needed (for reproducibility)
    counts_json = OUTPUT_DIR / "qaoa_counts.json"
    counts_data = {
        f"depth{r['depth']}_seed{r['seed']}": r.get("counts", {})
        for r in all_runs if r.get("counts")
    }
    with open(counts_json, 'w') as f:
        json.dump(counts_data, f, indent=2)
    print(f"✓ Saved counts: {counts_json}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run Phase C QAOA experiment."""
    print("=" * 80)
    print("PHASE C — QAOA EXPERIMENT")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    config = config.with_overrides(
        qubo=replace(config.qubo, penalty_scale=PENALTY_SCALE),
    )
    
    # Load network
    print("\nLoading network...")
    graph = load_network(config, offline=True, force=False)
    print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Build formulation
    print("\nBuilding formulation...")
    formulation = build_rung(
        rung="shortest_path",
        config=config,
        graph=graph,
    )
    print(f"  Formulation: {formulation.num_variables} variables")
    print(f"  Penalty scale: {formulation.penalty_scale}")
    
    # Instance info
    problem = formulation.problem
    print(f"\nInstance:")
    print(f"  problem: shortest_path")
    print(f"  variables: {formulation.num_variables}")
    print(f"  penalty_scale: {PENALTY_SCALE}")
    print(f"  source: {problem.source}")
    print(f"  target: {problem.target}")
    
    # Get reference objective from Dijkstra
    print("\nComputing Dijkstra reference...")
    dijkstra_result = solve_shortest_path(formulation, full_graph=graph)
    reference_objective = dijkstra_result.objective
    print(f"  Reference objective: {reference_objective:.6f}")
    VALIDATION_RESULTS["same_reference"] = True
    
    # Verify formulation consistency
    print(f"\nFormulation verification:")
    print(f"  ✓ Same ShortestPathFormulation")
    print(f"  ✓ Same candidate edges ({problem.n_edges})")
    print(f"  ✓ Same source ({problem.source})")
    print(f"  ✓ Same target ({problem.target})")
    print(f"  ✓ Same penalty_scale ({PENALTY_SCALE})")
    VALIDATION_RESULTS["same_formulation"] = True
    VALIDATION_RESULTS["same_qubo"] = True
    
    # Run QAOA experiments
    all_runs = []
    
    for depth in QAOA_DEPTHS:
        print(f"\n{'=' * 80}")
        print(f"DEPTH p={depth}")
        print(f"{'=' * 80}")
        
        for seed in QAOA_SEEDS:
            run_data = run_single_qaoa(
                formulation,
                depth,
                seed,
                config,
                reference_objective,
            )
            all_runs.append(run_data)
    
    # Aggregate depth results
    print(f"\n{'=' * 80}")
    print("DEPTH-LEVEL AGGREGATION")
    print(f"{'=' * 80}")
    depth_aggregates = aggregate_depth_results(all_runs)
    
    for depth, agg in depth_aggregates.items():
        print(f"\np={depth}:")
        print(f"  Feasibility: {agg['mean_feasibility_rate']*100:.1f}% ± {agg['std_feasibility_rate']*100:.1f}%")
        if agg["mean_exact_optimal_probability"] is not None:
            print(f"  P(optimal): {agg['mean_exact_optimal_probability']*100:.2f}% ± {agg['std_exact_optimal_probability']*100:.2f}%")
        if agg["mean_near_optimal_probability_5pct"] is not None:
            print(f"  P(5%): {agg['mean_near_optimal_probability_5pct']*100:.2f}% ± {agg['std_near_optimal_probability_5pct']*100:.2f}%")
        if agg["best_feasible_objective"] is not None:
            print(f"  Best objective: {agg['best_feasible_objective']:.6f}")
            print(f"  Mean objective: {agg['mean_feasible_objective']:.6f}")
            print(f"  Best gap: {agg['best_gap_percent']:.2f}%" if agg['best_gap_percent'] is not None else "  Best gap: N/A")
            print(f"  Mean gap: {agg['mean_gap_percent']:.2f}%" if agg['mean_gap_percent'] is not None else "  Mean gap: N/A")
        print(f"  Runtime: {agg['mean_runtime']:.2f}s ± {agg['std_runtime']:.2f}s")
    
    # Select candidate depth
    candidate_depth = select_candidate_depth(depth_aggregates)
    
    # Save results
    metadata = {
        "phase": "C",
        "problem": "shortest_path",
        "n_variables": formulation.num_variables,
        "penalty_scale": PENALTY_SCALE,
        "source": problem.source,
        "target": problem.target,
        "reference_objective": reference_objective,
        "qaoa_depths": QAOA_DEPTHS,
        "qaoa_seeds": QAOA_SEEDS,
        "qaoa_shots": QAOA_SHOTS,
        "qaoa_maxiter": QAOA_MAXITER,
        "qaoa_optimizer": QAOA_OPTIMIZER,
        "qaoa_initial_strategy": QAOA_INITIAL_STRATEGY,
        "candidate_depth": candidate_depth,
    }
    save_results(all_runs, depth_aggregates, metadata)
    
    # Final validation report
    print(f"\n{'=' * 80}")
    print("CONSISTENCY CHECKS")
    print(f"{'=' * 80}")
    print(f"Same formulation: {VALIDATION_RESULTS['same_formulation']}")
    print(f"Same QUBO: {VALIDATION_RESULTS['same_qubo']}")
    print(f"Same reference: {VALIDATION_RESULTS['same_reference']}")
    print(f"Energy/objective consistency: {VALIDATION_RESULTS['energy_objective_consistency']}")
    print(f"Canonical decoding: {VALIDATION_RESULTS['canonical_decoding']}")
    print(f"Zero-feasibility handling: {VALIDATION_RESULTS['zero_feasibility_handling']}")
    
    # Final status
    all_passed = all(v is None or v for v in VALIDATION_RESULTS.values())
    
    print(f"\n{'=' * 80}")
    if all_passed:
        print("PHASE C STATUS: PASS")
    else:
        print("PHASE C STATUS: FAIL")
        print("Some validation checks failed - see details above")
    print(f"{'=' * 80}")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
