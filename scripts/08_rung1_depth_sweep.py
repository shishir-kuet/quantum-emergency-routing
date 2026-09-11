"""
QUANTUM EMERGENCY ROUTING - RUNG 1 DEPTH SWEEP EXPERIMENT
File: scripts/08_rung1_depth_sweep.py

This script performs a depth sweep on the Rung 1 problem (Shortest Path QAOA).
Fixes: Corrects the metrics API call to use correct parameter order and keywords.

Key Fixes Applied:
- Changed: analyse_counts(counts, formulation, reference_objective=...)
- Removed: Silent exception handling
- Added: Explicit error reporting with traceback
- Added: Distinction between zero feasibility and metrics extraction errors
- Added: Progress tracking
"""

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qroute.config import load_config
from qroute.pipeline import load_network, build_rung
from qroute.quantum.qaoa import run_qaoa
from qroute.evaluation.metrics import analyse_counts


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Rung 1 Depth Sweep Experiment - Shortest Path QAOA"
    )
    
    parser.add_argument(
        "--penalty",
        type=float,
        default=5.0,
        help="QUBO penalty scale (default: 5.0)"
    )
    
    parser.add_argument(
        "--depths",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated list of QAOA depths to sweep (default: 1,2,3,4,5)"
    )
    
    parser.add_argument(
        "--seeds",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10",
        help="Comma-separated list of random seeds (default: 1..10)"
    )
    
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Number of measurement shots (default: 4096)"
    )
    
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Maximum optimizer iterations (default: 200)"
    )
    
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline mode (don't fetch fresh data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    return parser.parse_args()


def parse_int_list(s: str) -> List[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def load_problem(config, offline: bool = True):
    """
    Load the problem graph and build Rung 1 formulation.
    
    Returns:
        Tuple of (formulation, reference_objective)
    """
    print("Loading network...")
    graph = load_network(config, offline=offline, force=False)
    print(f"  ✓ Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    print("Building Rung 1 formulation (Shortest Path)...")
    formulation = build_rung(
        rung="shortest_path",
        config=config,
        graph=graph,
    )
    print(f"  ✓ Formulation built: {formulation.num_variables} variables")
    
    # Reference objective from Dijkstra
    reference_objective = getattr(formulation, "reference_objective", None)
    if reference_objective is None:
        # Try to compute Dijkstra if not available
        print("  ⚠ Warning: reference_objective not found in formulation")
        reference_objective = 160.0365  # Known value from penalty sweep
        print(f"  Using default reference: {reference_objective}")
    else:
        print(f"  ✓ Reference objective (Dijkstra): {reference_objective:.4f}")
    
    return formulation, reference_objective


def run_single_experiment(
    formulation,
    depth: int,
    seed: int,
    config,
    reference_objective: float,
) -> Dict:
    """
    Run a single QAOA experiment for given depth and seed.
    
    Returns:
        Dictionary with experiment results
    """
    try:
        # Run QAOA
        result = run_qaoa(
            formulation,
            config=config,
            reps=depth,
            optimizer="COBYLA",
            maxiter=config.qaoa.maxiter,
            shots=config.qaoa.shots,
            seed=seed,
            expectation_mode="shots",
            initial_strategy="ramp",
            schedule_couplings=True,
            max_qubits=40,
        )
        
        # Extract measurements
        counts = result.counts  # Dict[str, int]
        
        # CRITICAL FIX: Correct API call to analyse_counts
        # Parameter order: counts FIRST, formulation SECOND
        # Keyword: reference_objective (NOT reference)
        try:
            stats = analyse_counts(
                counts,                          # ← COUNTS FIRST
                formulation,                     # ← FORMULATION SECOND
                reference_objective=reference_objective,  # ← CORRECT KEYWORD
            )
        except Exception as e:
            print(f"\n✗ ERROR in metrics extraction for depth={depth}, seed={seed}:")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {str(e)}")
            print(f"  Traceback:")
            traceback.print_exc()
            raise
        
        # Build result dictionary
        experiment_result = {
            "depth": depth,
            "seed": seed,
            "penalty": config.qubo.penalty_scale,
            "shots": result.shots,
            "reference_objective": reference_objective,
            
            # QAOA performance
            "qaoa_best_energy": result.best_energy,
            "qaoa_optimal_value": result.optimal_value,
            "qaoa_n_evaluations": result.n_evaluations,
            "qaoa_runtime_seconds": result.seconds,
            
            # Sample statistics
            "n_shots": stats.n_shots,
            "n_unique": stats.n_unique,
            "feasible_shots": stats.feasible_shots,
            "optimal_shots": stats.optimal_shots,
            
            # Quality metrics
            "feasibility_rate": stats.feasibility_rate,
            "best_feasible_objective": stats.best_feasible_objective,
            "mean_feasible_objective": (
                np.mean(stats.feasible_objectives)
                if stats.feasible_objectives else None
            ),
            
            # Gap and approximation ratio
            "optimality_gap_percent": (
                (stats.best_feasible_objective - reference_objective) / reference_objective * 100
                if stats.best_feasible_objective is not None else None
            ),
            "approximation_ratio": (
                reference_objective / stats.best_feasible_objective
                if stats.best_feasible_objective is not None else None
            ),
            
            # Probability metrics
            "p_optimal": (
                stats.optimal_shots / stats.n_shots
                if stats.n_shots > 0 else None
            ),
            "p_1percent": (
                stats.near_optimal_shots_1pct / stats.n_shots
                if stats.n_shots > 0 else None
            ),
            "p_2percent": (
                stats.near_optimal_shots_2pct / stats.n_shots
                if stats.n_shots > 0 else None
            ),
            "p_5percent": (
                stats.near_optimal_shots_5pct / stats.n_shots
                if stats.n_shots > 0 else None
            ),
            "p_10percent": (
                stats.near_optimal_shots_10pct / stats.n_shots
                if stats.n_shots > 0 else None
            ),
            
            # Circuit statistics (from QAOAResult)
            "n_qubits": result.n_qubits,
            "circuit_depth": None,  # Not available in QAOAResult
            "gate_count": None,  # Not available in QAOAResult
            "cx_count": None,  # Not available in QAOAResult
            
            # Status
            "status": "success",
            "error": None,
        }
        
        return experiment_result
        
    except Exception as e:
        # Return error result
        return {
            "depth": depth,
            "seed": seed,
            "penalty": config.qubo.penalty_scale,
            "status": "error",
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback_info": traceback.format_exc(),
        }


def aggregate_results(
    raw_results: List[Dict],
) -> Dict:
    """
    Aggregate results across seeds for each depth.
    
    Returns:
        Dictionary with aggregated statistics by depth
    """
    # Group by depth
    by_depth = {}
    for result in raw_results:
        if result["status"] != "success":
            continue
        
        depth = result["depth"]
        if depth not in by_depth:
            by_depth[depth] = []
        by_depth[depth].append(result)
    
    # Aggregate
    summary = {}
    for depth in sorted(by_depth.keys()):
        results_at_depth = by_depth[depth]
        
        # Extract metrics
        feasibility_rates = [r["feasibility_rate"] for r in results_at_depth
                            if r["feasibility_rate"] is not None]
        best_objectives = [r["best_feasible_objective"] for r in results_at_depth
                          if r["best_feasible_objective"] is not None]
        gaps = [r["optimality_gap_percent"] for r in results_at_depth
               if r["optimality_gap_percent"] is not None]
        p_opts = [r["p_optimal"] for r in results_at_depth
                 if r["p_optimal"] is not None]
        p_5pcts = [r["p_5percent"] for r in results_at_depth
                  if r["p_5percent"] is not None]
        
        summary[depth] = {
            "n_seeds": len(results_at_depth),
            "feasibility_rate_mean": np.mean(feasibility_rates) if feasibility_rates else None,
            "feasibility_rate_std": np.std(feasibility_rates) if feasibility_rates else None,
            "best_objective_mean": np.mean(best_objectives) if best_objectives else None,
            "best_objective_std": np.std(best_objectives) if best_objectives else None,
            "gap_percent_mean": np.mean(gaps) if gaps else None,
            "gap_percent_std": np.std(gaps) if gaps else None,
            "p_optimal_mean": np.mean(p_opts) if p_opts else None,
            "p_optimal_std": np.std(p_opts) if p_opts else None,
            "p_5percent_mean": np.mean(p_5pcts) if p_5pcts else None,
            "p_5percent_std": np.std(p_5pcts) if p_5pcts else None,
        }
    
    return summary


def save_results(
    raw_results: List[Dict],
    summary: Dict,
    output_dir: str,
    penalty: float,
):
    """Save results to CSV, JSON, and generate figures."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw results to CSV
    print(f"\nSaving results to {output_dir}/")
    
    raw_df = pd.DataFrame([r for r in raw_results if r["status"] == "success"])
    raw_csv_path = output_path / "rung1_depth_sweep_raw.csv"
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"  ✓ Raw results: {raw_csv_path}")
    
    # Save summary CSV
    summary_rows = []
    for depth, agg in summary.items():
        row = {"depth": depth, "penalty": penalty}
        row.update(agg)
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_path / "rung1_depth_sweep_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  ✓ Summary: {summary_csv_path}")
    
    # Save full JSON
    json_data = {
        "experiment": "rung1_depth_sweep",
        "timestamp": datetime.now().isoformat(),
        "penalty": penalty,
        "raw_results": raw_results,
        "summary": summary,
    }
    
    json_path = output_path / "rung1_depth_sweep.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  ✓ Full data: {json_path}")
    
    return raw_csv_path, summary_csv_path, json_path


def generate_figures(
    summary: Dict,
    output_dir: str,
    penalty: float,
):
    """Generate publication-quality figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        
        # Set up plotting style
        mpl.use("Agg")
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            plt.style.use("seaborn-darkgrid")
        
        output_path = Path(output_dir)
        figures_dir = output_path / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        depths = sorted(summary.keys())
        
        if not depths:
            print("  ⚠ No successful runs; skipping figures")
            return
        
        # Figure 1: Best feasible objective vs depth
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [summary[d]["best_objective_mean"] for d in depths]
        stds = [summary[d]["best_objective_std"] for d in depths]
        # Filter out None values
        valid_data = [(d, m, s) for d, m, s in zip(depths, means, stds) if m is not None]
        if valid_data:
            valid_depths, valid_means, valid_stds = zip(*valid_data)
            ax.errorbar(valid_depths, valid_means, yerr=valid_stds, fmt="o-", capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel("QAOA Depth", fontsize=12)
        ax.set_ylabel("Best Feasible Objective", fontsize=12)
        ax.set_title(f"Depth vs Objective (Penalty={penalty})", fontsize=14)
        ax.grid(True)
        fig_path = figures_dir / "depth_vs_objective.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure: {fig_path}")
        
        # Figure 2: Feasibility rate vs depth
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [summary[d]["feasibility_rate_mean"] for d in depths]
        stds = [summary[d]["feasibility_rate_std"] for d in depths]
        valid_data = [(d, m, s) for d, m, s in zip(depths, means, stds) if m is not None]
        if valid_data:
            valid_depths, valid_means, valid_stds = zip(*valid_data)
            ax.errorbar(valid_depths, valid_means, yerr=valid_stds, fmt="o-", capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel("QAOA Depth", fontsize=12)
        ax.set_ylabel("Feasibility Rate", fontsize=12)
        ax.set_title(f"Depth vs Feasibility (Penalty={penalty})", fontsize=14)
        ax.set_ylim([0, 1])
        ax.grid(True)
        fig_path = figures_dir / "depth_vs_feasibility.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure: {fig_path}")
        
        # Figure 3: Optimality gap vs depth
        fig, ax = plt.subplots(figsize=(10, 6))
        means = [summary[d]["gap_percent_mean"] for d in depths]
        stds = [summary[d]["gap_percent_std"] for d in depths]
        valid_data = [(d, m, s) for d, m, s in zip(depths, means, stds) if m is not None]
        if valid_data:
            valid_depths, valid_means, valid_stds = zip(*valid_data)
            ax.errorbar(valid_depths, valid_means, yerr=valid_stds, fmt="o-", capsize=5, capthick=2, linewidth=2)
        ax.set_xlabel("QAOA Depth", fontsize=12)
        ax.set_ylabel("Optimality Gap (%)", fontsize=12)
        ax.set_title(f"Depth vs Gap (Penalty={penalty})", fontsize=14)
        ax.grid(True)
        fig_path = figures_dir / "depth_vs_gap.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure: {fig_path}")
        
        # Figure 4: Probability metrics vs depth
        fig, ax = plt.subplots(figsize=(10, 6))
        p_opt = [summary[d]["p_optimal_mean"] for d in depths]
        p_5pct = [summary[d]["p_5percent_mean"] for d in depths]
        valid_p_opt = [(d, p) for d, p in zip(depths, p_opt) if p is not None]
        valid_p_5pct = [(d, p) for d, p in zip(depths, p_5pct) if p is not None]
        if valid_p_opt:
            valid_depths, valid_p_opt_vals = zip(*valid_p_opt)
            ax.plot(valid_depths, valid_p_opt_vals, "o-", label="P(optimal)", linewidth=2)
        if valid_p_5pct:
            valid_depths, valid_p_5pct_vals = zip(*valid_p_5pct)
            ax.plot(valid_depths, valid_p_5pct_vals, "s-", label="P(≤5%)", linewidth=2)
        ax.set_xlabel("QAOA Depth", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"Depth vs Probability (Penalty={penalty})", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True)
        fig_path = figures_dir / "depth_vs_probability.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✓ Figure: {fig_path}")
        
    except ImportError:
        print("  ⚠ matplotlib not available; skipping figures")


def main():
    """Main experiment workflow."""
    args = parse_arguments()
    
    # Parse sweep parameters
    depths = parse_int_list(args.depths)
    seeds = parse_int_list(args.seeds)
    n_total = len(depths) * len(seeds)
    
    print("=" * 80)
    print("QUANTUM EMERGENCY ROUTING - RUNG 1 DEPTH SWEEP")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Penalty: {args.penalty}")
    print(f"  Depths: {depths}")
    print(f"  Seeds: {seeds}")
    print(f"  Shots: {args.shots}")
    print(f"  Max iterations: {args.maxiter}")
    print(f"  Total experiments: {n_total}")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    config = config.with_overrides(
        qubo=replace(config.qubo, penalty_scale=args.penalty),
        qaoa=replace(config.qaoa, shots=args.shots, maxiter=args.maxiter, reps=1),
    )
    
    # Load problem
    formulation, reference_objective = load_problem(config, offline=args.offline)
    
    # Run experiments
    print(f"\nRunning {n_total} experiments...")
    raw_results = []
    
    experiment_count = 0
    for depth in depths:
        for seed in seeds:
            experiment_count += 1
            progress_pct = (experiment_count / n_total) * 100
            print(f"\nProgress: {experiment_count}/{n_total} ({progress_pct:.1f}%)")
            print(f"  Depth={depth}, Seed={seed}")
            
            result = run_single_experiment(
                formulation,
                depth=depth,
                seed=seed,
                config=config,
                reference_objective=reference_objective,
            )
            
            if result["status"] == "success":
                print(f"  ✓ Feasibility: {result['feasibility_rate']:.4f}")
                best_obj = result['best_feasible_objective']
                print(f"  ✓ Best objective: {best_obj:.4f}" if best_obj is not None else "  ✓ Best objective: N/A (no feasible solutions)")
                gap = result['optimality_gap_percent']
                print(f"  ✓ Gap: {gap:.2f}%" if gap is not None else "  ✓ Gap: N/A")
                p_opt = result['p_optimal']
                print(f"  ✓ P(opt): {p_opt:.4f}" if p_opt is not None else "  ✓ P(opt): N/A")
            else:
                print(f"  ✗ Error: {result['error']}")
            
            raw_results.append(result)
    
    # Aggregate results
    print(f"\nAggregating results...")
    summary = aggregate_results(raw_results)
    
    # Save results
    save_results(raw_results, summary, args.output_dir, args.penalty)
    
    # Generate figures
    print(f"\nGenerating figures...")
    generate_figures(summary, args.output_dir, args.penalty)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    
    # Summary statistics
    successful = sum(1 for r in raw_results if r["status"] == "success")
    failed = len(raw_results) - successful
    print(f"Successful runs: {successful}/{len(raw_results)}")
    if failed > 0:
        print(f"Failed runs: {failed}")
    
    print("\nDepth Summary:")
    for depth in sorted(summary.keys()):
        agg = summary[depth]
        print(f"  Depth {depth}:")
        print(f"    Feasibility: {agg['feasibility_rate_mean']:.4f} ± {agg['feasibility_rate_std']:.4f}")
        best_obj_mean = agg['best_objective_mean']
        best_obj_std = agg['best_objective_std']
        print(f"    Best obj:    {best_obj_mean:.4f} ± {best_obj_std:.4f}" if best_obj_mean is not None else "    Best obj:    N/A")
        gap_mean = agg['gap_percent_mean']
        gap_std = agg['gap_percent_std']
        print(f"    Gap:         {gap_mean:.2f}% ± {gap_std:.2f}%" if gap_mean is not None else "    Gap:         N/A")
        p_opt_mean = agg['p_optimal_mean']
        p_opt_std = agg['p_optimal_std']
        print(f"    P(opt):      {p_opt_mean:.4f} ± {p_opt_std:.4f}" if p_opt_mean is not None else "    P(opt):      N/A")


if __name__ == "__main__":
    main()
