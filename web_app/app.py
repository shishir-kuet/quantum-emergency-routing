"""Flask web application for interactive emergency routing.

Provides a Google Maps-like interface for selecting origin/destination
and computing quantum-enhanced routes.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from flask import Flask, render_template, request, jsonify, send_from_directory
import yaml

from qroute.config import Config, load_config
from qroute.pipeline import load_network, run_experiment, write_figures
from qroute.exceptions import QRouteError
from qroute.logging_utils import configure_logging, get_logger

app = Flask(__name__)
app.secret_key = "emergency-routing-secret-key"

# Load configuration
config = load_config()
configure_logging("INFO")
_LOG = get_logger("web_app")

# Load Dhaka areas
AREAS_FILE = Path(__file__).resolve().parents[1] / "config" / "dhaka_areas.yaml"
with open(AREAS_FILE, 'r') as f:
    AREAS = yaml.safe_load(f)['areas']

# Cache for loaded graphs
GRAPH_CACHE: Dict[str, Any] = {}


@app.route('/')
def index():
    """Render the main interactive routing interface."""
    return render_template('index.html', areas=AREAS)


@app.route('/api/areas')
def get_areas():
    """Return available Dhaka areas."""
    return jsonify(AREAS)


@app.route('/api/compute_route', methods=['POST'])
def compute_route():
    """Compute route between selected points using QAOA."""
    try:
        data = request.json
        area_name = data.get('area', 'dhanmondi')
        use_qaoa = data.get('use_qaoa', True)
        
        # Generate unique ID for this computation
        run_id = str(uuid.uuid4())[:8]
        
        # Load area configuration
        area = AREAS[area_name]
        
        # Update config for selected area
        from dataclasses import replace
        area_config = config.with_overrides(
            data=replace(config.data, 
                        bbox=replace(config.data.bbox,
                            north=area['bbox']['north'],
                            south=area['bbox']['south'],
                            east=area['bbox']['east'],
                            west=area['bbox']['west']
                        ),
                        graph_filename=f"{area_name}_drive.graphml"
                    )
        )
        
        # Load network (use cache if available)
        if area_name not in GRAPH_CACHE:
            _LOG.info(f"Loading network for {area_name}...")
            graph = load_network(area_config, force=False)
            GRAPH_CACHE[area_name] = graph
        else:
            graph = GRAPH_CACHE[area_name]
            _LOG.info(f"Using cached network for {area_name}")

        # Snap the map clicks to the nearest road nodes so the experiment solves
        # the route the user selected rather than a randomly sampled pair.
        def nearest_node(point):
            if not point:
                return None
            latitude = float(point['lat'])
            longitude = float(point['lng'])
            return min(
                graph.nodes,
                key=lambda node: (
                    float(graph.nodes[node]['y']) - latitude
                ) ** 2 + (
                    float(graph.nodes[node]['x']) - longitude
                ) ** 2,
            )

        source = nearest_node(data.get('origin'))
        target = nearest_node(data.get('destination'))
        
        # Run experiment
        _LOG.info(f"Running routing experiment for {area_name}...")
        result = run_experiment(
            "shortest_path",
            area_config,
            graph,
            quantum=use_qaoa,
            source=source,
            target=target,
            max_qubits=40,
        )
        
        # Create unique output directories
        paths = area_config.build_paths(create=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = paths.results_dir / f"{area_name}_{run_id}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        figures_dir = paths.figures_dir / f"{area_name}_{run_id}_{timestamp}"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        from qroute.pipeline import save_experiment
        result_path = save_experiment(result, run_dir / f"{area_name}_route.json")
        
        # Generate figures with unique names
        figure_paths = []
        if area_config.output.save_figures:
            _LOG.info("Generating figures...")
            for figure_path in write_figures(area_config, result, graph):
                # Move to unique directory
                import shutil
                unique_figure_path = figures_dir / figure_path.name
                shutil.move(str(figure_path), str(unique_figure_path))
                # Return relative path from results directory
                relative_path = unique_figure_path.relative_to(paths.results_dir)
                figure_paths.append(str(relative_path))
        
        dijkstra_result = next(
            (item for item in result.classical if item.solver == 'dijkstra'), None
        )
        annealing_result = next(
            (item for item in result.classical if item.solver == 'annealing'), None
        )
        qaoa_solution = result.quantum.best_solution if result.quantum else None
        problem = getattr(result.formulation, 'problem', None)
        candidate_route = problem.candidate_paths[0] if problem and problem.candidate_paths else None

        def coordinates_for_nodes(nodes):
            if not nodes:
                return None
            return [
                [float(graph.nodes[node]['y']), float(graph.nodes[node]['x'])]
                for node in nodes
            ]

        def route_coordinates(solution):
            if solution is None or not solution.feasible or not solution.route:
                return None
            return coordinates_for_nodes(solution.route)

        # Prepare detailed solver metrics
        dijkstra_metrics = None
        if dijkstra_result:
            dijkstra_metrics = {
                'objective': dijkstra_result.objective,
                'solver_runtime_seconds': dijkstra_result.seconds,
                'runtime': dijkstra_result.seconds,
                'route': route_coordinates(dijkstra_result.solution),
                'optimization_source': 'Exact classical baseline',
                'optimal': dijkstra_result.optimal,
                'feasible': dijkstra_result.feasible,
            }

        annealing_metrics = None
        if annealing_result:
            scope = result.metadata.get('optimization_scope', 'Independent QUBO optimization')
            annealing_metrics = {
                'objective': annealing_result.objective,
                'runtime': annealing_result.seconds,
                'solver_runtime_seconds': annealing_result.seconds,
                'bitstring': annealing_result.bits,
                'solver_status': 'converged' if annealing_result.feasible else 'infeasible',
                'route': route_coordinates(annealing_result.solution),
                'optimization_source': f'Independent solver ({scope})',
                'feasible': annealing_result.feasible,
                'n_restarts': annealing_result.details.get('n_restarts'),
                'n_sweeps': annealing_result.details.get('n_sweeps'),
                'hit_count': annealing_result.details.get('hit_count'),
            }

        qaoa_metrics = None
        if result.quantum:
            scope = result.metadata.get('optimization_scope', 'Independent QUBO optimization')
            top_states = result.metadata.get('top_sampled_states', [])
            qaoa_metrics = {
                'best_sampled_objective': result.quantum.objective,
                'p_opt': result.metadata.get('success_probability'),
                'feasibility_rate': result.metadata.get('feasibility_rate'),
                'feasibility': result.quantum.feasible,
                'optimizer_iterations': result.quantum.n_evaluations,
                'qaoa_runtime': result.quantum.seconds,
                'optimizer_runtime_seconds': result.quantum.metadata.get(
                    'optimizer_runtime_seconds', result.quantum.seconds
                ),
                'circuit_sampling_seconds': result.quantum.metadata.get(
                    'circuit_sampling_seconds'
                ),
                'execution_runtime_seconds': result.quantum.metadata.get(
                    'execution_runtime_seconds'
                ),
                'top_sampled_states': top_states,
                'total_shots': result.metadata.get('sampled_shots', result.quantum.shots),
                'best_bits_in_counts': result.metadata.get('best_bits_in_counts', False),
                'best_sampled_bitstring': result.quantum.metadata.get('best_bitstring'),
                'route': route_coordinates(qaoa_solution),
                'optimization_source': f'Independent solver ({scope})',
                'reps': result.quantum.reps,
                'expectation_mode': result.quantum.expectation_mode,
                'n_qubits': result.quantum.n_qubits,
                'optimizer': result.quantum.optimizer,
            }

        # Prepare response
        response = {
            'success': True,
            'run_id': run_id,
            'area': area_name,
            'timestamp': timestamp,
            'formulation_info': {
                'n_variables': result.formulation.num_variables,
                'n_candidate_edges': problem.n_edges if problem else None,
                'n_candidate_paths': len(problem.candidate_paths) if problem else None,
                'candidate_paths': result.metadata.get('candidate_path_costs', []),
                'optimization_scope': result.metadata.get(
                    'optimization_scope', 'Assignment or distilled-instance QUBO'
                ),
                'candidate_generation': result.metadata.get(
                    'candidate_generation', 'See experiment metadata'
                ),
                'dijkstra_role': result.metadata.get(
                    'dijkstra_role', 'Reference solver or preprocessing as documented'
                ),
                'optimization_type': result.metadata.get(
                    'optimization_scope', 'Assignment or distilled-instance QUBO'
                ),
                'instance_id': result.metadata.get('instance_id'),
                'source': result.metadata.get('source'),
                'target': result.metadata.get('target'),
                'vehicles': result.metadata.get('vehicles'),
                'incidents': result.metadata.get('incidents'),
            },
            'results': {
                'dijkstra': dijkstra_metrics,
                'annealing': annealing_metrics,
                'qaoa': qaoa_metrics,
                'candidate_cost': problem.reference_cost() if problem else None,
                'candidate_route': coordinates_for_nodes(candidate_route),
            },
            'figures': figure_paths,
            'result_path': str(result_path)
        }
        
        return jsonify(response)
        
    except Exception as e:
        _LOG.error(f"Route computation failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/results/<path:filename>')
def serve_results(filename):
    """Serve result files including PNG images."""
    results_dir = Path(__file__).resolve().parents[1] / "results"
    return send_from_directory(results_dir, filename, mimetype='image/png' if filename.endswith('.png') else None)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
