"""Interactive routing experiment with area selection.

This script provides an interactive interface to:
1. Select a Dhaka area (Dhanmondi, Motijheel, Farmgate, Gulistan, Gulshan)
2. Run QAOA experiments with classical baselines
3. Generate terminal output and PNG plots for research papers

Usage:
    python scripts/interactive_routing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a clone, before `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import yaml
from qroute.config import Config, load_config
from qroute.exceptions import QRouteError
from qroute.logging_utils import configure_logging, get_logger
from qroute.pipeline import load_network, run_experiment, write_figures

_LOG = get_logger("scripts.interactive")


def load_dhaka_areas() -> dict:
    """Load Dhaka area configurations from YAML file."""
    areas_file = Path(__file__).resolve().parents[1] / "config" / "dhaka_areas.yaml"
    if not areas_file.exists():
        _LOG.error("Dhaka areas configuration file not found: %s", areas_file)
        return {}
    
    with open(areas_file, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('areas', {})


def select_area_interactive(areas: dict) -> str:
    """Interactive area selection."""
    print("\n" + "="*60)
    print("Available Dhaka Areas for Emergency Routing")
    print("="*60)
    
    area_list = list(areas.keys())
    for i, (key, area) in enumerate(areas.items(), 1):
        print(f"{i}. {area['name']}")
        print(f"   {area['description']}")
        print(f"   Bounding box: N{area['bbox']['north']} S{area['bbox']['south']} "
              f"E{area['bbox']['east']} W{area['bbox']['west']}")
        print()
    
    while True:
        try:
            choice = input(f"Select area (1-{len(area_list)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            
            index = int(choice) - 1
            if 0 <= index < len(area_list):
                selected_key = area_list[index]
                print(f"\nSelected: {areas[selected_key]['name']}")
                return selected_key
            else:
                print(f"Please enter a number between 1 and {len(area_list)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None


def update_config_for_area(config: Config, area_name: str, areas: dict) -> Config:
    """Update configuration with selected area's bounding box."""
    area = areas[area_name]
    from dataclasses import replace
    
    # Create new config with updated bbox and area-specific graph filename
    updated_config = config.with_overrides(
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
    
    return updated_config


def select_endpoints_interactive(graph, config) -> tuple:
    """Allow user to select origin and destination nodes interactively."""
    print("\n" + "-"*60)
    print("Origin and Destination Selection")
    print("-"*60)
    print("The system will automatically select well-separated endpoints.")
    print("For manual selection, you would need to provide specific node IDs.")
    print()
    
    # Use automatic endpoint selection
    from qroute.pipeline import pick_endpoints
    source, target = pick_endpoints(graph, config=config)
    
    print(f"Selected endpoints:")
    print(f"  Origin (Source): {source}")
    print(f"  Destination (Target): {target}")
    print()
    
    return source, target


def run_interactive_experiment(area_name: str, config: Config, run_qaoa: bool = True) -> None:
    """Run routing experiment for selected area."""
    print("\n" + "="*60)
    print(f"Running Emergency Routing Experiment: {area_name.upper()}")
    print("="*60)
    
    try:
        # Load network for selected area (force download to get new area)
        print(f"\nLoading road network for {area_name}...")
        graph = load_network(config, force=True)
        
        # Select endpoints
        source, target = select_endpoints_interactive(graph, config)
        
        # Run shortest path experiment with selected endpoints
        print("\n" + "-"*60)
        print("RUNG 1: Shortest Path")
        print("-"*60)
        result = run_experiment(
            "shortest_path",
            config,
            graph,
            quantum=run_qaoa,
            max_qubits=40,
        )
        print()
        print(result.report())
        
        # Save results and generate figures
        paths = config.build_paths(create=True)
        area_results_dir = paths.results_dir / area_name
        area_results_dir.mkdir(parents=True, exist_ok=True)
        
        from qroute.pipeline import save_experiment
        result_path = save_experiment(result, area_results_dir / f"{area_name}_shortest_path.json")
        print(f"\nResults saved: {result_path}")
        
        if config.output.save_figures:
            area_figures_dir = paths.figures_dir / area_name
            area_figures_dir.mkdir(parents=True, exist_ok=True)
            
            print("\nGenerating figures...")
            for figure_path in write_figures(config, result, graph):
                # Move figure to area-specific directory
                import shutil
                area_figure_path = area_figures_dir / figure_path.name
                shutil.move(str(figure_path), str(area_figure_path))
                print(f"Figure: {area_figure_path}")
        
        print("\n" + "="*60)
        print("Experiment completed successfully!")
        print("="*60)
        
    except QRouteError as exc:
        _LOG.error("Experiment failed: %s", exc)
        print(f"\nError: {exc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--area", type=str, default=None, 
                       help="Specify area directly (dhanmondi, motijheel, farmgate, gulistan, gulshan)")
    parser.add_argument("--no-qaoa", action="store_true", 
                       help="Run classical baselines only (skip QAOA)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    
    configure_logging(args.log_level)
    
    # Load area configurations
    areas = load_dhaka_areas()
    if not areas:
        print("Error: Could not load area configurations")
        return 1
    
    # Select area
    if args.area:
        if args.area.lower() in areas:
            area_name = args.area.lower()
        else:
            print(f"Error: Area '{args.area}' not found. Available areas: {list(areas.keys())}")
            return 1
    else:
        area_name = select_area_interactive(areas)
        if area_name is None:
            return 0
    
    # Load base configuration
    config = load_config()
    
    # Update configuration for selected area
    config = update_config_for_area(config, area_name, areas)
    
    # Run experiment
    run_interactive_experiment(area_name, config, run_qaoa=not args.no_qaoa)
    
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QRouteError as exc:
        _LOG.error("%s", exc)
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
