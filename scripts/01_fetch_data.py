"""Step 1: download the Dhaka study-area road network and cache it.

This is the only script that needs an internet connection, and it only needs one
once -- the graph is written to ``data/raw/`` as GraphML and every later step
reads it from there.

    python scripts/01_fetch_data.py
    python scripts/01_fetch_data.py --plot          # also write a map
    python scripts/01_fetch_data.py --force         # re-download
    python scripts/01_fetch_data.py --instances     # also sample and save instances

The study area is a small bbox over Dhanmondi / Kalabagan / New Market, set in
``config/default.yaml``. Small on purpose: a few thousand intersections downloads
in seconds and keeps every later experiment fast enough to iterate on a laptop.
Widen ``data.bbox`` when you want a harder instance -- nothing else needs to change.

OpenStreetMap data is © OpenStreetMap contributors, ODbL 1.0. Cite it in any
write-up that uses these graphs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a clone, before `pip install -e .`.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from qroute.config import load_config  # noqa: E402
from qroute.data import (  # noqa: E402
    build_dispatch_instance,
    build_tour_instance,
    format_graph_summary,
    graph_summary,
    save_instance,
)
from qroute.exceptions import QRouteError  # noqa: E402
from qroute.logging_utils import configure_logging, get_logger  # noqa: E402
from qroute.pipeline import load_network  # noqa: E402

_LOG = get_logger("scripts.fetch")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-c", "--config", type=Path, default=None)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--plot", action="store_true", help="write a map of the network")
    parser.add_argument(
        "--instances",
        action="store_true",
        help="also sample a tour instance and a dispatch instance and save them",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    config = load_config(args.config)
    paths = config.build_paths(create=True)

    print(config.summary())
    print()

    graph = load_network(config, force=args.force)
    print(f"DEBUG: graph type = {type(graph)}")
    print(f"DEBUG: graph is dict = {isinstance(graph, dict)}")
    print(format_graph_summary(graph_summary(graph)))
    print(f"\nCached at: {paths.graph_file}")

    if args.instances:
        tour = build_tour_instance(graph, config)
        dispatch = build_dispatch_instance(graph, config)
        tour_path = save_instance(tour, paths.processed_dir / "tour_instance.json")
        dispatch_path = save_instance(
            dispatch, paths.processed_dir / "dispatch_instance.json"
        )
        print()
        print("tour instance    : " + tour.describe())
        print(f"                   {tour_path}")
        print("dispatch instance: " + dispatch.describe())
        print(f"                   {dispatch_path}")

    if args.plot:
        from qroute.viz import plot_graph, use_headless_backend

        use_headless_backend()
        target = paths.figures_dir / "network.png"
        plot_graph(graph, path=target, title=f"{config.project.name}: road network")
        print(f"\nMap: {target}")

    print("\nNext: python scripts/02_classical_baseline.py")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except QRouteError as exc:
        _LOG.error("%s", exc)
        print(f"\nerror: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
