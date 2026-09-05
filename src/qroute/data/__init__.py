"""Road-network data layer: download, persistence, and instance sampling.

The three stages are deliberately separate so each can be tested and re-run on
its own:

1. :mod:`qroute.data.fetch` -- pull the study area from OpenStreetMap once and
   cache it as GraphML.
2. :mod:`qroute.data.graph_io` -- read/write that cache and repair the numeric
   edge attributes that GraphML and sparse OSM tagging mangle.
3. :mod:`qroute.data.instance` -- select a handful of well-separated
   intersections and collapse the network between them into a dense cost
   matrix that a QUBO can address.

:mod:`qroute.data._compat` absorbs the OSMnx 1.x/2.x API differences so no
other module needs to care which version is installed.
"""

from __future__ import annotations

from .fetch import FetchResult, fetch_road_network
from .graph_io import (
    coerce_numeric_edge_attributes,
    ensure_travel_times,
    format_graph_summary,
    graph_summary,
    largest_strongly_connected_subgraph,
    load_graph,
    save_graph,
    validate_edge_weight,
)
from .instance import (
    RoutingInstance,
    build_cost_matrix,
    build_dispatch_instance,
    build_tour_instance,
    haversine_m,
    load_instance,
    sample_instance,
    save_instance,
    select_nodes,
)

__all__ = [
    # fetch
    "FetchResult",
    "fetch_road_network",
    # graph io
    "save_graph",
    "load_graph",
    "coerce_numeric_edge_attributes",
    "ensure_travel_times",
    "validate_edge_weight",
    "largest_strongly_connected_subgraph",
    "graph_summary",
    "format_graph_summary",
    # instances
    "RoutingInstance",
    "select_nodes",
    "build_cost_matrix",
    "sample_instance",
    "build_tour_instance",
    "build_dispatch_instance",
    "save_instance",
    "load_instance",
    "haversine_m",
]
