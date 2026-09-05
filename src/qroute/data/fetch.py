"""Fetch the study-area road network from OpenStreetMap.

This is the only module in the project that touches the network. It downloads
the drivable graph for the configured bounding box exactly once, enriches it
with the attributes routing needs, and caches it as GraphML under
``data/raw/``. Every later stage reads that cache, so the rest of the pipeline
runs offline and is fully reproducible.

Typical use::

    from qroute.config import load_config
    from qroute.data.fetch import fetch_road_network

    config = load_config()
    result = fetch_road_network(config)
    print(result.describe())
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import networkx as nx

from .. import __version__
from ..config import Config
from ..exceptions import DataError
from ..logging_utils import get_logger
from . import _compat
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

__all__ = ["FetchResult", "fetch_road_network"]

_LOG = get_logger(__name__)

#: Below this many nodes the study area is almost certainly mis-configured
#: (wrong bbox ordering, ocean, or a typo in the coordinates).
_SUSPICIOUSLY_SMALL = 10


@dataclass(frozen=True)
class FetchResult:
    """Outcome of a fetch, including where the graph was cached."""

    graph: nx.MultiDiGraph
    path: Path
    from_cache: bool
    summary: Dict[str, Any]

    def describe(self) -> str:
        origin = "loaded from cache" if self.from_cache else "downloaded"
        return f"Road network {origin}: {format_graph_summary(self.graph)}\n  -> {self.path}"


# ---------------------------------------------------------------------------
# Enrichment pipeline
# ---------------------------------------------------------------------------
def _consolidate(graph: nx.MultiDiGraph, tolerance_m: float) -> nx.MultiDiGraph:
    """Merge near-coincident intersections, returning a lat/lon graph.

    OSM models a large intersection as several nodes a few metres apart. That
    inflates the node count and, more importantly, makes "distinct locations"
    ambiguous when sampling instances. Consolidation needs projected
    coordinates, so we project to UTM, merge, then project back to lat/lon.
    """
    _LOG.info("Consolidating intersections within %.1f m", tolerance_m)
    projected = _compat.project_graph(graph)
    merged = _compat.consolidate_intersections(
        projected, tolerance=tolerance_m, rebuild_graph=True, dead_ends=False
    )
    restored = _compat.project_graph(merged, to_latlong=True)
    if not isinstance(restored, nx.MultiDiGraph):
        restored = nx.MultiDiGraph(restored)
    _LOG.info(
        "Consolidation: %d -> %d nodes",
        graph.number_of_nodes(),
        restored.number_of_nodes(),
    )
    return restored


def _attach_speeds(graph: nx.MultiDiGraph, default_speed_kph: float) -> None:
    """Impute ``speed_kph`` and ``travel_time`` on every edge, in place.

    OSMnx does the clever part (inferring speeds per highway class from the
    tags that *are* present); we then backfill anything it left behind. The
    OSMnx call is wrapped because it raises on graphs with no usable
    ``maxspeed`` tags at all -- entirely possible for a small Dhaka window.
    """
    try:
        _compat.add_edge_speeds(graph, fallback_kph=default_speed_kph)
        _compat.add_edge_travel_times(graph)
    except Exception as exc:  # noqa: BLE001 - imputation is best-effort
        _LOG.warning(
            "OSMnx speed imputation failed (%s); falling back to the configured "
            "default of %.1f km/h for every edge",
            exc,
            default_speed_kph,
        )

    coerce_numeric_edge_attributes(graph)
    ensure_travel_times(graph, default_speed_kph=default_speed_kph)


def _prepare(graph: nx.MultiDiGraph, config: Config) -> nx.MultiDiGraph:
    """Turn a raw OSMnx download into a graph that is safe to route on."""
    if config.data.consolidate_tolerance_m > 0:
        graph = _consolidate(graph, config.data.consolidate_tolerance_m)

    _attach_speeds(graph, config.data.default_speed_kph)

    self_loops = list(nx.selfloop_edges(graph, keys=True))
    if self_loops:
        graph.remove_edges_from(self_loops)
        _LOG.info("Removed %d self-loop edge(s)", len(self_loops))

    graph = largest_strongly_connected_subgraph(graph)

    graph.graph.update(
        {
            "qroute_version": __version__,
            "qroute_fetched_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "qroute_bbox_north": config.data.bbox.north,
            "qroute_bbox_south": config.data.bbox.south,
            "qroute_bbox_east": config.data.bbox.east,
            "qroute_bbox_west": config.data.bbox.west,
            "qroute_network_type": config.data.network_type,
            "qroute_default_speed_kph": config.data.default_speed_kph,
        }
    )
    return graph


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def fetch_road_network(
    config: Config, *, force: bool = False, validate: bool = True
) -> FetchResult:
    """Return the study-area road network, downloading it if necessary.

    Parameters
    ----------
    config:
        Fully-resolved project configuration.
    force:
        Re-download even when a cached GraphML file exists.
    validate:
        Verify that ``config.instance.weight`` is present and positive on every
        edge. Leave this on unless you are deliberately inspecting a broken
        graph.
    """
    paths = config.build_paths(create=True)
    target = paths.graph_file

    if target.is_file() and not force:
        _LOG.info("Using cached road network at %s", target)
        graph = load_graph(target, default_speed_kph=config.data.default_speed_kph)
        if validate:
            validate_edge_weight(graph, config.instance.weight)
        return FetchResult(
            graph=graph, path=target, from_cache=True, summary=graph_summary(graph)
        )

    _LOG.info("Downloading road network for %s", config.data.bbox.describe())
    _compat.configure_osmnx(cache_folder=str(paths.root / "cache"))

    try:
        raw = _compat.graph_from_bbox(
            config.data.bbox.as_osmnx_bbox,
            network_type=config.data.network_type,
            simplify=config.data.simplify,
            truncate_by_edge=True,
        )
    except DataError:
        raise
    except Exception as exc:  # noqa: BLE001 - network/Overpass failures vary widely
        raise DataError(
            "Failed to download the road network from OpenStreetMap. Check your "
            "internet connection and that the Overpass API is reachable, then "
            f"retry. Underlying error: {exc}"
        ) from exc

    if raw is None or raw.number_of_nodes() == 0:
        raise DataError(
            f"OpenStreetMap returned an empty graph for {config.data.bbox.describe()}. "
            f"Verify the bounding box covers land with mapped roads and that "
            f"network_type='{config.data.network_type}' is appropriate."
        )

    if raw.number_of_nodes() < _SUSPICIOUSLY_SMALL:
        _LOG.warning(
            "Downloaded graph has only %d nodes. That is unusually small -- "
            "double-check the bounding box coordinates in your config.",
            raw.number_of_nodes(),
        )

    if not isinstance(raw, nx.MultiDiGraph):
        raw = nx.MultiDiGraph(raw)

    graph = _prepare(raw, config)

    if graph.number_of_nodes() < config.instance.n_nodes:
        raise DataError(
            f"The prepared graph has only {graph.number_of_nodes()} routable node(s), "
            f"fewer than instance.n_nodes={config.instance.n_nodes}. Widen the "
            f"bounding box or reduce the instance size."
        )

    save_graph(graph, target)
    if validate:
        validate_edge_weight(graph, config.instance.weight)

    _LOG.info("Prepared network: %s", format_graph_summary(graph))
    return FetchResult(
        graph=graph, path=target, from_cache=False, summary=graph_summary(graph)
    )
