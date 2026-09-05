"""Graph persistence and hygiene.

Two concerns live here, both deliberately independent of OSMnx so that the
QUBO/QAOA half of the project (and the test suite) can manipulate small
synthetic graphs without the geospatial stack installed:

**Persistence** -- :func:`save_graph` / :func:`load_graph` round-trip a
``MultiDiGraph`` through GraphML, preferring OSMnx's readers (which restore
attribute types correctly) and falling back to plain NetworkX.

**Hygiene** -- GraphML is a text format, so numeric edge attributes come back
as strings, and OSM tagging is patchy enough that ``travel_time`` is sometimes
missing entirely. Routing silently produces nonsense in both cases (NetworkX
substitutes a weight of 1 for a missing attribute), so every graph is pushed
through :func:`coerce_numeric_edge_attributes`, :func:`ensure_travel_times`
and :func:`validate_edge_weight` before it is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import networkx as nx

from ..exceptions import DataError
from ..logging_utils import get_logger

__all__ = [
    "NUMERIC_EDGE_ATTRS",
    "save_graph",
    "load_graph",
    "coerce_numeric_edge_attributes",
    "ensure_travel_times",
    "validate_edge_weight",
    "largest_strongly_connected_subgraph",
    "graph_summary",
    "format_graph_summary",
]

_LOG = get_logger(__name__)

#: Edge attributes that must be floats for routing to work.
NUMERIC_EDGE_ATTRS: tuple[str, ...] = ("length", "speed_kph", "travel_time")

_KPH_TO_MPS = 1000.0 / 3600.0


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------
def _to_float(value: Any) -> Optional[float]:
    """Best-effort conversion of an OSM attribute value to a float.

    OSM attributes are messy: a simplified edge that merged three OSM ways
    carries a *list* of the original tag values. For a list we average, which
    is right for speeds and harmless for lengths (OSMnx already sums lengths
    during simplification, so lists of lengths do not occur in practice).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            # Values like "30 mph" or "signals" are not usable numerics.
            head = text.split()[0]
            try:
                return float(head)
            except ValueError:
                return None
    if isinstance(value, (list, tuple, set)):
        numbers = [_to_float(item) for item in value]
        numbers = [n for n in numbers if n is not None]
        if not numbers:
            return None
        return sum(numbers) / len(numbers)
    return None


def coerce_numeric_edge_attributes(
    graph: nx.Graph, attributes: Sequence[str] = NUMERIC_EDGE_ATTRS
) -> Dict[str, int]:
    """Convert the given edge attributes to floats in place.

    Returns a mapping ``{attribute: number_of_edges_repaired}``. Values that
    cannot be interpreted as numbers are **removed** rather than left as
    strings, so that downstream validation notices them instead of NetworkX
    silently comparing strings.
    """
    repaired: Dict[str, int] = {attribute: 0 for attribute in attributes}
    dropped: Dict[str, int] = {attribute: 0 for attribute in attributes}

    for _, _, data in graph.edges(data=True):
        for attribute in attributes:
            if attribute not in data:
                continue
            current = data[attribute]
            if isinstance(current, float):
                continue
            converted = _to_float(current)
            if converted is None:
                del data[attribute]
                dropped[attribute] += 1
            else:
                data[attribute] = converted
                repaired[attribute] += 1

    for attribute in attributes:
        if repaired[attribute]:
            _LOG.debug("Coerced %d '%s' values to float", repaired[attribute], attribute)
        if dropped[attribute]:
            _LOG.warning(
                "Dropped %d unparseable '%s' value(s)", dropped[attribute], attribute
            )
    return repaired


def ensure_travel_times(graph: nx.Graph, *, default_speed_kph: float) -> int:
    """Fill in any missing ``speed_kph`` / ``travel_time`` edge attributes.

    ``travel_time`` is computed as ``length / (speed_kph * 1000 / 3600)`` and
    is therefore in **seconds**. Edges with no usable ``length`` are reported
    and left alone -- they will be caught by :func:`validate_edge_weight`.

    Returns the number of edges that were modified.
    """
    if default_speed_kph <= 0:
        raise DataError("default_speed_kph must be positive")

    modified = 0
    missing_length = 0

    for _, _, data in graph.edges(data=True):
        length = _to_float(data.get("length"))
        if length is None or length <= 0:
            missing_length += 1
            continue

        speed = _to_float(data.get("speed_kph"))
        if speed is None or speed <= 0:
            speed = float(default_speed_kph)
            data["speed_kph"] = speed
            modified += 1

        travel_time = _to_float(data.get("travel_time"))
        if travel_time is None or travel_time <= 0:
            data["travel_time"] = length / (speed * _KPH_TO_MPS)
            modified += 1

    if missing_length:
        _LOG.warning(
            "%d edge(s) have no usable 'length'; travel_time could not be derived "
            "for them",
            missing_length,
        )
    if modified:
        _LOG.info("Imputed speed/travel_time on %d edge attribute slot(s)", modified)
    return modified


def validate_edge_weight(graph: nx.Graph, weight: str) -> None:
    """Raise :class:`DataError` unless every edge carries a positive *weight*.

    This guard exists because NetworkX's shortest-path functions treat a
    missing edge attribute as a weight of ``1``. On a travel-time graph where
    real values are in the hundreds of seconds, a handful of missing attributes
    would turn those edges into near-free shortcuts and quietly corrupt every
    baseline the project compares against.
    """
    missing = 0
    non_positive = 0
    for _, _, data in graph.edges(data=True):
        value = data.get(weight)
        if not isinstance(value, (int, float)):
            missing += 1
        elif value <= 0:
            non_positive += 1

    if missing or non_positive:
        raise DataError(
            f"Edge weight '{weight}' is unusable: {missing} edge(s) missing it and "
            f"{non_positive} edge(s) with a non-positive value. Run "
            f"ensure_travel_times() (or re-fetch the graph) before routing."
        )


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------
def largest_strongly_connected_subgraph(graph: nx.Graph) -> nx.Graph:
    """Return a mutable copy of the largest strongly connected component.

    Undirected graphs fall back to connected components. Restricting to this
    component guarantees that a shortest path exists between *every* pair of
    sampled nodes, which the distance matrices depend on.
    """
    if graph.number_of_nodes() == 0:
        raise DataError("Cannot extract a component from an empty graph")

    if graph.is_directed():
        components = nx.strongly_connected_components(graph)
    else:
        components = nx.connected_components(graph)

    largest = max(components, key=len)
    subgraph = graph.subgraph(largest).copy()

    removed = graph.number_of_nodes() - subgraph.number_of_nodes()
    if removed:
        _LOG.info(
            "Kept largest %s component: %d node(s) (dropped %d)",
            "strongly connected" if graph.is_directed() else "connected",
            subgraph.number_of_nodes(),
            removed,
        )
    return subgraph


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _sanitise_for_graphml(graph: nx.Graph) -> nx.Graph:
    """Return a copy whose attribute values are all GraphML-writable scalars."""
    clean = graph.copy()

    def fix(container: Dict[str, Any]) -> None:
        for key, value in list(container.items()):
            if value is None:
                del container[key]
            elif not isinstance(value, (str, int, float, bool)):
                container[key] = str(value)

    for _, data in clean.nodes(data=True):
        fix(data)
    for edge in clean.edges(data=True):
        fix(edge[-1])
    fix(clean.graph)
    return clean


def save_graph(graph: nx.Graph, path: Path | str) -> Path:
    """Write *graph* to GraphML, creating parent directories as needed."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ._compat import save_graphml

        save_graphml(graph, str(target))
    except Exception as exc:  # noqa: BLE001 - fall back rather than fail the run
        _LOG.debug("OSMnx GraphML writer unavailable (%s); using NetworkX", exc)
        nx.write_graphml(_sanitise_for_graphml(graph), str(target))

    _LOG.info(
        "Saved graph (%d nodes, %d edges) to %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        target,
    )
    return target


def load_graph(path: Path | str, *, default_speed_kph: float = 20.0) -> nx.MultiDiGraph:
    """Read a GraphML graph and restore numeric attribute types.

    The returned graph is guaranteed to have float ``length`` and
    ``travel_time`` attributes on every edge that has a usable length.
    """
    source = Path(path)
    if not source.is_file():
        raise DataError(
            f"Graph file not found: {source}. Run the fetch step first "
            f"(python -m qroute fetch)."
        )

    try:
        from ._compat import load_graphml

        graph = load_graphml(str(source))
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("OSMnx GraphML reader unavailable (%s); using NetworkX", exc)
        graph = nx.read_graphml(str(source))
        if not graph.is_multigraph() or not graph.is_directed():
            graph = nx.MultiDiGraph(graph)

    # Ensure we have a proper NetworkX graph (OSMnx 2.x may return different types)
    if not isinstance(graph, nx.MultiDiGraph):
        if isinstance(graph, nx.Graph):
            graph = nx.MultiDiGraph(graph)
        else:
            _LOG.warning("Unexpected graph type %s, converting to MultiDiGraph", type(graph))
            graph = nx.MultiDiGraph(graph)

    coerce_numeric_edge_attributes(graph)
    ensure_travel_times(graph, default_speed_kph=default_speed_kph)
    _LOG.info(
        "Loaded graph (%d nodes, %d edges) from %s",
        graph.number_of_nodes(),
        graph.number_of_edges(),
        source,
    )
    return graph


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def graph_summary(graph: nx.Graph) -> Dict[str, Any]:
    """Collect descriptive statistics used in logs and result metadata."""
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    total_length_m = 0.0
    with_length = 0
    with_travel_time = 0
    for _, _, data in graph.edges(data=True):
        length = data.get("length")
        if isinstance(length, (int, float)):
            total_length_m += float(length)
            with_length += 1
        if isinstance(data.get("travel_time"), (int, float)):
            with_travel_time += 1

    if graph.is_directed():
        n_components = nx.number_strongly_connected_components(graph)
        component_kind = "strongly_connected"
    else:
        n_components = nx.number_connected_components(graph)
        component_kind = "connected"

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "component_kind": component_kind,
        "n_components": n_components,
        "mean_out_degree": (n_edges / n_nodes) if n_nodes else 0.0,
        "total_length_km": total_length_m / 1000.0,
        "edges_with_length": with_length,
        "edges_with_travel_time": with_travel_time,
    }


def format_graph_summary(graph: nx.Graph) -> str:
    """One-line human-readable rendering of :func:`graph_summary`."""
    stats = graph_summary(graph)
    return (
        f"{stats['n_nodes']} nodes, {stats['n_edges']} edges, "
        f"{stats['n_components']} {stats['component_kind']} component(s), "
        f"{stats['total_length_km']:.2f} km of road, "
        f"travel_time on {stats['edges_with_travel_time']}/{stats['n_edges']} edges"
    )
