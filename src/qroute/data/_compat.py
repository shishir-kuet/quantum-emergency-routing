"""Compatibility layer over the OSMnx 1.x -> 2.x API break.

OSMnx 2.0 (released late 2024) reorganised most top-level helpers into
submodules and -- critically -- **changed the meaning of the ``bbox``
argument** to the ``(left, bottom, right, top)`` convention, i.e.
``(west, south, east, north)``. OSMnx 1.x instead took four separate
``north/south/east/west`` keyword arguments.

Getting that ordering wrong does not raise: it silently downloads the wrong
patch of the planet, or an empty graph. This module therefore version-gates
the bbox call rather than relying on ``TypeError`` fallbacks, and resolves
every other moved helper by trying its known locations in order.

Everything here is intentionally dependency-light: :mod:`osmnx` is imported
lazily so that the rest of the package (QUBO, QAOA, tests) works on a machine
where the geospatial stack is not installed.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

from ..exceptions import DataError, MissingDependencyError
from ..logging_utils import get_logger

__all__ = [
    "import_osmnx",
    "osmnx_version",
    "configure_osmnx",
    "graph_from_bbox",
    "add_edge_speeds",
    "add_edge_travel_times",
    "save_graphml",
    "load_graphml",
    "largest_component",
    "project_graph",
    "consolidate_intersections",
    "graph_to_gdfs",
    "nearest_nodes",
]

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# Import & version handling
# ---------------------------------------------------------------------------
def import_osmnx() -> Any:
    """Import and return the :mod:`osmnx` module, or raise a helpful error."""
    try:
        import osmnx as ox  # noqa: PLC0415 (deliberately lazy)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MissingDependencyError(
            "osmnx", "downloading road networks from OpenStreetMap"
        ) from exc
    return ox


def osmnx_version() -> Tuple[int, ...]:
    """Return the installed OSMnx version as a tuple of ints, e.g. ``(2, 0, 1)``.

    Non-numeric suffixes (``rc1``, ``dev0``) are stripped so that release
    candidates compare sensibly against final releases.
    """
    ox = import_osmnx()
    raw = str(getattr(ox, "__version__", "0"))
    parts: list[int] = []
    for chunk in raw.split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _resolve(*dotted_candidates: str) -> Callable[..., Any]:
    """Return the first callable that exists at one of the given dotted paths.

    Paths are relative to the ``osmnx`` module, e.g.
    ``"routing.add_edge_speeds"`` then ``"add_edge_speeds"``.
    """
    ox = import_osmnx()
    for dotted in dotted_candidates:
        target: Any = ox
        for attribute in dotted.split("."):
            target = getattr(target, attribute, None)
            if target is None:
                break
        if callable(target):
            return target
    raise DataError(
        f"osmnx {'.'.join(str(p) for p in osmnx_version())} exposes none of the "
        f"expected helpers: {list(dotted_candidates)}. Please install a version "
        f"in the supported range (>=1.9,<3.0)."
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
def configure_osmnx(*, cache_folder: str | None = None, log_console: bool = False) -> None:
    """Apply the OSMnx global settings we rely on.

    Caching is switched **on** deliberately: re-running the pipeline should not
    hammer the Overpass API, and it makes offline re-runs possible once the
    first download has succeeded.
    """
    ox = import_osmnx()
    settings = getattr(ox, "settings", None)
    if settings is None:  # pragma: no cover - very old osmnx
        _LOG.warning("osmnx.settings not available; skipping settings configuration")
        return
    settings.use_cache = True
    settings.log_console = bool(log_console)
    if cache_folder is not None:
        settings.cache_folder = str(cache_folder)


# ---------------------------------------------------------------------------
# Graph download
# ---------------------------------------------------------------------------
def graph_from_bbox(
    bbox_wsen: Sequence[float],
    *,
    network_type: str = "drive",
    simplify: bool = True,
    retain_all: bool = False,
    truncate_by_edge: bool = True,
) -> Any:
    """Download a street network inside a bounding box.

    Parameters
    ----------
    bbox_wsen:
        ``(west, south, east, north)`` in WGS-84 degrees -- the OSMnx 2.x
        ordering. This function converts to the 1.x keyword form when needed,
        so callers only ever deal with one convention.
    """
    if len(bbox_wsen) != 4:
        raise DataError(
            f"bbox must have exactly 4 values (west, south, east, north), "
            f"got {len(bbox_wsen)}"
        )
    west, south, east, north = (float(v) for v in bbox_wsen)
    ox = import_osmnx()
    version = osmnx_version()

    common = {
        "network_type": network_type,
        "simplify": simplify,
        "retain_all": retain_all,
        "truncate_by_edge": truncate_by_edge,
    }

    if version >= (2,):
        _LOG.debug("Using OSMnx >= 2.0 bbox convention (west, south, east, north)")
        return ox.graph_from_bbox(bbox=(west, south, east, north), **common)

    _LOG.debug("Using OSMnx 1.x north/south/east/west keyword convention")
    try:
        return ox.graph_from_bbox(
            north=north, south=south, east=east, west=west, **common
        )
    except TypeError:
        # OSMnx 1.9.x transitional builds accepted a bbox tuple ordered
        # (north, south, east, west) before 2.0 flipped it.
        _LOG.debug("Falling back to OSMnx 1.9 bbox tuple (north, south, east, west)")
        return ox.graph_from_bbox(bbox=(north, south, east, west), **common)


# ---------------------------------------------------------------------------
# Edge attribute enrichment
# ---------------------------------------------------------------------------
def add_edge_speeds(graph: Any, *, fallback_kph: float) -> Any:
    """Attach a ``speed_kph`` attribute to every edge.

    OSMnx imputes missing ``maxspeed`` tags from the highway type; anything it
    still cannot infer falls back to *fallback_kph*. In Dhaka the OSM speed
    tagging is sparse, so the fallback does most of the work -- which is why it
    is a configurable parameter rather than a magic number.
    """
    function = _resolve("routing.add_edge_speeds", "add_edge_speeds", "speed.add_edge_speeds")
    return function(graph, fallback=float(fallback_kph))


def add_edge_travel_times(graph: Any) -> Any:
    """Attach a ``travel_time`` attribute (seconds) derived from length/speed."""
    function = _resolve(
        "routing.add_edge_travel_times",
        "add_edge_travel_times",
        "speed.add_edge_travel_times",
    )
    return function(graph)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_graphml(graph: Any, filepath: str) -> None:
    """Write *graph* to GraphML at *filepath*."""
    function = _resolve("io.save_graphml", "save_graphml")
    function(graph, filepath=str(filepath))


def load_graphml(filepath: str) -> Any:
    """Read a GraphML file written by :func:`save_graphml`."""
    function = _resolve("io.load_graphml", "load_graphml")
    return function(filepath=str(filepath))


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------
def largest_component(graph: Any, *, strongly: bool = True) -> Any:
    """Return the largest (strongly) connected component of *graph*.

    Strong connectivity is what routing needs: on a one-way network, weak
    connectivity does not guarantee that node *j* is reachable from node *i*.
    """
    function = _resolve("truncate.largest_component", "utils_graph.get_largest_component")
    return function(graph, strongly=strongly)


def project_graph(graph: Any, *, to_latlong: bool = False) -> Any:
    """Project *graph* to UTM, or back to lat/lon when ``to_latlong=True``."""
    function = _resolve("projection.project_graph", "project_graph")
    return function(graph, to_latlong=to_latlong)


def consolidate_intersections(
    graph: Any, *, tolerance: float, rebuild_graph: bool = True, dead_ends: bool = False
) -> Any:
    """Merge intersection nodes that lie within *tolerance* metres of each other.

    The input graph must already be projected (metres), which is why
    :func:`qroute.data.fetch` sandwiches this call between two
    :func:`project_graph` calls.
    """
    function = _resolve(
        "simplification.consolidate_intersections", "consolidate_intersections"
    )
    return function(
        graph, tolerance=float(tolerance), rebuild_graph=rebuild_graph, dead_ends=dead_ends
    )


def graph_to_gdfs(graph: Any, *, nodes: bool = True, edges: bool = True) -> Any:
    """Convert *graph* to GeoDataFrames (nodes, edges)."""
    function = _resolve("convert.graph_to_gdfs", "graph_to_gdfs", "utils_graph.graph_to_gdfs")
    return function(graph, nodes=nodes, edges=edges)


def nearest_nodes(graph: Any, longitudes: Any, latitudes: Any) -> Any:
    """Return the graph node(s) nearest to the given lon/lat coordinate(s).

    Note the argument order: OSMnx takes ``X`` (longitude) before ``Y``
    (latitude), which is the opposite of how coordinates are usually spoken.
    """
    function = _resolve("distance.nearest_nodes", "nearest_nodes")
    return function(graph, X=longitudes, Y=latitudes)
