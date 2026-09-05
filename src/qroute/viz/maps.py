"""Maps: road network, sampled instance nodes, and decoded routes.

Plots directly with matplotlib rather than ``osmnx.plot_graph`` so that a route
can be drawn *on top of* the network, node roles (depot, ambulance base,
incident) can be distinguished, and the same code works on a graph loaded from
GraphML with no OSMnx installed.

The key detail is that an abstract route ``(0, 3, 1, 2)`` over instance indices
must be expanded back onto real road geometry before it means anything. The
instance keeps that mapping in :attr:`~qroute.data.instance.RoutingInstance.paths`
-- one full node sequence per index pair -- so :func:`plot_route` draws the streets
the ambulance would actually drive, not straight lines between abstract points. A
straight-line plot of a road route is not just uglier, it is misleading: it hides
the one-way streets and detours that make the cost matrix asymmetric in the first
place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..data.instance import RoutingInstance
from ..exceptions import DataError
from ..logging_utils import get_logger
from .style import PALETTE, _equal_aspect_geographic, annotate_source, get_pyplot, save_figure

__all__ = ["plot_graph", "plot_instance", "plot_route", "plot_node_path", "plot_assignment"]

_LOG = get_logger(__name__)


def _node_xy(graph: nx.Graph, node: Any) -> Tuple[float, float]:
    """``(longitude, latitude)`` for *node*, i.e. plotting order, not geo order."""
    data = graph.nodes[node]
    try:
        return float(data["x"]), float(data["y"])
    except KeyError as exc:
        raise DataError(
            f"Node {node!r} has no x/y coordinates; the graph must come from OSMnx "
            f"or a GraphML file that preserved them"
        ) from exc


def _edge_segments(graph: nx.Graph) -> List[List[Tuple[float, float]]]:
    """Line segments for every edge, following ``geometry`` when present.

    OSMnx stores a curved street as a Shapely ``LineString`` on the edge. Using it
    makes the map look like a map; falling back to a straight line between
    endpoints is fine for a simplified graph.
    """
    segments: List[List[Tuple[float, float]]] = []
    for u, v, data in graph.edges(data=True):
        geometry = data.get("geometry")
        coordinates = getattr(geometry, "coords", None)
        if coordinates is not None:
            segments.append([(float(x), float(y)) for x, y in coordinates])
        else:
            segments.append([_node_xy(graph, u), _node_xy(graph, v)])
    return segments


def plot_graph(
    graph: nx.Graph,
    *,
    path: Optional[Path | str] = None,
    title: str = "Road network",
    figsize: Tuple[float, float] = (8.0, 8.0),
    node_size: float = 0.0,
    axes: Optional[Any] = None,
) -> Any:
    """Draw the road network as a grey basemap.

    ``node_size=0`` by default: on a few thousand intersections the dots swamp the
    streets and hide the structure you are trying to see.
    """
    plt = get_pyplot()
    from matplotlib.collections import LineCollection

    if axes is None:
        figure, axes = plt.subplots(figsize=figsize)
    else:
        figure = axes.get_figure()

    axes.add_collection(
        LineCollection(
            _edge_segments(graph), colors=PALETTE["road"], linewidths=0.6, zorder=1
        )
    )
    xs = [float(data["x"]) for _, data in graph.nodes(data=True) if "x" in data]
    ys = [float(data["y"]) for _, data in graph.nodes(data=True) if "y" in data]
    if not xs:
        raise DataError("Graph has no plottable coordinates")

    if node_size > 0:
        axes.scatter(xs, ys, s=node_size, c=PALETTE["muted"], zorder=2, linewidths=0)

    pad_x = (max(xs) - min(xs)) * 0.03 or 1e-4
    pad_y = (max(ys) - min(ys)) * 0.03 or 1e-4
    axes.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    axes.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    axes.set_title(title)
    axes.set_xlabel("longitude")
    axes.set_ylabel("latitude")
    axes.grid(False)
    _equal_aspect_geographic(axes, ys)

    save_figure(figure, path)
    return figure


def _scatter_nodes(
    axes: Any,
    instance: RoutingInstance,
    *,
    label_offset: float = 0.0004,
) -> None:
    """Draw instance nodes with role-specific markers and index labels."""
    latitudes = np.array([coordinate[0] for coordinate in instance.coords], dtype=float)
    longitudes = np.array([coordinate[1] for coordinate in instance.coords], dtype=float)

    vehicles = set(instance.vehicle_indices)
    incidents = set(instance.incident_indices)
    plain = [
        index
        for index in range(instance.n)
        if index != instance.depot and index not in vehicles and index not in incidents
    ]

    if plain:
        axes.scatter(
            longitudes[plain],
            latitudes[plain],
            s=70,
            c=PALETTE["classical"],
            edgecolors="white",
            linewidths=1.2,
            zorder=4,
            label="stop",
        )
    if vehicles:
        order = sorted(vehicles)
        axes.scatter(
            longitudes[order],
            latitudes[order],
            s=130,
            marker="s",
            c=PALETTE["annealing"],
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
            label="ambulance base",
        )
    if incidents:
        order = sorted(incidents)
        axes.scatter(
            longitudes[order],
            latitudes[order],
            s=130,
            marker="X",
            c=PALETTE["quantum"],
            edgecolors="white",
            linewidths=1.0,
            zorder=5,
            label="incident",
        )
    if instance.depot not in vehicles and instance.depot not in incidents:
        axes.scatter(
            [longitudes[instance.depot]],
            [latitudes[instance.depot]],
            s=170,
            marker="*",
            c=PALETTE["highlight"],
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label="depot",
        )

    for index in range(instance.n):
        axes.annotate(
            str(index),
            (longitudes[index], latitudes[index] + label_offset),
            fontsize=8,
            ha="center",
            zorder=7,
        )


def plot_instance(
    instance: RoutingInstance,
    graph: Optional[nx.Graph] = None,
    *,
    path: Optional[Path | str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
) -> Any:
    """Show which nodes were sampled, and what role each one plays."""
    plt = get_pyplot()
    figure, axes = plt.subplots(figsize=figsize)

    if graph is not None:
        plot_graph(graph, axes=axes, title="", node_size=0.0)
    axes.set_title(title or f"Instance: {instance.n} nodes ({instance.weight})")

    _scatter_nodes(axes, instance)
    if graph is None:
        latitudes = [coordinate[0] for coordinate in instance.coords]
        axes.set_xlabel("longitude")
        axes.set_ylabel("latitude")
        _equal_aspect_geographic(axes, latitudes)

    axes.legend(loc="best", fontsize=8)
    annotate_source(axes, instance.describe())
    save_figure(figure, path)
    return figure


def _route_geometry(
    instance: RoutingInstance,
    graph: Optional[nx.Graph],
    route: Sequence[int],
    *,
    closed: bool,
) -> List[Tuple[float, float]]:
    """Expand an index route into a polyline of ``(lon, lat)`` points.

    Follows real road geometry when both the stored path and the graph are
    available, and degrades to straight lines otherwise -- with a warning, because
    the difference matters when reading the picture.
    """
    order = list(route)
    if closed and len(order) > 1 and order[0] != order[-1]:
        order.append(order[0])

    points: List[Tuple[float, float]] = []
    straight_legs = 0
    for start, end in zip(order[:-1], order[1:]):
        nodes: Iterable[Any] = instance.paths.get((start, end), ())
        if graph is not None and nodes:
            leg = [_node_xy(graph, node) for node in nodes]
        else:
            straight_legs += 1
            leg = [
                (instance.coords[start][1], instance.coords[start][0]),
                (instance.coords[end][1], instance.coords[end][0]),
            ]
        points.extend(leg if not points else leg[1:])

    if straight_legs:
        _LOG.warning(
            "%d of %d legs drawn as straight lines (no stored path or no graph); "
            "the picture understates the real detours",
            straight_legs,
            len(order) - 1,
        )
    return points


def plot_route(
    instance: RoutingInstance,
    route: Sequence[int],
    graph: Optional[nx.Graph] = None,
    *,
    path: Optional[Path | str] = None,
    title: Optional[str] = None,
    closed: bool = True,
    figsize: Tuple[float, float] = (8.0, 8.0),
    colour: Optional[str] = None,
    comparison: Optional[Sequence[int]] = None,
    comparison_label: str = "classical",
) -> Any:
    """Draw a decoded route on the network.

    Passing *comparison* overlays a second route (dashed) so a quantum answer can
    be compared against the classical optimum in one figure. When the two routes
    coincide the dashed line sits exactly on the solid one, which is itself the
    clearest possible way to show agreement.
    """
    plt = get_pyplot()
    figure, axes = plt.subplots(figsize=figsize)

    if graph is not None:
        plot_graph(graph, axes=axes, title="", node_size=0.0)

    if comparison is not None:
        reference = np.asarray(_route_geometry(instance, graph, comparison, closed=closed))
        if reference.size:
            axes.plot(
                reference[:, 0],
                reference[:, 1],
                color=PALETTE["classical"],
                linewidth=3.6,
                linestyle="--",
                alpha=0.85,
                zorder=2,
                label=f"{comparison_label} ({instance.tour_cost(comparison, closed=closed):.0f} {instance.unit})",
            )

    polyline = np.asarray(_route_geometry(instance, graph, route, closed=closed))
    if polyline.size:
        axes.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=colour or PALETTE["quantum"],
            linewidth=2.2,
            zorder=3,
            label=f"route ({instance.tour_cost(route, closed=closed):.0f} {instance.unit})",
        )

    _scatter_nodes(axes, instance)
    axes.set_title(
        title
        or "Route: " + " -> ".join(str(index) for index in route) + ("" if not closed else " -> " + str(route[0]))
    )
    if graph is None:
        axes.set_xlabel("longitude")
        axes.set_ylabel("latitude")
        _equal_aspect_geographic(axes, [coordinate[0] for coordinate in instance.coords])
    axes.legend(loc="best", fontsize=8)
    annotate_source(axes, f"{instance.describe()} | cost unit: {instance.unit}")
    save_figure(figure, path)
    return figure


def plot_node_path(
    graph: nx.Graph,
    route: Sequence[Any],
    *,
    path: Optional[Path | str] = None,
    title: Optional[str] = None,
    candidate_edges: Optional[Sequence[Tuple[Any, Any, float]]] = None,
    candidate_paths: Optional[Sequence[Tuple[Sequence[Any], float]]] = None,
    comparison: Optional[Sequence[Any]] = None,
    comparison_label: str = "classical",
    label: str = "route",
    unit: str = "",
    figsize: Tuple[float, float] = (8.0, 8.0),
    zoom: bool = True,
) -> Any:
    """Draw a route given as a sequence of *graph node ids*.

    Rung 1 decodes to real OSM node ids rather than instance indices, so it needs
    this rather than :func:`plot_route`.

    Passing *candidate_edges* (i.e. ``problem.edges``) shades the sub-network the
    QUBO was allowed to choose from. Passing *candidate_paths* additionally draws
    and labels every enumerated path with its calculated cost, while highlighting
    the cheapest candidate. That shading is worth the extra line of code:
    it is the difference between "the solver found this route" and "the solver
    found this route *out of these options*", and the second claim is the one a
    reader should be able to check. A binary variable per candidate segment is
    also the project's whole scaling story, visible at a glance.
    """
    plt = get_pyplot()
    from matplotlib.collections import LineCollection

    figure, axes = plt.subplots(figsize=figsize)
    plot_graph(graph, axes=axes, title="", node_size=0.0)

    if candidate_edges:
        segments = []
        for u, v, _cost in candidate_edges:
            try:
                segments.append([_node_xy(graph, u), _node_xy(graph, v)])
            except DataError:  # pragma: no cover - defensive
                continue
        if segments:
            axes.add_collection(
                LineCollection(
                    segments,
                    colors=PALETTE["highlight"],
                    linewidths=1.4,
                    alpha=0.55,
                    zorder=2,
                    label=f"candidate segments ({len(segments)} variables)",
                )
            )

    def polyline(nodes: Sequence[Any]) -> np.ndarray:
        return np.asarray([_node_xy(graph, node) for node in nodes], dtype=float)

    if candidate_paths:
        ranked = sorted(candidate_paths, key=lambda item: float(item[1]))
        for index, (candidate, cost) in enumerate(ranked, start=1):
            candidate_points = polyline(candidate)
            is_optimum = index == 1
            axes.plot(
                candidate_points[:, 0],
                candidate_points[:, 1],
                color=PALETTE["highlight"],
                linewidth=3.0 if is_optimum else 1.3,
                linestyle="-" if is_optimum else ":",
                alpha=0.95 if is_optimum else 0.45,
                zorder=2.5 if is_optimum else 2,
                label=(
                    f"candidate {index} (optimum: {cost:.2f} {unit})"
                    if is_optimum
                    else f"candidate {index}: {cost:.2f} {unit}"
                ),
            )

    if comparison:
        reference = polyline(comparison)
        axes.plot(
            reference[:, 0],
            reference[:, 1],
            color=PALETTE["classical"],
            linewidth=3.8,
            linestyle="--",
            alpha=0.85,
            zorder=3,
            label=comparison_label,
        )

    points = polyline(route)
    axes.plot(
        points[:, 0],
        points[:, 1],
        color=PALETTE["quantum"],
        linewidth=2.2,
        zorder=4,
        label=label,
    )
    axes.scatter(
        [points[0, 0], points[-1, 0]],
        [points[0, 1], points[-1, 1]],
        s=[150, 150],
        marker="o",
        c=[PALETTE["annealing"], PALETTE["quantum"]],
        edgecolors="white",
        linewidths=1.2,
        zorder=5,
    )
    axes.annotate(
        "origin", (points[0, 0], points[0, 1]), fontsize=8,
        textcoords="offset points", xytext=(6, 6), zorder=6,
    )
    axes.annotate(
        "incident", (points[-1, 0], points[-1, 1]), fontsize=8,
        textcoords="offset points", xytext=(6, 6), zorder=6,
    )

    if zoom:
        # The candidate sub-network occupies a small corner of the city graph;
        # showing the whole basemap makes the route a barely visible squiggle.
        all_points = [points]
        if comparison:
            all_points.append(polyline(comparison))
        stacked = np.vstack(all_points)
        pad_x = max((stacked[:, 0].max() - stacked[:, 0].min()) * 0.25, 2e-3)
        pad_y = max((stacked[:, 1].max() - stacked[:, 1].min()) * 0.25, 2e-3)
        axes.set_xlim(stacked[:, 0].min() - pad_x, stacked[:, 0].max() + pad_x)
        axes.set_ylim(stacked[:, 1].min() - pad_y, stacked[:, 1].max() + pad_y)

    suffix = f" {unit}" if unit else ""
    axes.set_title(title or f"Shortest path: {len(route)} nodes{suffix}")
    axes.legend(loc="best", fontsize=8)
    save_figure(figure, path)
    return figure


def plot_assignment(
    instance: RoutingInstance,
    assignments: Dict[int, int],
    graph: Optional[nx.Graph] = None,
    *,
    alternatives: Optional[Sequence[Tuple[str, Dict[int, int], str]]] = None,
    path: Optional[Path | str] = None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8.0, 8.0),
) -> Any:
    """Draw ambulance-to-incident assignments as arrows.

    *assignments* maps incident index to vehicle index, matching
    :attr:`~qroute.qubo.base.RouteSolution.assignments`. Arrow width scales with
    how many incidents a vehicle took, so an unbalanced solution is visible at a
    glance -- which is exactly what the balance penalty exists to prevent.
    """
    plt = get_pyplot()
    figure, axes = plt.subplots(figsize=figsize)

    if graph is not None:
        plot_graph(graph, axes=axes, title="", node_size=0.0)

    loads: Dict[int, int] = {}
    for vehicle in assignments.values():
        loads[vehicle] = loads.get(vehicle, 0) + 1

    assignment_sets = [("selected", assignments, PALETTE["quantum"])]
    if alternatives:
        assignment_sets.extend(alternatives)

    for assignment_label, assignment_map, assignment_color in assignment_sets:
        local_loads: Dict[int, int] = {}
        for vehicle in assignment_map.values():
            local_loads[vehicle] = local_loads.get(vehicle, 0) + 1
        for incident, vehicle in sorted(assignment_map.items()):
            polyline = np.asarray(
                _route_geometry(instance, graph, [vehicle, incident], closed=False)
            )
            if not polyline.size:
                continue
            axes.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=assignment_color,
                linewidth=1.2 + 0.9 * local_loads.get(vehicle, 1),
                alpha=0.9,
                zorder=3,
                label=assignment_label,
            )
            axes.annotate(
                "",
                xy=(polyline[-1, 0], polyline[-1, 1]),
                xytext=(polyline[max(len(polyline) - 2, 0), 0], polyline[max(len(polyline) - 2, 0), 1]),
                arrowprops={"arrowstyle": "-|>", "color": assignment_color, "lw": 1.4},
                zorder=4,
            )

    _scatter_nodes(axes, instance)
    response = [
        instance.cost_between(vehicle, incident) for incident, vehicle in assignments.items()
    ]
    summary = (
        f"max {max(response):.0f} {instance.unit}, mean {np.mean(response):.0f} {instance.unit}"
        if response
        else "no assignments"
    )
    axes.set_title(title or f"Dispatch: {summary}")
    if graph is None:
        axes.set_xlabel("longitude")
        axes.set_ylabel("latitude")
        _equal_aspect_geographic(axes, [coordinate[0] for coordinate in instance.coords])
    axes.legend(loc="best", fontsize=8)
    annotate_source(
        axes,
        "loads per vehicle: "
        + ", ".join(f"{vehicle}:{count}" for vehicle, count in sorted(loads.items())),
    )
    save_figure(figure, path)
    return figure
