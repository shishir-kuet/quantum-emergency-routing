"""Carve small, quantum-scale routing instances out of a full road network.

Why this module exists
----------------------
A Dhanmondi-sized drivable graph has a few hundred nodes. QAOA formulations of
routing need one qubit per (location, position) pair, so a 5-location tour
already costs 25 qubits and a 6-location tour costs 36. The gap between "the
real network" and "what fits on a simulator" is bridged here: we choose a
handful of well-separated, genuinely important intersections, then collapse the
road network between them into a dense cost matrix whose entries are true
shortest-path costs on the real graph.

That two-level structure is what keeps the project honest. The quantum solver
optimises over a tiny complete graph, but every edge weight in it is a real
Dhaka travel time, and every abstract edge expands back into a concrete
turn-by-turn path (:attr:`RoutingInstance.paths`) that can be drawn on a map.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from ..config import Config
from ..exceptions import DataError
from ..logging_utils import get_logger
from ..rng import make_rng
from .graph_io import largest_strongly_connected_subgraph, validate_edge_weight

__all__ = [
    "RoutingInstance",
    "haversine_m",
    "select_nodes",
    "build_cost_matrix",
    "sample_instance",
    "build_tour_instance",
    "build_dispatch_instance",
    "save_instance",
    "load_instance",
]

_LOG = get_logger(__name__)

_EARTH_RADIUS_M = 6_371_008.8


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_M * math.asin(math.sqrt(min(1.0, a)))


def _node_coords(graph: nx.Graph, node: Any) -> Tuple[float, float]:
    """Return ``(latitude, longitude)`` for *node*.

    OSMnx stores longitude in ``x`` and latitude in ``y``. Values may come back
    as strings from a plain NetworkX GraphML read, hence the explicit casts.
    """
    data = graph.nodes[node]
    try:
        lon = float(data["x"])
        lat = float(data["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataError(
            f"Node {node!r} has no usable 'x'/'y' coordinates; cannot build an "
            f"instance from this graph."
        ) from exc
    return (lat, lon)


def _extent_m(coords: Sequence[Tuple[float, float]]) -> float:
    """Characteristic length scale (metres) of a set of lat/lon points.

    Uses the geometric mean of the bounding-box side lengths, which behaves
    better than the diagonal for elongated study areas.
    """
    if len(coords) < 2:
        return 0.0
    lats = [lat for lat, _ in coords]
    lons = [lon for _, lon in coords]
    height = haversine_m(min(lats), lons[0], max(lats), lons[0])
    width = haversine_m(lats[0], min(lons), lats[0], max(lons))
    if height <= 0 or width <= 0:
        return max(height, width)
    return math.sqrt(height * width)


# ---------------------------------------------------------------------------
# The instance object
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RoutingInstance:
    """A small complete-graph routing problem distilled from a road network.

    Attributes
    ----------
    node_ids:
        Original graph node identifiers. Index ``i`` in every matrix below
        refers to ``node_ids[i]``.
    coords:
        ``(latitude, longitude)`` per index, for plotting.
    cost:
        Dense ``(n, n)`` matrix of shortest-path costs in the units of
        :attr:`weight` (seconds for ``travel_time``, metres for ``length``).
        The diagonal is zero.
    weight:
        Which edge attribute the costs were computed from.
    paths:
        ``{(i, j): (node, node, ...)}`` -- the full node sequence realising the
        shortest path, so an abstract tour can be expanded back onto the map.
    depot:
        Index treated as the start/end of a tour.
    vehicle_indices:
        Indices acting as ambulance bases (dispatch formulation).
    incident_indices:
        Indices acting as incident sites (dispatch formulation).
    """

    node_ids: Tuple[Any, ...]
    coords: Tuple[Tuple[float, float], ...]
    cost: np.ndarray
    weight: str
    paths: Dict[Tuple[int, int], Tuple[Any, ...]] = field(default_factory=dict)
    depot: int = 0
    vehicle_indices: Tuple[int, ...] = ()
    incident_indices: Tuple[int, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.node_ids)
        if n < 2:
            raise DataError("A routing instance needs at least 2 nodes")
        if len(self.coords) != n:
            raise DataError("coords length does not match node_ids length")
        if self.cost.shape != (n, n):
            raise DataError(
                f"cost matrix shape {self.cost.shape} does not match {n} nodes"
            )
        if not 0 <= self.depot < n:
            raise DataError(f"depot index {self.depot} out of range for {n} nodes")
        for label, indices in (
            ("vehicle_indices", self.vehicle_indices),
            ("incident_indices", self.incident_indices),
        ):
            for index in indices:
                if not 0 <= index < n:
                    raise DataError(f"{label} contains out-of-range index {index}")

    # -- basic accessors ----------------------------------------------------
    @property
    def n(self) -> int:
        """Number of locations."""
        return len(self.node_ids)

    @property
    def unit(self) -> str:
        """Human-readable unit of the cost matrix."""
        return "s" if self.weight == "travel_time" else "m"

    def cost_between(self, i: int, j: int) -> float:
        return float(self.cost[i, j])

    def path_nodes(self, i: int, j: int) -> Tuple[Any, ...]:
        """Node sequence for the shortest path from index *i* to index *j*."""
        if i == j:
            return (self.node_ids[i],)
        try:
            return self.paths[(i, j)]
        except KeyError as exc:
            raise DataError(f"No stored path for pair ({i}, {j})") from exc

    # -- derived views ------------------------------------------------------
    def is_symmetric(self, tolerance: float = 1e-6) -> bool:
        """Whether the cost matrix is (numerically) symmetric.

        Real road networks are *not* symmetric -- one-way streets mean the trip
        back is often longer. The TSP QUBO in this project assumes symmetry, so
        it calls :meth:`symmetrised` first and records the distortion.
        """
        return bool(np.allclose(self.cost, self.cost.T, atol=tolerance, rtol=0.0))

    def asymmetry(self) -> float:
        """Largest relative gap between ``cost[i, j]`` and ``cost[j, i]``.

        Returned as a fraction, so ``0.18`` means the worst pair differs by 18%.
        Useful to report honestly alongside any symmetric-TSP result.
        """
        upper = self.cost
        lower = self.cost.T
        denominator = np.maximum(np.maximum(upper, lower), 1e-12)
        gaps = np.abs(upper - lower) / denominator
        np.fill_diagonal(gaps, 0.0)
        return float(gaps.max()) if gaps.size else 0.0

    def symmetrised(self) -> "RoutingInstance":
        """Return a copy whose cost matrix is ``(C + C.T) / 2``."""
        symmetric = (self.cost + self.cost.T) / 2.0
        np.fill_diagonal(symmetric, 0.0)
        metadata = dict(self.metadata)
        metadata["symmetrised"] = True
        metadata["asymmetry_before"] = self.asymmetry()
        return RoutingInstance(
            node_ids=self.node_ids,
            coords=self.coords,
            cost=symmetric,
            weight=self.weight,
            paths=dict(self.paths),
            depot=self.depot,
            vehicle_indices=self.vehicle_indices,
            incident_indices=self.incident_indices,
            metadata=metadata,
        )

    def tour_cost(self, order: Sequence[int], *, closed: bool = True) -> float:
        """Total cost of visiting *order*, returning to the start if *closed*."""
        if len(order) < 2:
            return 0.0
        total = sum(
            self.cost_between(order[k], order[k + 1]) for k in range(len(order) - 1)
        )
        if closed:
            total += self.cost_between(order[-1], order[0])
        return float(total)

    def describe(self) -> str:
        offdiag = self.cost[~np.eye(self.n, dtype=bool)]
        return (
            f"RoutingInstance(n={self.n}, weight={self.weight}, "
            f"cost {offdiag.min():.1f}-{offdiag.max():.1f} {self.unit}, "
            f"mean {offdiag.mean():.1f} {self.unit}, "
            f"asymmetry {self.asymmetry() * 100:.1f}%)"
        )

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_ids": [str(node) for node in self.node_ids],
            "coords": [list(pair) for pair in self.coords],
            "cost": self.cost.tolist(),
            "weight": self.weight,
            "paths": {
                f"{i},{j}": [str(node) for node in path]
                for (i, j), path in sorted(self.paths.items())
            },
            "depot": self.depot,
            "vehicle_indices": list(self.vehicle_indices),
            "incident_indices": list(self.incident_indices),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingInstance":
        paths: Dict[Tuple[int, int], Tuple[Any, ...]] = {}
        for key, value in (data.get("paths") or {}).items():
            left, _, right = str(key).partition(",")
            paths[(int(left), int(right))] = tuple(value)
        return cls(
            node_ids=tuple(data["node_ids"]),
            coords=tuple((float(a), float(b)) for a, b in data["coords"]),
            cost=np.asarray(data["cost"], dtype=float),
            weight=str(data["weight"]),
            paths=paths,
            depot=int(data.get("depot", 0)),
            vehicle_indices=tuple(int(v) for v in data.get("vehicle_indices", ())),
            incident_indices=tuple(int(v) for v in data.get("incident_indices", ())),
            metadata=dict(data.get("metadata") or {}),
        )


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------
def _importance_scores(
    graph: nx.Graph, strategy: str, weight: str, rng: np.random.Generator
) -> Dict[Any, float]:
    """Score every node so that the most 'useful' locations sort first."""
    if strategy == "degree":
        return {node: float(degree) for node, degree in graph.degree()}

    if strategy == "betweenness":
        # Exact betweenness is O(nm); the k-sample estimator is plenty accurate
        # for ranking a few hundred nodes and runs in well under a second.
        n_nodes = graph.number_of_nodes()
        k = int(min(n_nodes, max(32, n_nodes // 4)))
        seed = int(rng.integers(0, 2**31 - 1))
        _LOG.debug("Estimating betweenness centrality with k=%d pivots", k)
        return nx.betweenness_centrality(graph, k=k, weight=weight, seed=seed)

    # "random" (and anything else) gets a uniform random ordering.
    return {node: float(rng.random()) for node in graph.nodes}


def _greedy_spread(
    ordered_nodes: Sequence[Any],
    coords: Dict[Any, Tuple[float, float]],
    n: int,
    min_separation_m: float,
) -> List[Any]:
    """Take nodes in the given order, skipping any that crowd an earlier pick."""
    chosen: List[Any] = []
    for node in ordered_nodes:
        lat, lon = coords[node]
        if all(
            haversine_m(lat, lon, *coords[picked]) >= min_separation_m
            for picked in chosen
        ):
            chosen.append(node)
            if len(chosen) == n:
                break
    return chosen


def _farthest_point_sample(
    nodes: Sequence[Any],
    coords: Dict[Any, Tuple[float, float]],
    n: int,
    rng: np.random.Generator,
) -> List[Any]:
    """Classic max-min sampling: each new pick is as far as possible from the rest."""
    start = nodes[int(rng.integers(0, len(nodes)))]
    chosen = [start]
    best_distance = {
        node: haversine_m(*coords[node], *coords[start]) for node in nodes
    }
    while len(chosen) < n:
        candidate = max(
            (node for node in nodes if node not in chosen),
            key=lambda node: best_distance[node],
            default=None,
        )
        if candidate is None:
            break
        chosen.append(candidate)
        for node in nodes:
            best_distance[node] = min(
                best_distance[node], haversine_m(*coords[node], *coords[candidate])
            )
    return chosen


def select_nodes(
    graph: nx.Graph,
    n: int,
    *,
    strategy: str = "betweenness",
    weight: str = "travel_time",
    seed: int = 42,
    separation_factor: float = 0.5,
) -> List[Any]:
    """Choose *n* well-separated nodes from *graph*.

    The naive approach -- take the *n* highest-centrality nodes -- fails badly
    on road networks: the top of the betweenness ranking is a run of adjacent
    nodes along one arterial road, so the "instance" degenerates into a nearly
    collinear set of points metres apart. We therefore walk the ranking in order
    but reject any candidate closer than ``separation_factor * L / sqrt(n)`` to
    an already-chosen node (``L`` being the study-area length scale), relaxing
    the threshold only if that makes *n* picks impossible.
    """
    if n < 2:
        raise DataError(f"Need at least 2 nodes, asked for {n}")
    nodes = list(graph.nodes)
    if len(nodes) < n:
        raise DataError(
            f"Graph has only {len(nodes)} node(s); cannot select {n}. Widen the "
            f"bounding box or lower instance.n_nodes."
        )

    coords = {node: _node_coords(graph, node) for node in nodes}
    rng = make_rng(seed, "select_nodes", strategy)

    if strategy == "farthest":
        chosen = _farthest_point_sample(nodes, coords, n, rng)
        if len(chosen) < n:  # pragma: no cover - only if coords collapse
            raise DataError("Farthest-point sampling could not find enough nodes")
        _LOG.info("Selected %d node(s) by farthest-point sampling", len(chosen))
        return chosen

    scores = _importance_scores(graph, strategy, weight, rng)
    ordered = sorted(nodes, key=lambda node: scores.get(node, 0.0), reverse=True)

    extent = _extent_m(list(coords.values()))
    threshold = separation_factor * extent / math.sqrt(n) if extent > 0 else 0.0

    chosen: List[Any] = []
    for attempt in range(6):
        chosen = _greedy_spread(ordered, coords, n, threshold)
        if len(chosen) == n:
            break
        previous = threshold
        threshold *= 0.6
        _LOG.debug(
            "Only %d/%d nodes satisfied a %.0f m separation; relaxing to %.0f m",
            len(chosen),
            n,
            previous,
            threshold,
        )
    if len(chosen) < n:
        _LOG.warning(
            "Spatial-separation filter exhausted; falling back to the raw "
            "'%s' ranking for the top %d node(s)",
            strategy,
            n,
        )
        chosen = list(ordered[:n])

    _LOG.info(
        "Selected %d node(s) by '%s' with a %.0f m minimum separation",
        len(chosen),
        strategy,
        threshold,
    )
    return chosen


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------
def build_cost_matrix(
    graph: nx.Graph, node_ids: Sequence[Any], weight: str = "travel_time"
) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[Any, ...]]]:
    """Compute all-pairs shortest-path costs and paths between *node_ids*.

    One Dijkstra sweep per source is run over the **full** graph, so the entries
    are true network costs rather than straight-line approximations.
    """
    n = len(node_ids)
    cost = np.zeros((n, n), dtype=float)
    paths: Dict[Tuple[int, int], Tuple[Any, ...]] = {}
    index_of = {node: i for i, node in enumerate(node_ids)}
    if len(index_of) != n:
        raise DataError("node_ids contains duplicates")

    for i, source in enumerate(node_ids):
        distances, routes = nx.single_source_dijkstra(graph, source, weight=weight)
        for j, target in enumerate(node_ids):
            if i == j:
                continue
            if target not in distances:
                raise DataError(
                    f"No path from node {source!r} to node {target!r}. The graph is "
                    f"not strongly connected over the selected nodes -- restrict it "
                    f"to the largest strongly connected component first."
                )
            cost[i, j] = float(distances[target])
            paths[(i, j)] = tuple(routes[target])

    return cost, paths


# ---------------------------------------------------------------------------
# Instance builders
# ---------------------------------------------------------------------------
def sample_instance(
    graph: nx.Graph,
    config: Config,
    *,
    n_nodes: Optional[int] = None,
    restrict_to_component: bool = True,
    tag: str = "instance",
) -> RoutingInstance:
    """Sample a :class:`RoutingInstance` of ``n_nodes`` locations from *graph*."""
    n = int(n_nodes if n_nodes is not None else config.instance.n_nodes)
    weight = config.instance.weight

    working = (
        largest_strongly_connected_subgraph(graph) if restrict_to_component else graph
    )
    validate_edge_weight(working, weight)

    node_ids = select_nodes(
        working,
        n,
        strategy=config.instance.sampling,
        weight=weight,
        seed=config.project.seed,
    )
    cost, paths = build_cost_matrix(working, node_ids, weight=weight)
    coords = tuple(_node_coords(working, node) for node in node_ids)

    instance = RoutingInstance(
        node_ids=tuple(node_ids),
        coords=coords,
        cost=cost,
        weight=weight,
        paths=paths,
        depot=0,
        metadata={
            "tag": tag,
            "sampling": config.instance.sampling,
            "seed": config.project.seed,
            "source_nodes": working.number_of_nodes(),
            "source_edges": working.number_of_edges(),
        },
    )
    _LOG.info("Built %s", instance.describe())
    return instance


def build_tour_instance(graph: nx.Graph, config: Config) -> RoutingInstance:
    """Instance for the single-ambulance multi-stop tour (TSP) formulation.

    Index 0 is the ambulance base (depot); the remaining indices are stops that
    must all be visited exactly once before returning to base.
    """
    instance = sample_instance(graph, config, tag="tour")
    return RoutingInstance(
        node_ids=instance.node_ids,
        coords=instance.coords,
        cost=instance.cost,
        weight=instance.weight,
        paths=instance.paths,
        depot=0,
        vehicle_indices=(0,),
        incident_indices=tuple(range(1, instance.n)),
        metadata=instance.metadata,
    )


def build_dispatch_instance(graph: nx.Graph, config: Config) -> RoutingInstance:
    """Instance for the ambulance-to-incident assignment formulation.

    The first ``n_ambulances`` indices are bases, the next ``n_incidents`` are
    incident sites. Sampling them together (rather than independently) keeps
    them mutually well-separated.
    """
    n_vehicles = config.instance.n_ambulances
    n_incidents = config.instance.n_incidents
    total = n_vehicles + n_incidents
    if total < 2:
        raise DataError("Need at least 2 locations across ambulances and incidents")

    instance = sample_instance(graph, config, n_nodes=total, tag="dispatch")
    metadata = dict(instance.metadata)
    metadata.update({"n_ambulances": n_vehicles, "n_incidents": n_incidents})

    return RoutingInstance(
        node_ids=instance.node_ids,
        coords=instance.coords,
        cost=instance.cost,
        weight=instance.weight,
        paths=instance.paths,
        depot=0,
        vehicle_indices=tuple(range(n_vehicles)),
        incident_indices=tuple(range(n_vehicles, total)),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_instance(instance: RoutingInstance, path: Path | str) -> Path:
    """Write *instance* to JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(instance.to_dict(), handle, indent=2)
    _LOG.info("Saved instance (n=%d) to %s", instance.n, target)
    return target


def load_instance(path: Path | str) -> RoutingInstance:
    """Read an instance written by :func:`save_instance`."""
    source = Path(path)
    if not source.is_file():
        raise DataError(f"Instance file not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    instance = RoutingInstance.from_dict(data)
    _LOG.info("Loaded instance (n=%d) from %s", instance.n, source)
    return instance
