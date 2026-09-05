"""Shared fixtures -- every one of them offline.

No test in this suite touches the network. The road-network fixture is a
synthetic lattice with the same *shape* of attributes OSMnx produces (``x``,
``y``, ``length``, ``travel_time``, a ``MultiDiGraph`` with reciprocal edges), so
the data layer is exercised end to end without a download. A handful of edges are
deliberately made asymmetric, because a perfectly symmetric grid would hide every
bug that only appears on a one-way street.

Deterministic on purpose: fixed seeds throughout, so a failure is reproducible.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np
import pytest

from qroute.config import BBox, Config, DataConfig, InstanceConfig, OutputConfig, QaoaConfig
from qroute.data.instance import build_cost_matrix, haversine_m
from qroute.qubo.matrix import QUBO

# A small patch of Dhanmondi. 23.75 deg N matters: the lat/lon-to-metres
# conversion is latitude-dependent and a bug there is invisible at the equator.
ORIGIN_LAT = 23.7450
ORIGIN_LON = 90.3750
SPACING_M = 220.0
GRID_SIDE = 4
SPEED_MPS = 20.0 * 1000.0 / 3600.0


def _offset(row: int, column: int) -> Tuple[float, float]:
    """Lat/lon of grid cell ``(row, column)``, spaced *SPACING_M* apart."""
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = 111_320.0 * math.cos(math.radians(ORIGIN_LAT))
    latitude = ORIGIN_LAT + (row * SPACING_M) / metres_per_degree_lat
    longitude = ORIGIN_LON + (column * SPACING_M) / metres_per_degree_lon
    return latitude, longitude


@pytest.fixture(scope="session")
def grid_graph() -> nx.MultiDiGraph:
    """A 4x4 lattice of intersections, strongly connected, OSMnx-shaped.

    Node ids are ``row * GRID_SIDE + column`` so a test can reason about
    geometry: node 0 is the south-west corner and node 15 the north-east.
    """
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = "epsg:4326"

    for row in range(GRID_SIDE):
        for column in range(GRID_SIDE):
            latitude, longitude = _offset(row, column)
            graph.add_node(row * GRID_SIDE + column, x=longitude, y=latitude)

    def connect(first: int, second: int, *, both: bool = True, penalty: float = 1.0) -> None:
        lat_a, lon_a = graph.nodes[first]["y"], graph.nodes[first]["x"]
        lat_b, lon_b = graph.nodes[second]["y"], graph.nodes[second]["x"]
        length = haversine_m(lat_a, lon_a, lat_b, lon_b)
        graph.add_edge(
            first,
            second,
            key=0,
            length=length,
            speed_kph=20.0,
            travel_time=length / SPEED_MPS,
        )
        if both:
            # The reverse direction is slower on a few edges, which is what makes
            # the cost matrix asymmetric -- exactly like a congested one-way pair.
            graph.add_edge(
                second,
                first,
                key=0,
                length=length,
                speed_kph=20.0 / penalty,
                travel_time=penalty * length / SPEED_MPS,
            )

    for row in range(GRID_SIDE):
        for column in range(GRID_SIDE):
            node = row * GRID_SIDE + column
            if column + 1 < GRID_SIDE:
                connect(node, node + 1, penalty=1.6 if row == 1 else 1.0)
            if row + 1 < GRID_SIDE:
                connect(node, node + GRID_SIDE, penalty=1.4 if column == 2 else 1.0)

    assert nx.is_strongly_connected(graph)
    return graph


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """A config whose every output directory lives under ``tmp_path``.

    Absolute paths on purpose: :meth:`Config.build_paths` resolves relative
    directories against the *project* root, which would scribble real files into
    the repository during a test run.
    """
    return Config(
        data=DataConfig(
            bbox=BBox(north=23.7530, south=23.7400, east=90.3830, west=90.3700),
            raw_dir=str(tmp_path / "raw"),
            processed_dir=str(tmp_path / "processed"),
            graph_filename="test_grid.graphml",
        ),
        instance=InstanceConfig(n_nodes=4, n_incidents=2, n_ambulances=2),
        qaoa=QaoaConfig(reps=1, maxiter=30, shots=512),
        output=OutputConfig(
            results_dir=str(tmp_path / "results"),
            figures_dir=str(tmp_path / "results" / "figures"),
        ),
    )


@pytest.fixture
def tour_instance(grid_graph: nx.MultiDiGraph, config: Config):
    """A 4-node tour instance built from the lattice's corners.

    Corners rather than sampled nodes: an explicit node list makes the expected
    optimal tour something a human can verify by looking at the grid.
    """
    from qroute.data.instance import RoutingInstance

    node_ids = (0, 3, 15, 12)
    cost, paths = build_cost_matrix(grid_graph, node_ids, weight="travel_time")
    coords = tuple(
        (float(grid_graph.nodes[node]["y"]), float(grid_graph.nodes[node]["x"]))
        for node in node_ids
    )
    return RoutingInstance(
        node_ids=node_ids,
        coords=coords,
        cost=cost,
        weight="travel_time",
        paths=paths,
        depot=0,
    )


@pytest.fixture
def dispatch_instance(grid_graph: nx.MultiDiGraph):
    """Two ambulance bases (indices 0-1) and two incidents (indices 2-3)."""
    from qroute.data.instance import RoutingInstance

    node_ids = (0, 15, 3, 12)
    cost, paths = build_cost_matrix(grid_graph, node_ids, weight="travel_time")
    coords = tuple(
        (float(grid_graph.nodes[node]["y"]), float(grid_graph.nodes[node]["x"]))
        for node in node_ids
    )
    return RoutingInstance(
        node_ids=node_ids,
        coords=coords,
        cost=cost,
        weight="travel_time",
        paths=paths,
        depot=0,
        vehicle_indices=(0, 1),
        incident_indices=(2, 3),
    )


@pytest.fixture
def path_problem(grid_graph: nx.MultiDiGraph):
    """A source-to-target candidate set across the diagonal of the lattice."""
    from qroute.qubo.shortest_path import build_path_problem

    return build_path_problem(grid_graph, 0, 15, weight="travel_time", k_paths=3)


@pytest.fixture
def tiny_qubo() -> QUBO:
    r"""A two-variable QUBO whose four energies are hand-computable.

    :math:`Q = \begin{pmatrix} 1 & -1 \\ -1 & 2 \end{pmatrix}`, offset 0.5, giving
    :math:`E(x) = x_0 + 2 x_1 - 2 x_0 x_1 + 0.5` (the off-diagonal contributes
    :math:`2 Q_{01}`), so the energies for
    :math:`x = 00, 10, 01, 11` are :math:`0.5, 1.5, 2.5, 1.5`.
    """
    matrix = np.array([[1.0, -1.0], [-1.0, 2.0]])
    return QUBO(matrix, offset=0.5, labels=("a", "b"))


@pytest.fixture
def expected_tiny_energies() -> np.ndarray:
    """Energies of :func:`tiny_qubo` in variable-0-is-LSB order."""
    return np.array([0.5, 1.5, 2.5, 1.5])


@pytest.fixture
def path_formulation(path_problem, config: Config):
    from qroute.qubo.shortest_path import ShortestPathFormulation

    return ShortestPathFormulation.from_config(path_problem, config)


@pytest.fixture
def tsp_formulation(tour_instance, config: Config):
    from qroute.qubo.tsp import TSPFormulation

    return TSPFormulation.from_config(tour_instance, config)


@pytest.fixture
def assignment_formulation(dispatch_instance, config: Config):
    from qroute.qubo.assignment import AssignmentFormulation

    return AssignmentFormulation.from_config(dispatch_instance, config)


@pytest.fixture
def small_config(config: Config) -> Config:
    """Config tuned for the fastest possible QAOA smoke test."""
    return config.with_overrides(
        qaoa=replace(config.qaoa, reps=1, maxiter=12, shots=256),
        instance=replace(config.instance, n_nodes=3),
    )


def graph_attributes(graph: nx.Graph) -> Dict[str, Any]:
    """Helper: collect the attribute keys present on nodes and edges."""
    node_keys: set = set()
    edge_keys: set = set()
    for _, data in graph.nodes(data=True):
        node_keys |= set(data)
    for _, _, data in graph.edges(data=True):
        edge_keys |= set(data)
    return {"nodes": sorted(node_keys), "edges": sorted(edge_keys)}
