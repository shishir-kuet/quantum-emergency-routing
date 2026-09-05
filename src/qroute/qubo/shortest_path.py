r"""Rung 1 of the ladder: shortest path on the real road sub-network.

Why not use the collapsed instance graph
----------------------------------------
:class:`~qroute.data.instance.RoutingInstance` stores a dense matrix whose every
entry *is already* a shortest-path cost, so it satisfies the triangle inequality
and the shortest path from ``i`` to ``j`` is always the direct edge. Formulating
that as a QUBO would be busywork. So this rung drops down to the actual road
graph, where a route is a sequence of many short edges and the combinatorics are
real.

Encoding
--------
One binary variable per candidate road segment, :math:`x_e = 1` iff the route
uses segment :math:`e`. The objective is purely **linear**,

.. math::

    \sum_e c_e\, x_e

and the routing logic lives entirely in flow-conservation penalties, one per
node: outflow minus inflow must equal :math:`+1` at the source, :math:`-1` at
the target, and :math:`0` everywhere else.

.. math::

    A \sum_v \Bigl(
        \sum_{e \in \delta^{+}(v)} x_e - \sum_{e \in \delta^{-}(v)} x_e - b_v
    \Bigr)^2

Squaring those balance expressions is what generates the quadratic couplings, so
the *structure* of the road network -- not the cost function -- determines the
circuit's entangling pattern. That makes this rung a clean place to watch how
graph topology drives QAOA circuit depth.

Candidate segments
------------------
A whole city's worth of edges will not fit on a simulator, so
:func:`build_path_problem` restricts the variable set to the union of the *k*
shortest simple paths between source and target (Yen's algorithm). Two useful
consequences: the set is small and connected, and path 1 is the Dijkstra
answer. This provides a candidate-space reference, but it does not make the
QUBO a full-network optimization: routes outside the candidate union are
unrepresentable.

A caveat worth stating plainly: flow conservation is satisfied by a
source-to-target path *plus any number of disjoint cycles*. Positive edge costs
make cycles unprofitable, so the ground state is clean, but a sampled bitstring
from a shallow QAOA circuit may well contain one. :meth:`ShortestPathFormulation.decode`
therefore rejects them explicitly instead of quietly reporting a shorter path
than was actually encoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np

from ..config import Config
from ..exceptions import FormulationError
from ..logging_utils import get_logger
from .base import BitsLike, Formulation, RouteSolution, register
from .matrix import QUBO, QUBOBuilder

__all__ = [
    "PathProblem",
    "build_path_problem",
    "build_local_path_problem",
    "ShortestPathFormulation",
]

_LOG = get_logger(__name__)

#: Guard on register size. 20 qubits is a comfortable statevector simulation;
#: 24 needs ~256 MB per state vector and QAOA holds several.
MAX_VARIABLES = 40


# ---------------------------------------------------------------------------
# Problem extraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class PathProblem:
    """A small source-to-target sub-network, ready to encode.

    Attributes
    ----------
    edges:
        Candidate segments as ``(u, v, cost)`` triples, in variable order.
    nodes:
        Every node touched by :attr:`edges`, sorted for determinism.
    source, target:
        Endpoints, given as graph node ids (OSM ids on a real network).
    weight:
        Name of the edge attribute the costs came from -- so results can be
        reported in the right unit.
    candidate_paths:
        The paths whose union produced :attr:`edges`, cheapest first.
    """

    edges: Tuple[Tuple[int, int, float], ...]
    nodes: Tuple[int, ...]
    source: int
    target: int
    weight: str = "travel_time"
    candidate_paths: Tuple[Tuple[int, ...], ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def total_cost(self) -> float:
        """Sum of all candidate segment costs -- the penalty-weight bound."""
        return float(sum(cost for _, _, cost in self.edges))

    def path_cost(self, path: Sequence[int]) -> float:
        """Cost of a node sequence, using only candidate segments."""
        lookup = {(u, v): cost for u, v, cost in self.edges}
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            try:
                total += lookup[(int(u), int(v))]
            except KeyError as exc:
                raise FormulationError(
                    f"Segment {u} -> {v} is not a candidate in this problem"
                ) from exc
        return total

    def reference_cost(self) -> Optional[float]:
        """Cost of the cheapest candidate path, i.e. the Dijkstra optimum."""
        if not self.candidate_paths:
            return None
        return self.path_cost(self.candidate_paths[0])

    def describe(self) -> str:
        reference = self.reference_cost()
        text = (
            f"PathProblem({self.source} -> {self.target}: {self.n_edges} segments, "
            f"{self.n_nodes} nodes, {len(self.candidate_paths)} candidate path(s)"
        )
        if reference is not None:
            text += f", best {reference:.1f} {self.weight}"
        return text + ")"


def _to_simple_digraph(graph: nx.Graph, weight: str) -> nx.DiGraph:
    """Collapse a (possibly multi-, possibly undirected) graph to a DiGraph.

    OSMnx returns a ``MultiDiGraph`` because two intersections can be joined by
    several distinct ways. For a shortest-path QUBO only the cheapest of those
    parallel segments can ever be chosen, so keeping just the minimum-weight one
    loses nothing and saves a qubit per duplicate. Self-loops are dropped: they
    satisfy flow conservation trivially and would let free cycles into the
    solution space.
    """
    simple = nx.DiGraph()
    for node, data in graph.nodes(data=True):
        simple.add_node(node, **{key: data[key] for key in ("x", "y") if key in data})

    directed = graph.is_directed()
    for u, v, data in graph.edges(data=True):
        if u == v:
            continue
        try:
            cost = float(data[weight])
        except (KeyError, TypeError, ValueError) as exc:
            raise FormulationError(
                f"Edge {u} -> {v} has no usable '{weight}' attribute. Run the data "
                f"pipeline (qroute.data.graph_io.ensure_travel_times) first."
            ) from exc
        if cost < 0:
            raise FormulationError(f"Edge {u} -> {v} has negative cost {cost}")

        pairs = [(u, v)] if directed else [(u, v), (v, u)]
        for a, b in pairs:
            existing = simple.get_edge_data(a, b, default=None)
            if existing is None or cost < existing[weight]:
                simple.add_edge(a, b, **{weight: cost})
    return simple


def build_path_problem(
    graph: nx.Graph,
    source: int,
    target: int,
    *,
    weight: str = "travel_time",
    k_paths: int = 10,
    max_edges: int = MAX_VARIABLES,
) -> PathProblem:
    """Extract a QUBO-sized shortest-path problem between two graph nodes.

    Paths are enumerated cheapest-first and their edges unioned until either
    *k_paths* paths are collected or the next path would push the variable count
    past *max_edges*. The first path is always included, so the returned problem
    always contains the true optimum.

    Raises
    ------
    FormulationError
        If the endpoints coincide, are missing, or no route exists between them.
    """
    if source == target:
        raise FormulationError("source and target must be different nodes")
    for node in (source, target):
        if node not in graph:
            raise FormulationError(f"Node {node} is not in the graph")
    if k_paths < 1:
        raise FormulationError(f"k_paths must be at least 1, got {k_paths}")

    simple = _to_simple_digraph(graph, weight)
    if not nx.has_path(simple, source, target):
        raise FormulationError(
            f"No route from {source} to {target}. Restrict the graph to its "
            f"largest strongly connected component first."
        )

    edge_order: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    accepted: List[Tuple[int, ...]] = []

    generator = nx.shortest_simple_paths(simple, source, target, weight=weight)
    for path in generator:
        segments = list(zip(path[:-1], path[1:]))
        fresh = [pair for pair in segments if pair not in seen]
        if accepted and len(edge_order) + len(fresh) > max_edges:
            _LOG.debug(
                "Stopping path enumeration: path %d would need %d variable(s), "
                "over the %d limit",
                len(accepted) + 1,
                len(edge_order) + len(fresh),
                max_edges,
            )
            break
        for pair in fresh:
            seen.add(pair)
            edge_order.append(pair)
        accepted.append(tuple(int(node) for node in path))
        if len(accepted) >= k_paths:
            break

    if len(edge_order) > max_edges:
        raise FormulationError(
            f"Even the single shortest path from {source} to {target} needs "
            f"{len(edge_order)} variable(s), above the {max_edges} limit. Pick "
            f"closer endpoints or raise max_edges."
        )
    if len(accepted) < 2:
        _LOG.warning(
            "Only %d candidate path(s) fit within %d variable(s); the QUBO has "
            "little to choose between. Try a larger max_edges or closer endpoints.",
            len(accepted),
            max_edges,
        )

    edges = tuple(
        (int(u), int(v), float(simple[u][v][weight])) for u, v in edge_order
    )
    nodes = tuple(sorted({node for u, v, _ in edges for node in (u, v)}))

    problem = PathProblem(
        edges=edges,
        nodes=nodes,
        source=int(source),
        target=int(target),
        weight=weight,
        candidate_paths=tuple(accepted),
        metadata={"k_paths_requested": k_paths, "max_edges": max_edges},
    )
    _LOG.info("Built %s", problem.describe())
    return problem


def build_local_path_problem(
    graph: nx.Graph,
    source: int,
    target: int,
    *,
    weight: str = "travel_time",
    max_edges: int = MAX_VARIABLES,
) -> PathProblem:
    """Build a direct local edge-variable problem without path enumeration.

    The supplied graph is expected to be a small local subgraph. Every usable
    local edge becomes a QUBO variable; Dijkstra/Yen candidate-path generation
    is deliberately not used. Dijkstra may still be used later as a reference
    for validation and reporting.
    """
    if source == target:
        raise FormulationError("source and target must be different nodes")
    simple = _to_simple_digraph(graph, weight)
    if source not in simple or target not in simple or not nx.has_path(simple, source, target):
        raise FormulationError(f"No local route from {source} to {target}")

    edges = tuple(
        (int(u), int(v), float(data[weight]))
        for u, v, data in simple.edges(data=True)
    )
    if not edges:
        raise FormulationError("Local graph has no usable edges")
    if len(edges) > max_edges:
        raise FormulationError(
            f"Local graph has {len(edges)} edges, above the {max_edges}-variable limit"
        )

    reference_cost, reference_path = nx.single_source_dijkstra(
        simple, source, target=target, weight=weight
    )
    nodes = tuple(sorted({node for u, v, _ in edges for node in (u, v)}))
    return PathProblem(
        edges=edges,
        nodes=nodes,
        source=int(source),
        target=int(target),
        weight=weight,
        candidate_paths=(tuple(int(node) for node in reference_path),),
        metadata={
            "candidate_generation": "none; direct local edge variables",
            "local_edge_count": len(edges),
            "reference_cost": float(reference_cost),
        },
    )


# ---------------------------------------------------------------------------
# Formulation
# ---------------------------------------------------------------------------
@register
class ShortestPathFormulation(Formulation):
    """Edge-selection QUBO with flow-conservation penalties."""

    name = "shortest_path"
    description = "Single-vehicle fastest route between two intersections"

    def __init__(
        self,
        problem: PathProblem,
        *,
        penalty_scale: float = 2.0,
        normalise: bool = True,
        max_variables: int = MAX_VARIABLES,
    ) -> None:
        super().__init__(penalty_scale=penalty_scale, normalise=normalise)

        if problem.n_edges < 1:
            raise FormulationError("PathProblem has no candidate segments")
        if problem.n_edges > max_variables:
            raise FormulationError(
                f"This problem needs {problem.n_edges} qubits, above the "
                f"{max_variables}-qubit guard. Rebuild with a smaller max_edges."
            )

        self.problem = problem
        # (u, v) -> variable index. Parallel segments were already collapsed, so
        # this mapping is total and injective.
        self._index: Dict[Tuple[int, int], int] = {
            (u, v): k for k, (u, v, _) in enumerate(problem.edges)
        }

    # -- layout -------------------------------------------------------------
    @property
    def num_variables(self) -> int:
        return self.problem.n_edges

    def variable_index(self, u: int, v: int) -> int:
        """Variable index of segment ``u -> v``."""
        try:
            return self._index[(int(u), int(v))]
        except KeyError as exc:
            raise FormulationError(
                f"Segment {u} -> {v} is not a candidate in this problem"
            ) from exc

    def variable_labels(self) -> Tuple[str, ...]:
        return tuple(f"x[{u}->{v}]" for u, v, _ in self.problem.edges)

    def encode_path(self, path: Sequence[int]) -> np.ndarray:
        """Encode a node sequence into a bit array over the candidate segments.

        The inverse of :meth:`decode`. Used to score the Dijkstra answer on the
        QUBO: it must come out both feasible and at the ground-state energy, and
        if it doesn't, the encoding is wrong rather than the solver.
        """
        nodes = [int(node) for node in path]
        if len(nodes) < 2:
            raise FormulationError("A path needs at least two nodes")
        if nodes[0] != self.problem.source or nodes[-1] != self.problem.target:
            raise FormulationError(
                f"Path must run from {self.problem.source} to {self.problem.target}, "
                f"got {nodes[0]} to {nodes[-1]}"
            )
        x = np.zeros(self.num_variables, dtype=np.int8)
        for u, v in zip(nodes[:-1], nodes[1:]):
            x[self.variable_index(u, v)] = 1
        return x

    def _balance(self, node: int) -> int:
        """The required outflow minus inflow at *node*."""
        if node == self.problem.source:
            return 1
        if node == self.problem.target:
            return -1
        return 0

    # -- penalty ------------------------------------------------------------
    def penalty_weight(self) -> float:
        """``penalty_scale * (total candidate cost)``.

        Selecting *every* candidate segment is the most the linear objective can
        ever charge, so a penalty above that total means no amount of objective
        saving can buy a broken route. It is a loose bound, deliberately: a
        tighter one risks a ground state that cheats, and cheating is far worse
        than a slightly compressed energy landscape.
        """
        total = self.problem.total_cost()
        return self.penalty_scale * total if total > 0 else self.penalty_scale

    # -- encode -------------------------------------------------------------
    def _build(self) -> QUBO:
        builder = QUBOBuilder(self.num_variables, labels=self.variable_labels())

        # --- objective ---
        for index, (_, _, cost) in enumerate(self.problem.edges):
            builder.add_linear(index, float(cost))

        # --- flow conservation, one squared balance per node ---
        weight = self.penalty_weight()
        for node in self.problem.nodes:
            terms: Dict[int, float] = {}
            for u, v, _ in self.problem.edges:
                if u == node:
                    terms[self.variable_index(u, v)] = 1.0
                elif v == node:
                    terms[self.variable_index(u, v)] = -1.0
            if not terms:
                continue
            builder.add_penalty_equality(
                terms, target=self._balance(node), weight=weight
            )

        qubo = builder.build(
            metadata={
                "formulation": self.name,
                "source": self.problem.source,
                "target": self.problem.target,
                "n_segments": self.problem.n_edges,
                "n_nodes": self.problem.n_nodes,
                "penalty_weight": weight,
                "weight_attribute": self.problem.weight,
                "reference_cost": self.problem.reference_cost(),
            }
        )
        _LOG.debug("Built shortest-path QUBO: %s", qubo.describe())
        return qubo

    # -- decode -------------------------------------------------------------
    def decode(self, bits: BitsLike, *, qiskit_order: bool = True) -> RouteSolution:
        x = self.as_array(bits, qiskit_order=qiskit_order)
        energy, raw_energy = self.energies(x)

        selected = [
            (u, v, cost)
            for (u, v, cost), chosen in zip(self.problem.edges, x)
            if chosen
        ]
        violations: List[str] = []

        # --- flow conservation ---
        net: Dict[int, int] = {node: 0 for node in self.problem.nodes}
        for u, v, _ in selected:
            net[u] += 1
            net[v] -= 1
        for node in self.problem.nodes:
            required = self._balance(node)
            if net[node] != required:
                violations.append(
                    f"node {node} has net flow {net[node]:+d}, expected {required:+d}"
                )

        route: Optional[Tuple[int, ...]] = None
        objective: Optional[float] = None
        details: Dict[str, Any] = {
            "n_segments_selected": len(selected),
            "segments": [(u, v) for u, v, _ in selected],
        }

        if not violations:
            route, objective, walk_problem = self._walk(selected)
            if walk_problem is not None:
                violations.append(walk_problem)
                route, objective = None, None

        if not violations and objective is not None:
            reference = self.problem.reference_cost()
            if reference is not None:
                details["reference_cost"] = reference
                details["gap"] = objective - reference
                details["optimal"] = bool(np.isclose(objective, reference))

        return RouteSolution(
            formulation=self.name,
            bits=self.canonical_bits(x),
            energy=energy,
            raw_energy=raw_energy,
            feasible=not violations,
            violations=tuple(violations),
            objective=objective,
            route=route,
            details=details,
        )

    def _walk(
        self, selected: Iterable[Tuple[int, int, float]]
    ) -> Tuple[Optional[Tuple[int, ...]], Optional[float], Optional[str]]:
        """Trace the selected segments into one path, or explain why they aren't.

        Flow conservation is necessary but not sufficient: a path plus a disjoint
        cycle balances at every node. This walk consumes segments one at a time
        and insists that every selected segment ends up on the path, which rules
        cycles out.
        """
        out_edges: Dict[int, List[Tuple[int, float]]] = {}
        for u, v, cost in selected:
            out_edges.setdefault(u, []).append((v, cost))

        remaining = sum(len(values) for values in out_edges.values())
        node = self.problem.source
        path: List[int] = [node]
        total = 0.0

        while node != self.problem.target:
            options = out_edges.get(node, [])
            if len(options) > 1:
                return None, None, (
                    f"route branches at node {node} ({len(options)} outgoing "
                    f"segments selected)"
                )
            if not options:
                return None, None, f"route dead-ends at node {node}"
            next_node, cost = options.pop()
            total += cost
            remaining -= 1
            node = next_node
            path.append(node)
            if len(path) > self.num_variables + 1:
                return None, None, "route revisits nodes (cycle detected)"

        if remaining:
            return None, None, (
                f"{remaining} selected segment(s) are not on the source-to-target "
                f"route (disjoint cycle)"
            )
        return tuple(path), total, None

    # -- construction helpers -----------------------------------------------
    @classmethod
    def from_graph(
        cls,
        graph: nx.Graph,
        source: int,
        target: int,
        config: Optional[Config] = None,
        *,
        k_paths: int = 3,
        max_edges: int = MAX_VARIABLES,
        weight: Optional[str] = None,
        **overrides: Any,
    ) -> "ShortestPathFormulation":
        """Build straight from a road graph using settings from *config*."""
        problem = build_path_problem(
            graph,
            source,
            target,
            weight=weight or (config.instance.weight if config else "travel_time"),
            k_paths=k_paths,
            max_edges=max_edges,
        )
        options: Dict[str, Any] = {
            "penalty_scale": config.qubo.penalty_scale if config else 2.0,
            "normalise": config.qubo.normalise if config else True,
            "max_variables": max_edges,
        }
        options.update(overrides)
        return cls(problem, **options)

    @classmethod
    def from_config(
        cls, problem: PathProblem, config: Config, **overrides: Any
    ) -> "ShortestPathFormulation":
        options: Dict[str, Any] = {
            "penalty_scale": config.qubo.penalty_scale,
            "normalise": config.qubo.normalise,
        }
        options.update(overrides)
        return cls(problem, **options)
