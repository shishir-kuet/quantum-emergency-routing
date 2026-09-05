"""Validation for the direct/local Rung 2 routing QUBO."""

import networkx as nx

from qroute.config import load_config
from qroute.pipeline import build_rung


def test_local_routing_uses_direct_edges_not_yen_candidates():
    graph = nx.DiGraph()
    graph.add_edge(0, 1, travel_time=2.0)
    graph.add_edge(1, 3, travel_time=2.0)
    graph.add_edge(0, 2, travel_time=1.0)
    graph.add_edge(2, 3, travel_time=10.0)
    graph.add_edge(1, 2, travel_time=1.0)

    formulation = build_rung(
        "local_routing",
        load_config(),
        graph,
        source=0,
        target=3,
        max_edges=8,
        normalise=False,
    )

    assert formulation.problem.metadata["candidate_generation"] == (
        "none; direct local edge variables"
    )
    assert formulation.problem.n_edges == graph.number_of_edges()
    assert formulation.num_variables == graph.number_of_edges()
    assert formulation.problem.reference_cost() == 4.0
