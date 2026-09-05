"""The three rungs of the ladder: shortest path, tour, assignment.

Each rung gets the same four questions, because they are the ones that decide
whether a quantum result means anything:

1. Does ``encode -> decode`` return what went in?
2. Does a decoded feasible solution report the *same* cost the instance says it
   has? (An encoding can be self-consistent and still price routes wrongly.)
3. Is the QUBO ground state feasible -- i.e. is the penalty weight big enough
   that cheating never pays?
4. Does the QUBO optimum equal the true optimum found by an exact classical
   method? (A ground state can be feasible and still not be the best route if
   the objective terms are wrong.)
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qroute.exceptions import FormulationError
from qroute.qubo import available_formulations, build_formulation, get_formulation_class
from qroute.qubo.assignment import AssignmentFormulation
from qroute.qubo.matrix import all_energies, enumerate_assignments
from qroute.qubo.shortest_path import ShortestPathFormulation, build_path_problem
from qroute.qubo.tsp import TSPFormulation


def ground_state(formulation):
    """The exhaustive minimiser of the formulation's QUBO, as a bit array."""
    energies = all_energies(formulation.qubo())
    index = int(np.argmin(energies))
    n = formulation.num_variables
    return enumerate_assignments(n, index, index + 1)[0]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class TestRegistry:
    def test_all_three_rungs_are_registered(self) -> None:
        assert set(available_formulations()) >= {"shortest_path", "tsp", "assignment"}

    def test_lookup_returns_the_class(self) -> None:
        assert get_formulation_class("tsp") is TSPFormulation

    def test_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(FormulationError, match="Available"):
            get_formulation_class("travelling_salesbeing")

    def test_build_formulation_dispatches(self, tour_instance, config) -> None:
        built = build_formulation("tsp", tour_instance, config)
        assert isinstance(built, TSPFormulation)


# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------
class TestSharedContract:
    @pytest.fixture(params=["path_formulation", "tsp_formulation", "assignment_formulation"])
    def formulation(self, request):
        return request.getfixturevalue(request.param)

    def test_label_count_matches_variable_count(self, formulation) -> None:
        assert len(formulation.variable_labels()) == formulation.num_variables

    def test_labels_are_unique(self, formulation) -> None:
        labels = formulation.variable_labels()
        assert len(set(labels)) == len(labels)

    def test_qubo_size_matches_declared_variables(self, formulation) -> None:
        assert formulation.qubo().n_variables == formulation.num_variables

    def test_qubo_is_cached(self, formulation) -> None:
        """Rebuilding on every access would silently multiply the cost of a sweep."""
        assert formulation.qubo() is formulation.qubo()
        assert formulation.raw_qubo() is formulation.raw_qubo()

    def test_normalisation_scales_energies_uniformly(self, formulation) -> None:
        scale = formulation.energy_scale()
        assert scale == pytest.approx(1.0 / formulation.raw_qubo().max_abs_coefficient())
        rng = np.random.default_rng(0)
        for _ in range(10):
            x = rng.integers(0, 2, size=formulation.num_variables)
            energy, raw = formulation.energies(x)
            assert energy == pytest.approx(raw * scale)

    def test_normalised_qubo_has_unit_peak(self, formulation) -> None:
        assert formulation.qubo().max_abs_coefficient() == pytest.approx(1.0)

    def test_penalty_weight_is_positive(self, formulation) -> None:
        assert formulation.penalty_weight() > 0

    def test_ground_state_is_feasible(self, formulation) -> None:
        """The single most important property in the whole project.

        If the cheapest QUBO state breaks a constraint, then QAOA is being asked
        to minimise a function whose optimum is not a route, and every number
        downstream is measuring the wrong thing.
        """
        solution = formulation.decode(ground_state(formulation), qiskit_order=False)
        assert solution.feasible, (
            f"{formulation.name}: QUBO ground state violates "
            f"{solution.violations} -- raise penalty_scale"
        )

    def test_bit_order_flip_is_consistent(self, formulation) -> None:
        state = ground_state(formulation)
        by_array = formulation.decode(state, qiskit_order=False)
        by_string = formulation.decode(by_array.bits, qiskit_order=True)
        assert by_string.bits == by_array.bits
        assert by_string.energy == pytest.approx(by_array.energy)
        assert by_string.objective == pytest.approx(by_array.objective)

    def test_decode_rejects_the_wrong_length(self, formulation) -> None:
        with pytest.raises(FormulationError):
            formulation.decode([0] * (formulation.num_variables + 1), qiskit_order=False)

    def test_decode_rejects_non_binary_values(self, formulation) -> None:
        bad = [2] + [0] * (formulation.num_variables - 1)
        with pytest.raises(FormulationError, match="0s and 1s"):
            formulation.decode(bad, qiskit_order=False)

    def test_infeasible_solutions_have_no_objective(self, formulation) -> None:
        """A cost of ``None`` is the honest answer, not a zero to average in."""
        empty = np.zeros(formulation.num_variables, dtype=int)
        solution = formulation.decode(empty, qiskit_order=False)
        if not solution.feasible:
            assert solution.objective is None
            assert solution.violations

    def test_summary_reports_the_shape(self, formulation) -> None:
        text = formulation.summary()
        assert formulation.name in text
        assert "qubits" in text


# ---------------------------------------------------------------------------
# Rung 1: shortest path
# ---------------------------------------------------------------------------
class TestShortestPath:
    def test_variable_count_is_the_candidate_segment_count(self, path_formulation) -> None:
        assert path_formulation.num_variables == path_formulation.problem.n_edges

    def test_dijkstra_path_round_trips(self, path_formulation) -> None:
        best = path_formulation.problem.candidate_paths[0]
        solution = path_formulation.decode(
            path_formulation.encode_path(best), qiskit_order=False
        )
        assert solution.feasible
        assert solution.route == tuple(best)
        assert solution.objective == pytest.approx(path_formulation.problem.path_cost(best))

    def test_ground_state_is_the_dijkstra_optimum(self, path_formulation) -> None:
        reference = path_formulation.problem.reference_cost()
        solution = path_formulation.decode(ground_state(path_formulation), qiskit_order=False)
        assert solution.objective == pytest.approx(reference)
        assert solution.details.get("optimal") is True

    def test_route_is_graph_node_ids_not_variable_indices(self, path_formulation) -> None:
        """Rung 1 decodes to real graph nodes -- the plotting path depends on it."""
        solution = path_formulation.decode(ground_state(path_formulation), qiskit_order=False)
        assert solution.route[0] == path_formulation.problem.source
        assert solution.route[-1] == path_formulation.problem.target

    def test_selecting_nothing_breaks_flow_conservation(self, path_formulation) -> None:
        solution = path_formulation.decode(
            np.zeros(path_formulation.num_variables, dtype=int), qiskit_order=False
        )
        assert not solution.feasible
        assert any("net flow" in violation for violation in solution.violations)

    def test_selecting_everything_is_infeasible(self, path_formulation) -> None:
        solution = path_formulation.decode(
            np.ones(path_formulation.num_variables, dtype=int), qiskit_order=False
        )
        assert not solution.feasible

    def test_penalty_exceeds_the_total_candidate_cost(self, path_formulation) -> None:
        total = path_formulation.problem.total_cost()
        assert path_formulation.penalty_weight() >= total

    def test_endpoints_must_differ(self, grid_graph) -> None:
        with pytest.raises(FormulationError, match="different"):
            build_path_problem(grid_graph, 0, 0)

    def test_missing_node_is_reported(self, grid_graph) -> None:
        with pytest.raises(FormulationError, match="not in the graph"):
            build_path_problem(grid_graph, 0, 999)

    def test_first_candidate_path_is_always_included(self, grid_graph) -> None:
        """Even with a stingy budget, the true optimum must stay reachable."""
        problem = build_path_problem(grid_graph, 0, 15, k_paths=5, max_edges=7)
        assert problem.candidate_paths
        assert problem.n_edges <= 7 or len(problem.candidate_paths) == 1

    def test_candidate_paths_are_cheapest_first(self, path_formulation) -> None:
        problem = path_formulation.problem
        costs = [problem.path_cost(path) for path in problem.candidate_paths]
        assert costs == sorted(costs)

    def test_multi_edges_are_collapsed(self, grid_graph) -> None:
        """A MultiDiGraph must not produce one variable per parallel way."""
        problem = build_path_problem(grid_graph, 0, 5, k_paths=2)
        pairs = [(u, v) for u, v, _ in problem.edges]
        assert len(set(pairs)) == len(pairs)

    def test_from_graph_builds_directly(self, grid_graph, config) -> None:
        formulation = ShortestPathFormulation.from_graph(
            grid_graph, 0, 5, weight="travel_time", k_paths=2
        )
        assert formulation.num_variables >= 2


# ---------------------------------------------------------------------------
# Rung 2: tour
# ---------------------------------------------------------------------------
class TestTSP:
    def test_fixing_the_depot_saves_qubits(self, tour_instance, config) -> None:
        fixed = TSPFormulation.from_config(tour_instance, config, fix_depot=True)
        free = TSPFormulation.from_config(tour_instance, config, fix_depot=False)
        n = tour_instance.n
        assert fixed.num_variables == (n - 1) ** 2
        assert free.num_variables == n * n

    def test_route_round_trips(self, tsp_formulation) -> None:
        route = (0, 1, 2, 3)
        solution = tsp_formulation.decode(
            tsp_formulation.encode_route(route), qiskit_order=False
        )
        assert solution.feasible
        assert solution.route == route
        assert solution.objective == pytest.approx(
            tsp_formulation.instance.tour_cost(route, closed=True)
        )

    def test_every_permutation_round_trips(self, tsp_formulation) -> None:
        instance = tsp_formulation.instance
        for tail in itertools.permutations(range(1, instance.n)):
            route = (0,) + tail
            solution = tsp_formulation.decode(
                tsp_formulation.encode_route(route), qiskit_order=False
            )
            assert solution.feasible
            assert solution.objective == pytest.approx(instance.tour_cost(route, closed=True))

    def test_ground_state_matches_the_exhaustive_optimum(self, tsp_formulation) -> None:
        instance = tsp_formulation.instance
        best = min(
            ((0,) + tail for tail in itertools.permutations(range(1, instance.n))),
            key=lambda order: instance.tour_cost(order, closed=True),
        )
        expected = instance.tour_cost(best, closed=True)
        solution = tsp_formulation.decode(ground_state(tsp_formulation), qiskit_order=False)
        assert solution.objective == pytest.approx(expected)

    def test_depot_leads_the_decoded_route(self, tsp_formulation) -> None:
        solution = tsp_formulation.decode(ground_state(tsp_formulation), qiskit_order=False)
        assert solution.route[0] == tsp_formulation.depot

    def test_visiting_a_city_twice_is_infeasible(self, tsp_formulation) -> None:
        x = tsp_formulation.encode_route((0, 1, 2, 3)).copy()
        city, position = tsp_formulation.cities[0], tsp_formulation.positions[-1]
        x[tsp_formulation.variable_index(city, position)] = 1
        solution = tsp_formulation.decode(x, qiskit_order=False)
        assert not solution.feasible
        assert any("expected exactly 1" in violation for violation in solution.violations)

    def test_encode_rejects_a_partial_route(self, tsp_formulation) -> None:
        with pytest.raises(FormulationError, match="position"):
            tsp_formulation.encode_route((0, 1))

    def test_encode_rejects_a_repeated_city(self, tsp_formulation) -> None:
        with pytest.raises(FormulationError, match="exactly once"):
            tsp_formulation.encode_route((0, 1, 1, 2))

    def test_too_few_locations_is_refused(self, grid_graph, config) -> None:
        from qroute.data.instance import RoutingInstance, build_cost_matrix

        node_ids = (0, 5)
        cost, paths = build_cost_matrix(grid_graph, node_ids, weight="travel_time")
        pair = RoutingInstance(
            node_ids=node_ids,
            coords=((0.0, 0.0), (1.0, 1.0)),
            cost=cost,
            weight="travel_time",
            paths=paths,
        )
        with pytest.raises(FormulationError, match="at least 3"):
            TSPFormulation(pair)

    def test_qubit_guard_refuses_a_large_instance(self, tour_instance) -> None:
        with pytest.raises(FormulationError, match="qubit guard"):
            TSPFormulation(tour_instance, max_variables=4)

    def test_symmetrise_records_the_distortion(self, tour_instance, config) -> None:
        """The grid is deliberately asymmetric, so this must not be a no-op."""
        assert not tour_instance.is_symmetric()
        formulation = TSPFormulation.from_config(tour_instance, config, symmetrise=True)
        assert formulation.instance.is_symmetric()
        assert formulation.instance.metadata["asymmetry_before"] > 0

    def test_penalty_exceeds_the_worst_tour(self, tsp_formulation) -> None:
        instance = tsp_formulation.instance
        worst = max(
            instance.tour_cost((0,) + tail, closed=True)
            for tail in itertools.permutations(range(1, instance.n))
        )
        assert tsp_formulation.penalty_weight() > worst


# ---------------------------------------------------------------------------
# Rung 3: assignment
# ---------------------------------------------------------------------------
class TestAssignment:
    def test_variable_count_is_vehicles_times_incidents(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        assert formulation.num_variables == formulation.n_vehicles * formulation.n_incidents

    def test_auto_mode_picks_exactly_one_for_equal_counts(self, assignment_formulation) -> None:
        assert assignment_formulation.vehicle_mode == "exactly_one"

    def test_assignments_round_trip(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        vehicles, incidents = formulation.vehicles, formulation.incidents
        mapping = {incidents[0]: vehicles[0], incidents[1]: vehicles[1]}
        solution = formulation.decode(
            formulation.encode_assignments(mapping), qiskit_order=False
        )
        assert solution.feasible
        assert solution.assignments == mapping
        expected = sum(
            formulation.response_time(vehicle, incident)
            for incident, vehicle in mapping.items()
        )
        assert solution.objective == pytest.approx(expected)

    def test_ground_state_matches_the_exhaustive_optimum(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        vehicles, incidents = formulation.vehicles, formulation.incidents
        expected = min(
            sum(
                formulation.response_time(vehicle, incident)
                for vehicle, incident in zip(order, incidents)
            )
            for order in itertools.permutations(vehicles)
        )
        solution = formulation.decode(ground_state(formulation), qiskit_order=False)
        assert solution.objective == pytest.approx(expected)

    def test_double_booking_a_vehicle_is_infeasible(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        vehicle = formulation.vehicles[0]
        x = np.zeros(formulation.num_variables, dtype=int)
        for incident in formulation.incidents:
            x[formulation.variable_index(vehicle, incident)] = 1
        solution = formulation.decode(x, qiskit_order=False)
        assert not solution.feasible
        assert any("expected 1" in violation for violation in solution.violations)

    def test_unassigned_incident_is_infeasible(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        x = np.zeros(formulation.num_variables, dtype=int)
        x[formulation.variable_index(formulation.vehicles[0], formulation.incidents[0])] = 1
        solution = formulation.decode(x, qiskit_order=False)
        assert not solution.feasible

    def test_loads_are_reported_even_when_infeasible(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        x = np.zeros(formulation.num_variables, dtype=int)
        solution = formulation.decode(x, qiskit_order=False)
        assert set(solution.details["loads"]) == set(formulation.vehicles)
        assert solution.details["idle_vehicles"]

    def test_max_response_time_is_reported_not_optimised(self, assignment_formulation) -> None:
        solution = assignment_formulation.decode(
            ground_state(assignment_formulation), qiskit_order=False
        )
        times = solution.details["response_times"]
        assert solution.details["max_response_time"] == pytest.approx(max(times.values()))
        assert solution.objective == pytest.approx(sum(times.values()))

    def test_missing_vehicle_indices_is_refused(self, tour_instance) -> None:
        with pytest.raises(FormulationError, match="vehicle_indices"):
            AssignmentFormulation(tour_instance)

    def test_encode_requires_every_incident(self, assignment_formulation) -> None:
        formulation = assignment_formulation
        partial = {formulation.incidents[0]: formulation.vehicles[0]}
        with pytest.raises(FormulationError, match="No vehicle assigned"):
            formulation.encode_assignments(partial)

    def test_unknown_vehicle_is_reported(self, assignment_formulation) -> None:
        with pytest.raises(FormulationError, match="not a vehicle base"):
            assignment_formulation.variable_index(99, assignment_formulation.incidents[0])

    def test_exactly_one_requires_equal_counts(self, grid_graph) -> None:
        from qroute.data.instance import RoutingInstance, build_cost_matrix

        node_ids = (0, 15, 3)
        cost, paths = build_cost_matrix(grid_graph, node_ids, weight="travel_time")
        instance = RoutingInstance(
            node_ids=node_ids,
            coords=tuple((0.0, float(i)) for i in range(3)),
            cost=cost,
            weight="travel_time",
            paths=paths,
            vehicle_indices=(0, 1),
            incident_indices=(2,),
        )
        with pytest.raises(FormulationError, match="equal counts"):
            AssignmentFormulation(instance, vehicle_mode="exactly_one")

    def test_balance_mode_allows_spare_vehicles(self, grid_graph) -> None:
        from qroute.data.instance import RoutingInstance, build_cost_matrix

        node_ids = (0, 15, 3)
        cost, paths = build_cost_matrix(grid_graph, node_ids, weight="travel_time")
        instance = RoutingInstance(
            node_ids=node_ids,
            coords=tuple((0.0, float(i)) for i in range(3)),
            cost=cost,
            weight="travel_time",
            paths=paths,
            vehicle_indices=(0, 1),
            incident_indices=(2,),
        )
        formulation = AssignmentFormulation(instance)
        assert formulation.vehicle_mode == "balance"
        solution = formulation.decode(ground_state(formulation), qiskit_order=False)
        assert solution.feasible
        # One vehicle is left idle, and that is allowed in balance mode.
        assert len(solution.details["idle_vehicles"]) == 1
