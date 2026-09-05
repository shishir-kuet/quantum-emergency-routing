"""Rung 3 assignment-QUBO independence checks."""

from __future__ import annotations

import inspect
from unittest.mock import patch

import numpy as np

from qroute.classical import solve_assignment, solve_formulation_bruteforce, solve_formulation_annealing
from qroute.config import load_config
from qroute.data.instance import RoutingInstance
from qroute.qubo.assignment import AssignmentFormulation
from qroute.qubo.matrix import enumerate_assignments
from qroute.quantum import run_qaoa


def _assignment_formulation(costs: np.ndarray) -> AssignmentFormulation:
    instance = RoutingInstance(
        node_ids=(0, 1, 2, 3),
        coords=((0.0, 0.0),) * 4,
        cost=np.asarray(costs, dtype=float),
        weight="travel_time",
        vehicle_indices=(0, 1),
        incident_indices=(2, 3),
    )
    return AssignmentFormulation.from_config(
        instance,
        load_config(),
        vehicle_mode="exactly_one",
        normalise=False,
    )


def test_rung3_exact_ground_state_and_coefficient_response():
    """Assignment decisions follow QUBO coefficients, not a stored baseline."""
    first = _assignment_formulation(np.array([
        [0.0, 0.0, 10.0, 100.0],
        [0.0, 0.0, 90.0, 20.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    exact = solve_assignment(first, method="hungarian")
    brute = solve_formulation_bruteforce(first)
    assert exact.solution is not None and exact.solution.feasible
    assert brute.solution is not None and brute.solution.feasible
    assert exact.objective == brute.objective == 30.0
    assert brute.details["ground_state_feasible"] is True

    second = _assignment_formulation(np.array([
        [0.0, 0.0, 90.0, 20.0],
        [0.0, 0.0, 10.0, 100.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    changed = solve_assignment(second, method="hungarian")
    assert changed.solution is not None and changed.solution.feasible
    assert changed.objective == 30.0
    assert changed.solution.assignments != exact.solution.assignments


def test_rung3_optimizers_receive_formulation_not_classical_assignment():
    """SA and QAOA APIs expose no classical-assignment input channel."""
    annealing_parameters = inspect.signature(solve_formulation_annealing).parameters
    qaoa_parameters = inspect.signature(run_qaoa).parameters
    forbidden = {"assignment", "best_assignment", "dijkstra_assignment", "initial_state"}
    assert forbidden.isdisjoint(annealing_parameters)
    assert forbidden.isdisjoint(qaoa_parameters)


def test_rung3_qaoa_uses_assignment_qubo_samples():
    """QAOA's reported assignment is decoded from its measured samples."""
    formulation = _assignment_formulation(np.array([
        [0.0, 0.0, 10.0, 100.0],
        [0.0, 0.0, 90.0, 20.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    feasible_states = []
    exact_objective = None
    for state in enumerate_assignments(formulation.num_variables, 0, 1 << formulation.num_variables):
        decoded = formulation.decode(state, qiskit_order=False)
        if decoded.feasible:
            feasible_states.append((formulation.canonical_bits(state), decoded.objective))
            if exact_objective is None or decoded.objective < exact_objective:
                exact_objective = decoded.objective
    optimal_states = [state for state in feasible_states if np.isclose(state[1], exact_objective)]
    print(f"Exact feasible states: {len(feasible_states)}")
    print(f"Exact optimal states: {optimal_states}")
    print(f"Exact optimum: {exact_objective}")
    with patch(
        "qroute.classical.dijkstra.dijkstra_path",
        side_effect=AssertionError("Dijkstra must not solve the assignment"),
    ):
        annealing = solve_formulation_annealing(
            formulation, seed=7, n_restarts=3, n_sweeps=30
        )
        result = run_qaoa(
            formulation,
            load_config(),
            reps=1,
            maxiter=2,
            shots=20,
            expectation_mode="shots",
            max_qubits=8,
        )
    assert annealing.solution is not None
    assert result.counts
    feasible_shots = 0
    optimal_shots = 0
    total_shots = sum(result.counts.values())
    for bits, count in result.counts.items():
        decoded_state = formulation.decode(bits, qiskit_order=True)
        if decoded_state.feasible:
            feasible_shots += count
            if np.isclose(decoded_state.objective, exact_objective):
                optimal_shots += count
    print(f"QAOA total shots: {total_shots}")
    print(f"QAOA unique states: {len(result.counts)}")
    print(f"QAOA feasible shots: {feasible_shots}")
    print(f"QAOA optimal shots: {optimal_shots}")
    print(f"QAOA p(opt): {optimal_shots / total_shots if total_shots else 0.0}")
    print(f"QAOA best feasible objective: {result.objective}")
    assert result.best_bits is not None
    best_key = formulation.canonical_bits(result.best_bits)
    assert best_key in result.counts
    decoded = formulation.decode(result.best_bits, qiskit_order=False)
    assert decoded.feasible
    assert np.isclose(decoded.objective, result.objective)
