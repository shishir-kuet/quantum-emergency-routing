r"""The QUBO container, its bit-order conventions, and the builder.

Every convention this project relies on is pinned here, because a sign or
endianness error in this module produces results that look plausible and are
wrong: the optimiser converges, the circuit runs, the reported route is simply
not the route that was encoded.
"""

from __future__ import annotations

import numpy as np
import pytest

from qroute.exceptions import FormulationError
from qroute.qubo.matrix import (
    QUBO,
    QUBOBuilder,
    all_energies,
    array_to_bitstring,
    bitstring_to_array,
    enumerate_assignments,
)


# ---------------------------------------------------------------------------
# Bit order
# ---------------------------------------------------------------------------
class TestBitOrder:
    def test_qiskit_order_puts_variable_zero_on_the_right(self) -> None:
        # Qiskit prints qubit 0 last. "100" therefore means q2=1, q1=0, q0=0.
        assert bitstring_to_array("100", 3).tolist() == [0, 0, 1]

    def test_plain_order_reads_left_to_right(self) -> None:
        assert bitstring_to_array("100", 3, qiskit_order=False).tolist() == [1, 0, 0]

    @pytest.mark.parametrize("bits", ["0", "01", "1011", "110010"])
    def test_round_trip_is_the_identity(self, bits: str) -> None:
        array = bitstring_to_array(bits, len(bits))
        assert array_to_bitstring(array) == bits

    def test_spaces_from_multiple_registers_are_stripped(self) -> None:
        assert bitstring_to_array("10 1", 3).tolist() == [1, 0, 1]

    def test_wrong_length_is_rejected(self) -> None:
        with pytest.raises(FormulationError, match="3 variable"):
            bitstring_to_array("1010", 3)

    def test_non_binary_characters_are_rejected(self) -> None:
        with pytest.raises(FormulationError, match="non-binary"):
            bitstring_to_array("1x0", 3)


class TestEnumeration:
    def test_variable_zero_is_the_least_significant_bit(self) -> None:
        # This is the whole reason a statevector probability array can be dotted
        # straight into an energy array with no reindexing.
        assert enumerate_assignments(2, 0, 4).tolist() == [[0, 0], [1, 0], [0, 1], [1, 1]]

    def test_chunking_matches_a_single_pass(self) -> None:
        whole = enumerate_assignments(4, 0, 16)
        halves = np.vstack(
            [enumerate_assignments(4, 0, 7), enumerate_assignments(4, 7, 16)]
        )
        assert np.array_equal(whole, halves)

    def test_row_index_equals_the_integer_it_encodes(self) -> None:
        rows = enumerate_assignments(5, 0, 32)
        powers = 1 << np.arange(5)
        assert (rows @ powers).tolist() == list(range(32))


# ---------------------------------------------------------------------------
# Energy convention
# ---------------------------------------------------------------------------
class TestEnergyConvention:
    def test_hand_computed_energies(self, tiny_qubo: QUBO, expected_tiny_energies) -> None:
        r"""E(x) = x_0 + 2 x_1 - 2 x_0 x_1 + 0.5, with the off-diagonal doubled."""
        assert tiny_qubo.energy([0, 0]) == pytest.approx(0.5)
        assert tiny_qubo.energy([1, 0]) == pytest.approx(1.5)
        assert tiny_qubo.energy([0, 1]) == pytest.approx(2.5)
        assert tiny_qubo.energy([1, 1]) == pytest.approx(1.5)
        assert all_energies(tiny_qubo) == pytest.approx(expected_tiny_energies)

    def test_batch_matches_single(self, tiny_qubo: QUBO) -> None:
        rows = enumerate_assignments(2, 0, 4)
        batch = tiny_qubo.energies(rows)
        singles = [tiny_qubo.energy(row) for row in rows]
        assert batch == pytest.approx(singles)

    def test_quadratic_view_reports_the_physical_coefficient(self, tiny_qubo: QUBO) -> None:
        # Q[0, 1] = -1 is stored symmetrically, so x0*x1 is multiplied by -2.
        assert tiny_qubo.quadratic() == {(0, 1): pytest.approx(-2.0)}

    def test_linear_view_is_the_diagonal(self, tiny_qubo: QUBO) -> None:
        assert tiny_qubo.linear().tolist() == [1.0, 2.0]

    def test_energy_of_bitstring_uses_qiskit_order(self, tiny_qubo: QUBO) -> None:
        # "01" is x0=1, x1=0 -> energy 1.5, not 2.5.
        assert tiny_qubo.energy_of_bitstring("01") == pytest.approx(1.5)
        assert tiny_qubo.energy_of_bitstring("01", qiskit_order=False) == pytest.approx(2.5)

    def test_wrong_length_assignment_is_rejected(self, tiny_qubo: QUBO) -> None:
        with pytest.raises(FormulationError, match="2 variable"):
            tiny_qubo.energy([1, 0, 1])


class TestValidation:
    def test_asymmetric_matrix_is_rejected(self) -> None:
        with pytest.raises(FormulationError, match="symmetric"):
            QUBO(np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_non_square_matrix_is_rejected(self) -> None:
        with pytest.raises(FormulationError, match="square"):
            QUBO(np.zeros((2, 3)))

    def test_label_count_must_match(self) -> None:
        with pytest.raises(FormulationError, match="label"):
            QUBO(np.zeros((2, 2)), labels=("only-one",))

    def test_default_labels_are_generated(self) -> None:
        assert QUBO(np.zeros((2, 2))).label(1) == "x[1]"


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
class TestTransforms:
    def test_normalisation_caps_the_largest_coefficient(self, tiny_qubo: QUBO) -> None:
        scaled = tiny_qubo.normalised()
        assert scaled.max_abs_coefficient() == pytest.approx(1.0)

    def test_normalisation_preserves_the_argmin(self, tiny_qubo: QUBO) -> None:
        """The point of normalisation: change the numbers, never the answer."""
        raw = all_energies(tiny_qubo)
        scaled = all_energies(tiny_qubo.normalised())
        assert int(np.argmin(raw)) == int(np.argmin(scaled))
        # Energies stay proportional, offset included.
        factor = tiny_qubo.normalised().metadata["scale_applied"]
        assert scaled == pytest.approx(raw * factor)

    def test_scale_factor_is_recorded_and_compounds(self, tiny_qubo: QUBO) -> None:
        twice = tiny_qubo.scaled(2.0).scaled(3.0)
        assert twice.metadata["scale_applied"] == pytest.approx(6.0)

    def test_scaling_by_zero_is_refused(self, tiny_qubo: QUBO) -> None:
        with pytest.raises(FormulationError, match="zero"):
            tiny_qubo.scaled(0.0)

    def test_density_counts_realised_couplings(self) -> None:
        matrix = np.zeros((4, 4))
        matrix[0, 1] = matrix[1, 0] = 1.0
        # 1 of the 6 possible pairs.
        assert QUBO(matrix).density() == pytest.approx(1.0 / 6.0)

    def test_all_energies_refuses_an_oversized_problem(self) -> None:
        big = QUBO(np.zeros((30, 30)))
        with pytest.raises(FormulationError, match="limit"):
            all_energies(big, max_variables=20)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class TestQUBOBuilder:
    def test_symmetry_is_maintained_automatically(self) -> None:
        qubo = QUBOBuilder(2).add_quadratic(0, 1, 3.0).build()
        assert qubo.Q[0, 1] == pytest.approx(qubo.Q[1, 0])
        assert qubo.quadratic() == {(0, 1): pytest.approx(3.0)}

    def test_terms_accumulate(self) -> None:
        """Objective and penalty contributions must sum, not overwrite."""
        qubo = (
            QUBOBuilder(2)
            .add_linear(0, 1.0)
            .add_linear(0, 2.0)
            .add_quadratic(0, 1, 1.0)
            .add_quadratic(1, 0, 1.0)
            .build()
        )
        assert qubo.linear()[0] == pytest.approx(3.0)
        assert qubo.quadratic()[(0, 1)] == pytest.approx(2.0)

    def test_one_hot_penalty_energies(self) -> None:
        r"""``w (sum(x) - 1)^2`` is zero on one-hot states and ``w`` elsewhere."""
        qubo = QUBOBuilder(3).add_penalty_equality([0, 1, 2], target=1, weight=10.0).build()
        assert qubo.energy([1, 0, 0]) == pytest.approx(0.0)
        assert qubo.energy([0, 1, 0]) == pytest.approx(0.0)
        assert qubo.energy([0, 0, 0]) == pytest.approx(10.0)
        assert qubo.energy([1, 1, 0]) == pytest.approx(10.0)
        assert qubo.energy([1, 1, 1]) == pytest.approx(40.0)

    def test_penalty_with_target_two(self) -> None:
        qubo = QUBOBuilder(3).add_penalty_equality([0, 1, 2], target=2, weight=1.0).build()
        assert qubo.energy([1, 1, 0]) == pytest.approx(0.0)
        assert qubo.energy([1, 0, 0]) == pytest.approx(1.0)
        assert qubo.energy([0, 0, 0]) == pytest.approx(4.0)

    def test_weighted_penalty_terms(self) -> None:
        r"""``(2 x_0 + x_1 - 1)^2`` -- coefficients, not just membership."""
        qubo = (
            QUBOBuilder(2)
            .add_penalty_equality({0: 2.0, 1: 1.0}, target=1, weight=1.0)
            .build()
        )
        assert qubo.energy([0, 1]) == pytest.approx(0.0)
        assert qubo.energy([1, 0]) == pytest.approx(1.0)
        assert qubo.energy([0, 0]) == pytest.approx(1.0)
        assert qubo.energy([1, 1]) == pytest.approx(4.0)

    def test_index_bounds_are_checked(self) -> None:
        with pytest.raises(FormulationError, match="out of range"):
            QUBOBuilder(2).add_linear(5, 1.0)

    def test_self_coupling_is_rejected_or_folded(self) -> None:
        r"""``x_i^2 = x_i`` for binaries, so a self-coupling is a linear term."""
        builder = QUBOBuilder(2)
        try:
            builder.add_quadratic(0, 0, 4.0)
        except FormulationError:
            return  # Refusing is also a defensible contract.
        assert builder.build().linear()[0] == pytest.approx(4.0)

    def test_zero_variables_rejected(self) -> None:
        with pytest.raises(FormulationError, match="at least one variable"):
            QUBOBuilder(0)

    def test_metadata_reaches_the_built_qubo(self) -> None:
        qubo = QUBOBuilder(1).add_linear(0, 1.0).build({"formulation": "test"})
        assert qubo.metadata["formulation"] == "test"
