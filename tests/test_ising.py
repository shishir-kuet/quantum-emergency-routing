r"""QUBO <-> Ising conversion.

The mapping is :math:`x_i = (1 - z_i) / 2`, so :math:`x_i = 0` corresponds to
:math:`z_i = +1`. That orientation is chosen to match ``Z|0> = +|0>``, which is
what makes variable *i* map to qubit *i* with no extra bookkeeping. The tests
below check the round trip by *energy*, not by coefficient, so they stay valid
even if the internal storage changes.
"""

from __future__ import annotations

import numpy as np
import pytest

from qroute.exceptions import FormulationError
from qroute.qubo.ising import IsingModel, ising_to_qubo, qubo_to_ising
from qroute.qubo.matrix import QUBO, all_energies, enumerate_assignments


def spins_from_bits(x) -> np.ndarray:
    """``z = 1 - 2x``: x=0 -> z=+1, x=1 -> z=-1."""
    return 1 - 2 * np.asarray(x, dtype=int)


class TestOrientation:
    def test_zero_maps_to_spin_up(self) -> None:
        assert spins_from_bits([0, 1]).tolist() == [1, -1]

    def test_single_variable_field_sign(self) -> None:
        r"""``E = x`` becomes ``h = -1/2``, constant ``1/2``."""
        ising = qubo_to_ising(QUBO(np.array([[1.0]])))
        assert ising.h[0] == pytest.approx(-0.5)
        assert ising.constant == pytest.approx(0.5)
        # x=0 -> z=+1 -> E = -0.5 + 0.5 = 0; x=1 -> z=-1 -> E = 0.5 + 0.5 = 1.
        assert ising.energy([1]) == pytest.approx(0.0)
        assert ising.energy([-1]) == pytest.approx(1.0)


class TestEnergyPreservation:
    def test_tiny_qubo_energies_survive_the_map(self, tiny_qubo: QUBO) -> None:
        ising = qubo_to_ising(tiny_qubo)
        for x in enumerate_assignments(2, 0, 4):
            assert ising.energy(spins_from_bits(x)) == pytest.approx(tiny_qubo.energy(x))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_qubos_survive_the_map(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        n = 5
        matrix = rng.normal(size=(n, n))
        matrix = (matrix + matrix.T) / 2.0
        qubo = QUBO(matrix, offset=float(rng.normal()))
        ising = qubo_to_ising(qubo)
        for x in enumerate_assignments(n, 0, 1 << n):
            assert ising.energy(spins_from_bits(x)) == pytest.approx(qubo.energy(x))

    def test_full_round_trip_returns_the_same_energies(self, tiny_qubo: QUBO) -> None:
        recovered = ising_to_qubo(qubo_to_ising(tiny_qubo))
        assert all_energies(recovered) == pytest.approx(all_energies(tiny_qubo))

    def test_round_trip_recovers_the_coefficients(self, tiny_qubo: QUBO) -> None:
        recovered = ising_to_qubo(qubo_to_ising(tiny_qubo))
        assert recovered.Q == pytest.approx(tiny_qubo.Q)
        assert recovered.offset == pytest.approx(tiny_qubo.offset)

    def test_labels_are_carried_through(self, tiny_qubo: QUBO) -> None:
        assert qubo_to_ising(tiny_qubo).labels == ("a", "b")
        assert ising_to_qubo(qubo_to_ising(tiny_qubo)).labels == ("a", "b")

    def test_provenance_is_recorded(self, tiny_qubo: QUBO) -> None:
        assert ising_to_qubo(qubo_to_ising(tiny_qubo)).metadata.get("from_ising") is True


class TestCouplingStorage:
    def test_j_is_strictly_upper_triangular(self, tiny_qubo: QUBO) -> None:
        """One entry per pair, so there is no factor-of-two ambiguity."""
        ising = qubo_to_ising(tiny_qubo)
        assert np.allclose(np.tril(ising.J), 0.0)

    def test_couplings_view_is_keyed_by_ordered_pairs(self, tiny_qubo: QUBO) -> None:
        couplings = qubo_to_ising(tiny_qubo).couplings()
        assert set(couplings) == {(0, 1)}
        # q_01 = -2 for the tiny QUBO, and J = q/4.
        assert couplings[(0, 1)] == pytest.approx(-0.5)

    def test_lower_triangular_input_is_rejected(self) -> None:
        with pytest.raises(FormulationError):
            IsingModel(h=np.zeros(2), J=np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(FormulationError, match="incompatible"):
            IsingModel(h=np.zeros(3), J=np.zeros((2, 2)))

    def test_wrong_spin_count_is_rejected(self) -> None:
        ising = IsingModel(h=np.zeros(2), J=np.zeros((2, 2)))
        with pytest.raises(FormulationError, match="2 spin"):
            ising.energy([1, -1, 1])

    def test_describe_is_informative(self, tiny_qubo: QUBO) -> None:
        text = qubo_to_ising(tiny_qubo).describe()
        assert "n=2" in text and "couplings=1" in text


class TestFormulationIsingViews:
    """The three rungs must all expose a consistent Ising form."""

    def test_path_formulation_ising_matches_its_qubo(self, path_formulation) -> None:
        qubo = path_formulation.qubo()
        ising = path_formulation.ising()
        assert ising.n_spins == qubo.n_variables
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.integers(0, 2, size=qubo.n_variables)
            assert ising.energy(spins_from_bits(x)) == pytest.approx(qubo.energy(x))

    def test_tsp_formulation_ising_matches_its_qubo(self, tsp_formulation) -> None:
        qubo = tsp_formulation.qubo()
        ising = tsp_formulation.ising()
        rng = np.random.default_rng(1)
        for _ in range(20):
            x = rng.integers(0, 2, size=qubo.n_variables)
            assert ising.energy(spins_from_bits(x)) == pytest.approx(qubo.energy(x))

    def test_assignment_formulation_ising_matches_its_qubo(self, assignment_formulation) -> None:
        qubo = assignment_formulation.qubo()
        ising = assignment_formulation.ising()
        for x in enumerate_assignments(qubo.n_variables, 0, 1 << qubo.n_variables):
            assert ising.energy(spins_from_bits(x)) == pytest.approx(qubo.energy(x))
