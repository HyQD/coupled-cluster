import pytest

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.romp2.romp2 import ROMP2
from coupled_cluster.mix import DIIS


@pytest.fixture
def bh_groundstate_romp2():
    return -25.18727358121961


def test_romp2_groundstate_pyscf(bh_groundstate_romp2):

    molecule = "b 0.0 0.0 0.0;h 0.0 0.0 2.4"
    basis = "cc-pvdz"

    system = construct_pyscf_system_rhf(
        molecule,
        basis=basis,
        verbose=False,
        add_spin=False,
        anti_symmetrize=False,
    )

    romp2 = ROMP2(system, mixer=DIIS, verbose=True)
    romp2.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=1e-10,
        termination_tol=1e-10,
        tol_factor=1e-1,
    )

    energy_tol = 1e-10

    assert (
        abs((romp2.compute_energy().real) - bh_groundstate_romp2) < energy_tol
    )
