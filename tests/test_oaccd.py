import pytest

from coupled_cluster.ccd.oaccd import OACCD
from coupled_cluster.mix import DIIS


@pytest.fixture
def he_groundstate_oaccd():
    return -2.887594831090973


def test_he_oaccd_groundstate(helium_system, he_groundstate_oaccd):
    try:
        from tdhf import HartreeFock
    except ImportError:
        pytest.skip("Cannot import module tdhf")

    hf = HartreeFock(helium_system, verbose=True)
    C = hf.scf(tolerance=1e-10)
    helium_system.change_basis(C)

    energy_tol = 1e-8

    oaccd = OACCD(helium_system, mixer=DIIS, verbose=True)
    oaccd.compute_ground_state(
        max_iterations=100,
        num_vecs=10,
        tol=1e-10,
        termination_tol=1e-12,
        tol_factor=1e-1,
    )

    assert abs(oaccd.compute_energy() - he_groundstate_oaccd) < energy_tol
