import pytest

from quantum_systems import construct_pyscf_system_rhf

from coupled_cluster.ccd.oaccd import OACCD
from coupled_cluster.mix import DIIS


@pytest.fixture
def he_groundstate_oaccd():
    return -2.887594831090973


def test_he_oaccd_groundstate(he_groundstate_oaccd):
    helium_system = construct_pyscf_system_rhf("he")

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

    man_energy = oaccd.compute_one_body_expectation_value(
        helium_system.h
    ) + 0.5 * oaccd.compute_two_body_expectation_value(
        helium_system.u, asym=True
    )

    assert abs(he_groundstate_oaccd - man_energy) < energy_tol
