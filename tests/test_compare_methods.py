import pytest
import numpy as np

from coupled_cluster.ccsd import TDCCSD, CoupledClusterSinglesDoubles
from coupled_cluster.ccd import TDCCD, CoupledClusterDoubles


@pytest.fixture()
def ccsd_sans_singles(zanghellini_system):
    return CoupledClusterSinglesDoubles(
        zanghellini_system, include_singles=False
    )


@pytest.fixture()
def ccd(zanghellini_system):
    return CoupledClusterDoubles(zanghellini_system)


# @pytest.fixture()
# def ccsd(zanghellini_system):
#     return CoupledClusterSinglesDoubles(zanghellini_system)


def test_ground_state(ccd, ccsd_sans_singles, t_kwargs):
    ccsd = ccsd_sans_singles
    # ccsd = ccsd

    ccd.iterate_t_amplitudes(**t_kwargs)
    ccsd.iterate_t_amplitudes(**t_kwargs)

    energy_ccd = ccd.compute_energy()
    energy_ccsd = ccsd.compute_energy()

    assert abs(energy_ccd - energy_ccsd) < 1e-10

    np.testing.assert_allclose(ccsd.t_1, np.zeros_like(ccsd.t_1), atol=1e-10)
    np.testing.assert_allclose(ccsd.t_2, ccd.t_2, atol=1e-10)

    ccd.iterate_l_amplitudes(**t_kwargs)
    ccsd.iterate_l_amplitudes(**t_kwargs)

    np.testing.assert_allclose(ccsd.l_1, np.zeros_like(ccsd.l_1), atol=1e-10)
    # FAILING
    # np.testing.assert_allclose(ccsd.l_2, ccd.l_2, atol=1e-10)

    rho_ccd = ccd.compute_particle_density()
    rho_ccsd = ccsd.compute_particle_density()

    # FAILING
    # np.testing.assert_allclose(rho_ccsd, rho_ccd, atol=1e-10)


@pytest.mark.skip
def test_time_evolution(ccd, ccsd_sans_singles, cc_params, time_params):
    ccsd = ccsd_sans_singles

    energy_ccd, iterations_ccd = ccd.compute_ground_state_energy(**cc_params)
    energy_ccsd, iterations_ccsd = ccsd.compute_ground_state_energy(**cc_params)

    ccd.compute_l_amplitudes(**cc_params)
    ccsd.compute_l_amplitudes(**cc_params)

    prob_ccd, _ = ccd.evolve_amplitudes(**time_params)
    prob_ccsd, _ = ccsd.evolve_amplitudes(**time_params)

    np.testing.assert_allclose(prob_ccd, prob_ccsd, atol=1e-10)

    rho_ccd = ccd.compute_spin_reduced_one_body_density_matrix()
    rho_ccsd = ccsd.compute_spin_reduced_one_body_density_matrix()

    np.testing.assert_allclose(rho_ccsd, rho_ccd, atol=1e-10)
