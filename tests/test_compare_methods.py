import pytest
import numpy as np

from coupled_cluster import CCD, CCSD, TDCCD, TDCCSD

from coupled_cluster.mix import AlphaMixer, DIIS


@pytest.fixture()
def ccsd_sans_singles(zanghellini_system):
    return CCSD(zanghellini_system, include_singles=False, mixer=AlphaMixer)


@pytest.fixture()
def ccd(zanghellini_system):
    return CCD(zanghellini_system, mixer=AlphaMixer)


def test_ground_state(ccd, ccsd_sans_singles, t_kwargs):
    ccsd = ccsd_sans_singles

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
    np.testing.assert_allclose(ccsd.l_2, ccd.l_2, atol=1e-10)

    rho_ccd = ccd.compute_particle_density()
    rho_ccsd = ccsd.compute_particle_density()

    np.testing.assert_allclose(rho_ccsd, rho_ccd, atol=1e-10)
