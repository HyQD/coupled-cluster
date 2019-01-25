import pytest
import numpy as np

from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.ccsd import TDCCSD, CoupledClusterSinglesDoubles
from coupled_cluster.ccd import TDCCD, CoupledClusterDoubles


class LaserPulse:
    def __init__(self, laser_frequency=2, laser_strength=1):
        self.laser_frequency = laser_frequency
        self.laser_strength = laser_strength

    def __call__(self, t):
        return self.laser_strength * np.sin(self.laser_frequency * t)


@pytest.fixture
def zanghellini_system():
    n = 2
    l = 6
    length = 10
    num_grid_points = 400
    omega = 0.25
    laser_frequency = 8 * omega
    laser_strength = 1

    odho = OneDimensionalHarmonicOscillator(
        n, l, length, num_grid_points, omega=omega
    )
    odho.setup_system()
    laser = LaserField(
        LaserPulse(
            laser_frequency=laser_frequency, laser_strength=laser_strength
        )
    )
    odho.set_time_evolution_operator(laser)

    return odho


@pytest.fixture
def cc_params():
    theta = 0.6
    tol = 1e-4

    return {"theta": theta, "tol": tol}


@pytest.fixture
def time_params():
    t_start = 0
    t_end = 10
    num_timesteps = 1001

    return {"t_start": t_start, "t_end": t_end, "num_timesteps": num_timesteps}


@pytest.fixture()
def ccsd_sans_singles(zanghellini_system):
    return CoupledClusterSinglesDoubles(
        zanghellini_system, include_singles=False
    )


@pytest.fixture()
def ccd(zanghellini_system):
    return CoupledClusterDoubles(zanghellini_system)


@pytest.mark.skip
def test_ground_state(ccd, ccsd_sans_singles, cc_params):
    ccsd = ccsd_sans_singles

    energy_ccd, iterations_ccd = ccd.compute_ground_state_energy(**cc_params)
    energy_ccsd, iterations_ccsd = ccsd.compute_ground_state_energy(**cc_params)

    assert abs(energy_ccd - energy_ccsd) < 1e-10
    assert iterations_ccd == iterations_ccsd

    np.testing.assert_allclose(ccsd.t_1, np.zeros_like(ccsd.t_1), atol=1e-10)
    np.testing.assert_allclose(ccsd.t_2, ccd.t_2, atol=1e-10)

    ccd.compute_l_amplitudes(**cc_params)
    ccsd.compute_l_amplitudes(**cc_params)

    np.testing.assert_allclose(ccsd.l_1, np.zeros_like(ccsd.l_1), atol=1e-10)
    np.testing.assert_allclose(ccsd.l_2, ccd.l_2, atol=1e-10)

    rho_ccd = ccd.compute_spin_reduced_one_body_density_matrix()
    rho_ccsd = ccsd.compute_spin_reduced_one_body_density_matrix()

    np.testing.assert_allclose(rho_ccsd, rho_ccd, atol=1e-10)


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
