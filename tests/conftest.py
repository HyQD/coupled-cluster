import pytest
import numpy as np
import os

from quantum_systems import (
    # TwoDimensionalHarmonicOscillator,
    RandomBasisSet,
    GeneralOrbitalSystem,
    ODQD,
    construct_pyscf_system_rhf,
)
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

l = 12  # Number of orbitals
n = 2  # Number of particles
n_large = 4
n_larger = 22
l_large = 20

radius = 4
num_grid_points = 101


def get_random_doubles_amplitude(m, n):
    t = RandomBasisSet.get_random_elements((m, m, n, n), np)
    t = t + t.transpose(1, 0, 3, 2)
    t = t - t.transpose(0, 1, 3, 2)

    return t


# @pytest.fixture(params=[0.5, 1.0])
# def _omega(request):
#     return request.param


@pytest.fixture(params=[n_large, n_larger], scope="session")
def _n_large(request):
    return request.param


# @pytest.fixture
# def tdho(_omega):
#     _tdho = TwoDimensionalHarmonicOscillator(
#         n, l, radius, num_grid_points, omega=_omega
#     )
#     _tdho.setup_system()
#     return _tdho


@pytest.fixture(scope="session")
def large_system_ccs(_n_large):
    n = _n_large
    l = l_large * 2
    m = l - n
    dim = 3

    rs = GeneralOrbitalSystem(n, RandomBasisSet(l, dim, includes_spin=True))
    rs.f = rs.construct_fock_matrix(rs.h, rs.u)

    t_1 = RandomBasisSet.get_random_elements((m, n), np)
    l_1 = RandomBasisSet.get_random_elements((n, m), np)

    return t_1, l_1, rs


@pytest.fixture(scope="session")
def large_system_ccd(_n_large):
    n = _n_large
    l = l_large * 2
    m = l - n
    dim = 1

    rs = GeneralOrbitalSystem(n, RandomBasisSet(l, dim, includes_spin=True))
    rs.f = rs.construct_fock_matrix(rs.h, rs.u)

    t = get_random_doubles_amplitude(m, n)
    l = get_random_doubles_amplitude(n, m)

    return t, l, rs


@pytest.fixture(scope="session")
def large_system_ccsd(_n_large):
    n = _n_large
    l = l_large * 2
    m = l - n
    dim = 2

    rs = GeneralOrbitalSystem(n, RandomBasisSet(l, dim, includes_spin=True))
    rs.f = rs.construct_fock_matrix(rs.h, rs.u)

    t_1 = RandomBasisSet.get_random_elements((m, n), np)
    t_2 = get_random_doubles_amplitude(m, n)

    l_1 = RandomBasisSet.get_random_elements((n, m), np)
    l_2 = get_random_doubles_amplitude(n, m)

    return t_1, t_2, l_1, l_2, rs


# @pytest.fixture
# def ref_energy(_omega):
#     if _omega == 0.5:
#         return 1.8862268283560368
#     elif _omega == 1.0:
#         return 3.253314
#     else:
#         raise NotImplementedError(
#             "We do not have a test value for omega "
#             + "= {0} yet".format(_omega)
#         )


# @pytest.fixture
# def ccd_energy(_omega):
#     if _omega == 0.5:
#         return 1.7788892410077777
#     elif _omega == 1.0:
#         return 3.141829931728858
#     else:
#         raise NotImplementedError(
#             "We do not a have a test value for omega "
#             + "= {0} yet".format(_omega)
#         )


# @pytest.fixture
# def tdho_ccd_hf_energy(_omega):
#     if _omega == 0.5:
#         return 1.681979
#     elif _omega == 1.0:
#         return 3.039048
#     else:
#         raise NotImplementedError(
#             "We do not a have a test value for omega "
#             + "= {0} yet".format(_omega)
#         )


# @pytest.fixture
# def ccsd_energy(_omega):
#     if _omega == 0.5:
#         return 1.681608
#     elif _omega == 1.0:
#         return 3.038599
#     else:
#         raise NotImplementedError(
#             "We do not a have a test value for omega "
#             + "= {0} yet".format(_omega)
#         )


class LaserPulse:
    def __init__(self, laser_frequency=2, laser_strength=1):
        self.laser_frequency = laser_frequency
        self.laser_strength = laser_strength

    def __call__(self, t):
        return self.laser_strength * np.sin(self.laser_frequency * t)


@pytest.fixture
def zanghellini_system():
    n = 2
    l = 3
    length = 10
    num_grid_points = 400
    omega = 0.25
    laser_frequency = 8 * omega
    laser_strength = 1

    odho = GeneralOrbitalSystem(
        n,
        ODQD(
            l,
            length,
            num_grid_points,
            potential=ODQD.HOPotential(omega),
        ),
    )
    laser = DipoleFieldInteraction(
        LaserPulse(
            laser_frequency=laser_frequency, laser_strength=laser_strength
        )
    )
    odho.set_time_evolution_operator(laser)

    return odho


@pytest.fixture
def tdccd_zanghellini_ground_state_energy():
    return 1.1063


@pytest.fixture
def tdccd_zanghellini_ground_state_particle_density():
    filename = os.path.join(
        "tests", "dat", "tdccd_zanghellini_ground_state_particle_density.dat"
    )

    return np.loadtxt(filename, dtype=complex)


@pytest.fixture
def tdccd_zanghellini_psi_overlap():
    filename = os.path.join("tests", "dat", "tdccd_zanghellini_psi_overlap.dat")

    return np.loadtxt(filename)


@pytest.fixture
def tdccd_zanghellini_td_energies():
    filename = os.path.join("tests", "dat", "tdccd_zanghellini_td_energies.dat")

    return np.loadtxt(filename)


@pytest.fixture
def oatdccd_helium_td_energies():
    filename = os.path.join("tests", "dat", "oatdccd_helium_td_energies.dat")

    return np.loadtxt(filename)


@pytest.fixture
def oatdccd_helium_td_energies_imag():
    filename = os.path.join(
        "tests", "dat", "oatdccd_helium_td_energies_imag.dat"
    )

    return np.loadtxt(filename)


@pytest.fixture
def oatdccd_helium_dip_z():
    filename = os.path.join("tests", "dat", "oatdccd_helium_dip_z.dat")

    return np.loadtxt(filename)


@pytest.fixture
def oatdccd_helium_norm_t2():
    filename = os.path.join("tests", "dat", "oatdccd_helium_norm_t2.dat")

    return np.loadtxt(filename)


@pytest.fixture
def oatdccd_helium_norm_l2():
    filename = os.path.join("tests", "dat", "oatdccd_helium_norm_l2.dat")

    return np.loadtxt(filename)


@pytest.fixture
def t_kwargs():
    theta = 0.5
    tol = 1e-4

    return {"theta": theta, "tol": tol}


@pytest.fixture
def l_kwargs():
    theta = 0.8
    tol = 1e-4

    return {"theta": theta, "tol": tol}


@pytest.fixture
def time_params():
    t_start = 0
    t_end = 10
    num_timesteps = 1001

    return {"t_start": t_start, "t_end": t_end, "num_timesteps": num_timesteps}


@pytest.fixture
def oaccd_groundstate_helium_energy():
    return -2.8875947783360814
