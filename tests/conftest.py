import pytest
import numpy as np

from quantum_systems import TwoDimensionalHarmonicOscillator, CustomSystem

l = 12  # Number of orbitals
n = 2  # Number of particles
n_large = 6
n_larger = 12
l_large = 20

radius = 4
num_grid_points = 101


def get_random_doubles_amplitude(m, n):
    t = np.random.random((m, m, n, n)) + 1j * np.random.random((m, m, n, n))
    t = t + t.transpose(1, 0, 3, 2)
    t = t - t.transpose(0, 1, 3, 2)

    return t


@pytest.fixture(params=[0.5, 1.0])
def _omega(request):
    return request.param


@pytest.fixture(params=[n_large, n_larger], scope="session")
def _n_large(request):
    return request.param


@pytest.fixture
def tdho(_omega):
    _tdho = TwoDimensionalHarmonicOscillator(
        n, l, radius, num_grid_points, omega=_omega
    )
    _tdho.setup_system()
    return _tdho


@pytest.fixture(scope="session")
def large_system_ccd(_n_large):
    n = _n_large
    l = l_large
    m = l - n

    h = np.random.random((l, l)) + 1j * np.random.random((l, l))
    u = np.random.random((l, l, l, l)) + 1j * np.random.random((l, l, l, l))
    # Make u symmetric
    u = u + u.transpose(1, 0, 3, 2)

    cs = CustomSystem(n, l)
    cs.set_h(h, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.f = cs.construct_fock_matrix(cs.h, cs.u)

    t = get_random_doubles_amplitude(m, n)
    l = get_random_doubles_amplitude(n, m)

    return t, l, cs


@pytest.fixture(scope="session")
def large_system_ccsd(_n_large):
    n = _n_large
    l = l_large
    m = l - n

    h = np.random.random((l, l)) + 1j * np.random.random((l, l))
    u = np.random.random((l, l, l, l)) + 1j * np.random.random((l, l, l, l))
    # Make u symmetric
    u = u + u.transpose(1, 0, 3, 2)

    cs = CustomSystem(n, l)
    cs.set_h(h, add_spin=True)
    cs.set_u(u, add_spin=True, anti_symmetrize=True)
    cs.f = cs.construct_fock_matrix(cs.h, cs.u)

    t_1 = np.random.random((m, n)) + 1j * np.random.random((m, n))
    t_2 = get_random_doubles_amplitude(m, n)

    l_1 = np.random.random((n, m)) + 1j * np.random.random((n, m))
    l_2 = get_random_doubles_amplitude(n, m)

    return t_1, t_2, l_1, l_2, cs


@pytest.fixture
def ref_energy(_omega):
    if _omega == 0.5:
        return 1.8862268283560368
    elif _omega == 1.0:
        return 3.253314
    else:
        raise NotImplementedError(
            "We do not a have a test value for omega "
            + "= {0} yet".format(_omega)
        )


@pytest.fixture
def ccd_energy(_omega):
    if _omega == 0.5:
        return 1.7788892410077777
    elif _omega == 1.0:
        return 3.141829931728858
    else:
        raise NotImplementedError(
            "We do not a have a test value for omega "
            + "= {0} yet".format(_omega)
        )


@pytest.fixture
def ccsd_energy(_omega):
    if _omega == 0.5:
        return 1.681608
    elif _omega == 1.0:
        return 3.038599
    else:
        raise NotImplementedError(
            "We do not a have a test value for omega "
            + "= {0} yet".format(_omega)
        )
