import pytest
import numpy as np
import numba

from quantum_systems import TwoDimensionalHarmonicOscillator, CustomSystem

l = 12  # Number of orbitals
n = 2  # Number of particles
n_large = 6
n_larger = 12
l_large = 20

radius = 4
num_grid_points = 101


@numba.njit(cache=True)
def anti_symmetrize_t(t, m, n):
    for a in range(m):
        for b in range(a, m):
            for i in range(n):
                for j in range(i, n):
                    t[a, b, j, i] = -t[a, b, i, j]
                    t[b, a, i, j] = -t[a, b, i, j]
                    t[b, a, j, i] = t[a, b, i, j]

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

    cs = CustomSystem(n, l)
    cs.set_h(np.random.random((l, l)), add_spin=True)
    cs.set_u(
        np.random.random((l, l, l, l)), add_spin=True, anti_symmetrize=True
    )
    cs.construct_fock_matrix()

    t = np.random.random((m, m, n, n)).astype(np.complex128)
    t = anti_symmetrize_t(t, m, n)

    l = np.random.random((n, n, m, m)).astype(np.complex128)
    l = anti_symmetrize_t(l, n, m)

    return t, l, cs


@pytest.fixture(scope="session")
def large_system_ccsd(_n_large):
    n = _n_large
    l = l_large
    m = l - n

    cs = CustomSystem(n, l)
    cs.set_h(np.random.random((l, l)), add_spin=True)
    cs.set_u(
        np.random.random((l, l, l, l)), add_spin=True, anti_symmetrize=True
    )
    cs.construct_fock_matrix()

    t_1 = np.random.random((m, n)).astype(np.complex128)
    t_2 = np.random.random((m, m, n, n)).astype(np.complex128)
    t_2 = anti_symmetrize_t(t_2, m, n)

    l_1 = np.random.random((n, m)).astype(np.complex128)
    l_2 = np.random.random((n, n, m, m)).astype(np.complex128)
    l_2 = anti_symmetrize_t(l_2, n, m)

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
