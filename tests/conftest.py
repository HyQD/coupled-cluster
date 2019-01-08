import pytest
import numpy as np
import numba

from quantum_systems import TwoDimensionalHarmonicOscillator, CustomSystem

l = 12  # Number of orbitals
n = 2  # Number of particles
n_large = 2
l_large = 5

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


@pytest.fixture
def tdho(_omega):
    _tdho = TwoDimensionalHarmonicOscillator(
        n, l, radius, num_grid_points, omega=_omega
    )
    _tdho.setup_system()
    return _tdho


@pytest.fixture(scope="session")
def large_system():
    n = n_large
    l = l_large

    cs = CustomSystem(n, l)
    cs.set_h(np.random.random((l, l)), add_spin=True)
    cs.set_u(
        np.random.random((l, l, l, l)), add_spin=True, anti_symmetrize=True
    )
    cs.construct_fock_matrix()

    return cs


@pytest.fixture(scope="session")
def large_t():
    n = n_large
    m = l_large - n

    t = np.random.random((m, m, n, n)).astype(np.complex128)
    t = anti_symmetrize_t(t, m, n)

    return t


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
