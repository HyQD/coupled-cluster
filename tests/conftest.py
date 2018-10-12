import pytest
import numpy as np

from quantum_systems import TwoDimensionalHarmonicOscillator

l = 12  # Number of orbitals
n = 2  # Number of particles

radius = 4
num_grid_points = 101


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
