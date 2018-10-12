import pytest
import numpy as np

from coupled_cluster.ccsd import CoupledClusterSinglesDoubles


def test_mbpt_enegy(tdho):
    tol = 1e-4

    cc_scheme = CoupledClusterSinglesDoubles(tdho, verbose=True)

    energy, _ = cc_scheme.compute_ground_state_energy(max_iterations=0, tol=tol)
    assert True


def test_ccsd_energy(tdho, ccsd_energy):
    tol = 1e-4

    cc_scheme = CoupledClusterSinglesDoubles(tdho, verbose=True)
    energy, _ = cc_scheme.compute_ground_state_energy(tol=tol)

    assert abs(energy - ccsd_energy) < tol


def test_lambda_amplitude_iterations(tdho):
    cc_scheme = CoupledClusterSinglesDoubles(tdho, verbose=True)

    energy, _ = cc_scheme.compute_ground_state_energy()
    cc_scheme.compute_lambda_amplitudes()
    assert True
