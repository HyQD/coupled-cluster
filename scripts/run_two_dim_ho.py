from quantum_systems import TwoDimensionalHarmonicOscillator
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles

import numpy as np

n = 2
l = 12
omega = 1

theta = 0.6
tol = 1e-4

tdho = TwoDimensionalHarmonicOscillator(n, l, omega=omega)

tdho.setup_system()

ccsd = CoupledClusterSinglesDoubles(tdho, verbose=True)

ccsd.compute_ground_state_energy(theta=theta, tol=tol)
ccsd.compute_lambda_amplitudes(theta=theta, tol=tol)
