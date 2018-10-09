from quantum_systems import TwoDimensionalHarmonicOscillator
from coupled_cluster.ccd import CoupledClusterDoubles

import numpy as np

n = 2
l = 42
omega = 1

theta = 0.1
tol = 1e-4

tdho = TwoDimensionalHarmonicOscillator(n, l, omega=omega)

tdho.setup_system()

ccd = CoupledClusterDoubles(tdho, verbose=True)

ccd.compute_ground_state_energy(theta=theta, tol=tol)
