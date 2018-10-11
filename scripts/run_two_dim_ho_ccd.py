from quantum_systems import TwoDimensionalHarmonicOscillator
from coupled_cluster.ccd import CoupledClusterDoubles

# from coupled_cluster.schemes.ccd import CoupledClusterDoubles

import numpy as np

n = 2
l = 12
omega = 1
length = 4
num_grid_points = 101

theta = 0.1
tol = 1e-4

tdho = TwoDimensionalHarmonicOscillator(
    n, l, radius_length=length, num_grid_points=num_grid_points, omega=omega
)

tdho.setup_system()
h = tdho.h.copy().astype(float)
u = tdho.u.copy().astype(float)

ccd = CoupledClusterDoubles(tdho, verbose=True)
# ccd = CoupledClusterDoubles(h, u, n, verbose=True)
# np.testing.assert_allclose(
#    ccd.f,
#    tdho.f.copy().astype(float)
# )

ccd.compute_ground_state_energy(theta=theta, tol=tol)
