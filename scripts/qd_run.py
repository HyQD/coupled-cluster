import numpy as np
from quantum_systems import TwoDimensionalHarmonicOscillator

# from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from coupled_cluster.ccd import CoupledClusterDoubles
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 6
l = 30
omega = 1.0
theta = 0.1
tdho = TwoDimensionalHarmonicOscillator(n, l, 4, 101, omega=omega)
tdho.setup_system()

# plt.subplot(1, 1, 1, polar=True)
# plt.contourf(tdho.T, tdho.R, np.abs(tdho.spf[1]) ** 2)
# plt.show()

# ccsd = CoupledClusterSinglesDoubles(tdho, verbose=True)
# ccsd.compute_ground_state_energy(theta=theta)
# ccsd.compute_lambda_amplitudes(theta=theta)
#
# rho = ccsd.compute_one_body_density()

ccd = CoupledClusterDoubles(tdho, verbose=True)
ccd.compute_ground_state_energy(theta=theta)
ccd.compute_l_amplitudes(theta=0.6)

rho = ccd.compute_spin_reduced_one_body_density_matrix().real

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
# plt.subplot(1, 1, 1, polar=True)
# plt.subplot(1, 1, 1)
ax.plot_surface(tdho.R * np.cos(tdho.T), tdho.R * np.sin(tdho.T), rho)
# im = plt.contourf(tdho.R * np.cos(tdho.T), tdho.R * np.sin(tdho.T), rho)
# fig.colorbar(im)
plt.show()
