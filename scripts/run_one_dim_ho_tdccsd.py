from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles
from coupled_cluster.ccd import CoupledClusterDoubles
from tdhf import HartreeFock

import matplotlib.pyplot as plt
import numpy as np


class LaserPulse:
    def __init__(self, laser_frequency=2, laser_strength=1):
        self.laser_frequency = laser_frequency
        self.laser_strength = laser_strength

    def __call__(self, t):
        return self.laser_strength * np.sin(self.laser_frequency * t)


n = 2
l = 6
length = 10
num_grid_points = 400
omega = 0.25
laser_frequency = 8 * omega
laser_strength = 1
theta = 0.6
tol = 1e-4

odho_ccsd = OneDimensionalHarmonicOscillator(
    n, l, length, num_grid_points, omega=omega
)
odho_ccsd.setup_system()
laser = LaserField(
    LaserPulse(laser_frequency=laser_frequency, laser_strength=laser_strength)
)
odho_ccsd.set_time_evolution_operator(laser)

hf_ccsd = HartreeFock(odho_ccsd, verbose=True)
C = hf_ccsd.scf(tolerance=1e-6)
odho_ccsd.change_basis(C)

odho_ccd = OneDimensionalHarmonicOscillator(
    n, l, length, num_grid_points, omega=omega
)
odho_ccd.setup_system()
laser = LaserField(
    LaserPulse(laser_frequency=laser_frequency, laser_strength=laser_strength)
)
odho_ccd.set_time_evolution_operator(laser)

hf_ccd = HartreeFock(odho_ccd, verbose=True)
C = hf_ccd.scf(tolerance=1e-6)
odho_ccd.change_basis(C)

ccsd = CoupledClusterSinglesDoubles(
    odho_ccsd, verbose=True, include_singles=True
)
ccd = CoupledClusterDoubles(odho_ccd, verbose=True)

ccsd.iterate_t_amplitudes(theta=theta, tol=tol)
print("Ground state energy (CCSD): {0}".format(ccsd.compute_energy()))
ccd.iterate_t_amplitudes(theta=theta, tol=tol)
print("Ground state energy (CCD): {0}".format(ccd.compute_energy()))

ccsd.iterate_l_amplitudes(theta=theta, tol=tol, max_iterations=100)
ccd.iterate_l_amplitudes(theta=theta, tol=tol, max_iterations=100)

rho_ccsd = ccsd.compute_particle_density()
rho_ccd = ccd.compute_particle_density()

plt.plot(odho_ccsd.grid, rho_ccsd, label=r"$\rho_{CCSD}$")
plt.plot(odho_ccd.grid, rho_ccd, label=r"$\rho_{CCD}$")
plt.legend(loc="best")
plt.show()

t_start = 0
t_end = 10
num_timesteps = 10001
print("delta t = {0}".format((t_end - t_start) / (num_timesteps - 1)))

prob_ccsd, time_ccsd = ccsd.evolve_amplitudes(t_start, t_end, num_timesteps)
prob_ccd, time_ccd = ccd.evolve_amplitudes(t_start, t_end, num_timesteps)

plt.figure()
plt.plot(
    time_ccsd * laser_frequency / (2.0 * np.pi),
    prob_ccsd.real,
    label=r"$P_{CCSD}$",
)
plt.plot(
    time_ccd * laser_frequency / (2.0 * np.pi),
    prob_ccd.real,
    label=r"$P_{CCD}$",
)
plt.legend(loc="best")
plt.show()

rho_ccsd = ccsd.compute_spin_reduced_one_body_density_matrix()
rho_ccd = ccd.compute_spin_reduced_one_body_density_matrix()

plt.figure()
plt.plot(odho_ccsd.grid, rho_ccsd, label=r"$\rho_{CCSD}$")
plt.plot(odho_ccd.grid, rho_ccd, label=r"$\rho_{CCD}$")
plt.legend(loc="best")
plt.show()
