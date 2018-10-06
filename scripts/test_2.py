from quantum_systems import OneDimensionalHarmonicOscillator
from coupled_cluster.ccsd import CoupledClusterSinglesDoubles

import matplotlib.pyplot as plt
import numpy as np

n = 2
l = 6
length = 10
num_grid_points = 400
omega = 0.25
laser_frequency = 8 * omega
laser_strength = 1
theta = 0.6
tol = 1e-4

odho = OneDimensionalHarmonicOscillator(
    n,
    l,
    length,
    num_grid_points,
    omega=omega,
    laser_frequency=laser_frequency,
    laser_strength=laser_strength,
)
odho.setup_system()

ccsd = CoupledClusterSinglesDoubles(odho, verbose=True)

ccsd.compute_ground_state_energy(theta=theta, tol=tol)
ccsd.compute_lambda_amplitudes(theta=theta, tol=tol)

rho = ccsd.compute_one_body_density()

plt.plot(odho.grid, rho)
plt.show()

t_start = 0
t_end = 10
num_timesteps = 10001
print("delta t = {0}".format((t_end - t_start) / (num_timesteps - 1)))

prob, time = ccsd.evolve_amplitudes(t_start, t_end, num_timesteps)

fig, ax = plt.subplots()
ax.plot(time * laser_frequency / (2.0 * np.pi), prob.real)
ax.ticklabel_format(useOffset=False)
plt.show()

rho = ccsd.compute_one_body_density()

plt.plot(odho.grid, rho)
plt.show()
