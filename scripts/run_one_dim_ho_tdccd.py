from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.ccd import CoupledClusterDoubles
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.density_matrices import compute_one_body_density_matrix
from coupled_cluster.ccd.time_dependent_overlap import (
    compute_time_dependent_overlap,
)
from coupled_cluster.ccd.energies import compute_time_dependent_energy
from coupled_cluster.tdcc import TimeDependentCoupledCluster
from coupled_cluster.integrators import RungeKutta4

import matplotlib.pyplot as plt
import numpy as np
import tqdm


class LaserPulse:
    def __init__(self, laser_frequency=2, laser_strength=1):
        self.laser_frequency = laser_frequency
        self.laser_strength = laser_strength

    def __call__(self, t):
        if t < 100:
            return self.laser_strength * np.sin(self.laser_frequency * t)
        return 0


n = 2
l = 6
length = 10
num_grid_points = 400
omega = 0.25
laser_frequency = 8 * omega
laser_strength = 1
theta_t = 0.1
theta_l = 0.9
tol = 1e-5

odho = OneDimensionalHarmonicOscillator(
    n, l, length, num_grid_points, omega=omega
)
odho.setup_system()
laser = LaserField(
    LaserPulse(laser_frequency=laser_frequency, laser_strength=laser_strength)
)
odho.set_time_evolution_operator(laser)

ccd = CoupledClusterDoubles(odho, verbose=True)
ccd.compute_ground_state_energy(theta=theta_t, tol=tol)
ccd.compute_l_amplitudes(theta=theta_l, tol=tol)

rho = ccd.compute_spin_reduced_one_body_density_matrix()

plt.plot(odho.grid, rho)
plt.show()

t_start = 0
t_end = 10
num_timesteps = 1001
dt = (t_end - t_start) / (num_timesteps - 1)
print(f"dt = {dt}")

tdccd = TimeDependentCoupledCluster(
    compute_t_2_amplitudes,
    compute_l_2_amplitudes,
    compute_time_dependent_energy,
    odho,
    np=np,
    integrator=RungeKutta4,
)

u_0 = ccd.get_amplitudes()
psi_overlap = np.zeros(num_timesteps)
td_energies = np.zeros(num_timesteps)
time = np.zeros(num_timesteps)

td_energies[0] = tdccd.compute_time_dependent_energy(u_0)
psi_overlap[0] = compute_time_dependent_overlap(
    *u_0.unpack(), *u_0.unpack(), np=np
).real
current_time = t_start
time[0] = current_time

u_new = u_0
for i in tqdm.tqdm(range(1, num_timesteps)):
    u_new = tdccd.step(u_new, current_time, dt)

    td_energies[i] = tdccd.compute_time_dependent_energy(u_new)
    psi_overlap[i] = compute_time_dependent_overlap(
        *u_0.unpack(), *u_new.unpack(), np=np
    ).real

    current_time += dt
    time[i] = current_time

plt.figure()
plt.plot(time, psi_overlap)
plt.title("Time-dependent overlap with ground state")
plt.figure()
plt.plot(time, td_energies)
plt.title("Time-dependent energy")
plt.show()
