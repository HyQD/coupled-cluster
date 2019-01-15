from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.cc_helper import AmplitudeContainer
from coupled_cluster.ccd import CoupledClusterDoubles
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes
from coupled_cluster.ccd.density_matrices import compute_one_body_density_matrix
from coupled_cluster.tdcc import TimeDependentCoupledCluster

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
theta_t = 0.1
theta_l = 0.9
tol = 1e-4

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
num_timesteps = 10001
dt = (t_end - t_start) / (num_timesteps - 1)
print(f"dt = {dt}")

tdccd = TimeDependentCoupledCluster(
    compute_t_2_amplitudes, compute_l_2_amplitudes, odho, np=np
)
u_0 = AmplitudeContainer(l=ccd._get_l_copy(), t=ccd._get_t_copy())
u_new = tdccd.rk4_step(u_0, t_start, dt)
u_new = tdccd.rk4_step(u_new, t_start + dt, dt)
