from quantum_systems import OneDimensionalHarmonicOscillator
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.ccd import CoupledClusterDoubles, TDCCD

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
theta_t = 0.5
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

cc_kwargs = dict(verbose=True)

tdccd = TDCCD(CoupledClusterDoubles, odho, np=np, **cc_kwargs)
t_kwargs = dict(theta=theta_t, tol=tol)
l_kwargs = dict(theta=theta_l, tol=tol)
tdccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)

print("Ground state energy: {0}".format(tdccd.compute_ground_state_energy()))

rho = tdccd.compute_ground_state_particle_density()

plt.plot(odho.grid, rho)
plt.show()

tdccd.set_initial_conditions()

num_timesteps = 1001
time_points = np.linspace(0, 10, num_timesteps)
psi_overlap = np.zeros(num_timesteps)
td_energies = np.zeros(num_timesteps)

psi_overlap[0] = tdccd.compute_time_dependent_overlap().real
td_energies[0] = tdccd.compute_energy().real

for i, amp in enumerate(tdccd.solve(time_points)):
    psi_overlap[i + 1] = tdccd.compute_time_dependent_overlap().real
    td_energies[i + 1] = tdccd.compute_energy().real


plt.figure()
plt.plot(time_points, psi_overlap)
plt.title("Time-dependent overlap with ground state")
plt.figure()
plt.plot(time_points, td_energies)
plt.title("Time-dependent energy")
plt.show()
