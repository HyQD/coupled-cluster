import tqdm
import numpy as np
import matplotlib.pyplot as plt

from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator
from coupled_cluster.sampler import TDCCSampleAll

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField


class LaserPulse:
    def __init__(self, t0=0, td=5, omega=0.1, E=0.03):
        self.t0 = t0
        self.td = td
        self.omega = omega
        self.E = E  # Field strength

    def __call__(self, t):
        T = self.td
        delta_t = t - self.t0
        return (
            -(np.sin(np.pi * delta_t / T) ** 2)
            * np.heaviside(delta_t, 1.0)
            * np.heaviside(T - delta_t, 1.0)
            * np.cos(self.omega * delta_t)
            * self.E
        )


pol = np.zeros(3)
pol[2] = 1

system = construct_pyscf_system_rhf("he")
system.set_time_evolution_operator(
    LaserField(LaserPulse(), polarization_vector=pol)
)

tdccsd = TDCCSD(
    system, integrator=GaussIntegrator(s=3, eps=1e-6, np=np), verbose=True
)
tdccsd.compute_ground_state()
tdccsd.set_initial_conditions()

t_final = 10
dt = 1e-2
num_timesteps = int(t_final / dt + 1)

time_steps = np.linspace(0, t_final, num_timesteps)

sampler = TDCCSampleAll(tdccsd, num_timesteps, np)
sampler.add_sample("time_points", time_steps)

sampler.sample(0)

for i, amp in tqdm.tqdm(
    enumerate(tdccsd.solve(time_steps)), total=num_timesteps - 1
):
    sampler.sample(i + 1)

sampler.dump()
