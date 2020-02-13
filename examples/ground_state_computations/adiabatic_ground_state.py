import tqdm
import numpy as np
import matplotlib.pyplot as plt

from coupled_cluster.integrators import GaussIntegrator
from coupled_cluster import OATDCCD
from coupled_cluster.cc_helper import OACCVector

from quantum_systems import ODQD
from quantum_systems.time_evolution_operators import AdiabaticSwitching


class FermiFunction:
    def __init__(self, tau=3, half_time=25):
        self.tau = tau
        self.half_time = half_time

    def __call__(self, current_time):
        denom = 1 + np.exp((current_time - self.half_time) / self.tau)

        return 1 - 1 / denom


n = 4
l = 10

system = ODQD(n, l, 10, 201)
system.setup_system(potential=ODQD.HOPotential(omega=1))
system.set_time_evolution_operator(AdiabaticSwitching(FermiFunction()))
print(f"Reference energy: {system.compute_reference_energy()}")

integrator = GaussIntegrator(s=3, eps=1e-6, np=np)
oa = OATDCCD(system, verbose=True, integrator=integrator)
oa.compute_ground_state(change_system_basis=False)
oa.set_initial_conditions(
    OACCVector.zeros_like(oa.cc.get_amplitudes(get_t_0=True))
)

dt = 1e-1
time_points = np.arange(0, 100 + dt, dt)

energy = np.zeros(len(time_points), dtype=np.complex128)
oa.u = np.zeros_like(oa.u)
oa.f = oa.h
energy[0] = oa.compute_energy()
print(energy[0])

for i, amp in tqdm.tqdm(
    enumerate(oa.solve(time_points)), total=len(time_points) - 1
):
    energy[i + 1] = oa.compute_energy()


print(f"Final energy: {energy[-1]}")
plt.plot(time_points, energy.real)
plt.show()
