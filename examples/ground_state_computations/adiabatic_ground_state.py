import tqdm
import numpy as np
import matplotlib.pyplot as plt

from coupled_cluster.integrators import GaussIntegrator
from coupled_cluster import OATDCCD
from coupled_cluster.cc_helper import OACCVector
from coupled_cluster.ccd.rhs_t import compute_t_2_amplitudes
from coupled_cluster.ccd.rhs_l import compute_l_2_amplitudes

from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import AdiabaticSwitching


class FermiFunction:
    def __init__(self, tau=3, half_time=25):
        self.tau = tau
        self.half_time = half_time

    def __call__(self, current_time):
        denom = 1 + np.exp((current_time - self.half_time) / self.tau)
        return 1 - 1 / denom


n = 6
l = 8

system = GeneralOrbitalSystem(
    n, ODQD(l, 10, 801, a=1, potential=ODQD.AtomicPotential(Za=n, c=1))
)
system.set_time_evolution_operator(AdiabaticSwitching(FermiFunction()))
print(f"Reference energy: {system.compute_reference_energy()}")

integrator = GaussIntegrator(s=3, eps=1e-10, np=np)

oa = OATDCCD(system, verbose=True, integrator=integrator)
oa.compute_ground_state(change_system_basis=False)

oa.set_initial_conditions(
    OACCVector.zeros_like(oa.cc.get_amplitudes(get_t_0=True))
)

dt = 1e-1
time_points = np.arange(0, 50 + dt, dt)

energy = np.zeros(len(time_points), dtype=np.complex128)
norm_res_t2 = np.zeros(len(time_points))
norm_res_l2 = np.zeros(len(time_points))

oa.u = np.zeros_like(oa.u)
oa.f = oa.h

energy[0] = oa.compute_energy()

t, l, C, C_tilde = oa.amplitudes
t2, l2 = t[1], l[0]
norm_res_t2[0] = np.linalg.norm(
    compute_t_2_amplitudes(oa.f, oa.u, t2, system.o, system.v, np=np)
)
norm_res_l2[0] = np.linalg.norm(
    compute_l_2_amplitudes(oa.f, oa.u, t2, l2, system.o, system.v, np=np)
)

print(energy[0])
print(norm_res_t2[0], norm_res_l2[0])

for i, amp in tqdm.tqdm(
    enumerate(oa.solve(time_points)), total=len(time_points) - 1
):
    t, l, C, C_tilde = amp
    energy[i + 1] = oa.compute_energy()
    t2, l2 = t[1], l[0]
    norm_res_t2[i + 1] = np.linalg.norm(
        compute_t_2_amplitudes(oa.f, oa.u, t2, system.o, system.v, np=np)
    )
    norm_res_l2[i + 1] = np.linalg.norm(
        compute_l_2_amplitudes(oa.f, oa.u, t2, l2, system.o, system.v, np=np)
    )

print(f"Final energy: {energy[-1]}")
plt.figure()
plt.plot(time_points, energy.real)

plt.figure()
plt.semilogy(time_points, norm_res_t2, label="t2 residual")
plt.semilogy(time_points, norm_res_t2, label="l2 residual")
plt.legend()
plt.show()
