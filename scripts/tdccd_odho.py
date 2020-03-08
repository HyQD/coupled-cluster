import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import complex_ode

from quantum_systems import ODQD, GeneralOrbitalSystem
from quantum_systems.time_evolution_operators import LaserField

from hartree_fock import GHF
from coupled_cluster import CCD, TDCCD
from gauss_integrator import GaussIntegrator

n = 2
l = 6
omega = 1

gos = GeneralOrbitalSystem(n, ODQD(l, 11, 201))
gos.set_time_evolution_operator(LaserField(lambda t: np.sin(omega * t)))

basis_set = ODQD(l=20, grid_length=10, num_grid_points=801)

hf = GHF(gos)
hf.compute_ground_state(change_system_basis=True)

ccd = CCD(gos, verbose=True)
ccd.compute_ground_state()

tdcc = TDCCD(gos)

r = complex_ode(tdcc).set_integrator("GaussIntegrator")
r.set_initial_value(ccd.get_amplitudes(get_t_0=True).asarray())

t_final = 5
dt = 1e-2

num_steps = int(t_final / dt) + 1
energy = np.zeros(num_steps, dtype=np.complex128)

i = 0

while r.successful() and r.t <= t_final:
    if i % 100 == 0:
        print(f"{i} / {num_steps}")

    energy[i] = tdcc.compute_energy(r.t, r.y)
    r.integrate(r.t + dt)

    i += 1

plt.plot(np.linspace(0, t_final, num_steps), energy.real)
plt.show()
