import numpy as np
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import tqdm

from quantum_systems import construct_psi4_system
from quantum_systems.time_evolution_operators import LaserField
from tdhf import HartreeFock
from coupled_cluster.ccsd import TDCCSD
from coupled_cluster.integrators import GaussIntegrator, RungeKutta4


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


# System parameters
He = """
    He 0.0 0.0 0.0
    symmetry c1
    """

options = {"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-6}
omega = 2.873_564_3
E = 5
laser_duration = 5

system = construct_psi4_system(He, options)
hf = HartreeFock(system, verbose=True)
C = hf.scf(tolerance=1e-15)
system.change_basis(C)

integrator = GaussIntegrator(s=3, np=np, eps=1e-6)
# integrator = RungeKutta4(np=np)
tdccsd = TDCCSD(system, integrator=integrator, np=np, verbose=True)
t_kwargs = dict(theta=0, tol=1e-10)
tdccsd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=t_kwargs)
print(f"Ground state CCSD energy: {tdccsd.compute_ground_state_energy()}")

polarization = np.zeros(3)
polarization[2] = 1
system.set_time_evolution_operator(
    LaserField(
        LaserPulse(td=laser_duration, omega=omega, E=E),
        polarization_vector=polarization,
    )
)

tdccsd.set_initial_conditions()
dt = 1e-3
T = 5
num_steps = int(T // dt) + 1
t_stop_laser = int(laser_duration // dt)

print(f"Laser stops @ {t_stop_laser}")

time_points = np.linspace(0, T, num_steps)

td_energies = np.zeros(len(time_points), dtype=np.complex128)
dip_z = np.zeros(len(time_points))
td_energies[0] = tdccsd.compute_energy()
phase = np.zeros_like(td_energies)

for i, amp in tqdm.tqdm(
    enumerate(tdccsd.solve(time_points)), total=num_steps - 1
):
    t, l = amp
    energy = tdccsd.compute_energy()
    td_energies[i + 1] = energy
    phase[i + 1] = t[0][0]
    rho_qp = tdccsd.compute_one_body_density_matrix()
    rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

    z = system.dipole_moment[2].copy()
    dip_z[i + 1] = (np.einsum("qp,pq->", rho_qp_hermitian, z)).real

plt.figure()
plt.plot(time_points, td_energies.real)
plt.title("Time-dependent energy")
plt.grid()

plt.figure()
plt.plot(time_points, np.abs(np.exp(phase)) ** 2)
plt.title("Phase")
plt.grid()

plt.figure()
plt.plot(time_points, dip_z)
plt.title("Induced dipole moment")
plt.grid()

plt.show()
