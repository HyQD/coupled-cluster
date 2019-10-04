import numpy as np
from pyscf import gto, scf, ao2mo
from quantum_systems import CustomSystem, construct_pyscf_system_rhf

from coupled_cluster import TDCCSD, OATDCCD

from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.integrators import GaussIntegrator, RungeKutta4

import matplotlib.pyplot as plt
import tqdm


system = construct_pyscf_system_rhf(
    molecule="he 0.0 0.0 0.0", basis="cc-pvdz", symmetry=False
)


class laser_pulse:
    def __init__(self, t_0=0, t_l=5, omega=0.1, E=0.03, np=None):
        if np is None:
            import numpy as np

        self.np = np
        self.t_0 = t_0
        self.t_l = t_l
        self.omega = omega
        self.E = E  # Field strength

    def __call__(self, t):
        np = self.np

        delta_t = t - self.t_0
        pulse = (
            -(np.sin(np.pi * delta_t / self.t_l) ** 2)
            * np.heaviside(delta_t, 1.0)
            * np.heaviside(self.t_l - delta_t, 1.0)
            * np.cos(self.omega * delta_t)
            * self.E
        )

        return pulse


"""
Set peak electric field strength and laser frequency. Define polarization axis and 
set the time evolution operator of system.
"""
omega = 2.8735643
E = 1
laser_duration = 5

polarization = np.zeros(3)
polarization[2] = 1

system.set_time_evolution_operator(
    LaserField(
        laser_pulse(omega=omega, E=E, t_l=laser_duration),
        polarization_vector=polarization,
    )
)

"""
Choose time integrator and setup OATDCCD instance and compute initial condition (ground state).
"""
integrator = GaussIntegrator(s=3, np=np, eps=1e-10)
# integrator = RungeKutta4(np=np)

oatdccd = OATDCCD(system, integrator=integrator)


oatdccd.compute_ground_state(tol=1e-15, termination_tol=1e-15)
print(
    "Ground state CCSD energy: {0}".format(
        oatdccd.compute_ground_state_energy()
    )
)
oatdccd.set_initial_conditions()

"""
Define integration paramters
"""
dt = 1e-2
Tfinal = 100
num_steps = int(Tfinal / dt) + 1
timestep_stop_laser = int(laser_duration / dt)
time_points = np.linspace(0, Tfinal, num_steps)

energy = np.zeros(num_steps, dtype=np.complex128)
exp_tau0 = np.zeros(num_steps, dtype=np.complex128)
norm_t2 = np.zeros(num_steps)
dip_z = np.zeros(len(time_points), dtype=np.complex128)

energy[0] = oatdccd.compute_energy()

rho_qp = oatdccd.compute_one_body_density_matrix()
rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)
z = system.dipole_moment[2].copy()
dip_z[0] = np.einsum("qp,pq->", rho_qp_hermitian, z)

"""
If oatdccd

t[0][0] = tau_0
t[1] = tau-doubles

l[0] = lambda-doubles

If ccsd
t[0][0] = tau_0
t[1] = tau-singles
t[2] = tau-doubles

l[0] = lambda-singles
l[1] = lambda-doubles
"""

t, l, C, C_tilde = oatdccd.amplitudes
exp_tau0[0] = np.exp(t[0][0])
norm_t2[0] = np.linalg.norm(t[1])

for i, amp in tqdm.tqdm(
    enumerate(oatdccd.solve(time_points)), total=num_steps - 1
):
    t, l, C, C_tilde = amp
    energy[i + 1] = oatdccd.compute_energy()

    rho_qp = oatdccd.compute_one_body_density_matrix()
    rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)
    z = system.dipole_moment[2].copy()
    z = C_tilde @ z @ C
    dip_z[i + 1] = np.einsum("qp,pq->", rho_qp_hermitian, z)

    exp_tau0[i + 1] = np.exp(t[0][0])
    norm_t2[i + 1] = np.linalg.norm(t[1])


plt.figure()
plt.title("Re(E(t))")
plt.plot(time_points, energy.real)
plt.grid()

plt.figure()
plt.title("|Im(E(t))|")
plt.semilogy(time_points, np.abs(energy.imag))
plt.grid()

plt.figure()
plt.title("<z(t)>")
plt.plot(time_points, dip_z.real)
plt.grid()

plt.figure()
plt.title("Im(|<z(t)>|)")
plt.semilogy(time_points, np.abs(dip_z.imag))
plt.grid()

plt.figure()
plt.plot(time_points, np.abs(exp_tau0) ** 2, label=r"$|e^{\tau_0(t)}|^2$")
plt.plot(time_points, norm_t2, label=r"$||\tau_2(t)||$")
plt.legend()
plt.grid()

from scipy.fftpack import fft, ifft, fftshift, fftfreq

# Fourier transform of dip_z after pulse.
freq = fftshift(fftfreq(len(time_points[timestep_stop_laser:]))) * (
    2 * np.pi / dt
)
a = np.abs(fftshift(fft(dip_z[timestep_stop_laser:].real)))
amax = a.max()
a = a / amax

plt.figure()
plt.plot(freq, a, label=r"$\vert \tilde{d_z} \vert$")
plt.title(r"Fourier transform of $\langle z(t)\rangle$")
plt.legend()
plt.xlim(0, 6)
plt.xlabel("frequency/au")

plt.show()
