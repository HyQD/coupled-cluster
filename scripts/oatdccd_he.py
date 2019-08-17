import numpy as np
import matplotlib.pyplot as plt
import time
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.ccd import OATDCCD
from coupled_cluster.integrators import GaussIntegrator


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


# Define system paramters
He = """
He 0.0 0.0 0.0
"""


omega = 2.873_564_3
E = 100  # 0.05-5
laser_duration = 5


system = construct_pyscf_system_rhf(He)

integrator = GaussIntegrator(np=np, s=3, eps=1e-10)
oatdccd = OATDCCD(system, integrator=integrator, np=np, verbose=True)
oatdccd.compute_ground_state()
print(
    "Ground state CCD energy: {0}".format(oatdccd.compute_ground_state_energy())
)

polarization = np.zeros(3)
polarization[2] = 1
system.set_time_evolution_operator(
    LaserField(
        LaserPulse(td=laser_duration, omega=omega, E=E),
        polarization_vector=polarization,
    )
)

oatdccd.set_initial_conditions()
dt = 1e-2
Tfinal = 10
Nsteps = int(Tfinal / dt) + 1
timestep_stop_laser = int(laser_duration / dt)

time_points = np.linspace(0, Tfinal, Nsteps)
print("Nsteps: %d" % Nsteps)
print("step stop laser: %d" % timestep_stop_laser)
print("dt = {0}".format(dt))

td_energies = np.zeros(len(time_points))
td_energies_imag = np.zeros(len(time_points))
norm_t2 = np.zeros(len(time_points))
norm_l2 = np.zeros(len(time_points))
dip_z = np.zeros(len(time_points))
td_energies[0] = oatdccd.compute_energy()

for i, amp in enumerate(oatdccd.solve(time_points)):
    t, l, C, C_tilde = amp
    energy = oatdccd.compute_energy()
    td_energies[i + 1] = energy.real
    td_energies_imag[i + 1] = energy.imag

    rho_qp = oatdccd.compute_one_body_density_matrix()
    rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

    z = system.dipole_moment[2].copy()
    z = C_tilde @ z @ C

    dip_z[i + 1] = (np.einsum("qp,pq->", rho_qp_hermitian, z)).real

    norm_t2[i + 1] = np.linalg.norm(t[1])
    norm_l2[i + 1] = np.linalg.norm(l[0])

    if i % 100 == 0:
        print(f"i = {i}")
        eye = C_tilde @ C
        print(
            "C_tilde C is identity: %s" % np.allclose(eye, np.eye(eye.shape[0]))
        )
        print(
            "rho_qp_hermitian is hermitian: %s"
            % np.allclose(rho_qp_hermitian, rho_qp_hermitian.conj().T)
        )
        print("norm(t2): %g" % np.linalg.norm(t[1]))
        print("norm(l2): %g" % np.linalg.norm(l[0]))


plt.figure()
plt.plot(time_points, td_energies)
plt.title("Time-dependent energy")
plt.grid()

plt.figure()
plt.plot(time_points[time_points > 5], td_energies[time_points > 5])
plt.title("Time-dependent energy, after laser is turned off")
plt.grid()

plt.figure()
plt.plot(time_points, td_energies_imag)
plt.title("Time-dependent energy (imaginary component)")
plt.grid()

plt.figure()
plt.plot(time_points[time_points > 5], td_energies_imag[time_points > 5])
plt.title(
    "Time-dependent energy, after laser is turned off (imaginary component)"
)
plt.grid()

plt.figure()
plt.plot(time_points, dip_z)
plt.title(r"$\langle z(t) \rangle$")
plt.grid()

plt.figure()
plt.plot(time_points, norm_t2)
plt.title(r"Norm of $\tau_2$-amplitudes")
plt.grid()

plt.figure()
plt.plot(time_points, norm_l2)
plt.title(r"Norm of $\lambda_2$-amplitudes")
plt.grid()

from scipy.fftpack import fft, ifft, fftshift, fftfreq

"""
Fourier transform of dip_z after pulse.
"""
freq = fftshift(fftfreq(len(time_points[timestep_stop_laser:]))) * (
    2 * np.pi / dt
)
a = np.abs(fftshift(fft(dip_z[timestep_stop_laser:])))
amax = a.max()
a = a / amax
plt.figure()
plt.plot(freq, a, label=r"$\vert \tilde{d_z} \vert$")
plt.title(r"Fourier transform of $\langle z(t)\rangle$")
plt.legend()
plt.xlim(0, 6)
plt.xlabel("frequency/au")
plt.show()
