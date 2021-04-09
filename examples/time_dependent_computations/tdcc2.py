import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tqdm
import os

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from coupled_cluster.cc2 import CC2, TDCC2
from gauss_integrator import GaussIntegrator
from tdhf import HartreeFock, TimeDependentHartreeFock
from scipy.integrate import complex_ode

from coupled_cluster.mix import DIIS, AlphaMixer


class sine_square_laser:
    def __init__(self, F_str, omega, tprime, phase=0):
        self.F_str = F_str
        self.omega = omega
        self.tprime = tprime
        self.phase = phase

    def __call__(self, t):
        pulse = (
            (np.sin(np.pi * t / self.tprime) ** 2)
            * np.heaviside(t, 1.0)
            * np.heaviside(self.tprime - t, 1.0)
            * np.sin(self.omega * t + self.phase)
            * self.F_str
        )
        #        print("pulse ", pulse)
        return pulse


molecule = "li 0.0 0.0 0.0;h 0.0 0.0 3.08"
basis = "6-31G"
system = construct_pyscf_system_rhf(
    molecule,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=True,
    anti_symmetrize=True,
)

conv_tol = 1e-14
cc2 = CC2(system, mixer=AlphaMixer, verbose=False)
t_kwargs = dict(tol=conv_tol, theta=0)
l_kwargs = dict(tol=conv_tol, theta=0)

cc2.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
print("Ground state energy: {0}".format(cc2.compute_energy()))
F_str = 0.10
omega = 0.2
t_cycle = 2 * np.pi / omega

tprime = t_cycle
phase = 0
polarization = np.zeros(3)
polarization_direction = 2
polarization[polarization_direction] = 1

time_after_pulse = 100
tfinal = np.floor(tprime) + time_after_pulse

system.set_time_evolution_operator(
    DipoleFieldInteraction(
        sine_square_laser(F_str=F_str, omega=omega, tprime=tprime, phase=phase),
        polarization_vector=polarization,
    )
)

dt = 1e-1

cc_kwargs = dict(verbose=False)
amps0 = cc2.get_amplitudes(get_t_0=True)
y0 = amps0.asarray()

print("TDRCCSD initiated")
tdcc2 = TDCC2(system)

r = complex_ode(tdcc2).set_integrator("not_GaussIntegrator", s=3, eps=1e-6)
print("r is initiated, based on call")

r.set_initial_value(y0)

num_steps = int(tfinal / dt) + 1
time_points = np.linspace(0, tfinal, num_steps)
timestep_stop_laser = int(tprime / dt)

# Initialize arrays to hold different "observables".
energy = np.zeros(num_steps, dtype=np.complex128)
dip_z = np.zeros(num_steps, dtype=np.complex128)
tau0 = np.zeros(num_steps, dtype=np.complex128)
auto_corr = np.zeros(num_steps, dtype=np.complex128)
reference_weight = np.zeros(num_steps, dtype=np.complex128)

# Set initial values
t, l = amps0


energy[0] = tdcc2.compute_energy(r.t, r.y)
dip_z[0] = tdcc2.compute_one_body_expectation_value(
    r.t,
    r.y,
    system.dipole_moment[polarization_direction],
    make_hermitian=False,
)
print("initial dipole z")
print(dip_z[0])

tau0[0] = t[0][0]
auto_corr[0] = tdcc2.compute_overlap(r.t, y0, r.y)
reference_weight[0] = (
    0.5 * np.exp(tau0[0])
    + 0.5
    * (np.exp(-tau0[0]) * tdcc2.compute_left_reference_overlap(r.t, r.y)).conj()
)

for i, _t in tqdm.tqdm(enumerate(time_points[:-1])):
    r.integrate(r.t + dt)
    if not r.successful():
        break
    # use amps0 as template
    t, l = amps0.from_array(r.y)
    energy[i + 1] = tdcc2.compute_energy(r.t, r.y)
    dip_z[i + 1] = tdcc2.compute_one_body_expectation_value(
        r.t,
        r.y,
        system.dipole_moment[polarization_direction],
        make_hermitian=False,
    )
    tau0[i + 1] = t[0][0]
    auto_corr[i + 1] = tdcc2.compute_overlap(r.t, y0, r.y)
    reference_weight[i + 1] = (
        0.5 * np.exp(tau0[i + 1])
        + 0.5
        * (
            np.exp(-tau0[i + 1])
            * tdcc2.compute_left_reference_overlap(r.t, r.y)
        ).conj()
    )

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

np.save("a_oatdccd", a)
np.save("freq_oatdccd", freq)

plt.figure()
plt.plot(freq, a, label=r"$\vert \tilde{d_z} \vert$")
plt.title(r"Fourier transform of $\langle z(t)\rangle$")
plt.legend()
plt.xlim(0, 6)
plt.xlabel("frequency/au")
plt.show()
