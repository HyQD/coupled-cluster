from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.integrators import GaussIntegrator
import numpy as np
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from coupled_cluster.rccsd import TDRCCSD
import tqdm
import matplotlib.pyplot as plt


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
        return pulse


name = "beryllium"
atom = "be 0.0 0.0 0.0"
basis = "cc-pvdz"
charge = 0

system = construct_pyscf_system_rhf(
    atom,
    basis=basis,
    np=np,
    verbose=False,
    add_spin=False,
    anti_symmetrize=False,
)

F_str = 0.01
omega = 0.2
t_cycle = 2 * np.pi / omega
print(f"1 optical cycle={t_cycle}")

tprime = t_cycle
phase = 0
polarization = np.zeros(3)
polarization_direction = 2
polarization[polarization_direction] = 1

time_after_pulse = 0
tfinal = np.floor(tprime)
print(f"tfinal={tfinal}")

system.set_time_evolution_operator(
    DipoleFieldInteraction(
        sine_square_laser(F_str=F_str, omega=omega, tprime=tprime, phase=phase),
        polarization_vector=polarization,
    )
)

s = 3
eps = 1e-5
dt = 1e-1
integrator = GaussIntegrator(s=s, np=np, eps=eps)

cc_kwargs = dict(verbose=False)
tdrccsd = TDRCCSD(system, integrator=integrator, **cc_kwargs)

ground_state_tolerance = 1e-8
tdrccsd.compute_ground_state(
    t_kwargs=dict(tol=ground_state_tolerance),
    l_kwargs=dict(tol=ground_state_tolerance),
)
print(
    "Ground state RCCSD energy: {0}".format(
        tdrccsd.compute_ground_state_energy().real
        # + system.nuclear_repulsion_energy
    )
)
tdrccsd.set_initial_conditions()

num_steps = int(tfinal / dt) + 1
print(f"num_steps={num_steps}")

time_points = np.linspace(0, tfinal, num_steps)

# Initialize arrays to hold different "observables".
energy = np.zeros(num_steps, dtype=np.complex128)
dip_z = np.zeros(num_steps, dtype=np.complex128)
tau0 = np.zeros(num_steps, dtype=np.complex128)
auto_corr = np.zeros(num_steps, dtype=np.complex128)
reference_weight = np.zeros(num_steps, dtype=np.complex128)

# Set initial values
t, l = tdrccsd.amplitudes
energy[0] = tdrccsd.compute_energy()
print(f"E(0)={energy[0].real}")
rho_qp = tdrccsd.compute_one_body_density_matrix()
z = system.dipole_moment[polarization_direction].copy()
dip_z[0] = np.trace(np.dot(rho_qp, z))
print(f"dip_z(0)={dip_z[0].real}")
tau0[0] = t[0][0]
auto_corr[0] = tdrccsd.compute_time_dependent_overlap()
reference_weight[0] = (
    0.5 * np.exp(tau0[0])
    + 0.5 * (np.exp(-tau0[0]) * tdrccsd.left_reference_overlap()).conj()
)

for i, amp in tqdm.tqdm(
    enumerate(tdrccsd.solve(time_points)), total=num_steps - 1
):
    t, l = amp
    energy[i + 1] = tdrccsd.compute_energy()
    rho_qp = tdrccsd.compute_one_body_density_matrix()
    z = system.dipole_moment[polarization_direction].copy()
    dip_z[i + 1] = np.trace(np.dot(rho_qp, z))
    tau0[i + 1] = t[0][0]
    auto_corr[i + 1] = tdrccsd.compute_time_dependent_overlap()
    reference_weight[i + 1] = (
        0.5 * np.exp(tau0[i + 1])
        + 0.5 * (np.exp(-tau0[i + 1]) * tdrccsd.left_reference_overlap()).conj()
    )

np.save("dip_z_rccsd.npy", dip_z)
np.save("energy_rccsd.npy", energy)
np.save("tau0_rccsd.npy", tau0)
np.save("auto_corr_rccsd.npy", auto_corr)
np.save("reference_weight_rccsd.npy", reference_weight)

plt.figure()
plt.plot(time_points, dip_z.real, label=r"$d_z(t)$")
plt.legend()
plt.grid()

plt.figure()
plt.subplot(211)
plt.plot(time_points, energy.real, label=r"$\Re(\langle \hat{H}(t) \rangle)$")
plt.grid()
plt.subplot(212)
plt.plot(time_points, energy.imag, label=r"$\Im(\langle \hat{H}(t) \rangle)$")
plt.grid()

plt.figure()
plt.plot(time_points, np.abs(np.exp(tau0)) ** 2, label=r"$|\exp(\tau_0)|^2$")
plt.legend()
plt.grid()

plt.figure()
plt.plot(
    time_points,
    np.abs(auto_corr) ** 2,
    label=r"$|\langle \langle \tilde{\Psi}(t_0) | \Psi(t_1)|^2$",
)
plt.legend()
plt.grid()

plt.figure()
plt.plot(
    time_points,
    np.abs(reference_weight) ** 2,
    label=r"$|\langle \langle R(t) | S(t)|^2$",
)
plt.legend()
plt.grid()

plt.show()
