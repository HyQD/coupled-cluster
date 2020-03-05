from quantum_systems import construct_pyscf_system_rhf
from coupled_cluster.integrators import GaussIntegrator, RungeKutta4
import numpy as np
from quantum_systems.time_evolution_operators import LaserField
from coupled_cluster.rccsd import TDRCCSD
import tqdm
import matplotlib.pyplot as plt
import os


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


def test_tdrccsd_vs_tdccsd():
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

    tprime = t_cycle
    phase = 0
    polarization = np.zeros(3)
    polarization_direction = 2
    polarization[polarization_direction] = 1

    time_after_pulse = 0
    tfinal = np.floor(tprime)

    system.set_time_evolution_operator(
        LaserField(
            sine_square_laser(
                F_str=F_str, omega=omega, tprime=tprime, phase=phase
            ),
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
    tdrccsd.set_initial_conditions()

    num_steps = int(tfinal / dt) + 1
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
    rho_qp = tdrccsd.compute_one_body_density_matrix()
    z = system.dipole_moment[polarization_direction].copy()
    dip_z[0] = np.trace(np.dot(rho_qp, z))
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
            + 0.5
            * (np.exp(-tau0[i + 1]) * tdrccsd.left_reference_overlap()).conj()
        )

    dip_z_ccsd = np.load(
        os.path.join("tests", "dat", f"{name}", "dip_z_ccsd.npy"),
        allow_pickle=True,
    )
    energy_ccsd = np.load(
        os.path.join("tests", "dat", f"{name}", "energy_ccsd.npy"),
        allow_pickle=True,
    )
    auto_corr_ccsd = np.load(
        os.path.join("tests", "dat", f"{name}", "auto_corr_ccsd.npy"),
        allow_pickle=True,
    )
    reference_weight_ccsd = np.load(
        os.path.join("tests", "dat", f"{name}", "reference_weight_ccsd.npy"),
        allow_pickle=True,
    )

    np.testing.assert_allclose(dip_z, dip_z_ccsd, atol=1e-7)
    np.testing.assert_allclose(energy, energy_ccsd, atol=1e-8)
    np.testing.assert_allclose(
        np.abs(auto_corr) ** 2, auto_corr_ccsd, atol=1e-8
    )
    np.testing.assert_allclose(
        reference_weight, reference_weight_ccsd, atol=1e-8
    )


if __name__ == "__main__":
    test_tdrccsd_vs_tdccsd()
