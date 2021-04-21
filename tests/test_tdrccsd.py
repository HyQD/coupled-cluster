import numpy as np
import tqdm
import matplotlib.pyplot as plt
import os

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from coupled_cluster.rccsd import RCCSD, TDRCCSD
from gauss_integrator import GaussIntegrator
from scipy.integrate import complex_ode


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


def test_energy_conservation_after_pulse():

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

    tprime = 1
    phase = 0
    polarization = np.zeros(3)
    polarization_direction = 2
    polarization[polarization_direction] = 1

    time_after_pulse = 0
    tfinal = 5

    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            sine_square_laser(
                F_str=F_str, omega=omega, tprime=tprime, phase=phase
            ),
            polarization_vector=polarization,
        )
    )

    s = 3
    eps = 1e-5
    dt = 1e-1

    cc_kwargs = dict(verbose=False)
    rccsd = RCCSD(system, **cc_kwargs)

    ground_state_tolerance = 1e-8
    rccsd.compute_ground_state(
        t_kwargs=dict(tol=ground_state_tolerance),
        l_kwargs=dict(tol=ground_state_tolerance),
    )

    amps0 = rccsd.get_amplitudes(get_t_0=True)
    y0 = amps0.asarray()

    tdrccsd = TDRCCSD(system)

    r = complex_ode(tdrccsd).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(y0)

    num_steps = int(tfinal / dt) + 1
    time_points = np.linspace(0, tfinal, num_steps)

    # Initialize arrays to hold different "observables".
    energy = np.zeros(num_steps, dtype=np.complex128)

    # Set initial values
    t, l = amps0
    energy[0] = tdrccsd.compute_energy(r.t, r.y)

    for i, _t in enumerate(time_points[:-1]):
        r.integrate(r.t + dt)

        if not r.successful():
            break
        t, l = amps0.from_array(r.y)
        energy[i + 1] = tdrccsd.compute_energy(r.t, r.y)

    assert np.linalg.norm(energy[11:].real - energy[11].real) < 1e-9


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
        DipoleFieldInteraction(
            sine_square_laser(
                F_str=F_str, omega=omega, tprime=tprime, phase=phase
            ),
            polarization_vector=polarization,
        )
    )

    s = 3
    eps = 1e-5
    dt = 1e-1

    cc_kwargs = dict(verbose=False)
    rccsd = RCCSD(system, **cc_kwargs)

    ground_state_tolerance = 1e-8
    rccsd.compute_ground_state(
        t_kwargs=dict(tol=ground_state_tolerance),
        l_kwargs=dict(tol=ground_state_tolerance),
    )

    amps0 = rccsd.get_amplitudes(get_t_0=True)
    y0 = amps0.asarray()

    tdrccsd = TDRCCSD(system)

    r = complex_ode(tdrccsd).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(y0)

    num_steps = int(tfinal / dt) + 1
    time_points = np.linspace(0, tfinal, num_steps)

    # Initialize arrays to hold different "observables".
    energy = np.zeros(num_steps, dtype=np.complex128)
    dip_z = np.zeros(num_steps, dtype=np.complex128)
    tau0 = np.zeros(num_steps, dtype=np.complex128)
    auto_corr = np.zeros(num_steps, dtype=np.complex128)
    reference_weight = np.zeros(num_steps, dtype=np.complex128)

    # Set initial values
    t, l = amps0
    energy[0] = tdrccsd.compute_energy(r.t, r.y)
    dip_z[0] = tdrccsd.compute_one_body_expectation_value(
        r.t,
        r.y,
        system.position[polarization_direction],
        make_hermitian=False,
    )
    tau0[0] = t[0][0]
    auto_corr[0] = tdrccsd.compute_overlap(r.t, y0, r.y)
    reference_weight[0] = (
        0.5 * np.exp(tau0[0])
        + 0.5
        * (
            np.exp(-tau0[0]) * tdrccsd.compute_left_reference_overlap(r.t, r.y)
        ).conj()
    )

    # for i, amp in tqdm.tqdm(
    #     enumerate(tdrccsd.solve(time_points)), total=num_steps - 1
    # ):
    for i, _t in enumerate(time_points[:-1]):
        r.integrate(r.t + dt)

        if not r.successful():
            break
        # use amps0 as template
        t, l = amps0.from_array(r.y)
        energy[i + 1] = tdrccsd.compute_energy(r.t, r.y)
        dip_z[i + 1] = tdrccsd.compute_one_body_expectation_value(
            r.t,
            r.y,
            system.position[polarization_direction],
            make_hermitian=False,
        )
        tau0[i + 1] = t[0][0]
        auto_corr[i + 1] = tdrccsd.compute_overlap(r.t, y0, r.y)
        reference_weight[i + 1] = (
            0.5 * np.exp(tau0[i + 1])
            + 0.5
            * (
                np.exp(-tau0[i + 1])
                * tdrccsd.compute_left_reference_overlap(r.t, r.y)
            ).conj()
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
    test_energy_conservation_after_pulse()
