import os
import pytest

import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from coupled_cluster.ccd import OATDCCD, OACCD
from gauss_integrator import GaussIntegrator
from scipy.integrate import complex_ode


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


def test_oatdccd_energy_conservation():

    omega = 0.2
    E = 0.5
    laser_duration = 1

    system = construct_pyscf_system_rhf(
        molecule="be 0.0 0.0 0.0",
        basis="cc-pvdz",
        np=np,
    )

    polarization = np.zeros(3)
    polarization[2] = 1
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            LaserPulse(td=laser_duration, omega=omega, E=E),
            polarization_vector=polarization,
        )
    )

    dt = 1e-1
    T = 5
    num_steps = int(T // dt) + 1
    t_stop_laser = int(laser_duration // dt) + 1

    time_points = np.linspace(0, T, num_steps)

    oaccd = OACCD(system, verbose=True)
    oaccd.compute_ground_state(tol=1e-8)

    oatdccd = OATDCCD(system)

    r = complex_ode(oatdccd).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(oaccd.get_amplitudes(get_t_0=True).asarray())

    td_energies_oatdccd = np.zeros(len(time_points), dtype=np.complex128)
    dip_z_oatdccd = np.zeros(len(time_points), dtype=np.complex128)
    td_energies_oatdccd[0] = oatdccd.compute_energy(r.t, r.y)
    dip_z_oatdccd[0] = oatdccd.compute_one_body_expectation_value(
        r.t, r.y, system.position[2]
    )

    man_energy = oatdccd.compute_one_body_expectation_value(
        r.t, r.y, system.h_t(r.t), make_hermitian=False
    ) + 0.5 * oatdccd.compute_two_body_expectation_value(r.t, r.y, system.u)
    assert abs(td_energies_oatdccd[0] - man_energy) < 1e-12

    for i, _t in enumerate(time_points[:-1]):

        r.integrate(r.t + dt)

        td_energies_oatdccd[i + 1] = oatdccd.compute_energy(r.t, r.y)
        dip_z_oatdccd[i + 1] = oatdccd.compute_one_body_expectation_value(
            r.t, r.y, system.position[2]
        )

        # When comparing the two ways of computing the energy, we should not
        # explicitly make the one-body density matrix Hermitian as the energy
        # calculation using the Lagrangian functional does no such thing.
        man_energy = oatdccd.compute_one_body_expectation_value(
            r.t, r.y, system.h_t(r.t), make_hermitian=False
        ) + 0.5 * oatdccd.compute_two_body_expectation_value(r.t, r.y, system.u)
        assert abs(td_energies_oatdccd[i + 1] - man_energy) < 1e-12

    energy_conservation = np.linalg.norm(
        td_energies_oatdccd[11:].real - td_energies_oatdccd[11].real
    )
    assert energy_conservation < 1e-6


def test_oatdccd_helium():
    omega = 2.873_564_3
    E = 0.1
    laser_duration = 5

    system = construct_pyscf_system_rhf(
        molecule="he 0.0 0.0 0.0", basis="cc-pvdz"
    )

    oaccd = OACCD(system, verbose=True)
    oaccd.compute_ground_state()
    assert abs(oaccd.compute_energy() - -2.887_594_831_090_936) < 1e-6

    oatdccd = OATDCCD(system)

    r = complex_ode(oatdccd).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(oaccd.get_amplitudes(get_t_0=True).asarray())

    polarization = np.zeros(3)
    polarization[2] = 1
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            LaserPulse(td=laser_duration, omega=omega, E=E),
            polarization_vector=polarization,
        )
    )

    dt = 1e-3
    T = 1
    num_steps = int(T // dt) + 1
    t_stop_laser = int(laser_duration // dt) + 1

    time_points = np.linspace(0, T, num_steps)

    td_energies = np.zeros(len(time_points), dtype=np.complex128)
    dip_z = np.zeros(len(time_points))

    i = 0

    while r.successful() and r.t < T:
        assert abs(time_points[i] - r.t) < dt * 1e-1

        td_energies[i] = oatdccd.compute_energy(r.t, r.y)
        dip_z[i] = oatdccd.compute_one_body_expectation_value(
            r.t, r.y, system.position[2]
        )

        man_energy = oatdccd.compute_one_body_expectation_value(
            r.t, r.y, system.h_t(r.t), make_hermitian=False
        ) + 0.5 * oatdccd.compute_two_body_expectation_value(r.t, r.y, system.u)
        assert abs(td_energies[i] - man_energy) < 1e-12

        i += 1
        r.integrate(time_points[i])

    td_energies[i] = oatdccd.compute_energy(r.t, r.y)
    dip_z[i] = oatdccd.compute_one_body_expectation_value(
        r.t, r.y, system.position[2]
    )

    man_energy = oatdccd.compute_one_body_expectation_value(
        r.t, r.y, system.h_t(r.t), make_hermitian=False
    ) + 0.5 * oatdccd.compute_two_body_expectation_value(r.t, r.y, system.u)
    assert abs(td_energies[i] - man_energy) < 1e-12

    np.testing.assert_allclose(
        td_energies.real,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_energies_real_0.1.dat")
        ),
        atol=1e-5,
    )

    np.testing.assert_allclose(
        dip_z,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_dipole_z_0.1.dat")
        ),
        atol=1e-5,
    )


if __name__ == "__main__":
    test_oatdccd_energy_conservation()
