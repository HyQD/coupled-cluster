import os
import pytest

import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import LaserField

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
        LaserField(
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
        rho_qp = oatdccd.compute_one_body_density_matrix(r.t, r.y)
        rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

        t, l, C, C_tilde = oatdccd.amplitudes_from_array(r.y)
        z = C_tilde @ system.dipole_moment[2] @ C

        dip_z[i] = np.trace(rho_qp_hermitian @ z).real

        i += 1
        r.integrate(time_points[i])

    td_energies[i] = oatdccd.compute_energy(r.t, r.y)
    rho_qp = oatdccd.compute_one_body_density_matrix(r.t, r.y)
    rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

    t, l, C, C_tilde = oatdccd.amplitudes_from_array(r.y)
    z = C_tilde @ system.dipole_moment[2] @ C

    dip_z[i] = np.trace(rho_qp_hermitian @ z).real

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
