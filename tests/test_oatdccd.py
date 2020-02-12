import os
import pytest

import numpy as np
from quantum_systems import construct_pyscf_system
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


@pytest.mark.skip
def test_oatdccd(
    helium_system,
    oaccd_groundstate_helium_energy,
    oatdccd_helium_td_energies,
    oatdccd_helium_td_energies_imag,
    oatdccd_helium_dip_z,
    oatdccd_helium_norm_t2,
    oatdccd_helium_norm_l2,
):
    omega = 2.873_564_3
    E = 100  # 0.05-5
    laser_duration = 5
    tol = 1e-7

    system = helium_system
    system.change_to_hf_basis(verbose=True, tolerance=1e-8)

    integrator = GaussIntegrator(np=np, eps=1e-10)
    cc_kwargs = dict(verbose=True)
    oatdccd = OATDCCD(system, integrator=integrator, **cc_kwargs)
    oatdccd.compute_ground_state()

    assert (
        abs(
            oaccd_groundstate_helium_energy
            - oatdccd.compute_ground_state_energy()
        )
        < tol
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
    t_final = 10
    num_steps = int(t_final / dt) + 1
    timestep_stop_laser = int(laser_duration / dt)

    time_points = np.linspace(0, t_final, num_steps)

    td_energies = np.zeros(len(time_points))
    td_energies_imag = np.zeros(len(time_points))
    norm_t2 = np.zeros(len(time_points))
    norm_l2 = np.zeros(len(time_points))
    dip_z = np.zeros(len(time_points))

    t, l, C, C_tilde = oatdccd.amplitudes

    energy = oatdccd.compute_energy()
    td_energies[0] = energy.real
    td_energies_imag[0] = energy.imag

    rho_qp = oatdccd.compute_one_body_density_matrix()
    rho_qp = 0.5 * (rho_qp.conj().T + rho_qp)

    z = C_tilde @ system.dipole_moment[2] @ C

    dip_z[0] = np.trace(rho_qp @ z).real

    norm_t2[0] = np.linalg.norm(t[1])
    norm_l2[0] = np.linalg.norm(l)

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
        norm_l2[i + 1] = np.linalg.norm(l)

        if i % 100 == 0:
            eye = C_tilde @ C
            np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-8)
            np.testing.assert_allclose(
                rho_qp_hermitian, rho_qp_hermitian.conj().T, atol=1e-8
            )

    np.testing.assert_allclose(
        td_energies, oatdccd_helium_td_energies, atol=1e-7
    )
    np.testing.assert_allclose(
        td_energies_imag, oatdccd_helium_td_energies_imag, atol=1e-7
    )
    np.testing.assert_allclose(dip_z, oatdccd_helium_dip_z, atol=1e-7)
    np.testing.assert_allclose(norm_t2, oatdccd_helium_norm_t2, atol=1e-7)
    np.testing.assert_allclose(norm_l2, oatdccd_helium_norm_l2, atol=1e-7)


def test_oatdccd_helium():
    omega = 2.873_564_3
    E = 0.1
    laser_duration = 5

    system = construct_pyscf_system(molecule="he 0.0 0.0 0.0", basis="cc-pvdz")

    integrator = GaussIntegrator(s=3, np=np, eps=1e-6)
    oatdccd = OATDCCD(system, integrator=integrator, verbose=True)
    oatdccd.compute_ground_state()
    assert (
        abs(oatdccd.compute_ground_state_energy() - -2.887_594_831_090_936)
        < 1e-6
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
    dt = 1e-3
    T = 1
    num_steps = int(T // dt) + 1
    t_stop_laser = int(laser_duration // dt) + 1

    time_points = np.linspace(0, T, num_steps)

    td_energies = np.zeros(len(time_points), dtype=np.complex128)
    dip_z = np.zeros(len(time_points))

    rho_qp = oatdccd.compute_one_body_density_matrix()
    rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

    td_energies[0] = oatdccd.compute_energy()

    t, l, C, C_tilde = oatdccd.amplitudes

    z = C_tilde @ system.dipole_moment[2] @ C

    dip_z[0] = np.einsum("qp,pq->", rho_qp_hermitian, z).real

    for i, amp in enumerate(oatdccd.solve(time_points)):
        td_energies[i + 1] = oatdccd.compute_energy()

        rho_qp = oatdccd.compute_one_body_density_matrix()
        rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

        t, l, C, C_tilde = amp
        z = C_tilde @ system.dipole_moment[2] @ C

        dip_z[i + 1] = np.einsum("qp,pq->", rho_qp_hermitian, z).real

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
