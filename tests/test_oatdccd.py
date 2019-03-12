import pytest

import numpy as np
from quantum_systems import construct_psi4_system
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
    ccd_groundstate_He_energy,
    oatdccd_helium_td_energies,
    oatdccd_helium_td_energies_imag,
    oatdccd_helium_dip_z,
    oatdccd_helium_norm_t2,
    oatdccd_helium_norm_l2,
):
    try:
        from tdhf import HartreeFock
    except ImportError:
        pytest.skip("Cannot import module tdhf")

    omega = 2.873_564_3
    E = 100  # 0.05-5
    laser_duration = 5
    tol = 1e-7

    system = helium_system
    hf = HartreeFock(system, verbose=True)
    C = hf.scf(tolerance=1e-8)
    system.change_basis(C)

    integrator = GaussIntegrator(np=np, eps=1e-10)
    cc_kwargs = dict(verbose=True)
    oatdccd = OATDCCD(system, integrator=integrator, np=np, **cc_kwargs)
    oatdccd.compute_ground_state(tol=1e-10, termination_tol=1e-12)

    assert (
        abs(ccd_groundstate_He_energy - oatdccd.compute_ground_state_energy())
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
    td_energies[0] = oatdccd.compute_energy()

    for i, amp in enumerate(oatdccd.solve(time_points)):
        t, l, C, C_tilde = amp
        energy = oatdccd.compute_energy()
        td_energies[i + 1] = energy.real
        td_energies_imag[i + 1] = energy.imag

        rho_qp = oatdccd.one_body_density_matrix(t, l)
        rho_qp_hermitian = 0.5 * (rho_qp.conj().T + rho_qp)

        z = system.dipole_moment[2].copy()
        z = C_tilde @ z @ C

        dip_z[i + 1] = (np.einsum("qp,pq->", rho_qp_hermitian, z)).real

        norm_t2[i + 1] = np.linalg.norm(t)
        norm_l2[i + 1] = np.linalg.norm(l)

        if i % 100 == 0:
            eye = C_tilde @ C
            np.testing.assert_allclose(eye, np.eye(eye.shape[0]), atol=1e-8)
            np.testing.assert_allclose(
                rho_qp_hermitian, rho_qp_hermitian.conj().T, atol=1e-8
            )

    np.testing.assert_allclose(
        td_energies, oatdccd_helium_td_energies, atol=tol
    )

    np.testing.assert_allclose(
        td_energies_imag, oatdccd_helium_td_energies_imag, atol=tol
    )

    np.testing.assert_allclose(dip_z, oatdccd_helium_dip_z, atol=tol)

    np.testing.assert_allclose(norm_t2, oatdccd_helium_norm_t2, atol=tol)

    np.testing.assert_allclose(norm_l2, oatdccd_helium_norm_l2, atol=tol)
