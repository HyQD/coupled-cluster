import os
import pytest
import numpy as np

from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from coupled_cluster.ccs.energies import compute_lagrangian_functional
from coupled_cluster.ccs import CCS, TDCCS
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


def test_lagrangian_functional(large_system_ccs):
    t_1, l_1, cs = large_system_ccs

    f = cs.f
    u = cs.u
    o = cs.o
    v = cs.v

    # L <- f^{a}_{i} l^{i}_{a}
    result = np.einsum("ai, ia->", f[v, o], l_1, optimize=True)
    # L <- f^{i}_{a} t^{a}_{i}
    result += np.einsum("ia, ai->", f[o, v], t_1, optimize=True)

    # L <- f^{a}_{b} l^{i}_{a} t^{b}_{i}
    result += np.einsum("ab, ia, bi->", f[v, v], l_1, t_1)

    # L <- -f^{j}_{i} l^{i}_{a} t^{a}_{j}
    result += (-1) * np.einsum("ji, ia, aj->", f[o, o], l_1, t_1, optimize=True)

    # L <- -l^{i}_{a} t^{b}_{j} u^{aj}_{bi}
    result += (-1) * np.einsum(
        "ia, bj, ajbi->", l_1, t_1, u[v, o, v, o], optimize=True
    )

    # L <- -0.5 t^{a}_{j} t^{b}_{i} u^{ij}_{ab}
    result += (-0.5) * np.einsum(
        "aj, bi, ijab->", t_1, t_1, u[o, o, v, v], optimize=True
    )

    # L <- l^{i}_{a} t^{a}_{j} t^{b}_{k} u^{jk}_{bi}
    result += np.einsum(
        "ia, aj, bk, jkbi->", l_1, t_1, t_1, u[o, o, v, o], optimize=True
    )
    # L <- l^{i}_{a} t^{b}_{i} t^{c}_{j} u^{aj}_{bc}
    result += np.einsum(
        "ia, bi, cj, ajbc->", l_1, t_1, t_1, u[v, o, v, v], optimize=True
    )

    # L <- -f^{i}_{a} l^{j}_{b} t^{a}_{j} t^{b}_{i}
    result += (-1) * np.einsum(
        "ia, jb, aj, bi->", f[o, v], l_1, t_1, t_1, optimize=True
    )

    # L <- -l^{i}_{a} t^{a}_{k} t^{b}_{j} t^{c}_{i} u^{jk}_{bc}
    result += (-1) * np.einsum(
        "ia, ak, bj, ci, jkbc->",
        l_1,
        t_1,
        t_1,
        t_1,
        u[o, o, v, v],
        optimize=True,
    )

    energy = compute_lagrangian_functional(f, u, t_1, l_1, o, v, np=np)

    assert abs(result - energy) < 1e-8


@pytest.mark.skip
def test_tdccs():
    omega = 2.873_564_3
    E = 0.1
    laser_duration = 5

    system = construct_pyscf_system_rhf(
        molecule="he 0.0 0.0 0.0", basis="cc-pvdz"
    )

    ccs = CCS(system, verbose=True)
    ccs.compute_ground_state()
    assert abs(ccs.compute_energy() - -2.887_594_831_090_936) < 1e-6

    y0 = ccs.get_amplitudes(get_t_0=True).asarray()

    tdccs = TDCCS(system)

    r = complex_ode(tdccs).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(y0)

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
    td_overlap = np.zeros_like(dip_z)

    td_energies[0] = tdccs.compute_energy(r.t, r.y)
    dip_z[0] = tdccs.compute_one_body_expectation_value(
        r.t, r.y, system.dipole_moment[2]
    ).real
    td_overlap[0] = tdccs.compute_overlap(r.t, y0, r.y)

    for i, t in enumerate(time_points[:-1]):
        r.integrate(r.t + dt)

        if not r.successful():
            break
        td_energies[i + 1] = tdccs.compute_energy(r.t, r.y)
        dip_z[i + 1] = tdccs.compute_one_body_expectation_value(
            r.t, r.y, system.dipole_moment[2]
        ).real
        td_overlap[i + 1] = tdccs.compute_overlap(r.t, y0, r.y)

    np.testing.assert_allclose(
        td_energies.real,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_energies_real_0.1.dat")
        ),
        atol=1e-7,
    )

    np.testing.assert_allclose(
        td_overlap,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_overlap_0.1.dat")
        ),
        atol=1e-7,
    )

    np.testing.assert_allclose(
        dip_z,
        np.loadtxt(
            os.path.join("tests", "dat", "tdcisd_helium_dipole_z_0.1.dat")
        ),
        atol=1e-7,
    )


@pytest.mark.skip
def test_tdccs_phase():
    t_f = 10
    dt = 1e-2
    num_timesteps = int(t_f / dt + 1)
    time_points = np.linspace(0, t_f, num_timesteps)

    polarization_vector = np.zeros(3)
    polarization_vector[2] = 1

    system = construct_pyscf_system_rhf("he", "cc-pvdz")
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            LaserPulse(E=100, omega=2.8735643, td=5, t0=0),
            polarization_vector=polarization_vector,
        )
    )

    ccs = CCS(system, verbose=True)
    ccs.compute_ground_state()
    y0 = ccs.get_amplitudes(get_t_0=True).asarray()

    tdccs = TDCCS(system)

    r = complex_ode(tdccs).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(y0)

    phase = np.zeros(num_timesteps, dtype=np.complex128)
    phase[0] = tdccs.compute_right_phase(r.t, r.y) * tdccs.compute_left_phase(
        r.t, r.y
    )

    i = 0

    for i, t in enumerate(time_points[:-1]):
        r.integrate(r.t + dt)

        if not r.successful():
            phase = phase[:i]
            time_points = time_points[:i]
            # Should break after 88 points to match data set
            break

        phase[i + 1] = tdccs.compute_left_phase(
            r.t, r.y
        ) * tdccs.compute_right_phase(r.t, r.y)

    test_dat_real = np.loadtxt(
        os.path.join("tests", "dat", "he_tdccsd_phase_real.dat")
    )[:, 1]

    test_dat_imag = np.loadtxt(
        os.path.join("tests", "dat", "he_tdccsd_phase_imag.dat")
    )[:, 1]
    print(phase.shape, test_dat_real.shape)

    np.testing.assert_allclose(phase.real, test_dat_real, atol=1e-7)
    np.testing.assert_allclose(phase.imag, test_dat_imag, atol=1e-7)


if __name__ == "__main__":
    test_tdccs_phase()
