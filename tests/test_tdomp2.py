import os
import pytest

import numpy as np
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction

from coupled_cluster.omp2 import OMP2, TDOMP2
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


def test_tdomp2_helium():
    omega = 2.873_564_3
    E = 1
    laser_duration = 5

    system = construct_pyscf_system_rhf(
        molecule="he 0.0 0.0 0.0", basis="cc-pvdz"
    )

    omp2 = OMP2(system, verbose=True)
    omp2.compute_ground_state()
    print(f"EOMP2: {omp2.compute_energy().real}")

    tdomp2 = TDOMP2(system)

    r = complex_ode(tdomp2).set_integrator("GaussIntegrator", s=3, eps=1e-6)
    r.set_initial_value(omp2.get_amplitudes(get_t_0=True).asarray())

    polarization = np.zeros(3)
    polarization[2] = 1
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            LaserPulse(td=laser_duration, omega=omega, E=E),
            polarization_vector=polarization,
        )
    )

    dt = 1e-2
    T = 10
    num_steps = int(T // dt) + 1
    t_stop_laser = int(laser_duration // dt) + 1

    time_points = np.linspace(0, T, num_steps)

    td_energies = np.zeros(len(time_points), dtype=np.complex128)
    dip_z = np.zeros(len(time_points))

    i = 0

    while r.successful() and r.t < T:
        assert abs(time_points[i] - r.t) < dt * 1e-1

        td_energies[i] = tdomp2.compute_energy(r.t, r.y)
        # dip_z[i] = oatdccd.compute_one_body_expectation_value(
        #    r.t, r.y, system.dipole_moment[2]
        # )

        i += 1
        r.integrate(time_points[i])

    td_energies[i] = tdomp2.compute_energy(r.t, r.y)


#     from matplotlib import pyplot as plt
#
#     plt.figure()
#     plt.subplot(211)
#     plt.plot(time_points, td_energies.real)
#     plt.subplot(212)
#     plt.semilogy(time_points, np.abs(td_energies.imag))
#     plt.show()
#
#
# if __name__ == "__main__":
#     test_tdomp2_helium()
