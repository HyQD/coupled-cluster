import pytest
import numpy as np

from coupled_cluster.ccd import CCD, TDCCD
from coupled_cluster.mix import AlphaMixer, DIIS
from scipy.integrate import complex_ode
from gauss_integrator import GaussIntegrator


def test_time_dependent_observables(
    zanghellini_system,
    tdccd_zanghellini_ground_state_energy,
    tdccd_zanghellini_ground_state_particle_density,
    tdccd_zanghellini_psi_overlap,
    tdccd_zanghellini_td_energies,
    t_kwargs,
    l_kwargs,
    time_params,
):

    ccd = CCD(zanghellini_system, mixer=AlphaMixer)
    ccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)
    y0 = ccd.get_amplitudes(get_t_0=True).asarray()

    tdccd = TDCCD(zanghellini_system)
    r = complex_ode(tdccd).set_integrator("dopri5")
    r.set_initial_value(y0)

    rho = tdccd.compute_particle_density(0, y0)

    np.testing.assert_allclose(
        rho, tdccd_zanghellini_ground_state_particle_density, atol=1e-10
    )

    time_points, dt = np.linspace(
        time_params["t_start"],
        time_params["t_end"],
        time_params["num_timesteps"],
        retstep=True,
    )

    psi_overlap = np.zeros(time_params["num_timesteps"])
    td_energies = np.zeros(time_params["num_timesteps"])

    for i, t in enumerate(time_points):
        assert abs(time_points[i] - r.t) < 1e-4
        if not r.successful():
            break

        psi_overlap[i] = tdccd.compute_overlap(r.t, y0, r.y).real
        td_energies[i] = tdccd.compute_energy(r.t, r.y).real
        man_energy = tdccd.compute_one_body_expectation_value(
            r.t, r.y, zanghellini_system.h_t(r.t)
        ) + 0.5 * tdccd.compute_two_body_expectation_value(
            r.t, r.y, zanghellini_system.u
        )
        assert abs(td_energies[i] - man_energy.real) < 1e-12

        # avoid extra integration step
        if t != time_points[-1]:
            r.integrate(r.t + dt)

    np.testing.assert_allclose(
        psi_overlap, tdccd_zanghellini_psi_overlap, atol=1e-06
    )

    np.testing.assert_allclose(
        td_energies, tdccd_zanghellini_td_energies, atol=1e-06
    )
