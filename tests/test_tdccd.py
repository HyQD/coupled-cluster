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
    r = complex_ode(tdccd).set_integrator("dopri5") # "GaussIntegrator")
    r.set_initial_value(y0)
    t0,t2,l2 = ccd.get_amplitudes(get_t_0=True).unpack()
    print(t0.shape, t2.shape, l2.shape)
    print(tdccd.system.m, tdccd.system.n)
    rho = tdccd.compute_particle_density(y0)

    np.testing.assert_allclose(
        rho, tdccd_zanghellini_ground_state_particle_density, atol=1e-10
    )

    # tdccd.set_initial_conditions()
    time_points, dt = np.linspace(
        time_params["t_start"],
        time_params["t_end"],
        time_params["num_timesteps"],
        retstep=True
    )

    psi_overlap = np.zeros(time_params["num_timesteps"])
    td_energies = np.zeros(time_params["num_timesteps"])

    psi_overlap[0] = tdccd.compute_overlap(y0, y0).real
    td_energies[0] = tdccd.compute_energy(y0).real

    for i, t in enumerate(time_points[:-1]):
        r.integrate(r.t + dt)

        if not r.successful():
            break

        psi_overlap[i + 1] = tdccd.compute_overlap(y0, r.y).real
        td_energies[i + 1] = tdccd.compute_energy(r.y).real

    np.testing.assert_allclose(
        psi_overlap, tdccd_zanghellini_psi_overlap, atol=1e-10
    )

    np.testing.assert_allclose(
        td_energies, tdccd_zanghellini_td_energies, atol=1e-10
    )
