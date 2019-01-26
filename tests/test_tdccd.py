import pytest
import numpy as np

from coupled_cluster.ccd import TDCCD, CoupledClusterDoubles


def test_time_dependent_observables(
    zanghellini_system,
    zanghellini_ground_state_energy,
    zanghellini_ground_state_particle_density,
    zanghellini_psi_overlap,
    zanghellini_td_energies,
    t_kwargs,
    l_kwargs,
    time_params,
):

    tdccd = TDCCD(CoupledClusterDoubles, zanghellini_system, np=np)
    tdccd.compute_ground_state(t_kwargs=t_kwargs, l_kwargs=l_kwargs)

    assert (
        abs(
            zanghellini_ground_state_energy
            - tdccd.compute_ground_state_energy()
        )
        < t_kwargs["tol"]
    )

    rho = tdccd.compute_ground_state_particle_density()

    np.testing.assert_allclose(
        rho, zanghellini_ground_state_particle_density, atol=1e-10
    )

    tdccd.set_initial_conditions()
    time_points = np.linspace(
        time_params["t_start"],
        time_params["t_end"],
        time_params["num_timesteps"],
    )

    psi_overlap = np.zeros(time_params["num_timesteps"])
    td_energies = np.zeros(time_params["num_timesteps"])

    psi_overlap[0] = tdccd.compute_time_dependent_overlap().real
    td_energies[0] = tdccd.compute_energy().real

    for i, amp in enumerate(tdccd.solve(time_points)):
        psi_overlap[i + 1] = tdccd.compute_time_dependent_overlap().real
        td_energies[i + 1] = tdccd.compute_energy().real

    np.testing.assert_allclose(psi_overlap, zanghellini_psi_overlap, atol=1e-10)

    np.testing.assert_allclose(td_energies, zanghellini_td_energies, atol=1e-10)
