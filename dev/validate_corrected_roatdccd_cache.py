#!/usr/bin/env python
"""Validate split-cache ROATDCCD against a fully uncached control."""

import numpy as np
import rk4_integrator  # noqa: F401 - registers Rk4Integrator with SciPy
from electromagnetic_fields.dipole_fields.delta_pulses import (
    DiscreteDeltaPulse,
)
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction
from scipy.integrate import complex_ode

from corrected_roatdccd import CorrectedROATDCCD, UncachedROATDCCD
from coupled_cluster.rccd import ROACCD


def main():
    dt = 0.2
    # Two complete RK4 steps exercise repeated abscissae within a step and the
    # shared endpoint between steps without making the deliberately uncached
    # control dominate the production calculations.
    t_final = 0.4
    system = construct_pyscf_system_rhf(
        molecule="He 0.0 0.0 0.0",
        basis="cc-pvdz",
        add_spin=False,
        anti_symmetrize=False,
    )
    ground_state = ROACCD(system)
    ground_state.compute_ground_state(tol=1e-10)
    initial_state = ground_state.get_amplitudes(get_t_0=True).asarray()

    polarization = np.zeros(3)
    polarization[2] = 1.0
    field = DiscreteDeltaPulse(F_str=1e-3, dt=dt)
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            field,
            polarization_vector=polarization,
            gauge="length",
        )
    )
    assert system.has_one_body_time_evolution_operator
    assert not system.has_two_body_time_evolution_operator

    cached_solver = CorrectedROATDCCD(system)
    uncached_solver = UncachedROATDCCD(system)
    cached_integrator = complex_ode(cached_solver).set_integrator(
        "Rk4Integrator", dt=dt
    )
    uncached_integrator = complex_ode(uncached_solver).set_integrator(
        "Rk4Integrator", dt=dt
    )
    cached_integrator.set_initial_value(initial_state.copy(), 0.0)
    uncached_integrator.set_initial_value(initial_state.copy(), 0.0)

    max_state_difference = 0.0
    max_energy_difference = 0.0
    max_dipole_difference = 0.0
    num_steps = int(round(t_final / dt))

    for step in range(num_steps):
        next_time = (step + 1) * dt
        cached_integrator.integrate(next_time)
        uncached_integrator.integrate(next_time)
        if (
            not cached_integrator.successful()
            or not uncached_integrator.successful()
        ):
            raise RuntimeError(f"RK4 failed at step {step + 1}")

        max_state_difference = max(
            max_state_difference,
            np.max(np.abs(cached_integrator.y - uncached_integrator.y)),
        )
        cached_energy = cached_solver.compute_energy(
            next_time, cached_integrator.y
        )
        uncached_energy = uncached_solver.compute_energy(
            next_time, uncached_integrator.y
        )
        max_energy_difference = max(
            max_energy_difference, abs(cached_energy - uncached_energy)
        )
        cached_dipole = cached_solver.compute_one_body_expectation_value(
            next_time,
            cached_integrator.y,
            system.dipole_moment[2],
            make_hermitian=False,
        )
        uncached_dipole = uncached_solver.compute_one_body_expectation_value(
            next_time,
            uncached_integrator.y,
            system.dipole_moment[2],
            make_hermitian=False,
        )
        max_dipole_difference = max(
            max_dipole_difference, abs(cached_dipole - uncached_dipole)
        )

    print(f"Maximum state difference:  {max_state_difference:.12e}")
    print(f"Maximum energy difference: {max_energy_difference:.12e}")
    print(f"Maximum dipole difference: {max_dipole_difference:.12e}")
    print(
        "Split-cache lab/transform updates: "
        f"{cached_solver.lab_hamiltonian_updates}/"
        f"{cached_solver.orbital_transform_updates}"
    )
    print(
        "Uncached lab/transform updates:    "
        f"{uncached_solver.lab_hamiltonian_updates}/"
        f"{uncached_solver.orbital_transform_updates}"
    )

    assert max_state_difference < 1e-13
    assert max_energy_difference < 1e-13
    assert max_dipole_difference < 1e-13
    assert cached_solver.lab_hamiltonian_updates < (
        uncached_solver.lab_hamiltonian_updates
    )
    assert cached_solver.orbital_transform_updates == (
        uncached_solver.orbital_transform_updates
    )


if __name__ == "__main__":
    main()
