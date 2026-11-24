#!/usr/bin/env python
"""Check OATDCC Hamiltonian transformations at classical RK4 stages."""

import numpy as np

from coupled_cluster import OACCD, OATDCCD
from quantum_systems import construct_pyscf_system_rhf
from quantum_systems.time_evolution_operators import DipoleFieldInteraction


def main():
    system = construct_pyscf_system_rhf("he", basis="cc-pvdz")
    ground_state = OACCD(system)
    ground_state.compute_ground_state(tol=1e-8)

    polarization = np.zeros(3)
    polarization[2] = 1.0
    system.set_time_evolution_operator(
        DipoleFieldInteraction(
            lambda time: 1.0, polarization_vector=polarization
        )
    )

    assert system.has_one_body_time_evolution_operator
    assert not system.has_two_body_time_evolution_operator

    solver = OATDCCD(system)
    state = ground_state.get_amplitudes(get_t_0=True).asarray()

    # Move away from stationary ground-state orbitals so the RK stages have
    # measurably different C and C_tilde matrices.
    amplitudes = solver.amp_template.from_array(state)
    C = np.eye(system.l, dtype=state.dtype)
    C[0, 1] = 0.05
    amplitudes.C[:] = C
    amplitudes.C_tilde[:] = np.linalg.inv(C)

    records = []

    def checked_rhs(current_state, time, stage):
        current = solver.amp_template.from_array(current_state)
        expected_h_prime = system.transform_one_body_elements(
            system.h_t(time), current.C, current.C_tilde
        )
        expected_u_prime = system.transform_two_body_elements(
            system.u, current.C, current.C_tilde
        )
        expected_f_prime = system.construct_fock_matrix(
            expected_h_prime, expected_u_prime
        )

        derivative = solver(time, current_state)
        stage_errors = (
            np.linalg.norm(solver.h_prime - expected_h_prime),
            np.linalg.norm(solver.u_prime - expected_u_prime),
            np.linalg.norm(solver.f_prime - expected_f_prime),
        )
        records.append((stage, time, *stage_errors))
        return derivative

    def rk4_step(current_state, time, step_size, step):
        k1 = checked_rhs(current_state, time, f"step {step}: k1")
        k2 = checked_rhs(
            current_state + 0.5 * step_size * k1,
            time + 0.5 * step_size,
            f"step {step}: k2",
        )
        k3 = checked_rhs(
            current_state + 0.5 * step_size * k2,
            time + 0.5 * step_size,
            f"step {step}: k3",
        )
        k4 = checked_rhs(
            current_state + step_size * k3,
            time + step_size,
            f"step {step}: k4",
        )
        return current_state + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    dt = 1e-2
    state = rk4_step(state, 0.0, dt, 1)
    rk4_step(state, dt, dt, 2)

    by_stage = {}
    for stage, time, h_error, u_error, f_error in records:
        by_stage[stage] = max(h_error, u_error, f_error)
        print(
            f"{stage:12s} t={time:.7f}  "
            f"h_error={h_error:.6e}  u_error={u_error:.6e}  "
            f"f_error={f_error:.6e}"
        )

    assert by_stage["step 1: k2"] < 1e-12
    assert by_stage["step 1: k3"] < 1e-12
    assert by_stage["step 1: k4"] < 1e-12
    assert by_stage["step 2: k1"] < 1e-12
    assert by_stage["step 2: k3"] < 1e-12


if __name__ == "__main__":
    main()
